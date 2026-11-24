"""Tracking por asignación óptima con modelo de movimiento (sin IA).

1. MOVIMIENTO  El salto esperado depende de la posición (el agua acelera) y cada
   pez conserva su desviación: tras su primer salto la predicción es más precisa.
2. ENTRADA     Un pez sólo puede aparecer a menos de un salto del borde izquierdo.
3. SALIDA      Un pez sólo puede desaparecer a menos de un salto del derecho.
4. ASIGNACIÓN  Para cada pareja se compara "es el mismo pez" contra "uno salió y otro
   entró" (razón de verosimilitudes). El algoritmo húngaro maximiza la suma, y
   las parejas menos probables que entrada+salida se rechazan. No hay tolerancia fija.
5. CONTEO      Cada trayectoria vale 1, sin importar el tamaño del pez.

Los parámetros salen de pares de peces aislados, NO de los totales por video.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import erf

import numpy as np
from scipy.optimize import linear_sum_assignment

W, H = 1280, 720
BORDE = 4.0
VACIO = -1e6


@dataclass
class Movimiento:
    coef: list = field(default_factory=lambda: [484.0, 0, 0, 0, 0])   # 1, u, u2, w, w2
    sx: float = 60.0          # desviación del primer salto
    sx2: float = 33.0         # desviación cuando ya se conoce el salto anterior
    rho: float = 0.86
    my: float = 9.0
    sy: float = 22.0
    sy2: float = 14.0
    rhoy: float = 0.77
    slw: float = 0.25         # cambio de log-ancho y log-alto de la caja
    slh: float = 0.33
    pob: list = field(default_factory=lambda: [4.9, .35, 4.2, .45])   # talla de la población
    p_perdido: float = 0.02   # probabilidad de no detectar un pez visible
    p_medio: float = 0.004    # densidad relativa de apariciones fuera de la entrada

    def salto(self, x, y):
        u, w = np.clip(x, 0, 700) / 1000.0 - .3, y / 720.0 - .5   # x > 700: sin datos no censurados
        c = self.coef
        return c[0] + c[1] * u + c[2] * u * u + c[3] * w + c[4] * w * w

    def guardar(self, p):
        with open(p, "w") as fh:
            json.dump(asdict(self), fh, indent=1)

    @classmethod
    def cargar(cls, p):
        with open(p) as fh:
            return cls(**json.load(fh))


def _cdf(z):
    return 0.5 * (1 + erf(z / 1.41421356))


def _lognormal(d, s):
    """Log-densidad normal con colas pesadas (5 % de casos con 3 veces la desviación)."""
    def g(q):
        return np.exp(-.5 * (d / q) ** 2) / (q * 2.5066)
    return float(np.log(.95 * g(s) + .05 * g(3 * s) + 1e-12))


def centro(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def desplazamiento(a, b):
    """Salto en x robusto a cortes: usa el borde de la caja que sí se ve en ambos cuadros."""
    izq, der = a[0] < BORDE, b[2] > W - BORDE
    if izq and der:
        return (b[0] + b[2] - a[0] - a[2]) / 2, 2.0
    if izq:
        return b[2] - a[2], 1.1
    if der:
        return b[0] - a[0], 1.1
    return (b[0] + b[2] - a[0] - a[2]) / 2, 1.0


def _mad(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def ajustar_movimiento(pares, ternas) -> Movimiento:
    """pares: (caja_t, caja_t+1) aislados; ternas: (caja_t, t+1, t+2) aisladas."""
    m = Movimiento()
    P = [(a, b) for a, b in pares if centro(a)[0] < 560]          # evita censura por salida
    dx = np.array([desplazamiento(a, b)[0] for a, b in P])
    c = np.array([centro(a) for a, _ in P])
    u, w = np.clip(c[:, 0], 0, 700) / 1000 - .3, c[:, 1] / 720 - .5
    A = np.stack([np.ones_like(u), u, u * u, w, w * w], 1)
    ok = np.ones(len(dx), bool)
    for _ in range(4):
        m.coef = np.linalg.lstsq(A[ok], dx[ok], rcond=None)[0].tolist()
        r = dx - A @ np.array(m.coef)
        m.sx = float(1.4826 * np.median(np.abs(r[ok])))
        ok = np.abs(r) < 3.5 * m.sx
    dy = np.array([centro(b)[1] - centro(a)[1] for a, b in P])
    m.my, m.sy = float(np.median(dy)), _mad(dy)

    def ex(a, b):
        return desplazamiento(a, b)[0] - m.salto(*centro(a))

    def ey(a, b):
        return centro(b)[1] - centro(a)[1] - m.my

    r1 = np.array([ex(a, b) for a, b, _ in ternas])
    r2 = np.array([ex(b, d) for _, b, d in ternas])
    g1 = np.array([ey(a, b) for a, b, _ in ternas])
    g2 = np.array([ey(b, d) for _, b, d in ternas])
    k = (np.abs(r1) < 4 * m.sx) & (np.abs(r2) < 4 * m.sx)
    m.rho = float(np.clip((r1[k] @ r2[k]) / (r1[k] @ r1[k]), 0, 1))
    m.sx2 = float(1.4826 * np.median(np.abs(r2[k] - m.rho * r1[k])))
    m.rhoy = float(np.clip((g1[k] @ g2[k]) / (g1[k] @ g1[k]), 0, 1))
    m.sy2 = float(1.4826 * np.median(np.abs(g2[k] - m.rhoy * g1[k])))
    ent = [(a, b) for a, b in P if a[0] > BORDE and b[2] < W - BORDE]
    m.slw = _mad(np.array([np.log((b[2] - b[0]) / (a[2] - a[0])) for a, b in ent]))
    m.slh = _mad(np.array([np.log((b[3] - b[1]) / (a[3] - a[1])) for a, b in ent]))
    tw = np.log([b[2] - b[0] for _, b in ent])
    th = np.log([b[3] - b[1] for _, b in ent])
    m.pob = [float(v) for v in (tw.mean(), tw.std(), th.mean(), th.std())]
    return m


class Pista:
    __slots__ = ("id", "k", "b", "rx", "ry", "perdida", "n")

    def __init__(self, i, k, b):
        self.id, self.k, self.b = i, k, b
        self.rx = self.ry = None
        self.perdida, self.n = 0, 1


class Rastreador:
    def __init__(self, mov: Movimiento, max_perdidos: int = 1):
        self.m, self.maxp = mov, max_perdidos
        self.act: list[Pista] = []
        self.conteo, self.nid = 0, 0          # conteo = todas las trayectorias
        self.confirmadas = 0                  # vistas en 2 cuadros o mas
        self.tasa = 1.0                       # peces nuevos por cuadro (media móvil)
        self.nac, self.mue = [], []           # (cuadro, x) para control de calidad

    def _prediccion(self, t: Pista, pasos: int):
        m, (x, y) = self.m, centro(t.b)
        con = t.rx is not None
        dx = dy = 0.0
        for p in range(1, pasos + 1):      # el residuo decae rho por CADA paso
            dx += m.salto(x + dx, y + dy) + (m.rho ** p * t.rx if con else 0.0)
            dy += m.my + (m.rhoy ** p * t.ry if con else 0.0)
        s = pasos ** .75
        return dx, dy, (m.sx2 if con else m.sx) * s, (m.sy2 if con else m.sy) * s

    def _p_visible(self, t, dx, sx):
        """Probabilidad de que el pez siga a la vista en el cuadro siguiente."""
        return min(max(_cdf((W + 40 - centro(t.b)[0] - dx) / sx), .01), .995)

    def _densidad_entrada(self, b):
        m, (x, y) = self.m, centro(b)
        v0 = m.salto(0, y)
        s = 1 - _cdf((x + 30 - v0) / m.sx)
        return float(np.log(max(self.tasa, .2) * max(s, m.p_medio) / (v0 * H)))

    def _puntaje(self, t, pasos, b, entrada):
        m = self.m
        dx, dy, sx, sy = self._prediccion(t, pasos)
        pv = self._p_visible(t, dx, sx)
        d, inc = desplazamiento(t.b, b)
        ddx, ddy = d - dx, centro(b)[1] - centro(t.b)[1] - dy
        if abs(ddx) > 5 * sx * inc or abs(ddy) > 6 * sy:
            return VACIO
        if pasos == 1:
            s = np.log(pv * (1 - m.p_perdido)) - np.log(1 - pv + m.p_perdido * pv)
        else:
            s = np.log(m.p_perdido * pv) - np.log(.5)
        s += _lognormal(ddx, sx * inc) + _lognormal(ddy, sy * inc) - entrada
        if inc == 1.0 and t.b[0] > BORDE:      # la talla sólo ayuda si ninguna caja está cortada
            for lo, hi, sd, mu, sp in ((0, 2, m.slw, m.pob[0], m.pob[1]),
                                       (1, 3, m.slh, m.pob[2], m.pob[3])):
                lb, la = np.log(max(b[hi] - b[lo], 1.0)), np.log(max(t.b[hi] - t.b[lo], 1.0))
                s += _lognormal(lb - la, sd) - _lognormal(lb - mu, sp)
        return float(s)

    def paso(self, k: int, cajas: np.ndarray) -> list[int]:
        m = self.m
        cand = [(t, 1 + t.perdida) for t in self.act]
        n, q = len(cand), len(cajas)
        ids = [-1] * q
        entrada = [self._densidad_entrada(b) for b in cajas]
        G = np.full((n, q), VACIO)
        for i, (t, pasos) in enumerate(cand):
            for j, b in enumerate(cajas):
                G[i, j] = self._puntaje(t, pasos, b, entrada[j])
        usado = set()
        if n and q:
            for i, j in zip(*linear_sum_assignment(-np.maximum(G, 0))):
                if G[i, j] <= 0:
                    continue
                t, pasos = cand[i]
                d, _ = desplazamiento(t.b, cajas[j])
                if pasos == 1:
                    x, y = centro(t.b)
                    t.rx = float(np.clip(d - m.salto(x, y), -3 * m.sx, 3 * m.sx))
                    t.ry = float(np.clip(centro(cajas[j])[1] - y - m.my, -3 * m.sy, 3 * m.sy))
                t.k, t.b, t.perdida, t.n = k, cajas[j], 0, t.n + 1
                self.confirmadas += t.n == 2
                ids[j] = t.id
                usado.add(i)
        sig = []
        for i, (t, pasos) in enumerate(cand):
            if i in usado:
                sig.append(t)
                continue
            dx, _, sx, _ = self._prediccion(t, pasos)
            if t.perdida < self.maxp and self._p_visible(t, dx, sx) > .3:
                t.perdida += 1
                sig.append(t)
            else:
                self.mue.append((k, centro(t.b)[0]))
        nuevos = 0
        for j, b in enumerate(cajas):
            if ids[j] < 0:
                sig.append(Pista(self.nid, k, b))
                ids[j] = self.nid
                self.nid += 1
                self.conteo += 1
                nuevos += 1
                self.nac.append((k, centro(b)[0]))
        self.act = sig
        self.tasa = .7 * self.tasa + .3 * max(nuevos, q / 2.8)
        return ids


def contar(cajas_por_cuadro: dict, mov: Movimiento, **kw) -> Rastreador:
    r = Rastreador(mov, **kw)
    r.ids = {}
    for k in sorted(cajas_por_cuadro):
        r.ids[k] = r.paso(k, cajas_por_cuadro[k])
    return r
