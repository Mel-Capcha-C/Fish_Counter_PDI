"""Tracking por flujo de costo mínimo con capacidad (EXPERIMENTAL, alternativa a temporal.py).

Todas las detecciones del video se explican a la vez con el menor costo total:
  * una trayectoria = una unidad de flujo que entra, salta de detección en detección y sale;
  * una pieza puede llevar 0 unidades (falsa o sobre-separada), 1, 2 o 3 (grupo no separado);
  * cada salto usa la velocidad del salto anterior (variables por pares y ternas).
Todos los costos son -log(probabilidad) con las mismas unidades, así que no hay
pesos que ajustar salvo las probabilidades de la tabla ``Evidencia``.
Se resuelve con programación entera (HiGHS, scipy.optimize.milp).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from segmentacion import Mancha, caja
from separacion import separar, solidez
from tracking import BORDE, H, W, Movimiento, _cdf, _lognormal, centro, desplazamiento

# columnas de una detección
AREA, RAZON, SOL, TOCA, AMANCHA = 4, 5, 6, 7, 8


def detecciones_flujo(ms: list[Mancha]) -> np.ndarray:
    """caja(4), área, razón, solidez de la mancha, toca borde, área de la caja de la mancha (1280x720)."""
    L = []
    for m in ms:
        s = solidez(m)
        h, w = m.mask.shape
        for x, y, a, b, ar, q in separar(m):
            L.append([*caja(m.x + x, m.y + y, a, b), ar, q, s, float(any(m.toca)), w * h * 1.11])
    return np.array(L, float).reshape(-1, 9)


def _tramo(v, cortes):
    return int(np.searchsorted(cortes, v))


@dataclass
class Evidencia:
    """p_falsa: P(la pieza sobra); p_doble: P(la pieza oculta otro pez). Tablas por tramos."""
    razon: list = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8])        # piezas separadas
    sol: list = field(default_factory=lambda: [0.75, 0.82, 0.88, 0.94])      # manchas enteras
    area: list = field(default_factory=lambda: [900.0])                       # enteras: pequeñas
    falsa_sep: list = field(default_factory=lambda: [.05] * 5)
    falsa_ent: list = field(default_factory=lambda: [[.01, .01], [.01, .01]])  # [toca][área]
    doble_sep: list = field(default_factory=lambda: [.05] * 5)                # por solidez
    doble_ent: list = field(default_factory=lambda: [.05] * 5)

    def p_falsa(self, d):
        if d[RAZON] > 0:
            return self.falsa_sep[_tramo(d[RAZON], self.razon)]
        return self.falsa_ent[int(d[TOCA])][_tramo(d[AREA], self.area)]

    def p_doble(self, d):
        t = self.doble_sep if d[RAZON] > 0 else self.doble_ent
        return t[_tramo(d[SOL], self.sol)]

    def guardar(self, p):
        with open(p, "w") as fh:
            json.dump(asdict(self), fh, indent=1)

    @classmethod
    def cargar(cls, p):
        with open(p) as fh:
            return cls(**json.load(fh))


def aprender_evidencia(manchas_reales) -> Evidencia:
    """manchas_reales: [(piezas (n, 9) de una mancha, número real de peces)] de los videos de ajuste."""
    e = Evidencia()
    fs, ds = np.zeros((5, 2)), np.zeros((5, 2))
    fe, de = np.zeros((2, 2, 2)), np.zeros((5, 2))
    for p, n in manchas_reales:
        k = len(p)
        if k > 1:
            q = _tramo(p[:, RAZON].max(), e.razon)
            fs[q] += [max(0, k - n), k]
            ds[_tramo(p[0, SOL], e.sol)] += [max(0, n - k), k]
        else:
            fe[int(p[0, TOCA]), _tramo(p[0, AREA], e.area)] += [n == 0, 1]
            if n:
                de[_tramo(p[0, SOL], e.sol)] += [n - 1, 1]
    s = lambda t: np.clip((t[..., 0] + .5) / (t[..., 1] + 20), .0005, .6).round(4).tolist()
    e.falsa_sep, e.falsa_ent, e.doble_sep, e.doble_ent = s(fs), s(fe), s(ds), s(de)
    return e


class _Modelo:
    def __init__(self, mov, ev, dets):
        self.m, self.ev = mov, ev
        self.n, self.k, self.d = [], [], []
        for k in sorted(dets):
            for d in dets[k]:
                self.k.append(k)
                self.d.append(d)
        self.porcuadro = {}
        for i, k in enumerate(self.k):
            self.porcuadro.setdefault(k, []).append(i)
        self.pos = [centro(d) for d in self.d]

    def entrada(self, i):
        m, (x, y), k = self.m, self.pos[i], self.k[i]
        tasa = max(.3, np.mean([len(self.porcuadro.get(q, ())) for q in range(k - 2, k + 3)]) / 2.8)
        v0 = m.salto(0, y)
        s = 1 - _cdf((x + 30 - v0) / m.sx)
        return -np.log(tasa * max(s, m.p_medio) / (v0 * H))

    def _visible(self, x, dx, sx):
        return min(max(_cdf((W + 40 - x - dx) / sx), .01), .995)

    def salida(self, x, dx, sx):
        pv = self._visible(x, dx, sx)
        return -np.log(1 - pv + self.m.p_perdido * pv)

    def ancho(self, j):
        """Desviación extra si la pieza puede ocultar a otro pez (su centro no es el del pez)."""
        d = self.d[j]
        return ((d[2] - d[0]) / 4, (d[3] - d[1]) / 4) if self.ev.p_doble(d) > .05 else (0.0, 0.0)

    def enlace(self, i, j, res=None):
        """Costo de i->j y residuo (rx, ry). res = residuo del salto anterior, si existe."""
        m, a, b = self.m, self.d[i], self.d[j]
        pasos = self.k[j] - self.k[i]
        x, y = self.pos[i]
        dx = dy = 0.0
        for p in range(1, pasos + 1):
            dx += m.salto(x + dx, y + dy) + (m.rho ** p * res[0] if res else 0.0)
            dy += m.my + (m.rhoy ** p * res[1] if res else 0.0)
        esc = pasos ** .75
        sx, sy = ((m.sx2, m.sy2) if res else (m.sx, m.sy))
        sx, sy = sx * esc, sy * esc
        pv = self._visible(x, dx, sx)
        d, inc = desplazamiento(a, b)
        ddx, ddy = d - dx, self.pos[j][1] - y - dy
        ex, ey = self.ancho(j)
        fx, fy = self.ancho(i)
        sx, sy = np.hypot(np.hypot(sx * inc, ex), fx), np.hypot(np.hypot(sy * inc, ey), fy)
        if abs(ddx) > 4.5 * sx or abs(ddy) > 5 * sy:
            return None
        c = -(_lognormal(ddx, sx) + _lognormal(ddy, sy))
        c -= np.log(pv * (1 - m.p_perdido)) if pasos == 1 else np.log(m.p_perdido * pv)
        if inc == 1.0 and a[0] > BORDE and not (ex or fx):
            for lo, hi, sd, mu, sp in ((0, 2, m.slw, m.pob[0], m.pob[1]), (1, 3, m.slh, m.pob[2], m.pob[3])):
                lb, la = np.log(max(b[hi] - b[lo], 1.0)), np.log(max(a[hi] - a[lo], 1.0))
                c -= _lognormal(lb - la, sd) - _lognormal(lb - mu, sp)
        r = None
        if pasos == 1:
            r = (float(np.clip(d - m.salto(x, y), -3 * m.sx, 3 * m.sx)),
                 float(np.clip(self.pos[j][1] - y - m.my, -3 * m.sy, 3 * m.sy)))
        return float(c), r, dx


def resolver(dets: dict, mov: Movimiento, ev: Evidencia, tiempo: float = 120.0):
    """-> (trayectorias = [[(cuadro, índice en el cuadro), ...]], unidades por detección, estado)."""
    M = _Modelo(mov, ev, dets)
    n = len(M.d)
    if n == 0:
        return [], {}, "vacío"
    ini = {k: v[0] for k, v in M.porcuadro.items()}
    pares, suc = {}, {}
    for i in range(n):
        for paso in (1, 2):
            for j in M.porcuadro.get(M.k[i] + paso, ()):
                e = M.enlace(i, j)
                if e is not None:
                    pares[(i, j)] = (len(pares), *e)
                    suc.setdefault(i, []).append(j)
    P = len(pares)
    lp = list(pares)
    # variables: s_i | a_p | z_p | t_q | w1_i w2_i w3_i
    tri = []
    for (i, j), (p, _, r, _) in pares.items():
        for k in suc.get(j, ()):
            e = M.enlace(j, k, r)
            if e is not None:
                tri.append((p, pares[(j, k)][0], e[0]))
    T = len(tri)
    oS, oA, oZ, oT, oW = 0, n, n + P, n + 2 * P, n + 2 * P + T
    N = oW + 3 * n
    c = np.zeros(N)
    fijo = 0.0
    for i in range(n):
        d, (x, y) = M.d[i], M.pos[i]
        ent = M.entrada(i)
        c[oS + i] = ent + M.salida(x, mov.salto(x, y), mov.sx)
        pf, pd = ev.p_falsa(d), ev.p_doble(d)
        zona = W * H if d[RAZON] == 0 else d[AMANCHA]
        c0 = -np.log(pf / (1 - pf)) + np.log(zona)          # explicarla como falsa
        c[oW + 3 * i] = -c0
        fijo += c0
        caj = max((d[2] - d[0]) * (d[3] - d[1]), 400.0)
        c2 = -np.log(pd / (1 - pd)) - np.log(caj)           # segundo pez oculto
        c[oW + 3 * i + 1], c[oW + 3 * i + 2] = c2, c2 + .7
    for (i, j), (p, ce, r, _) in pares.items():
        c[oA + p] = M.entrada(i) + ce
        x, y = M.pos[j]
        if r:
            c[oZ + p] = M.salida(x, mov.salto(x, y) + mov.rho * r[0], mov.sx2)
        else:
            c[oZ + p] = M.salida(x, mov.salto(x, y), mov.sx)
    for q, (_, _, ct) in enumerate(tri):
        c[oT + q] = ct
    f, col, val = [], [], []

    def pon(a, b, v):
        f.append(a); col.append(b); val.append(v)

    # conservación por par: a_p + sum(t entra) - z_p - sum(t sale) = 0
    for p in range(P):
        pon(p, oA + p, 1); pon(p, oZ + p, -1)
    for q, (p1, p2, _) in enumerate(tri):
        pon(p2, oT + q, 1); pon(p1, oT + q, -1)
    # unidades por detección: s + sum(a_jk) + sum(y_ij) = w1 + w2 + w3
    for i in range(n):
        pon(P + i, oS + i, 1)
        for b in range(3):
            pon(P + i, oW + 3 * i + b, -1)
    for p, (i, j) in enumerate(lp):
        pon(P + i, oA + p, 1)          # empieza en i
        pon(P + j, oA + p, 1)          # llega a j
    for q, (_, p2, _) in enumerate(tri):
        pon(P + lp[p2][1], oT + q, 1)
    A = csr_matrix((val, (f, col)), shape=(P + n, N))
    r = milp(c, constraints=LinearConstraint(A, 0, 0), integrality=np.ones(N),
             bounds=Bounds(0, np.r_[np.full(n, 1), np.full(2 * P + T, 2), np.ones(3 * n)]),
             options=dict(time_limit=tiempo, mip_rel_gap=1e-4))
    if r.x is None:
        return [], {}, r.message
    x = np.round(r.x).astype(int)
    sale = {}
    for q, (p1, p2, _) in enumerate(tri):
        if x[oT + q]:
            sale.setdefault(p1, []).extend([p2] * x[oT + q])
    fin = {p: x[oZ + p] for p in range(P)}
    pos = lambda i: (M.k[i], i - ini[M.k[i]])
    out = []
    for p in range(P):
        for _ in range(x[oA + p]):
            cam, q = [lp[p][0], lp[p][1]], p
            while True:
                if sale.get(q):
                    q = sale[q].pop()
                    cam.append(lp[q][1])
                else:
                    fin[q] -= 1
                    break
            out.append([pos(i) for i in cam])
    for i in range(n):
        if x[oS + i]:
            out.append([pos(i)])
    u = {pos(i): int(x[oW + 3 * i:oW + 3 * i + 3].sum()) for i in range(n)}
    return out, u, r.message
