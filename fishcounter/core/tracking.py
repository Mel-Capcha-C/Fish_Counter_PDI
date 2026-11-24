"""Tracking por asignación óptima con modelo de movimiento AR(1).

Para cada pareja pista-detección se compara "es el mismo pez" contra "uno salió y otro entró"
(razón de verosimilitudes); el algoritmo húngaro maximiza la suma. Coordenadas: 1280x720.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf

import numpy as np
from scipy.optimize import linear_sum_assignment

from .perfil import Perfil

W, H = 1280, 720
VACIO = -1e6


def _cdf(z):
    return 0.5 * (1 + erf(z / 1.41421356))


def _log_colas(d, s):
    """Log-densidad normal con colas pesadas (5 % de casos con 3 veces la desviación)."""
    def g(q):
        return np.exp(-.5 * (d / q) ** 2) / (q * 2.5066)
    return float(np.log(.95 * g(s) + .05 * g(3 * s) + 1e-12))


def centro(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def desplazamiento(a, b, borde):
    """Salto en x robusto a cortes: usa el borde de la caja que sí se ve en ambos cuadros."""
    izq, der = a[0] < borde, b[2] > W - borde
    if izq and der:
        return (b[0] + b[2] - a[0] - a[2]) / 2, 2.0
    if izq:
        return b[2] - a[2], 1.1
    if der:
        return b[0] - a[0], 1.1
    return (b[0] + b[2] - a[0] - a[2]) / 2, 1.0


class Pista:
    __slots__ = ("id", "k", "b", "rx", "ry", "perdida", "n")

    def __init__(self, i, k, b):
        self.id, self.k, self.b = i, k, b
        self.rx = self.ry = None
        self.perdida, self.n = 0, 1


@dataclass
class Paso:
    """Qué hizo el tracking en un cuadro (para las vistas)."""
    ids: list = field(default_factory=list)                # id de cada detección
    predicciones: list = field(default_factory=list)       # (id, x previa, y previa, x, y, sx, sy)
    parejas: list = field(default_factory=list)            # (id, índice de detección, puntaje)
    nuevas: list = field(default_factory=list)             # ids
    perdidas: list = field(default_factory=list)           # ids que siguen vivas sin detección
    terminadas: list = field(default_factory=list)         # ids


class Rastreador:
    def __init__(self, p: Perfil):
        self.p = p
        self.act: list[Pista] = []
        self.nid = 0
        self.confirmadas = 0                  # vistas en 2 cuadros o más
        self.tasa = 1.0                       # peces nuevos por cuadro (media móvil)

    def _prediccion(self, t: Pista, pasos: int):
        p, (x, y) = self.p, centro(t.b)
        con = t.rx is not None
        dx = dy = 0.0
        for i in range(1, pasos + 1):         # el residuo decae rho por CADA paso
            dx += p.salto(x + dx, y + dy) + (p.rho ** i * t.rx if con else 0.0)
            dy += p.my + (p.rhoy ** i * t.ry if con else 0.0)
        s = pasos ** .75
        return dx, dy, (p.sx2 if con else p.sx) * s, (p.sy2 if con else p.sy) * s

    def _p_visible(self, t, dx, sx):
        return min(max(_cdf((W + self.p.margen_salida - centro(t.b)[0] - dx) / sx), .01), .995)

    def _densidad_entrada(self, b):
        p, (x, y) = self.p, centro(b)
        v0 = p.salto(0, y)
        s = 1 - _cdf((x + p.margen_entrada - v0) / p.sx)
        return float(np.log(max(self.tasa, .2) * max(s, p.p_medio) / (v0 * H)))

    def _puntaje(self, t, pasos, b, entrada, pred):
        p = self.p
        dx, dy, sx, sy = pred
        pv = self._p_visible(t, dx, sx)
        d, inc = desplazamiento(t.b, b, p.borde)
        ddx, ddy = d - dx, centro(b)[1] - centro(t.b)[1] - dy
        if abs(ddx) > 5 * sx * inc or abs(ddy) > 6 * sy:
            return VACIO
        if pasos == 1:
            s = np.log(pv * (1 - p.p_perdido)) - np.log(1 - pv + p.p_perdido * pv)
        else:
            s = np.log(p.p_perdido * pv) - np.log(.5)
        s += _log_colas(ddx, sx * inc) + _log_colas(ddy, sy * inc) - entrada
        if inc == 1.0 and t.b[0] > p.borde:      # la talla sólo ayuda si ninguna caja está cortada
            for lo, hi, sd, mu, sp in ((0, 2, p.slw, p.pob_w, p.pob_sw), (1, 3, p.slh, p.pob_h, p.pob_sh)):
                lb, la = np.log(max(b[hi] - b[lo], 1.0)), np.log(max(t.b[hi] - t.b[lo], 1.0))
                s += _log_colas(lb - la, sd) - _log_colas(lb - mu, sp)
        return float(s)

    def paso(self, k: int, cajas: np.ndarray) -> Paso:
        p, res = self.p, Paso()
        cand = [(t, 1 + t.perdida) for t in self.act]
        n, q = len(cand), len(cajas)
        ids = [-1] * q
        entrada = [self._densidad_entrada(b) for b in cajas]
        G = np.full((n, q), VACIO)
        for i, (t, pasos) in enumerate(cand):
            pred = self._prediccion(t, pasos)
            x, y = centro(t.b)
            res.predicciones.append((t.id, x, y, x + pred[0], y + pred[1], pred[2], pred[3]))
            for j, b in enumerate(cajas):
                G[i, j] = self._puntaje(t, pasos, b, entrada[j], pred)
        usado = set()
        if n and q:
            for i, j in zip(*linear_sum_assignment(-np.maximum(G, 0))):
                if G[i, j] <= 0:
                    continue
                t, pasos = cand[i]
                d, _ = desplazamiento(t.b, cajas[j], p.borde)
                if pasos == 1:
                    x, y = centro(t.b)
                    t.rx = float(np.clip(d - p.salto(x, y), -3 * p.sx, 3 * p.sx))
                    t.ry = float(np.clip(centro(cajas[j])[1] - y - p.my, -3 * p.sy, 3 * p.sy))
                t.k, t.b, t.perdida, t.n = k, cajas[j], 0, t.n + 1
                self.confirmadas += t.n == 2
                ids[j] = t.id
                res.parejas.append((t.id, int(j), float(G[i, j])))
                usado.add(i)
        sig = []
        for i, (t, pasos) in enumerate(cand):
            if i in usado:
                sig.append(t)
                continue
            dx, _, sx, _ = self._prediccion(t, pasos)
            if t.perdida < p.max_perdidos and self._p_visible(t, dx, sx) > .3:
                t.perdida += 1
                sig.append(t)
                res.perdidas.append(t.id)
            else:
                res.terminadas.append(t.id)
        nuevos = 0
        for j, b in enumerate(cajas):
            if ids[j] < 0:
                sig.append(Pista(self.nid, k, b))
                ids[j] = self.nid
                res.nuevas.append(self.nid)
                self.nid += 1
                nuevos += 1
        self.act = sig
        self.tasa = .7 * self.tasa + .3 * max(nuevos, q / p.vistas_por_pez)
        res.ids = ids
        return res
