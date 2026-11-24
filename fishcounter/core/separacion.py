"""Separación de peces en contacto dentro de una mancha.

1. Solidez            sólo se analizan las manchas poco convexas
2. Relleno de huecos  tapa brillos cerrados y pequeños (crean cuellos falsos)
3. Distancia          euclidiana, promediada sobre tres umbrales
4. Persistencia       dos cimas son dos peces si el paso < razón x cima menor
5. Crecimiento        cada píxel se asigna a un pez, por niveles descendentes
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .perfil import Perfil
from .segmentacion import Mancha

GRIS_CUERPO = 90
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@dataclass
class Analisis:
    """Pasos intermedios, para las vistas. ``piezas``: (x, y, w, h, área, razón) en la mancha."""
    solidez: float
    sospechosa: bool
    piezas: list
    rellena: Mancha | None = None
    campo: np.ndarray | None = None
    cimas: list | None = None
    etiquetas: np.ndarray | None = None


def solidez(m: Mancha) -> float:
    c = max(cv2.findContours(m.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    return m.area / max(1.0, cv2.contourArea(cv2.convexHull(c)))


def rellenar_huecos(g: np.ndarray, p: Perfil) -> np.ndarray:
    o = g.copy()
    h, w = g.shape
    for T in p.umbrales:
        n, lab, st, _ = cv2.connectedComponentsWithStats((o >= T).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            x, y, a, b, c = st[i]
            if c <= p.hueco_max and x > 0 and y > 0 and x + a < w and y + b < h:
                o[lab == i] = min(GRIS_CUERPO, p.umbral3 - 1)
    return o


def campo_distancia(m: Mancha, p: Perfil) -> np.ndarray:
    """Distancia media al fondo; un corte en el borde de la ROI no cuenta como fondo."""
    h, w = m.mask.shape
    izq, der, arr, aba = m.toca
    b = [12 if t else 1 for t in (arr, aba, izq, der)]
    tipo = cv2.BORDER_REPLICATE if any(m.toca) else cv2.BORDER_CONSTANT
    o = np.zeros((h, w), np.float32)
    for T in p.umbrales:
        k = (m.gris < T).astype(np.uint8)
        if T < p.umbral:
            k = cv2.morphologyEx(k, cv2.MORPH_OPEN, _K3)
        q = cv2.copyMakeBorder(k, *b, tipo, value=0)
        for lado, t in zip((np.s_[0], np.s_[-1], np.s_[:, 0], np.s_[:, -1]), (arr, aba, izq, der)):
            if not t:
                q[lado] = 0
        d = cv2.distanceTransform(q, cv2.DIST_L2, 5)
        o += d[b[0]:b[0] + h, b[2]:b[2] + w]
    return o / len(p.umbrales)


def cimas(f: np.ndarray, p: Perfil, paso: float = 0.4) -> list[tuple[int, int, float]]:
    """[(fila, col, razón)] de los cuerpos aceptados; razón baja = separación clara."""
    f = cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mx = float(f.max())
    if mx <= 0:
        return []
    comps, aceptadas = [], []
    for t in np.arange(mx - paso, 0.2, -paso):
        n, lab, st, _ = cv2.connectedComponentsWithStats((f > t).astype(np.uint8), connectivity=8)
        nuevas = []
        for i in range(1, n):
            dentro = [c for c in comps if lab[c[1]] == i]
            if not dentro:
                x, y, w, h = st[i, :4]
                sub = np.where(lab[y:y + h, x:x + w] == i, f[y:y + h, x:x + w], 0)
                r, c = np.unravel_index(sub.argmax(), sub.shape)
                nuevas.append((float(sub[r, c]), (int(r + y), int(c + x)), int(st[i, 4])))
                continue
            dentro.sort(key=lambda c: -c[0])
            for c in dentro[1:]:
                if t < p.razon * c[0] and c[0] >= p.cima_min and c[2] >= p.area_cima:
                    aceptadas.append((*c[1], float(t / c[0])))
            nuevas.append((dentro[0][0], dentro[0][1], int(st[i, 4])))
        comps = nuevas
    # Cuerpos unidos por 1-2 px no llegan a fundirse en el barrido: es la separación más clara.
    comps.sort(key=lambda c: -c[0])
    for c in comps[1:]:
        if c[0] >= p.cima_min and c[2] >= p.area_cima:
            aceptadas.append((*c[1], 0.01))
    if not comps:
        return []
    lista = [(*comps[0][1], min([q[2] for q in aceptadas], default=0.0))] + aceptadas
    return [(r - 1, c - 1, q) for r, c, q in lista]


def repartir(f: np.ndarray, s: list, mask: np.ndarray, paso: float = 0.8) -> np.ndarray:
    """Asigna cada píxel de la mancha a una cima, creciendo por niveles descendentes."""
    k = np.ones((3, 3), np.uint8)
    lab = np.zeros(f.shape, np.uint8)
    for i, (r, c, _) in enumerate(s):
        cv2.circle(lab, (c, r), 2, i + 1, -1)
    lab[mask == 0] = 0
    for t in [*np.arange(float(f.max()) - paso, 0, -paso), -1.0]:
        region = (f > t) & (mask > 0)
        while True:
            crece = cv2.dilate(lab, k)
            nuevo = region & (lab == 0) & (crece > 0)
            if not nuevo.any():
                break
            lab[nuevo] = crece[nuevo]
    return lab


def analizar(m: Mancha, p: Perfil) -> Analisis:
    h, w = m.mask.shape
    sol = solidez(m)
    entera = [(0, 0, w, h, m.area, 0.0)]
    if sol > p.solidez_libre:
        return Analisis(sol, False, entera)
    g = rellenar_huecos(m.gris, p)
    q = replace(m, gris=g, mask=m.mask | (g < p.umbral).astype(np.uint8))
    f = campo_distancia(q, p)
    s = cimas(f, p)
    if len(s) < 2:
        return Analisis(sol, True, entera, q, f, s)
    lab = repartir(f, s, q.mask)
    out = []
    for i, (_, _, r) in enumerate(s):
        ys, xs = np.nonzero(lab == i + 1)
        if len(xs):
            out.append((int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1, len(xs), r))
    return Analisis(sol, True, out or entera, q, f, s, lab)


def detecciones(ms: list[Mancha], p: Perfil) -> tuple[np.ndarray, list[Analisis]]:
    """Filas: xmin, ymin, xmax, ymax (1280x720), área, razón; y el análisis de cada mancha."""
    L, A = [], []
    for m in ms:
        a = analizar(m, p)
        A.append(a)
        for x, y, w, h, ar, q in a.piezas:
            x, y = m.x + x, m.y + y
            L.append([x * p.fx, y * p.fy, (x + w) * p.fx, (y + h) * p.fy, ar, q])
    return np.array(L, float).reshape(-1, 6), A
