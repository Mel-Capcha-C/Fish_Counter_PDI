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
_K33 = np.ones((3, 3), np.uint8)


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
    gris = min(GRIS_CUERPO, p.umbral3 - 1)
    for T in p.umbrales:
        n, lab, st, _ = cv2.connectedComponentsWithStats((o >= T).astype(np.uint8), connectivity=4)
        x, y, a, b, c = st.T
        sel = (c <= p.hueco_max) & (x > 0) & (y > 0) & (x + a < w) & (y + b < h)
        sel[0] = False
        for i in np.flatnonzero(sel):
            xi, yi, ai, bi = (int(v) for v in st[i, :4])
            z = o[yi:yi + bi, xi:xi + ai]
            z[lab[yi:yi + bi, xi:xi + ai] == i] = gris
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


def _miembros(comps, lab) -> dict[int, list]:
    d: dict[int, list] = {}
    for c in comps:
        d.setdefault(int(lab[c[1]]), []).append(c)
    return d


class _Empate(Exception):
    pass


def cimas_completo(f: np.ndarray, p: Perfil, paso: float = 0.4) -> list[tuple[int, int, float]]:
    """Barrido de todos los niveles (algoritmo-v1). Se usa si hay empates y en las pruebas."""
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


def cimas(f: np.ndarray, p: Perfil, paso: float = 0.4) -> list[tuple[int, int, float]]:
    """[(fila, col, razón)] de los cuerpos aceptados; razón baja = separación clara.

    Da lo mismo que recorrer TODOS los niveles (así lo hacía algoritmo-v1), pero salta los que no
    pueden aceptar ningún cuerpo: uno nacido por debajo del nivel ``a`` tiene cima <= a y, para
    aceptarse, debe fundirse por debajo de razón*a. El nivel exacto de una fusión sólo se busca
    (por bisección) cuando el cuerpo menor llega a ``cima_min``. Si dos cimas separadas empatan en el
    máximo, cuál se queda depende del orden del barrido: entonces se hace el barrido completo.
    """
    try:
        return _cimas_rapido(f, p, paso)
    except _Empate:
        return cimas_completo(f, p, paso)


def _cimas_rapido(f: np.ndarray, p: Perfil, paso: float) -> list[tuple[int, int, float]]:
    f = cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mx = float(f.max())
    if mx <= 0:
        return []
    ts = np.arange(mx - paso, 0.2, -paso)
    ult = len(ts) - 1
    aceptadas: list[tuple[int, int, float]] = []

    def nivel(i):
        return cv2.connectedComponentsWithStats((f > ts[i]).astype(np.uint8), connectivity=8)

    def relevante(comps, lab):
        """¿Se fundieron dos cuerpos conocidos y el menor podría aceptarse?"""
        return any(len(v) > 1 and sorted(c[0] for c in v)[-2] >= p.cima_min
                   for v in _miembros(comps, lab).values())

    def avanzar(comps, i, N, aceptar):
        """Estado de los cuerpos en el nivel ``i``."""
        n, lab, st, _ = N
        d, nuevas = _miembros(comps, lab), []
        for q in range(1, n):
            dentro = d.get(q)
            if not dentro:
                x, y, w, h = st[q, :4]
                sub = np.where(lab[y:y + h, x:x + w] == q, f[y:y + h, x:x + w], 0)
                r, c = np.unravel_index(sub.argmax(), sub.shape)
                pico = sub[r, c]
                if cv2.countNonZero((sub == pico).astype(np.uint8)) > 1:
                    i0 = int(np.searchsorted(-ts, -float(pico), side="right"))      # nivel en que nació
                    if i0 < i:
                        z = nivel(i0)[1][y:y + h, x:x + w]
                        if len(np.unique(z[sub == pico])) > 1:
                            raise _Empate
                nuevas.append((float(pico), (int(r + y), int(c + x)), int(st[q, 4])))
                continue
            if len(dentro) > 1:
                dentro.sort(key=lambda c: -c[0])
                if aceptar:
                    for c in dentro[1:]:
                        if ts[i] < p.razon * c[0] and c[0] >= p.cima_min and c[2] >= p.area_cima:
                            aceptadas.append((*c[1], float(ts[i] / c[0])))
            nuevas.append((dentro[0][0], dentro[0][1], int(st[q, 4])))
        return nuevas

    comps, a = [], -1
    while a < ult:
        ta = mx if a < 0 else float(ts[a])
        if ta < p.cima_min:
            if sum(c[0] >= p.cima_min for c in comps) < 2:
                break                                    # ya no puede aparecer ni fundirse otro cuerpo
            b = ult
        else:                                            # salto seguro: ts[b] >= razón * ta
            b = min(max(int(np.searchsorted(-ts, -p.razon * ta, side="right")) - 1, a + 1), ult)
        N = nivel(b)
        if not relevante(comps, N[1]):
            comps, a = avanzar(comps, b, N, False), b
            continue
        lo, hi, Nlo, Nhi = a, b, None, N                 # primera fusión relevante en (lo, hi]
        while hi - lo > 1:
            m = (lo + hi) // 2
            M = nivel(m)
            if relevante(comps, M[1]):
                hi, Nhi = m, M
            else:
                lo, Nlo = m, M
        if lo > a:                                       # estado justo antes: da el área de cada cuerpo
            comps = avanzar(comps, lo, Nlo, False)
        comps, a = avanzar(comps, hi, Nhi, True), hi
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
    lab = np.zeros(f.shape, np.uint8)
    for i, (r, c, _) in enumerate(s):
        cv2.circle(lab, (c, r), 2, i + 1, -1)
    m255 = cv2.compare(mask, 0, cv2.CMP_GT)
    lab = cv2.bitwise_and(lab, m255)
    libre = cv2.bitwise_and(cv2.compare(lab, 0, cv2.CMP_EQ), m255)     # dentro de la mancha y sin asignar
    crece, nuevo = np.empty_like(lab), np.empty_like(lab)
    for t in [*np.arange(float(f.max()) - paso, 0, -paso), -1.0]:
        region = cv2.bitwise_and((f > t).astype(np.uint8) * 255, libre)
        while True:
            cv2.dilate(lab, _K33, crece)
            cv2.bitwise_and(cv2.compare(crece, 0, cv2.CMP_GT), region, nuevo)
            if not cv2.countNonZero(nuevo):
                break
            cv2.copyTo(crece, nuevo, lab)
            cv2.subtract(region, nuevo, region)
            cv2.subtract(libre, nuevo, libre)
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
