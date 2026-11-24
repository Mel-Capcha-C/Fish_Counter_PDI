"""Separación de peces en contacto dentro de una mancha (técnicas 10-12, rediseñadas).

1. CAMPO    Mapa de distancia al borde promediado sobre 3 umbrales (146, 125, 105): una
            franja algo más clara entre dos peces baja el campo aunque no los separe.
2. CRESTAS  Se baja un nivel desde el máximo. Cada cima es un cuerpo; cuando dos
            cimas se juntan, la altura del paso dice cuán profundo es el cuello.
3. DECISIÓN Son dos peces si  paso < RAZON * cima_menor, y la cima menor es un cuerpo
            real (cima >= CIMA_MIN, área >= AREA_CIMA). Es una razón: no depende de la talla.
4. REPARTO  Desde las cimas aceptadas, por niveles, cada píxel se asigna a un pez (repartir).
0. HUECOS   Antes de todo se tapan los agujeros claros, pequeños y CERRADOS del cuerpo (brillos):
            un agujero crea un borde falso que parece un cuello. No toca rendijas abiertas.
Un pez curvado o con la cola doblada tiene una sola cima, así que puede devolver 1.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from segmentacion import Mancha, caja

UMBRALES = (146, 125, 105)
RAZON, CIMA_MIN, AREA_CIMA = 0.88, 7.5, 60
SOLIDEZ_LIBRE = 0.94                 # por encima no se analiza (sólo 2 % de los grupos)
HUECO_MAX = 60                       # px; un hueco mayor puede ser fondo entre dos peces
GRIS_CUERPO = 90
_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def solidez(m: Mancha) -> float:
    c = max(cv2.findContours(m.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    return m.area / max(1.0, cv2.contourArea(cv2.convexHull(c)))


def rellenar_huecos(g: np.ndarray) -> np.ndarray:
    """Tapa los huecos claros cerrados de hasta HUECO_MAX px, en cada umbral del separador."""
    o = g.copy()
    h, w = g.shape
    for T in UMBRALES:
        n, lab, st, _ = cv2.connectedComponentsWithStats((o >= T).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            x, y, a, b, c = st[i]
            if c <= HUECO_MAX and x > 0 and y > 0 and x + a < w and y + b < h:
                o[lab == i] = GRIS_CUERPO
    return o


def campo(m: Mancha) -> np.ndarray:
    """Distancia media al fondo; un corte en el borde del ROI no cuenta como fondo."""
    h, w = m.mask.shape
    izq, der, arr, aba = m.toca
    b = [12 if t else 1 for t in (arr, aba, izq, der)]
    tipo = cv2.BORDER_REPLICATE if any(m.toca) else cv2.BORDER_CONSTANT
    o = np.zeros((h, w), np.float32)
    for T in UMBRALES:
        k = (m.gris < T).astype(np.uint8)
        if T < UMBRALES[0]:
            k = cv2.morphologyEx(k, cv2.MORPH_OPEN, _K)
        q = cv2.copyMakeBorder(k, *b, tipo, value=0)
        for lado, t in zip((np.s_[0], np.s_[-1], np.s_[:, 0], np.s_[:, -1]), (arr, aba, izq, der)):
            if not t:
                q[lado] = 0
        d = cv2.distanceTransform(q, cv2.DIST_L2, 5)
        o += d[b[0]:b[0] + h, b[2]:b[2] + w]
    return o / len(UMBRALES)


def cimas(f: np.ndarray, paso: float = 0.4):
    """[(fila, col, razon)]: cimas aceptadas; razon = paso/cima menor (bajo = separación clara)."""
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
                if t < RAZON * c[0] and c[0] >= CIMA_MIN and c[2] >= AREA_CIMA:
                    aceptadas.append((*c[1], float(t / c[0])))
            nuevas.append((dentro[0][0], dentro[0][1], int(st[i, 4])))
        comps = nuevas
    # Cuerpos unidos por 1-2 px no llegan a fundirse en el barrido: es la separación más clara.
    comps.sort(key=lambda c: -c[0])
    for c in comps[1:]:
        if c[0] >= CIMA_MIN and c[2] >= AREA_CIMA:
            aceptadas.append((*c[1], 0.01))
    p = comps[0][1] if comps else None
    lista = ([(*p, min([q[2] for q in aceptadas], default=0.0))] if p else []) + aceptadas
    return [(r - 1, c - 1, q) for r, c, q in lista]


def repartir(f: np.ndarray, s: list, mask: np.ndarray, paso: float = 0.8) -> np.ndarray:
    """Asigna CADA píxel de la mancha a una cima, creciendo por niveles descendentes del campo.

    Sustituye a cv2.watershed, que ordena por diferencia entre vecinos (no por nivel): el
    "fondo" invadía los cuerpos y sólo se repartía el 22 % del área de la mancha.
    """
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


def separar(m: Mancha) -> list[tuple[int, int, int, int, int, float]]:
    """(x, y, w, h, área, razón) relativos a la mancha, uno por pez. Razón 0 = mancha entera."""
    h, w = m.mask.shape
    entera = [(0, 0, w, h, m.area, 0.0)]
    if solidez(m) > SOLIDEZ_LIBRE:
        return entera
    g = rellenar_huecos(m.gris)
    m = replace(m, gris=g, mask=m.mask | (g < UMBRALES[0]).astype(np.uint8))
    f = campo(m)
    s = cimas(f)
    if len(s) < 2:
        return entera
    lab = repartir(f, s, m.mask)
    out = []
    for i, (_, _, q) in enumerate(s):
        ys, xs = np.nonzero(lab == i + 1)
        if len(xs):
            out.append((int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1, len(xs), q))
    return out or entera


def detecciones(ms: list[Mancha]) -> np.ndarray:
    """Filas: xmin, ymin, xmax, ymax (1280x720), área, razón de separación."""
    L = []
    for m in ms:
        for x, y, w, h, a, q in separar(m):
            L.append([*caja(m.x + x, m.y + y, w, h), a, q])
    return np.array(L, float).reshape(-1, 6)
