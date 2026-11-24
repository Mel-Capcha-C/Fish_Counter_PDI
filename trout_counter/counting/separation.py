"""T11 - Transformada de distancia + Watershed (SELECTIVO).

Sólo se ejecuta sobre blobs que la validación (T10) marcó como sospechosos,
por eso su costo no pesa en el promedio.

Pasos:
  1. Remuestreo isotrópico: el eje temporal del blob se estira por v (px/cuadro)
     para que una fila valga lo mismo que una columna. Sin esto, la distancia
     euclídea no tendría sentido geométrico.
  2. Transformada de distancia: cada píxel recibe su distancia al fondo. El eje
     de cada pez forma una "cresta"; el contacto entre dos peces es un "cuello"
     con distancia baja.
  3. Semillas: dist > frac * max(dist). Cada región conexa de semillas es un pez.
  4. Watershed (Meyer, cv2.watershed) inunda el relieve -dist desde las
     semillas; las líneas divisorias quedan en los cuellos.
  5. PRUEBA DE CUELLO (añadida): un pez grande y robusto puede producir dos
     semillas (cabeza y tronco) y el Watershed lo cortaría transversalmente.
     Un corte real entre dos peces es más ANGOSTO que los cuerpos que separa;
     uno falso atraviesa el cuerpo entero. Si
         largo_divisoria > neck_frac * ancho_del_trozo_más_delgado
     los dos trozos se vuelven a unir.
  6. Se cuentan los trozos con área >= min_piece_frac * área individual.
"""
from __future__ import annotations

import cv2
import numpy as np


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def watershed_split(mask: np.ndarray, v: float, single_area: float,
                    dist_threshold_frac: float, min_piece_frac: float,
                    neck_frac: float = 0.75) -> tuple[int, np.ndarray]:
    """Devuelve (número de trozos válidos, etiquetas en la imagen isotrópica)."""
    # 1. Remuestreo isotrópico (+ borde para que la distancia no toque el marco)
    h, w = mask.shape
    new_h = max(1, int(round(h * v)))
    iso = cv2.resize(mask, (w, new_h), interpolation=cv2.INTER_NEAREST)
    iso = cv2.copyMakeBorder(iso, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    # 2. Transformada de distancia (L2, máscara 5x5)
    dist = cv2.distanceTransform(iso, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return 1, np.zeros_like(iso, np.int32)

    # 3. Semillas
    sure_fg = (dist > dist_threshold_frac * dist.max()).astype(np.uint8)
    n_seeds, seeds = cv2.connectedComponents(sure_fg, connectivity=8)
    if n_seeds - 1 <= 1:
        return 1, seeds.astype(np.int32)

    # 4. Watershed sobre el relieve invertido de la distancia
    markers = seeds.astype(np.int32) + 1            # fondo seguro = 1, semillas = 2..
    markers[(iso > 0) & (sure_fg == 0)] = 0         # 0 = a decidir por watershed
    relief = cv2.normalize(-dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    markers = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)

    labels = list(range(2, n_seeds + 1))
    # Ancho de cada trozo = 2 * distancia máxima dentro de él
    width = {lab: 2.0 * float(dist[markers == lab].max(initial=0)) for lab in labels}

    # 5. Prueba de cuello: largo de la divisoria entre cada par de trozos
    neck: dict[tuple[int, int], int] = {}
    ys, xs = np.nonzero((markers == -1) & (iso > 0))
    H, W = markers.shape
    for y, x in zip(ys, xs):
        y0, y1, x0, x1 = max(0, y - 1), min(H, y + 2), max(0, x - 1), min(W, x + 2)
        neigh = {int(l) for l in np.unique(markers[y0:y1, x0:x1]) if l >= 2}
        if len(neigh) == 2:
            a, b = sorted(neigh)
            neck[(a, b)] = neck.get((a, b), 0) + 1
    parent = {lab: lab for lab in labels}
    parent_list = list(range(n_seeds + 1))
    for (a, b), length in neck.items():
        if length > neck_frac * min(width[a], width[b]):
            ra, rb = _find(parent_list, a), _find(parent_list, b)
            parent_list[rb] = ra
    for lab in labels:
        parent[lab] = _find(parent_list, lab)

    # 6. Trozos (ya unidos) con área suficiente
    min_piece = min_piece_frac * single_area
    area: dict[int, int] = {}
    for lab in labels:
        area[parent[lab]] = area.get(parent[lab], 0) + int(np.count_nonzero(markers == lab))
    n_valid = sum(1 for a in area.values() if a >= min_piece)
    return max(1, n_valid), markers
