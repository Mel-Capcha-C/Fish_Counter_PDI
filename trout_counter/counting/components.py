"""T9 - Componentes conexos (con filtro por área).

Cada blob de la imagen espacio-temporal es una región de píxeles conectados.
``cv2.connectedComponentsWithStats`` (algoritmo de Bolelli/Grana, SAUF/BBDT)
etiqueta la imagen en una sola pasada y devuelve, por blob: caja envolvente,
área y centroide, sin recorrer contornos.

El filtro por área se hace en dos tiempos:
  1. Aquí, un mínimo absoluto muy pequeño (descarta píxeles aislados).
  2. En counter.py, ya normalizada por la velocidad y comparada con el área
     de un pez individual del lote (``blobs.min_area_frac``).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Blob:
    t0: int                 # primera fila (cuadro) dentro de la pasada
    c0: int                 # primera columna transversal
    mask: np.ndarray        # recorte binario (0/255) SÓLO de este blob
    area_st: int            # área en la imagen espacio-temporal (px * cuadros)
    t_centroid: float       # centroide temporal (en cuadros, sub-cuadro)
    c_centroid: float

    @property
    def n_frames(self) -> int:
        return self.mask.shape[0]

    @property
    def c1(self) -> int:
        return self.c0 + self.mask.shape[1]


def extract_blobs(st: np.ndarray, min_area_st: int = 3, connectivity: int = 8) -> list[Blob]:
    n, labels, stats, cents = cv2.connectedComponentsWithStats(st, connectivity=connectivity)
    blobs: list[Blob] = []
    for i in range(1, n):  # 0 = fondo
        x, y, w, h, area = stats[i]
        if area < min_area_st:
            continue
        roi = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        blobs.append(Blob(int(y), int(x), roi, int(area), float(cents[i][1]), float(cents[i][0])))
    blobs.sort(key=lambda b: b.t_centroid)
    return blobs
