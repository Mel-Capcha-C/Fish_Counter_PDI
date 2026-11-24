"""TÉCNICA AÑADIDA - Doble línea de barrido: velocidad y sentido de cada pez.

Problema que resuelve: en la imagen espacio-temporal el eje vertical es TIEMPO,
no distancia. Un pez lento deja un blob alto y uno rápido un blob bajo, aunque
midan lo mismo. Sin corregirlo, el área del blob (T9, T12) y la forma usada por
la transformada de distancia (T11) dependen de la velocidad, y la estimación
por área queda inservible.

Solución: una segunda línea B, D píxeles aguas abajo de A. El mismo pez aparece
en ST_A y en ST_B, desplazado en el tiempo:

        dt = t_centroide_B - t_centroide_A      [cuadros]
        v  = D / dt                             [px / cuadro]

Con v, cada fila de la imagen espacio-temporal equivale a v píxeles de canal:
    largo_px  = n_cuadros * v
    area_px2  = area_st  * v

El signo de dt da el sentido: dt < 0 => el pez cruzó B antes que A (retrocede).
Si no hay pareja (pez cortado, choque de blobs) se usa la mediana reciente.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .components import Blob


@dataclass
class VelocityMatch:
    v: float            # px / cuadro (siempre positiva)
    direction: int      # +1 sentido normal, -1 retroceso
    measured: bool      # False si se usó la mediana por falta de pareja


class VelocityEstimator:
    def __init__(self, line_distance_px: float, default: float, vmin: float, vmax: float,
                 history: int):
        self.D = float(line_distance_px)
        self.vmin, self.vmax = vmin, vmax
        self.hist: deque[float] = deque([default], maxlen=history)

    @property
    def typical(self) -> float:
        return float(np.median(self.hist))

    @staticmethod
    def _overlap(a: Blob, b: Blob) -> float:
        inter = min(a.c1, b.c1) - max(a.c0, b.c0)
        return max(0, inter) / max(1, min(a.c1 - a.c0, b.c1 - b.c0))

    def match(self, blobs_a: list[Blob], blobs_b: list[Blob]) -> list[VelocityMatch]:
        max_dt = self.D / self.vmin
        used: set[int] = set()
        out: list[VelocityMatch] = []
        for a in blobs_a:
            best, best_score = None, 0.0
            for j, b in enumerate(blobs_b):
                if j in used:
                    continue
                dt = b.t_centroid - a.t_centroid
                if abs(dt) > max_dt or abs(dt) < 1e-3:
                    continue
                ov = self._overlap(a, b)
                if ov < 0.3:
                    continue
                # Preferir mayor solape transversal y área parecida (mismo pez)
                ratio = min(a.area_st, b.area_st) / max(a.area_st, b.area_st)
                score = ov + ratio
                if score > best_score:
                    best, best_score = j, score
            if best is None:
                out.append(VelocityMatch(self.typical, +1, False))
                continue
            used.add(best)
            dt = blobs_b[best].t_centroid - a.t_centroid
            v = float(np.clip(self.D / abs(dt), self.vmin, self.vmax))
            direction = 1 if dt > 0 else -1
            self.hist.append(v)
            out.append(VelocityMatch(v, direction, True))
        return out
