"""T7 - Cierre y apertura morfológicos.

- CIERRE (dilatación -> erosión): rellena huecos y une el cuerpo con la cola o
  con las aletas translúcidas que quedaron por encima del umbral.
- APERTURA (erosión -> dilatación): borra los puntos sueltos más pequeños que
  el elemento estructurante (burbujas, partículas, ruido residual).

Se aplica primero el cierre para no "abrir" (y perder) la cola fina antes de
reconectarla. Elementos elípticos: no privilegian ninguna dirección.
Un kernel grande fusionaría peces muy cercanos -> se mantienen pequeños.
"""
from __future__ import annotations

import cv2
import numpy as np


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


class MorphologyCleaner:
    def __init__(self, close_ksize: int, open_ksize: int):
        self.k_close = _ellipse(close_ksize) if close_ksize > 1 else None
        self.k_open = _ellipse(open_ksize) if open_ksize > 1 else None

    def apply(self, mask: np.ndarray) -> np.ndarray:
        if self.k_close is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        if self.k_open is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        return mask
