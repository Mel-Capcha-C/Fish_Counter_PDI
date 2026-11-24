"""T5 - Filtro gaussiano o de mediana (3x3).

- Gaussiano: promedio ponderado; atenúa el ruido del sensor y las cáusticas
  suaves del agua. Es separable y muy barato.
- Mediana: reemplaza cada píxel por la mediana de su vecindad; elimina mejor el
  ruido impulsivo (reflejos puntuales, burbujas) y conserva bordes.

El kernel se mantiene en 3x3: uno mayor borra la cola fina de los alevines de 4 cm.
"""
from __future__ import annotations

import cv2
import numpy as np


def denoise(img: np.ndarray, method: str = "gaussian", ksize: int = 3) -> np.ndarray:
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "median":
        return cv2.medianBlur(img, ksize)
    return img
