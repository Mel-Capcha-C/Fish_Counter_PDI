"""T6 - Umbral global fijo, calibrado UNA vez con Otsu.

Con retroiluminación + exposición fija (T2) + flat-field (T4) el histograma
de la franja es bimodal y estable: un modo brillante (acrílico iluminado,
~nivel_objetivo) y uno oscuro (silueta del pez). Otsu elige el umbral ``k``
que maximiza la varianza entre clases:

        sigma_B^2(k) = [mu_T * w(k) - mu(k)]^2 / ( w(k) * (1 - w(k)) )

Calcularlo en cada cuadro sería un error: en un cuadro SIN peces el
histograma es unimodal y Otsu partiría el ruido del fondo en dos. Por eso se
calcula sólo en la calibración, con cuadros que contienen peces, y luego el
umbral queda fijo.

La "efectividad" eta = sigma_B^2 / sigma_T^2 (0..1) mide qué tan bimodal es el
histograma; si es baja, la iluminación o el enfoque deben revisarse.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OtsuResult:
    threshold: int
    effectiveness: float
    dark_mean: float
    bright_mean: float


def otsu_threshold(pixels: np.ndarray) -> OtsuResult:
    """Otsu (1979) sobre un histograma de 256 niveles, con su métrica eta."""
    hist = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)                 # probabilidad de la clase oscura
    mu = np.cumsum(p * levels)           # media acumulada
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    # Si los modos están muy separados hay una MESETA de máximos (niveles vacíos
    # entre pez y fondo). argmax tomaría su borde, pegado al pez; se toma el centro.
    peak = sigma_b2.max()
    k = int(round(np.mean(np.flatnonzero(sigma_b2 >= peak * (1 - 1e-9)))))
    sigma_t2 = float(np.sum(((levels - mu_t) ** 2) * p))
    eta = float(sigma_b2[k] / sigma_t2) if sigma_t2 > 0 else 0.0
    dark_mean = float(mu[k] / omega[k]) if omega[k] > 0 else 0.0
    bright_mean = float((mu_t - mu[k]) / (1 - omega[k])) if omega[k] < 1 else 255.0
    return OtsuResult(k, eta, dark_mean, bright_mean)


def apply_threshold(img: np.ndarray, threshold: int) -> np.ndarray:
    """Máscara binaria 0/255: el pez es OSCURO sobre fondo retroiluminado."""
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask
