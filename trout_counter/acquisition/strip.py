"""Utilidades de la franja (ROI) común a todas las fuentes de imagen.

Convención interna del pipeline (independiente de cómo esté montada la cámara):
    franja.shape == (strip_rows, ancho_transversal)
    - eje 0 (filas)    -> dirección del flujo; el pez avanza hacia filas crecientes
    - eje 1 (columnas) -> dirección transversal del canal
"""
from __future__ import annotations

import numpy as np

from ..config import CameraCfg


def orient_strip(strip: np.ndarray, cam: CameraCfg) -> np.ndarray:
    """Gira/voltea la franja para que cumpla la convención interna."""
    if cam.flow_axis == "x":
        strip = strip.T
    if cam.flow_reversed:
        strip = strip[::-1, :]
    return np.ascontiguousarray(strip)


def strip_bounds(full_len_along_flow: int, cam: CameraCfg, rows: int) -> tuple[int, int]:
    """Índices [ini, fin) de la franja a lo largo del flujo en una imagen completa."""
    center = int(round(cam.strip_center_frac * full_len_along_flow))
    start = max(0, min(full_len_along_flow - rows, center - rows // 2))
    return start, start + rows


def crop_strip(y_full: np.ndarray, cam: CameraCfg) -> np.ndarray:
    """Recorta la franja desde un cuadro completo (fuentes de video / simulación).

    En la Raspberry Pi este recorte lo hace el ISP (ScalerCrop); aquí se emula
    por software para poder probar el pipeline con videos grabados.
    """
    if cam.flow_axis == "y":
        a, b = strip_bounds(y_full.shape[0], cam, cam.strip_rows)
        strip = y_full[a:b, :]
    else:
        a, b = strip_bounds(y_full.shape[1], cam, cam.strip_rows)
        strip = y_full[:, a:b]
    return orient_strip(strip, cam)
