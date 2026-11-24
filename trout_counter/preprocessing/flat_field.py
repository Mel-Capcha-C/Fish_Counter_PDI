"""T4 - Corrección flat-field.

Los paneles LED bajo el acrílico nunca iluminan de forma perfectamente
uniforme (centro más brillante, bordes más oscuros, manchas). La corrección
divide cada cuadro por una imagen de referencia del canal vacío:

        C = (I - D) / (B - D) * nivel_objetivo

    I : cuadro actual          B : fondo (canal vacío, LED encendidos)
    D : cuadro oscuro (LED apagados, opcional; si no se mide D = 0)

Tras corregir, el fondo vale ``nivel_objetivo`` en TODO el ancho del canal, de
modo que un único umbral global (T6) sirve en el centro y en los bordes.

Extensión añadida: actualización lenta del fondo en cuadros sin peces, para
seguir la deriva térmica de los LED y la suciedad que se acumula en el acrílico.
"""
from __future__ import annotations

import numpy as np


class FlatFieldCorrector:
    def __init__(self, background: np.ndarray, dark: np.ndarray | None = None,
                 target_level: int = 200):
        self.target = float(target_level)
        self.dark = (np.zeros_like(background, dtype=np.float32) if dark is None
                     else dark.astype(np.float32))
        self.background = background.astype(np.float32)
        self._recompute_gain()

    @classmethod
    def from_frames(cls, frames: list[np.ndarray], dark_frames: list[np.ndarray] | None = None,
                    target_level: int = 200) -> "FlatFieldCorrector":
        """La mediana temporal elimina ruido y cualquier objeto que pase."""
        bg = np.median(np.stack(frames).astype(np.float32), axis=0)
        dark = (np.median(np.stack(dark_frames).astype(np.float32), axis=0)
                if dark_frames else None)
        return cls(bg, dark, target_level)

    @classmethod
    def identity(cls, shape: tuple[int, int], target_level: int = 200) -> "FlatFieldCorrector":
        """Sin calibración: no corrige nada (fondo supuesto = target)."""
        return cls(np.full(shape, float(target_level), np.float32), None, target_level)

    def _recompute_gain(self) -> None:
        span = np.maximum(self.background - self.dark, 1.0)
        self.gain = (self.target / span).astype(np.float32)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        out = (frame.astype(np.float32) - self.dark) * self.gain
        return np.clip(out, 0, 255).astype(np.uint8)

    def update(self, frame: np.ndarray, alpha: float) -> None:
        """Media móvil exponencial del fondo. Llamar SOLO con cuadros vacíos."""
        self.background += alpha * (frame.astype(np.float32) - self.background)
        self._recompute_gain()
