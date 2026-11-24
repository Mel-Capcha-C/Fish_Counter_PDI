"""Calibración: se ejecuta UNA vez por instalación (o al cambiar los LED).

Paso 1 - Fondo (T4): N cuadros con el canal vacío y los LED encendidos.
         Opcional: cuadros oscuros con los LED apagados.
Paso 2 - Umbral (T6): cuadros con peces pasando. Se corrigen con el fondo,
         se suavizan (T5) y se calcula Otsu sobre los cuadros que contienen
         pez. Se guarda el umbral y la efectividad del histograma.

Resultado: ``calibration.npz`` (fondo, oscuro, umbral, eta).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Config
from .preprocessing import FlatFieldCorrector, denoise
from .segmentation import OtsuResult, otsu_threshold

log = logging.getLogger(__name__)


@dataclass
class Calibration:
    background: np.ndarray
    dark: np.ndarray | None
    threshold: int
    effectiveness: float

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, background=self.background,
            dark=self.dark if self.dark is not None else np.zeros(0),
            threshold=self.threshold, effectiveness=self.effectiveness)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = np.load(path)
        dark = d["dark"] if d["dark"].size else None
        return cls(d["background"], dark, int(d["threshold"]), float(d["effectiveness"]))


def calibrate_background(strips: Iterable[np.ndarray], n: int) -> np.ndarray:
    frames = []
    for s in strips:
        frames.append(s)
        if len(frames) >= n:
            break
    return np.median(np.stack(frames).astype(np.float32), axis=0)


def calibrate_threshold(strips: Iterable[np.ndarray], flat: FlatFieldCorrector, cfg: Config,
                        n_with_fish: int = 60, max_frames: int = 5000) -> OtsuResult:
    """Otsu sobre los píxeles de cuadros que contienen pez."""
    target = cfg.preprocess.target_level
    selected = []
    for i, s in enumerate(strips):
        img = denoise(flat.apply(s), cfg.preprocess.filter, cfg.preprocess.ksize)
        # Un cuadro "tiene pez" si una fracción apreciable cae muy por debajo del fondo
        if np.mean(img < 0.6 * target) > 0.01:
            selected.append(img)
        if len(selected) >= n_with_fish or i >= max_frames:
            break
    if not selected:
        raise RuntimeError("No se detectaron peces durante la calibración del umbral.")
    res = otsu_threshold(np.concatenate([x.ravel() for x in selected]))
    log.info("Otsu: umbral=%d  eta=%.3f  media_pez=%.1f  media_fondo=%.1f  (%d cuadros)",
             res.threshold, res.effectiveness, res.dark_mean, res.bright_mean, len(selected))
    if res.effectiveness < cfg.segmentation.otsu_min_effectiveness:
        log.warning("Histograma poco bimodal (eta=%.2f). Revisa iluminación, enfoque o "
                    "exposición antes de confiar en el umbral.", res.effectiveness)
    return res
