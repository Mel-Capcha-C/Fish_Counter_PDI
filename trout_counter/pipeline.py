"""Orquestador: aplica las técnicas en orden a cada franja recibida.

    franja Y (T1, T2, T3)
      -> flat-field (T4) -> suavizado (T5)
      -> umbral fijo (T6) -> cierre/apertura (T7)
      -> líneas A y B -> imagen espacio-temporal (T8)
      -> al cerrar una pasada: FishCounter (T9, T10, T11, T12 + velocidad)
"""
from __future__ import annotations

import logging

import numpy as np

from .calibration import Calibration
from .config import Config
from .counting import CountRecord, FishCounter, SpatioTemporalAccumulator, line_from_mask
from .preprocessing import FlatFieldCorrector, denoise
from .segmentation import MorphologyCleaner, apply_threshold

log = logging.getLogger(__name__)


class TroutCountingPipeline:
    def __init__(self, cfg: Config, calib: Calibration | None, strip_shape: tuple[int, int]):
        self.cfg = cfg
        pp = cfg.preprocess
        if calib is not None:
            if calib.background.shape != strip_shape:
                raise ValueError(f"La calibración es de {calib.background.shape} y la franja "
                                 f"es de {strip_shape}: recalibra.")
            self.flat = FlatFieldCorrector(calib.background, calib.dark, pp.target_level)
            self.threshold = calib.threshold
        else:
            log.warning("Sin calibración: se usa flat-field identidad y umbral de config.")
            self.flat = FlatFieldCorrector.identity(strip_shape, pp.target_level)
            self.threshold = cfg.segmentation.threshold
        self.morph = MorphologyCleaner(cfg.morphology.close_ksize, cfg.morphology.open_ksize)
        self.acc = SpatioTemporalAccumulator(cfg.scanline.gap_frames,
                                             cfg.scanline.max_event_frames)
        self.counter = FishCounter(cfg)
        self.last_idx = -1
        self.last_mask: np.ndarray | None = None  # para depuración/visualización

    # ------------------------------------------------------------------ etapas
    def segment(self, strip: np.ndarray) -> np.ndarray:
        pp = self.cfg.preprocess
        img = self.flat.apply(strip)                           # T4
        img = denoise(img, pp.filter, pp.ksize)                # T5
        mask = apply_threshold(img, self.threshold)            # T6
        return self.morph.apply(mask)                          # T7

    def process(self, frame_idx: int, strip: np.ndarray) -> list[CountRecord]:
        g = self.cfg.geometry
        mask = self.segment(strip)
        self.last_mask = mask

        # Fondo adaptativo: sólo si no hay NADA en la franja
        if self.cfg.preprocess.background_update and not mask.any():
            self.flat.update(strip, self.cfg.preprocess.background_alpha)

        # Cuadros perdidos (lo detecta el índice derivado del timestamp)
        if self.last_idx >= 0 and frame_idx - self.last_idx > 1:
            self.acc.fill_dropped(frame_idx - self.last_idx - 1)
        self.last_idx = frame_idx

        row_a = line_from_mask(mask, g.line_a_row, g.line_thickness)   # T8
        row_b = line_from_mask(mask, g.line_b_row, g.line_thickness)
        event = self.acc.push(row_a, row_b, frame_idx)
        return self.counter.process_event(event) if event is not None else []

    def flush(self) -> list[CountRecord]:
        event = self.acc.flush()
        return self.counter.process_event(event) if event is not None else []

    @property
    def count(self) -> int:
        return self.counter.net_count
