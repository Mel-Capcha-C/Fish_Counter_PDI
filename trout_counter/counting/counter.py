"""Lógica de decisión del conteo: une T9, T10, T11, T12 y la doble línea.

Para cada pasada (evento) de la imagen espacio-temporal:

    ST_A, ST_B --cierre suave--> componentes conexos (T9)
        -> velocidad y sentido por pareja de blobs A/B (técnica añadida)
        -> medidas normalizadas: área, largo, ancho, relación de aspecto
        -> clasificación (T10): noise | single | wide | long | large
             single -> 1 pez (y alimenta el modelo del lote, T12)
             otros  -> Watershed (T11) + estimaciones por forma y área (T12)

Regla de decisión para blobs sospechosos:
    n_ws    = trozos válidos del Watershed
    n_shape = round(r) si "wide"  |  round(1/r) si "long"   (r = aspecto relativo)
    n_area  = round(área / área_individual)                  (T12)

    "wide" / "long": n_ws si el Watershed encontró >= 2 cuellos; si no, n_shape.
    "large"        : n_ws si >= 2; si no, n_area cuando el área supera
                     area_strong_factor (grupo compacto); si no, 1 (pez grande).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Config
from .batch_area import BatchAreaModel
from .components import Blob, extract_blobs
from .scanline import ScanEvent
from .separation import watershed_split
from .validation import BlobMeasures, classify_blob, measure_blob
from .velocity import VelocityEstimator


@dataclass
class CountRecord:
    frame: int
    count: int
    direction: int
    x_mm: float
    klass: str
    method: str
    area_mm2: float
    length_mm: float
    width_mm: float
    speed_mm_s: float
    speed_measured: bool


class FishCounter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        g = cfg.geometry
        self.velocity = VelocityEstimator(
            g.line_b_row - g.line_a_row, cfg.velocity.default_px_per_frame,
            cfg.velocity.min_px_per_frame, cfg.velocity.max_px_per_frame, cfg.velocity.history)
        self.batch = BatchAreaModel(cfg.batch, g.mm_per_px)
        k = cfg.morphology.st_close_ksize
        self.k_st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 1 else None
        self.forward = 0
        self.reverse = 0

    @property
    def net_count(self) -> int:
        return self.forward - self.reverse

    def _close(self, st: np.ndarray) -> np.ndarray:
        if self.k_st is None:
            return st
        # Borde de relleno para que el cierre no se "pegue" al marco
        pad = cv2.copyMakeBorder(st, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        return cv2.morphologyEx(pad, cv2.MORPH_CLOSE, self.k_st)[2:-2, 2:-2]

    def _decide(self, klass: str, blob: Blob, v: float, meas: BlobMeasures,
                a1: float, asp1: float) -> tuple[int, str]:
        cfg = self.cfg
        if klass == "single":
            self.batch.add_single(meas.area_px2, meas.width_px, meas.aspect)
            return 1, "single"

        # T11: Watershed sólo sobre blobs sospechosos
        n_ws, _ = watershed_split(blob.mask, v, a1, cfg.separation.dist_threshold_frac,
                                  cfg.separation.min_piece_frac, cfg.separation.neck_frac)
        if n_ws >= 2:
            return n_ws, "watershed"

        r = meas.aspect_ratio_to(asp1)
        if klass == "wide":
            return max(2, int(round(r))), "shape"
        if klass == "long":
            return max(2, int(round(1.0 / r))), "shape"
        # "large": T12, estimación por área del lote
        if meas.area_px2 >= cfg.validation.area_strong_factor * a1:
            return self.batch.estimate_count(meas.area_px2), "area"
        return 1, "single-large"

    def process_event(self, ev: ScanEvent) -> list[CountRecord]:
        cfg = self.cfg
        blobs_a = extract_blobs(self._close(ev.st_a), 3, cfg.blobs.connectivity)
        blobs_b = extract_blobs(self._close(ev.st_b), 3, cfg.blobs.connectivity)
        matches = self.velocity.match(blobs_a, blobs_b)

        mm = cfg.geometry.mm_per_px
        records: list[CountRecord] = []
        for blob, vm in zip(blobs_a, matches):
            meas = measure_blob(blob, vm.v)
            a1, asp1 = self.batch.single_area, self.batch.single_aspect
            klass = classify_blob(meas, a1, asp1, cfg.blobs.min_area_frac, cfg.validation)
            if klass == "noise":
                continue
            n, method = self._decide(klass, blob, vm.v, meas, a1, asp1)

            if vm.direction > 0:
                self.forward += n
            else:
                self.reverse += n
            records.append(CountRecord(
                frame=ev.start_frame + blob.t0, count=n, direction=vm.direction,
                x_mm=(blob.c0 + float(np.mean(np.nonzero(blob.mask)[1]))) * mm,
                klass=klass, method=method,
                area_mm2=meas.area_px2 * mm * mm, length_mm=meas.length_px * mm,
                width_mm=meas.width_px * mm,
                speed_mm_s=vm.v * mm * cfg.camera.fps, speed_measured=vm.measured))
        return records
