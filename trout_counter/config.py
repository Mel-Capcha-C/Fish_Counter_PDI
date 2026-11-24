"""Carga de configuración (config.toml) en dataclasses tipadas.

Se usa ``tomllib`` (biblioteca estándar desde Python 3.11, incluida en
Raspberry Pi OS Bookworm), así que no se necesita ninguna dependencia extra.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class CameraCfg:
    sensor_mode_size: tuple[int, int] = (1536, 864)
    output_width: int = 640
    strip_rows: int = 64
    strip_center_frac: float = 0.5
    fps: int = 100
    flow_axis: str = "y"
    flow_reversed: bool = False
    exposure_us: int = 400
    analogue_gain: float = 1.5
    colour_gains: tuple[float, float] = (1.6, 1.6)
    buffer_count: int = 6
    lens_position: float = 4.0


@dataclass
class GeometryCfg:
    mm_per_px: float = 0.40
    line_a_row: int = 8
    line_b_row: int = 56
    line_thickness: int = 3


@dataclass
class PreprocessCfg:
    target_level: int = 200
    background_update: bool = True
    background_alpha: float = 0.02
    filter: str = "gaussian"
    ksize: int = 3


@dataclass
class SegmentationCfg:
    threshold: int = 120
    otsu_min_effectiveness: float = 0.70


@dataclass
class MorphologyCfg:
    close_ksize: int = 5
    open_ksize: int = 3
    st_close_ksize: int = 3


@dataclass
class ScanlineCfg:
    gap_frames: int = 4
    max_event_frames: int = 600


@dataclass
class VelocityCfg:
    default_px_per_frame: float = 10.0
    min_px_per_frame: float = 1.0
    max_px_per_frame: float = 60.0
    history: int = 30


@dataclass
class BlobsCfg:
    min_area_frac: float = 0.25
    connectivity: int = 8


@dataclass
class BatchCfg:
    length_range_mm: tuple[float, float] = (40.0, 70.0)
    width_ratio: float = 0.16
    shape_factor: float = 0.70
    history: int = 60


@dataclass
class ValidationCfg:
    aspect_wide_factor: float = 1.45
    aspect_long_factor: float = 0.65
    area_factor: float = 1.60
    area_strong_factor: float = 2.20


@dataclass
class SeparationCfg:
    dist_threshold_frac: float = 0.60
    min_piece_frac: float = 0.35
    neck_frac: float = 0.75


@dataclass
class OutputCfg:
    csv_path: str = "conteo.csv"
    report_every_s: float = 5.0


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    segmentation: SegmentationCfg = field(default_factory=SegmentationCfg)
    morphology: MorphologyCfg = field(default_factory=MorphologyCfg)
    scanline: ScanlineCfg = field(default_factory=ScanlineCfg)
    velocity: VelocityCfg = field(default_factory=VelocityCfg)
    blobs: BlobsCfg = field(default_factory=BlobsCfg)
    batch: BatchCfg = field(default_factory=BatchCfg)
    validation: ValidationCfg = field(default_factory=ValidationCfg)
    separation: SeparationCfg = field(default_factory=SeparationCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    def validate(self) -> None:
        g, c = self.geometry, self.camera
        if not (0 <= g.line_a_row < g.line_b_row < c.strip_rows):
            raise ValueError("Se requiere 0 <= line_a_row < line_b_row < strip_rows")
        if c.flow_axis not in ("x", "y"):
            raise ValueError("camera.flow_axis debe ser 'x' o 'y'")
        if self.preprocess.filter not in ("gaussian", "median", "none"):
            raise ValueError("preprocess.filter debe ser gaussian | median | none")
        if self.preprocess.ksize % 2 == 0:
            raise ValueError("preprocess.ksize debe ser impar")


def _fill(dc, values: dict):
    """Copia los valores del diccionario TOML sobre una dataclass."""
    known = {f.name: f for f in fields(dc)}
    for key, val in values.items():
        if key not in known:
            raise KeyError(f"Clave desconocida en config: {type(dc).__name__}.{key}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(val, dict):
            _fill(current, val)
        elif isinstance(current, tuple):
            setattr(dc, key, tuple(val))
        else:
            setattr(dc, key, type(current)(val) if current is not None else val)
    return dc


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is not None and Path(path).exists():
        with open(path, "rb") as fh:
            _fill(cfg, tomllib.load(fh))
    cfg.validate()
    return cfg
