"""Etapa de SEGMENTACIÓN: técnicas 6 (umbral Otsu fijo) y 7 (morfología)."""
from .morphology import MorphologyCleaner
from .threshold import OtsuResult, apply_threshold, otsu_threshold

__all__ = ["MorphologyCleaner", "OtsuResult", "apply_threshold", "otsu_threshold"]
