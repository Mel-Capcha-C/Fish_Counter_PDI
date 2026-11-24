"""Etapa de ADQUISICIÓN: técnicas 1 (ROI), 2 (controles fijos) y 3 (canal Y)."""
from .strip import crop_strip, orient_strip
from .video_source import ArraySource, VideoFileSource, bgr_to_y

__all__ = ["crop_strip", "orient_strip", "ArraySource", "VideoFileSource", "bgr_to_y"]
