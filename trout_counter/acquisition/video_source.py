"""Fuente de imágenes desde un video grabado (pruebas en PC).

Emula en software lo que en la Pi hace el hardware: recorte de la franja (T1)
y extracción del canal Y (T3) mediante la conversión BGR -> YUV.
"""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from ..config import CameraCfg
from .strip import crop_strip


def bgr_to_y(frame: np.ndarray) -> np.ndarray:
    """Luminancia Y (BT.601), la misma que entrega el plano Y de la cámara."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]


class VideoFileSource:
    def __init__(self, path: str, cam: CameraCfg):
        self.cam = cam
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or cam.fps

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if frame.shape[1] != self.cam.output_width and self.cam.flow_axis == "y":
                scale = self.cam.output_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield idx, idx / self.fps, crop_strip(bgr_to_y(frame), self.cam)
            idx += 1

    def close(self) -> None:
        self.cap.release()


class ArraySource:
    """Fuente a partir de cuadros Y en memoria (simulación y pruebas)."""

    def __init__(self, frames_y, cam: CameraCfg, fps: float | None = None):
        self.frames_y = frames_y
        self.cam = cam
        self.fps = fps or cam.fps

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        for idx, y in enumerate(self.frames_y):
            yield idx, idx / self.fps, crop_strip(y, self.cam)

    def close(self) -> None:
        pass
