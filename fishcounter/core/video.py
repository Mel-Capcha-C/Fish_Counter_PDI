"""Lectura exacta de video: cualquier cuadro, en cualquier orden, siempre el mismo."""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .perfil import Perfil


class VideoInvalido(RuntimeError):
    pass


class LectorVideo:
    """Saltar con CAP_PROP_POS_FRAMES no es exacto en mp4: se comprueba con una firma del cuadro
    (guardada la primera vez que se leyó en orden) y, si no coincide, se relee desde el inicio."""

    def __init__(self, ruta: str | Path):
        self.ruta = str(ruta)
        self._cap = None
        self._sig = 0                      # índice del próximo cuadro
        self._firmas: dict[int, int] = {}
        self._ultimo: tuple[int, np.ndarray] | None = None
        self._previo: tuple[int, np.ndarray] | None = None    # el tracking pide también el cuadro k-1
        self._abrir()
        self.ancho = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))   # se corrige en ``contar_cuadros``

    def _abrir(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.ruta)
        if not self._cap.isOpened():
            raise VideoInvalido(f"No se pudo abrir el video:\n{self.ruta}")
        self._sig = 0

    def _siguiente(self) -> np.ndarray | None:
        ok, f = self._cap.read()
        if not ok:
            return None
        f_ = zlib.crc32(f[::16, ::16].tobytes())
        self._firmas.setdefault(self._sig, f_)
        self._previo, self._ultimo = self._ultimo, (self._sig, f)
        self._sig += 1
        return f

    def contar_cuadros(self, progreso: Callable[[float], bool] | None = None) -> int:
        """Cuenta los cuadros reales decodificando (el dato del contenedor puede ser aproximado)."""
        self._abrir()
        n, est = 0, max(self.total, 1)
        while self._cap.grab():
            n += 1
            if progreso and n % 25 == 0 and not progreso(min(n / est, 1.0)):
                return 0
        self.total = n
        self._abrir()
        if not n:
            raise VideoInvalido("El video no tiene cuadros legibles")
        return n

    def leer(self, k: int) -> np.ndarray | None:
        if self._ultimo and self._ultimo[0] == k:
            return self._ultimo[1]
        if self._previo and self._previo[0] == k:
            return self._previo[1]
        if not 0 <= k < max(self.total, k + 1):
            return None
        if k < self._sig or k - self._sig > 90:
            esperada = self._firmas.get(k)
            if esperada is not None:                  # sólo se salta si hay con qué comprobar
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                self._sig = k
                f = self._siguiente()
                if f is not None and zlib.crc32(f[::16, ::16].tobytes()) == esperada:
                    return f
            self._abrir()
        while self._sig <= k:
            f = self._siguiente()
            if f is None:
                return None
        return self._ultimo[1]

    def cerrar(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def estimar_fondo(video: LectorVideo, p: Perfil,
                  progreso: Callable[[float], bool] | None = None) -> np.ndarray | None:
    """Estimación de fondo por mediana temporal. Sólo guarda en memoria los cuadros elegidos.

    ``progreso(fracción)`` devuelve False para cancelar (entonces devuelve None).
    """
    total = video.contar_cuadros(lambda f: progreso(f * .4) if progreso else True)
    if not total:
        return None
    idx = set(np.linspace(0, total - 1, p.fondo_cuadros).astype(int).tolist())
    L = []
    ult = max(idx)
    for k in range(ult + 1):
        f = video._siguiente()
        if f is None:
            break
        if k in idx:
            L.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[p.roi_y0:, p.roi_x0:])
        if progreso and k % 25 == 0 and not progreso(.4 + .6 * k / total):
            return None
    video._abrir()
    video._ultimo = video._previo = None
    if not L:
        raise VideoInvalido("No se pudo estimar el fondo")
    return np.median(np.stack(L), axis=0).astype(np.float32)
