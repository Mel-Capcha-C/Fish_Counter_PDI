"""T8 - Imagen espacio-temporal (línea de barrido virtual).

En vez de seguir (tracking) a cada pez en 2D, se observa una sola línea
transversal del canal y se apila su contenido cuadro tras cuadro:

        fila t de la imagen espacio-temporal  =  línea de barrido en el cuadro t

Un pez que cruza la línea deja UNA mancha (blob) en esa imagen: su eje
vertical es el tiempo de cruce y su eje horizontal la posición transversal.
Contar peces se reduce a contar blobs (T9), sin asociar detecciones entre
cuadros.

Mejoras sobre la técnica original:
  * Cada línea combina (OR) ``line_thickness`` filas de la franja, para que un
    pez rápido no "salte" la línea entre dos cuadros.
  * Se usan DOS líneas (A y B) separadas D píxeles: A cuenta; B permite medir
    velocidad y sentido (ver velocity.py).
  * La imagen se corta en "pasadas" (eventos) cuando la línea queda vacía
    ``gap_frames`` cuadros; cada pasada se analiza por separado, así la memoria
    es acotada y el conteo sale casi en tiempo real.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


def line_from_mask(mask: np.ndarray, row: int, thickness: int) -> np.ndarray:
    """Línea virtual = OR de ``thickness`` filas centradas en ``row``."""
    half = thickness // 2
    r0 = max(0, row - half)
    r1 = min(mask.shape[0], row + half + 1)
    return mask[r0:r1].max(axis=0)


@dataclass
class ScanEvent:
    start_frame: int
    st_a: np.ndarray      # (T, W) uint8, línea A apilada en el tiempo
    st_b: np.ndarray      # (T, W) uint8, línea B apilada en el tiempo
    forced: bool = False  # True si se cortó por longitud máxima


class SpatioTemporalAccumulator:
    def __init__(self, gap_frames: int, max_event_frames: int):
        self.gap = gap_frames
        self.max_len = max_event_frames
        self._reset()

    def _reset(self) -> None:
        self.rows_a: list[np.ndarray] = []
        self.rows_b: list[np.ndarray] = []
        self.start = -1
        self.empty_a = 0
        self.empty_b = 0
        self.last_a: np.ndarray | None = None
        self.last_b: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.start >= 0

    def push(self, row_a: np.ndarray, row_b: np.ndarray, frame_idx: int) -> ScanEvent | None:
        has_a, has_b = bool(row_a.any()), bool(row_b.any())
        if not self.active:
            if not (has_a or has_b):
                return None
            self.start = frame_idx
        self.rows_a.append(row_a)
        self.rows_b.append(row_b)
        self.last_a, self.last_b = row_a, row_b
        self.empty_a = 0 if has_a else self.empty_a + 1
        self.empty_b = 0 if has_b else self.empty_b + 1

        if self.empty_a >= self.gap and self.empty_b >= self.gap:
            return self._emit(len(self.rows_a) - min(self.empty_a, self.empty_b), forced=False)
        if len(self.rows_a) >= self.max_len:
            return self._forced_cut()
        return None

    def fill_dropped(self, n: int) -> None:
        """Cuadros perdidos: se repite la última línea para no deformar el tiempo."""
        if self.active and self.last_a is not None:
            for _ in range(n):
                self.rows_a.append(self.last_a)
                self.rows_b.append(self.last_b)

    def flush(self) -> ScanEvent | None:
        if not self.active:
            return None
        return self._emit(len(self.rows_a), forced=False)

    def _emit(self, end: int, forced: bool, keep_from: int | None = None) -> ScanEvent:
        st_a = np.stack(self.rows_a[:end])
        st_b = np.stack(self.rows_b[:end])
        event = ScanEvent(self.start, st_a, st_b, forced)
        if keep_from is None:
            self._reset()
        else:  # el resto pasa a ser el inicio de la siguiente pasada
            self.rows_a = self.rows_a[keep_from:]
            self.rows_b = self.rows_b[keep_from:]
            self.start += keep_from
        return event

    def _forced_cut(self) -> ScanEvent:
        """Flujo continuo de peces: se corta en la última fila vacía de la línea A."""
        n = len(self.rows_a)
        for i in range(n - 1, n // 2, -1):
            if not self.rows_a[i].any():
                return self._emit(i, forced=True, keep_from=i)
        log.warning("Línea A ocupada %d cuadros seguidos: ¿pez detenido o suciedad "
                    "sobre la línea? Se corta la pasada.", n)
        return self._emit(n, forced=True)
