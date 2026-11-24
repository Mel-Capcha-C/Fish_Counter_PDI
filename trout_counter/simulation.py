"""Simulador de video del canal retroiluminado (vista superior), con verdad de campo.

Sirve para probar el pipeline completo en un PC sin la Raspberry Pi. Reproduce
las condiciones que motivan cada técnica:
  * Iluminación LED NO uniforme (centro brillante, bordes oscuros)  -> T4
  * Ruido de sensor y cáusticas del agua                            -> T5
  * Aletas y cola translúcidas (más claras que el cuerpo)           -> T7
  * Velocidades distintas por pez                                   -> doble línea
  * Peces pegados lado a lado y en fila                             -> T10, T11, T12

NO usa inteligencia artificial: sólo geometría y ruido aleatorio.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SimFish:
    spawn: int          # cuadro en que aparece (cabeza entrando por y = 0)
    x: float            # centro transversal (px)
    length: float       # px
    speed: float        # px / cuadro
    angle: float        # rad, desvío respecto del eje del canal
    lag: float = 0.0    # px de retraso extra (para peces en fila)


def _fish_polygons(length: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Silueta dorsal (cuerpo) y aleta caudal en coordenadas del pez (cabeza en +y)."""
    s = np.linspace(0, 1, 24)                        # 0 = hocico, 1 = pedúnculo
    prof = np.where(s < 0.3, np.sqrt(s / 0.3), 1 - 0.78 * (np.clip(s - 0.3, 0, None) / 0.7) ** 1.3)
    half = 0.5 * width * np.maximum(prof, 0.08)
    yb = length * (0.5 - 0.82 * s)                   # el cuerpo ocupa el 82 % del largo
    body = np.concatenate([np.stack([half, yb], 1), np.stack([-half, yb], 1)[::-1]])
    y_ped, y_end = length * (0.5 - 0.82), -0.5 * length
    tail = np.array([[0.08 * width, y_ped], [0.45 * width, y_end], [0.0, y_end + 0.06 * length],
                     [-0.45 * width, y_end], [-0.08 * width, y_ped]])
    return body, tail


class ChannelSimulator:
    def __init__(self, width: int = 640, height: int = 480, mm_per_px: float = 0.4,
                 length_range_mm=(40, 70), speed_range=(5.0, 14.0), width_ratio: float = 0.16,
                 seed: int = 0):
        self.W, self.H = width, height
        self.mm = mm_per_px
        self.lr = length_range_mm
        self.sr = speed_range
        self.wr = width_ratio
        self.rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        r2 = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
        # LED: 235 en el centro, ~150 en las esquinas, + franjas de los módulos LED
        self.led = (235 - 55 * r2 + 6 * np.sin(xx / 23.0)).astype(np.float32)
        self.xx, self.yy = xx, yy
        self.fish: list[SimFish] = []

    # ---------------------------------------------------------------- escenario
    def build_schedule(self, n_fish: int, p_side: float = 0.12, p_tail: float = 0.08,
                       mean_gap: float = 18.0) -> int:
        """Genera peces aislados, parejas lado a lado y parejas en fila."""
        t = 10
        count = 0
        while count < n_fish:
            L = self.rng.uniform(*self.lr) / self.mm
            v = self.rng.uniform(*self.sr)
            ang = self.rng.normal(0, 0.08)
            x = self._free_x(t, L * self.wr * 2.2)
            if x is None:
                t += 5
                continue
            u = self.rng.random()
            if u < p_side and count + 2 <= n_fish:            # lado a lado, tocándose
                w = L * self.wr
                self.fish.append(SimFish(t, x - 0.48 * w, L, v, 0.0))
                self.fish.append(SimFish(t, x + 0.48 * w, L * self.rng.uniform(0.9, 1.1), v, 0.0))
                count += 2
            elif u < p_side + p_tail and count + 2 <= n_fish:  # en fila, cabeza con cola
                self.fish.append(SimFish(t, x, L, v, 0.0))
                self.fish.append(SimFish(t, x, L, v, 0.0, lag=0.97 * L))
                count += 2
            else:
                self.fish.append(SimFish(t, x, L, v, ang))
                count += 1
            t += int(self.rng.exponential(mean_gap)) + 3
        last = max(f.spawn + (self.H + 2 * f.length + f.lag) / f.speed for f in self.fish)
        return int(last) + 20

    def _free_x(self, t: int, need: float) -> float | None:
        """Elige una posición transversal que no choque con peces aún visibles."""
        for _ in range(30):
            x = self.rng.uniform(0.12 * self.W, 0.88 * self.W)
            ok = True
            for f in self.fish:
                still_visible = f.spawn + (self.H + 2.5 * f.length + f.lag) / f.speed > t
                if still_visible and abs(f.x - x) < need + f.length * self.wr * 1.3:
                    ok = False
                    break
            if ok:
                return x
        return None

    # ---------------------------------------------------------------- render
    def render(self, t: int, with_fish: bool = True) -> np.ndarray:
        trans = np.full((self.H, self.W), 255, np.uint8)  # transmisión (255 = sin pez)
        if with_fish:
            for f in self.fish:
                cy = (t - f.spawn) * f.speed - f.length / 2 - f.lag
                if cy < -f.length or cy > self.H + f.length:
                    continue
                body, tail = _fish_polygons(f.length, f.length * self.wr)
                c, s = np.cos(f.angle), np.sin(f.angle)
                rot = np.array([[c, -s], [s, c]])
                for poly, level in ((tail, 120), (body, 35)):   # cola translúcida
                    p = poly @ rot.T + [f.x, cy]
                    layer = np.full_like(trans, 255)
                    cv2.fillPoly(layer, [np.round(p * 4).astype(np.int32)], level,
                                 lineType=cv2.LINE_AA, shift=2)
                    np.minimum(trans, layer, out=trans)
        caustic = 5 * np.sin(self.xx / 17.0 + t * 0.35) * np.sin(self.yy / 13.0 - t * 0.2)
        img = self.led * (trans.astype(np.float32) / 255.0) + caustic
        img += self.rng.normal(0, 2.5, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    def frames(self, n_frames: int, with_fish: bool = True):
        for t in range(n_frames):
            yield self.render(t, with_fish)
