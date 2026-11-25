"""Preprocesamiento y segmentación: de un cuadro BGR a manchas de peces."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .perfil import Perfil


@dataclass
class Mancha:
    x: int                              # esquina en el ROI
    y: int
    mask: np.ndarray                    # uint8 0/1
    gris: np.ndarray                    # intensidad corregida (resto 255)
    area: int
    toca: tuple                         # (izq, der, arriba, abajo)


def recortar(bgr: np.ndarray, p: Perfil) -> np.ndarray:
    """Recorte de la ROI y conversión a luminancia."""
    return cv2.cvtColor(bgr[p.roi_y0:, p.roi_x0:], cv2.COLOR_BGR2GRAY)


def ganancia(fondo: np.ndarray, p: Perfil) -> np.ndarray:
    return p.nivel / np.maximum(fondo, 1.0)


_TIRAS = 4                              # la imagen se corrige en tiras, en paralelo
_hilos = ThreadPoolExecutor(_TIRAS, thread_name_prefix="campo")


def campo_plano(roi: np.ndarray, gan: np.ndarray) -> np.ndarray:
    """trunc(min(roi * gan, 255)); igual que ``np.clip(roi * gan, 0, 255).astype(np.uint8)``."""
    gan = gan.astype(np.float32, copy=False)
    h = roi.shape[0]
    out = np.empty(roi.shape, np.uint8)

    def tira(i):
        z = slice(h * i // _TIRAS, h * (i + 1) // _TIRAS)
        b = cv2.multiply(roi[z], gan[z], dtype=cv2.CV_32F)
        cv2.min(b, 255.0, b)
        np.copyto(out[z], b, casting="unsafe")
    list(_hilos.map(tira, range(_TIRAS)))
    return out


def suavizar(img: np.ndarray, p: Perfil) -> np.ndarray:
    return cv2.GaussianBlur(img, (p.gauss, p.gauss), 0) if p.gauss > 1 else img


def umbralizar(g: np.ndarray, p: Perfil) -> np.ndarray:
    return cv2.threshold(g, p.umbral - 1, 1, cv2.THRESH_BINARY_INV)[1]      # 1 donde g < umbral


def abrir(m: np.ndarray, p: Perfil) -> np.ndarray:
    if p.apertura <= 1:
        return m
    return cv2.morphologyEx(m, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.apertura, p.apertura)))


def componentes(m: np.ndarray, g: np.ndarray, p: Perfil) -> tuple[list[Mancha], list[tuple]]:
    """(manchas válidas, [(x, y, w, h, área) descartadas por tamaño])."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    H, W = g.shape
    ok, chicas = [], []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < p.area_min:
            chicas.append((x, y, w, h, a))
            continue
        r = lab[y:y + h, x:x + w] == i
        gr = np.where(r, g[y:y + h, x:x + w], 255).astype(np.uint8)
        ok.append(Mancha(x, y, r.astype(np.uint8), gr, a, (x <= 1, x + w >= W - 1, y <= 1, y + h >= H - 1)))
    return ok, chicas


class Estaticos:
    """Supresión de objetos estáticos por persistencia temporal (causal).

    Un pez no puede oscurecer el mismo píxel en ``quietos_min`` de los últimos
    ``quietos_ventana`` cuadros. Se quita la mancha sólo si la mayoría de su área es persistente.
    """
    ESCALA = 4

    def __init__(self, p: Perfil):
        self.p = p
        self.hist: deque = deque()              # [binaria, reducida o None] de los últimos cuadros

    def _reducida(self, e) -> np.ndarray:
        """Versión reducida de un cuadro; sólo se calcula si hace falta (es lo más caro)."""
        if e[1] is None:
            H, W = e[0].shape
            ch = cv2.resize(e[0] * 255, (W // self.ESCALA, H // self.ESCALA), interpolation=cv2.INTER_AREA)
            e[1] = (ch >= 128).astype(np.int16)
        return e[1]

    def _posible(self, m: Mancha) -> bool:
        """Condición necesaria y barata: algo oscuro junto a la mancha en ``quietos_min`` cuadros."""
        h, w = m.mask.shape
        M = 2 * self.ESCALA
        z = np.s_[max(m.y - M, 0):m.y + h + M, max(m.x - M, 0):m.x + w + M]
        faltan, quedan = self.p.quietos_min, len(self.hist)
        for e in self.hist:
            if faltan > quedan:
                return False
            quedan -= 1
            faltan -= bool(cv2.countNonZero(e[0][z]))
            if faltan <= 0:
                return True
        return False

    def filtrar(self, g: np.ndarray, ms: list[Mancha], binaria: np.ndarray | None = None
                ) -> tuple[list[Mancha], list[Mancha]]:
        """(móviles, estáticas). ``binaria`` es la umbralización de ``g``, si ya se calculó."""
        H, W = g.shape
        if binaria is None:
            binaria = umbralizar(g, self.p)
        mov, est = ms, []
        dudosas = [m for m in ms if self._posible(m)] if len(self.hist) >= self.p.quietos_min else []
        if dudosas:
            suma = sum(self._reducida(e) for e in self.hist)
            if suma.max() >= self.p.quietos_min:
                per = cv2.resize((suma >= self.p.quietos_min).astype(np.uint8), (W, H),
                                 interpolation=cv2.INTER_NEAREST)
                ids = {id(m) for m in dudosas}
                mov = []
                for m in ms:
                    h, w = m.mask.shape
                    quieta = id(m) in ids and \
                        per[m.y:m.y + h, m.x:m.x + w][m.mask > 0].mean() >= self.p.quietos_fraccion
                    (est if quieta else mov).append(m)
        self.hist.append([binaria, None])
        if len(self.hist) > self.p.quietos_ventana:
            self.hist.popleft()
        return mov, est
