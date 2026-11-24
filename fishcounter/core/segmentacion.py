"""Preprocesamiento y segmentación: de un cuadro BGR a manchas de peces."""
from __future__ import annotations

from collections import deque
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
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[p.roi_y0:, p.roi_x0:]


def ganancia(fondo: np.ndarray, p: Perfil) -> np.ndarray:
    return p.nivel / np.maximum(fondo, 1.0)


def campo_plano(roi: np.ndarray, gan: np.ndarray) -> np.ndarray:
    return np.clip(roi * gan, 0, 255).astype(np.uint8)


def suavizar(img: np.ndarray, p: Perfil) -> np.ndarray:
    return cv2.GaussianBlur(img, (p.gauss, p.gauss), 0) if p.gauss > 1 else img


def umbralizar(g: np.ndarray, p: Perfil) -> np.ndarray:
    return (g < p.umbral).astype(np.uint8)


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
        self.hist: deque = deque()
        self.suma = None

    def filtrar(self, g: np.ndarray, ms: list[Mancha]) -> tuple[list[Mancha], list[Mancha]]:
        """(móviles, estáticas)."""
        H, W = g.shape
        ch = cv2.resize(umbralizar(g, self.p) * 255, (W // self.ESCALA, H // self.ESCALA),
                        interpolation=cv2.INTER_AREA)
        osc = (ch >= 128).astype(np.int16)
        mov, est = ms, []
        if self.suma is not None and self.suma.max() >= self.p.quietos_min:
            per = cv2.resize((self.suma >= self.p.quietos_min).astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST)
            mov = []
            for m in ms:
                h, w = m.mask.shape
                quieta = per[m.y:m.y + h, m.x:m.x + w][m.mask > 0].mean() >= self.p.quietos_fraccion
                (est if quieta else mov).append(m)
        self.suma = osc.copy() if self.suma is None else self.suma + osc
        self.hist.append(osc)
        if len(self.hist) > self.p.quietos_ventana:
            self.suma -= self.hist.popleft()
        return mov, est
