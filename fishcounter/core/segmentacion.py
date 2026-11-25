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


# La corrección se hace por bloques de filas (caben en la caché) y en dos hilos: en la Raspberry Pi 5
# el límite es la memoria, y con más hilos va más lento.
_FILAS, _HILOS = 64, 2
_hilos = ThreadPoolExecutor(_HILOS, thread_name_prefix="campo")


def campo_plano(roi: np.ndarray, gan: np.ndarray) -> np.ndarray:
    """trunc(min(roi * gan, 255)); igual que ``np.clip(roi * gan, 0, 255).astype(np.uint8)``."""
    gan = gan.astype(np.float32, copy=False)
    h, w = roi.shape
    out = np.empty((h, w), np.uint8)

    def parte(i):
        buf = np.empty((_FILAS, w), np.float32)
        fin = h * (i + 1) // _HILOS
        for y in range(h * i // _HILOS, fin, _FILAS):
            y1 = min(y + _FILAS, fin)
            b = buf[:y1 - y]
            cv2.multiply(roi[y:y1], gan[y:y1], b, dtype=cv2.CV_32F)
            cv2.min(b, 255.0, b)
            np.copyto(out[y:y1], b, casting="unsafe")
    list(_hilos.map(parte, range(_HILOS)))
    return out


def suavizar(img: np.ndarray, p: Perfil) -> np.ndarray:
    return cv2.GaussianBlur(img, (p.gauss, p.gauss), 0) if p.gauss > 1 else img


def umbralizar(g: np.ndarray, p: Perfil) -> np.ndarray:
    return cv2.threshold(g, p.umbral - 1, 1, cv2.THRESH_BINARY_INV)[1]      # 1 donde g < umbral


def _zona(m: np.ndarray, margen: int) -> tuple | None:
    """(y, x) como cortes pares que contienen todo lo no nulo, o None si no hay nada."""
    x, y, w, h = cv2.boundingRect(m)
    if not w:
        return None
    H, W = m.shape
    return (slice(max(y - margen, 0) & ~1, min(y + h + margen, H)),
            slice(max(x - margen, 0) & ~1, min(x + w + margen, W)))


def abrir(m: np.ndarray, p: Perfil) -> np.ndarray:
    if p.apertura <= 1:
        return m
    z = _zona(m, p.apertura + 1)
    out = np.zeros_like(m)
    if z:
        out[z] = cv2.morphologyEx(m[z], cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.apertura, p.apertura)))
    return out


def componentes(m: np.ndarray, g: np.ndarray, p: Perfil) -> tuple[list[Mancha], list[tuple]]:
    """(manchas válidas, [(x, y, w, h, área) descartadas por tamaño])."""
    z = _zona(m, 1)
    if z is None:
        return [], []
    y0, x0 = z[0].start, z[1].start
    n, lab, st, _ = cv2.connectedComponentsWithStats(m[z], connectivity=4)
    H, W = g.shape
    ok, chicas = [], []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        r = lab[y:y + h, x:x + w] == i
        x, y = x + x0, y + y0
        if a < p.area_min:
            chicas.append((x, y, w, h, a))
            continue
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
        self.hist: deque = deque()              # [binaria, reducida, holgada] de los últimos cuadros

    def _reducida(self, e) -> np.ndarray:
        """Versión reducida de un cuadro; sólo se calcula si hace falta (es lo más caro)."""
        if e[1] is None:
            H, W = e[0].shape
            ch = cv2.resize(e[0] * 255, (W // self.ESCALA, H // self.ESCALA), interpolation=cv2.INTER_AREA)
            e[1] = (ch >= 128).astype(np.int16)
        return e[1]

    def _holgada(self, e) -> np.ndarray:
        """Cota superior de ``_reducida``, 3 veces más barata: celdas enteras de 4x4 más sus vecinas.

        Una celda exacta con la mitad oscura se reparte como mucho entre 2x2 celdas enteras: alguna
        tiene al menos 1/8 oscuro. El borde que las celdas enteras no cubren se da por posible.
        """
        if e[2] is None:
            E = self.ESCALA
            h, w = e[0].shape[0] // E, e[0].shape[1] // E
            a = cv2.resize(e[0][:h * E, :w * E] * 255, (w, h), interpolation=cv2.INTER_AREA)
            q = cv2.dilate((a >= 24).astype(np.uint8), np.ones((3, 3), np.uint8))
            q[-1], q[:, -1] = 1, 1
            e[2] = q
        return e[2]

    def _posibles(self, ms: list[Mancha], forma) -> list[Mancha]:
        """Manchas que PODRÍAN ser estáticas (condición necesaria, nunca descarta una que lo sea)."""
        if len(self.hist) < self.p.quietos_min:
            return []
        ms = [m for m in ms if self._posible(m)]
        if not ms:
            return []
        s = self._holgada(self.hist[0]).copy()
        for e in list(self.hist)[1:]:
            cv2.add(s, self._holgada(e), s)
        s = s >= self.p.quietos_min
        (H, W), (h, w) = forma, s.shape
        return [m for m in ms if s[m.y * h // H:(m.y + m.mask.shape[0] - 1) * h // H + 1,
                                   m.x * w // W:(m.x + m.mask.shape[1] - 1) * w // W + 1].any()]

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
        dudosas = self._posibles(ms, g.shape)
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
        self.hist.append([binaria, None, None])
        if len(self.hist) > self.p.quietos_ventana:
            self.hist.popleft()
        return mov, est
