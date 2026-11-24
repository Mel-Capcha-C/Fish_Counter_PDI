"""Segmentación desde los píxeles del video (técnicas 1-5, 7 y 8). Sin etiquetas.

ROI -> intensidad -> flat-field -> gaussiano 3x3 -> umbral fijo -> apertura -> componentes.

Cambios respecto del diseño anterior:
  * Fondo = mediana temporal de cuadros repartidos por TODO el video (no exige que
    empiece vacío: los peces cubren poco y la mediana los ignora).
  * Sin cierre morfológico y con conexión de 4 vecinos: el cierre volvía a unir
    peces separados por una franja fina.
  * Objetos quietos (gotas, suciedad): se descartan por persistencia temporal, ver Quietos.
Las coordenadas de salida son las del ROI llevado a 1280x720.
"""
from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dataset import ROOT, todos

X0, Y0 = 66, 36                         # pared izquierda y riel superior
FX, FY = 1280 / (1280 - X0), 720 / (720 - Y0)
NIVEL = 200.0
UMBRAL = 146
AREA_MIN = 350                          # px del video; el pez más pequeño visto tiene ~1500

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@dataclass
class Mancha:
    x: int                              # esquina en el ROI
    y: int
    mask: np.ndarray                    # uint8 0/1
    gris: np.ndarray                    # intensidad corregida (sólo esta mancha; resto 255)
    area: int
    toca: tuple                         # (izq, der, arriba, abajo)


def cuadros(video: str):
    ruta = ROOT / str(todos()[video]) / "videos" / f"{video}.mp4"
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir {ruta}")
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[Y0:, X0:]
    finally:
        cap.release()


def fondo(video: str, n: int = 60) -> np.ndarray:
    """Mediana de n cuadros repartidos. Sólo guarda esos n (no el video entero) en memoria."""
    total = sum(1 for _ in cuadros(video))
    if not total:
        raise ValueError(f"El video {video} no tiene cuadros")
    idx = set(np.linspace(0, total - 1, n).astype(int).tolist())
    return np.median(np.stack([f for k, f in enumerate(cuadros(video)) if k in idx]), axis=0).astype(np.float32)


def corregir(f: np.ndarray, ganancia: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(np.clip(f * ganancia, 0, 255).astype(np.uint8), (3, 3), 0)


def manchas(g: np.ndarray, umbral: int = UMBRAL) -> list[Mancha]:
    m = cv2.morphologyEx((g < umbral).astype(np.uint8), cv2.MORPH_OPEN, _K3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    H, W = g.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < AREA_MIN:
            continue
        r = (lab[y:y + h, x:x + w] == i)
        gr = np.where(r, g[y:y + h, x:x + w], 255).astype(np.uint8)
        out.append(Mancha(x, y, r.astype(np.uint8), gr, a, (x <= 1, x + w >= W - 1, y <= 1, y + h >= H - 1)))
    return out


class Quietos:
    """Descarta manchas formadas por píxeles que llevan oscuros casi todo el último tramo.

    Un pez avanza unos 3 cuerpos por cuadro: no puede oscurecer el mismo píxel en
    MIN_OSCURO de los últimos VENTANA cuadros. Una gota o suciedad sí. La mancha se quita
    sólo si la MAYORÍA de su área es persistente: un pez que pasa sobre la gota se conserva.
    Es causal (mira hacia atrás): tarda MIN_OSCURO cuadros en reconocer un objeto nuevo.
    SUPONE que los peces no se detienen: en un canal lento habría que ampliar VENTANA.
    """
    VENTANA, MIN_OSCURO, FRACCION, ESC = 10, 6, 0.6, 4

    def __init__(self):
        self.hist: deque = deque()
        self.suma = None
        self.descartadas = 0

    def filtrar(self, g: np.ndarray, ms: list[Mancha]) -> list[Mancha]:
        H, W = g.shape
        ch = cv2.resize((g < UMBRAL).astype(np.uint8) * 255, (W // self.ESC, H // self.ESC),
                        interpolation=cv2.INTER_AREA)
        osc = (ch >= 128).astype(np.int16)
        out = ms
        if self.suma is not None and self.suma.max() >= self.MIN_OSCURO:
            per = cv2.resize((self.suma >= self.MIN_OSCURO).astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST)
            out = []
            for m in ms:
                h, w = m.mask.shape
                if per[m.y:m.y + h, m.x:m.x + w][m.mask > 0].mean() >= self.FRACCION:
                    self.descartadas += 1
                else:
                    out.append(m)
        self.suma = osc.copy() if self.suma is None else self.suma + osc
        self.hist.append(osc)
        if len(self.hist) > self.VENTANA:
            self.suma -= self.hist.popleft()
        return out


def procesar(video: str, quietos: bool = True) -> list[list[Mancha]]:
    ganancia = NIVEL / np.maximum(fondo(video), 1.0)
    q, R = Quietos(), []
    for f in cuadros(video):
        g = corregir(f, ganancia)
        ms = manchas(g)
        R.append(q.filtrar(g, ms) if quietos else ms)
    return R


def caja(x, y, w, h) -> np.ndarray:
    """Caja del ROI (px del video) -> coordenadas 1280x720."""
    return np.array([x * FX, y * FY, (x + w) * FX, (y + h) * FY])


def etiquetas_a_roi(b: np.ndarray) -> np.ndarray:
    """Cajas anotadas (recorte images/) -> coordenadas del ROI. Sólo para EVALUAR."""
    q = b.copy()
    q[:, [0, 2]] = (b[:, [0, 2]] * .962 + 68 - X0) * FX
    q[:, [1, 3]] = (b[:, [1, 3]] * .917 + 40 - Y0) * FY
    return np.clip(q, 0, [1280, 720, 1280, 720])


def cache(video: str, carpeta: str | Path) -> list[list[Mancha]]:
    p = Path(carpeta) / f"{video}.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    r = procesar(video)
    p.parent.mkdir(exist_ok=True, parents=True)
    with open(p, "wb") as fh:
        pickle.dump(r, fh)
    return r
