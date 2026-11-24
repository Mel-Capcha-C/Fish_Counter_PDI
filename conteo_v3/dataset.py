"""Lectura del dataset FCDataZJU (MicropterusSalmoides) y división de videos.

Las cajas están en coordenadas de ``images/`` (recorte de 1280x720 sin la pared).
Las cajas ``attach`` delimitan GRUPOS, no peces, y se excluyen.
El total por video es el nombre de la carpeta (inferido: no hay documentación).
"""
from __future__ import annotations

import os
import random
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

# Otra especie del dataset: definir la variable de entorno FC_ESPECIE=CarassiusAuratus
ESPECIE = os.environ.get("FC_ESPECIE", "MicropterusSalmoides")
ROOT = Path(__file__).resolve().parents[1] / "21601548" / "FCDataZJU" / ESPECIE
W, H = 1280, 720
VISTOS = ("m26", "m13", "m8", "m15", "m19")          # ya analizados en detalle
N_RESERVA = {200: 2, 500: 3, 800: 2, 1000: 3}        # 10 videos


def todos() -> dict[str, int]:
    return {p.stem: int(p.parent.parent.name) for p in sorted(ROOT.glob("*/videos/*.mp4"))}


def division() -> tuple[list[str], list[str]]:
    """(ajuste, reserva). Sorteo estratificado con semilla fija; los vistos nunca van a reserva."""
    rng, v = random.Random(31), todos()
    res = []
    if ESPECIE != "MicropterusSalmoides":       # la división sólo está definida para lubina
        return sorted(v), []
    for g, n in N_RESERVA.items():
        res += rng.sample(sorted(k for k in v if v[k] == g and k not in VISTOS), n)
    return sorted(set(v) - set(res)), sorted(res)


@lru_cache(None)
def cajas(video: str) -> dict[int, np.ndarray]:
    """{cuadro: array (n, 4) = xmin, ymin, xmax, ymax}. Cuadro sin XML -> sin peces."""
    d = ROOT / str(todos()[video]) / "labels" / video
    out = {}
    for f in os.listdir(d):
        sel: list[list[float]] = []
        for o in ET.parse(d / f).iter("object"):
            if o.find("name").text.startswith("attach"):
                continue
            b = [float(o.find("bndbox").find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
            if all(np.abs(np.subtract(b, q)).max() > 4 for q in sel):   # duplicadas exactas
                sel.append(b)
        out[int(re.sub(r"\D", "", f))] = np.array(sel, float).reshape(-1, 4)
    return {k: out.get(k, np.zeros((0, 4))) for k in range(min(out), max(out) + 1)}
