"""Recuperación de trayectorias ocluidas por consistencia temporal (antes «técnica 15»).

Un pez puede verse suelto en un cuadro y dentro de un grupo en el vecino: su trayectoria tiene una
sola observación. Se recupera si (1) su observación es clara, (2) su posición predicha o
retropredicha cae en otra detección y (3) esa detección es anormalmente grande para el pez que la
ocupa, comparada con las demás vistas COMPLETAS de ese pez.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perfil import Perfil
from .tracking import W, centro


@dataclass(frozen=True)
class Motivo:
    cuadro: int                 # cuadro donde quedó oculto
    deteccion: int              # índice de la detección que lo oculta
    pista: int                  # pista que ocupa esa detección
    razon_area: float           # área / mediana de las otras vistas de esa pista


def cortada(d, p: Perfil) -> bool:
    return d[0] < p.borde or d[2] > W - p.borde


def buscar_oclusion(i: int, obs: dict, dets: dict, ids: dict, p: Perfil) -> Motivo | None:
    """Motivo por el que la trayectoria ``i`` (de una sola vista) se recupera, o None."""
    k, j = obs[i][0]
    d = dets[k][j]
    if d[5] > p.razon_clara:
        return None
    x, y = centro(d)
    for sg in (1, -1):
        v = p.salto(x, y) if sg > 0 else p.salto(max(x - p.retro_salto, 0), y)
        xp, yp = x + sg * v, y + sg * p.my
        if not -20 < xp < W + 20 or k + sg not in dets or not len(dets[k + sg]):
            continue
        q = dets[k + sg]
        en = ((q[:, 0] - p.sx < xp) & (xp < q[:, 2] + p.sx) &
              (q[:, 1] - p.margen_oclusion < yp) & (yp < q[:, 3] + p.margen_oclusion))
        for t in np.flatnonzero(en):
            if cortada(q[t], p):
                continue
            otra = ids[k + sg][t]
            A = [dets[a][b][4] for a, b in obs[otra] if a != k + sg and not cortada(dets[a][b], p)]
            if A and q[t][4] >= p.exceso * np.median(A):
                return Motivo(k + sg, int(t), otra, float(q[t][4] / np.median(A)))
    return None
