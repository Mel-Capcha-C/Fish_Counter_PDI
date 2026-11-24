"""Técnica 15: usa los cuadros vecinos para decidir las trayectorias vistas UNA sola vez.

Un pez real puede verse separado en un cuadro y dentro de un grupo en el vecino. Su
trayectoria tiene entonces una sola observación y la regla "vista 2+" la descarta.
Se recupera sólo si se cumplen las tres:
  1. Su observación es clara: mancha entera o separación con razón <= RAZON_CLARA.
  2. La posición predicha (cuadro siguiente) o retropredicha (anterior) cae dentro de la
     caja de otra detección.
  3. Esa detección es anormalmente grande para el pez que la ocupa: área >= EXCESO veces
     la que ese mismo pez tenía en sus otras vistas COMPLETAS. Una vista cortada por el borde
     es pequeña de por sí y haría parecer "grande" a un pez normal.
"""
from __future__ import annotations

import numpy as np

from tracking import BORDE, W, centro, contar

RAZON_CLARA, EXCESO, MARGEN = 0.4, 1.25, 10.0


def _cortada(d):
    return d[0] < BORDE or d[2] > W - BORDE


def contar_con_grupos(dets: dict, mov, razon=None, exceso=None):
    """dets: {cuadro: (n, 6)} = caja, área, razón. Devuelve (rastreador, total, recuperadas)."""
    razon, exceso = razon or RAZON_CLARA, exceso or EXCESO
    r = contar({k: d[:, :4] for k, d in dets.items()}, mov)
    obs: dict[int, list] = {}
    for k, ids in r.ids.items():
        for j, i in enumerate(ids):
            obs.setdefault(i, []).append((k, j))
    recup = 0
    r.contadas = {i for i, o in obs.items() if len(o) > 1}
    r.recuperadas = set()
    for i, o in obs.items():
        if len(o) != 1:
            continue
        k, j = o[0]
        d = dets[k][j]
        if d[5] > razon:
            continue
        x, y = centro(d)
        for sg in (1, -1):
            v = mov.salto(x, y) if sg > 0 else mov.salto(max(x - 450, 0), y)
            xp, yp = x + sg * v, y + sg * mov.my
            if not -20 < xp < W + 20 or k + sg not in dets or not len(dets[k + sg]):
                continue
            q = dets[k + sg]
            en = ((q[:, 0] - mov.sx < xp) & (xp < q[:, 2] + mov.sx) &
                  (q[:, 1] - MARGEN < yp) & (yp < q[:, 3] + MARGEN))
            hecho = False
            for t in np.flatnonzero(en):
                if _cortada(q[t]):
                    continue
                otra = r.ids[k + sg][t]
                A = [dets[a][b][4] for a, b in obs[otra] if a != k + sg and not _cortada(dets[a][b])]
                if A and q[t][4] >= exceso * np.median(A):
                    hecho = True
                    break
            if hecho:
                recup += 1
                r.contadas.add(i)
                r.recuperadas.add(i)
                break
    return r, r.confirmadas + recup, recup
