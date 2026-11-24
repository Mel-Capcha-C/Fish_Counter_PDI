"""Audita pez por pez: trayectorias de píxeles contra trayectorias de las cajas anotadas.

conteo = peces reales encontrados + duplicados + sin pez real. Expone compensaciones.
La 'verdad' por pez es el tracking de las cajas (error ~1 %), no una anotación de identidad.
"""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from dataset import cajas, todos
from segmentacion import etiquetas_a_roi
from tracking import contar


def _puntaje(a, b):
    """IoU, o fracción contenida atenuada (cajas de tamaños distintos por colas translúcidas)."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    i, A, B = ix * iy, (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return max(i / (A + B - i), .6 * i / min(A, B))


def auditar(v, dets, r, mov_cajas):
    L = cajas(v)
    rl = contar(L, mov_cajas)
    nobs = Counter(i for ids in rl.ids.values() for i in ids)
    real = {i for i, n in nobs.items() if n > 1}
    voto = {}
    for k in r.ids:
        if k not in L or not len(L[k]) or not len(dets[k]):
            continue
        G = etiquetas_a_roi(L[k])
        S = np.array([[_puntaje(a, b) for b in G] for a in dets[k][:, :4]])
        for a, b in zip(*linear_sum_assignment(-S)):
            if S[a, b] > .2:
                voto.setdefault(r.ids[k][a], Counter())[rl.ids[k][b]] += 1
    base = set(r.contadas) - set(r.recuperadas)

    def pares(tr):
        """Máximo de trayectorias con un pez real distinto: un intercambio de identidad
        entre dos peces contados no resta ni suma, porque no altera el conteo."""
        tr, pez = sorted(tr), sorted(real)
        ip = {x: j for j, x in enumerate(pez)}
        M = np.zeros((len(tr), len(pez)), np.float32)
        for a, t in enumerate(tr):
            for x, n in voto.get(t, {}).items():
                if x in ip:
                    M[a, ip[x]] = 1 + .01 * n
        a, b = linear_sum_assignment(-M)
        return int((M[a, b] > 0).sum())

    e1, e0 = pares(r.contadas), pares(base)
    return dict(real=len(real), total=todos()[v], conteo=len(r.contadas), encontrados=e1,
                de_mas=len(r.contadas) - e1, perdidos=len(real) - e1,
                base_conteo=len(base), base_encontrados=e0, recuperadas=len(r.recuperadas), recuperadas_ok=e1 - e0)
