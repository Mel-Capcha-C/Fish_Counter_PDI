"""Compara la segmentación (sólo píxeles) con las cajas anotadas, mancha por mancha."""
from __future__ import annotations

from collections import Counter

import numpy as np

from calibrar_movimiento import aislados
from dataset import cajas, todos
from segmentacion import FX, FY, caja, etiquetas_a_roi
from tracking import ajustar_movimiento, contar


def verdad_por_mancha(video, ms):
    """[cuadro] -> (lista de cajas reales por mancha, cajas sin mancha)."""
    L, out = cajas(video), []
    for k, M in enumerate(ms):
        b = etiquetas_a_roi(L[k]) if k in L and len(L[k]) else np.zeros((0, 4))
        due, sin = [[] for _ in M], []
        for q in b:
            u = [q[0] / FX, q[1] / FY, q[2] / FX, q[3] / FY]
            mejor = (0, -1)
            for j, m in enumerate(M):
                h, w = m.mask.shape
                i0, i1 = int(max(u[0] - m.x, 0)), int(min(u[2] - m.x, w))
                j0, j1 = int(max(u[1] - m.y, 0)), int(min(u[3] - m.y, h))
                if i1 > i0 and j1 > j0:
                    ov = int(m.mask[j0:j1, i0:i1].sum())
                    if ov > mejor[0]:
                        mejor = (ov, j)
            if mejor[0] > 150:
                due[mejor[1]].append(q)
            else:
                sin.append(q)
        out.append((due, sin))
    return out


def comparar(videos, cargar, separar):
    """separar(mancha) -> lista de cajas (x, y, w, h en px del ROI)."""
    conf, dets, extra = Counter(), {}, Counter()
    for v in videos:
        ms = cargar(v)
        dets[v] = {}
        for k, (M, (due, sin)) in enumerate(zip(ms, verdad_por_mancha(v, ms))):
            B = []
            extra["sin detectar"] += len(sin)
            for m, d in zip(M, due):
                p = separar(m)
                conf[(min(len(d), 5), min(len(p), 6))] += 1
                B += [[*caja(m.x + x, m.y + y, w, h), a, q] for x, y, w, h, a, q in p]
            dets[v][k] = np.array(B, float).reshape(-1, 6)
    return conf, dets, extra


def resumen(conf, extra):
    tot = sum(t * c for (t, p), c in conf.items())
    print(f"  peces en manchas {tot}, sin detectar {extra['sin detectar']}, manchas sin pez {sum(c for (t, p), c in conf.items() if t == 0)}")
    for t in range(0, 6):
        f = {p: conf[(t, p)] for p in range(0, 7) if conf[(t, p)]}
        n = sum(f.values())
        if n:
            print(f"  real {t}{'+' if t == 5 else ' '}: {n:5d} manchas, exactas {100 * f.get(t, 0) / n:5.1f} %  predichas {f}")
    de_mas = sum(max(0, p - t) * c for (t, p), c in conf.items() if t)
    de_menos = sum(max(0, t - p) * c for (t, p), c in conf.items() if t)
    print(f"  piezas de más {100 * de_mas / tot:.1f} %  de menos {100 * de_menos / tot:.1f} %")


def solo_cajas(d):
    return {k: b[:, :4] for k, b in d.items()}


def conteo(videos, dets, mov=None):
    if mov is None:
        mov = ajustar_movimiento(*aislados([solo_cajas(dets[v]) for v in videos]))
    a, b = [], []
    for v in videos:
        r = contar(solo_cajas(dets[v]), mov)
        g = todos()[v]
        a.append(100 * (r.confirmadas / g - 1))
        b.append(100 * (r.conteo / g - 1))
    a, b = np.array(a), np.array(b)
    print(f"  CONTEO vistas 2+: sesgo {a.mean():+.1f} % medio {abs(a).mean():.1f} % peor {abs(a).max():.1f} %"
          f" | todas: sesgo {b.mean():+.1f} % medio {abs(b).mean():.1f} % peor {abs(b).max():.1f} %")
    return mov, a, b
