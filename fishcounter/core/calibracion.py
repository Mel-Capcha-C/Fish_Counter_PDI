"""Calibración del movimiento a partir de peces aislados (sin etiquetas)."""
from __future__ import annotations

import numpy as np

from .tracking import W, centro, desplazamiento

MIN_PARES, MIN_TERNAS = 150, 40


class DatosInsuficientes(ValueError):
    pass


def _siguiente(c, k, i, ventana=110, aislado=160):
    A = c[k]
    p = A[i]
    y = centro(p)[1]
    if any(j != i and abs(centro(A[j])[1] - y) < aislado for j in range(len(A))):
        return None
    q = [j for j, b in enumerate(c.get(k + 1, []))
         if abs(centro(b)[1] - y) < ventana and b[0] > p[0] + 50]
    return q[0] if len(q) == 1 else None


def aislados(secuencias, borde=4.0):
    """secuencias: lista de {cuadro: cajas}."""
    pares, ternas = [], []
    for c in secuencias:
        for k in sorted(c):
            for i in range(len(c[k])):
                j = _siguiente(c, k, i)
                if j is None or c[k + 1][j][2] > W - borde or c[k][i][0] < borde:
                    continue
                pares.append((c[k][i], c[k + 1][j]))
                n = _siguiente(c, k + 1, j)
                if n is not None and c[k + 2][n][2] <= W - borde:
                    ternas.append((c[k][i], c[k + 1][j], c[k + 2][n]))
    return pares, ternas


def _mad(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def ajustar(pares, ternas, borde=4.0) -> dict:
    """Valores de los parámetros de movimiento del perfil."""
    if len(pares) < MIN_PARES or len(ternas) < MIN_TERNAS:
        raise DatosInsuficientes(f"Hay {len(pares)} pares y {len(ternas)} ternas de peces aislados; se "
                                 f"necesitan al menos {MIN_PARES} y {MIN_TERNAS}.")
    P = [(a, b) for a, b in pares if centro(a)[0] < 560]          # evita censura por salida
    dx = np.array([desplazamiento(a, b, borde)[0] for a, b in P])
    c = np.array([centro(a) for a, _ in P])
    u, w = np.clip(c[:, 0], 0, 700) / 1000 - .3, c[:, 1] / 720 - .5
    A = np.stack([np.ones_like(u), u, u * u, w, w * w], 1)
    ok = np.ones(len(dx), bool)
    for _ in range(4):
        coef = np.linalg.lstsq(A[ok], dx[ok], rcond=None)[0]
        r = dx - A @ coef
        sx = float(1.4826 * np.median(np.abs(r[ok])))
        ok = np.abs(r) < 3.5 * sx
    dy = np.array([centro(b)[1] - centro(a)[1] for a, b in P])
    my, sy = float(np.median(dy)), _mad(dy)

    def salto(x, y):
        a, b = min(max(x, 0), 700) / 1000 - .3, y / 720 - .5
        return coef[0] + coef[1] * a + coef[2] * a * a + coef[3] * b + coef[4] * b * b

    def ex(a, b):
        return desplazamiento(a, b, borde)[0] - salto(*centro(a))

    def ey(a, b):
        return centro(b)[1] - centro(a)[1] - my

    r1 = np.array([ex(a, b) for a, b, _ in ternas])
    r2 = np.array([ex(b, d) for _, b, d in ternas])
    g1 = np.array([ey(a, b) for a, b, _ in ternas])
    g2 = np.array([ey(b, d) for _, b, d in ternas])
    k = (np.abs(r1) < 4 * sx) & (np.abs(r2) < 4 * sx)
    rho = float(np.clip((r1[k] @ r2[k]) / (r1[k] @ r1[k]), 0, 1))
    rhoy = float(np.clip((g1[k] @ g2[k]) / (g1[k] @ g1[k]), 0, 1))
    ent = [(a, b) for a, b in P if a[0] > borde and b[2] < W - borde]
    tw = np.log([b[2] - b[0] for _, b in ent])
    th = np.log([b[3] - b[1] for _, b in ent])
    return dict(
        c0=float(coef[0]), c1=float(coef[1]), c2=float(coef[2]), c3=float(coef[3]), c4=float(coef[4]),
        sx=sx, sx2=float(1.4826 * np.median(np.abs(r2[k] - rho * r1[k]))), rho=rho,
        my=my, sy=sy, sy2=float(1.4826 * np.median(np.abs(g2[k] - rhoy * g1[k]))), rhoy=rhoy,
        slw=_mad(np.array([np.log((b[2] - b[0]) / (a[2] - a[0])) for a, b in ent])),
        slh=_mad(np.array([np.log((b[3] - b[1]) / (a[3] - a[1])) for a, b in ent])),
        pob_w=float(tw.mean()), pob_sw=float(tw.std()), pob_h=float(th.mean()), pob_sh=float(th.std()))
