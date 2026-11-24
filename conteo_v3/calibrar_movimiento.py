"""Aprende el movimiento con pares de peces aislados de los videos de AJUSTE (sin totales)."""
from pathlib import Path

from dataset import W, cajas, division
from tracking import ajustar_movimiento, centro


def _siguiente(c, k, i, ventana=110, aislado=160):
    A = c[k]
    p = A[i]
    y = centro(p)[1]
    if any(j != i and abs(centro(A[j])[1] - y) < aislado for j in range(len(A))):
        return None
    q = [j for j, b in enumerate(c.get(k + 1, []))
         if abs(centro(b)[1] - y) < ventana and b[0] > p[0] + 50]
    return q[0] if len(q) == 1 else None


def aislados(secuencias):
    """secuencias: lista de {cuadro: cajas} (anotadas o detectadas)."""
    pares, ternas = [], []
    for c in secuencias:
        for k in sorted(c):
            for i in range(len(c[k])):
                j = _siguiente(c, k, i)
                if j is None or c[k + 1][j][2] > W - 4 or c[k][i][0] < 4:
                    continue
                pares.append((c[k][i], c[k + 1][j]))
                n = _siguiente(c, k + 1, j)
                if n is not None and c[k + 2][n][2] <= W - 4:
                    ternas.append((c[k][i], c[k + 1][j], c[k + 2][n]))
    return pares, ternas


if __name__ == "__main__":
    p, t = aislados([cajas(v) for v in division()[0]])
    m = ajustar_movimiento(p, t)
    m.guardar(Path(__file__).with_name("movimiento.json"))
    print(len(p), "pares", len(t), "ternas", m)
