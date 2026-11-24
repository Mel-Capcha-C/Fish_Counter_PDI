"""Conteo con cajas anotadas. Uso: python evaluar_tracking.py ajuste|reserva"""
import sys
from pathlib import Path

import numpy as np

from dataset import cajas, division, todos
from tracking import Movimiento, contar


def evaluar(videos, mov):
    e = {"confirmadas": [], "conteo": []}
    for v in videos:
        r, g = contar(cajas(v), mov), todos()[v]
        for k in e:
            e[k].append(100 * (getattr(r, k) / g - 1))
        print(f"{v:4s} real {g:4d} | PRINCIPAL vistas 2+: {r.confirmadas:4d} ({e['confirmadas'][-1]:+.1f} %)"
              f" | todas: {r.conteo:4d} ({e['conteo'][-1]:+.1f} %)")
    for k, x in e.items():
        x = np.array(x)
        print(f"{k:12s} sesgo {x.mean():+.2f} %  error medio {abs(x).mean():.2f} %  peor {abs(x).max():.2f} %")


if __name__ == "__main__":
    sel = division()[0 if sys.argv[1] == "ajuste" else 1]
    evaluar(sel, Movimiento.cargar(Path(__file__).with_name("movimiento.json")))
