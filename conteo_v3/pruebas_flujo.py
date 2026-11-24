"""Pruebas sintéticas del flujo de costo mínimo: python pruebas_flujo.py"""
import unittest
from pathlib import Path

import numpy as np

from flujo import Evidencia, resolver
from tracking import Movimiento

MOV = Movimiento.cargar(Path(__file__).with_name("movimiento_pixeles.json"))
EV = Evidencia()
EV.doble_ent = [.4, .4, .3, .1, .01]           # solidez baja -> probable grupo
EV.falsa_sep = [.02, .05, .1, .2, .3]


def det(x, y, w=120, h=40, area=3000, razon=0.0, sol=.97):
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2, area, razon, sol, 0.0, w * h]


def pez(x0, y0, v=0.0, n=4):
    """Centros sucesivos de un pez según el modelo de movimiento, con desvío v de velocidad."""
    out, x, y = [], float(x0), float(y0)
    for _ in range(n):
        if x > 1240:
            break
        out.append((x, y))
        x, y = x + MOV.salto(x, y) + v, y + MOV.my
    return out


def arma(*listas):
    D = {}
    for L in listas:
        for k, d in L:
            D.setdefault(k, []).append(d)
    return {k: np.array(v, float) for k, v in D.items()}


class Flujo(unittest.TestCase):
    def test_un_pez(self):
        D = arma([(k, det(*p)) for k, p in enumerate(pez(200, 300))])
        self.assertEqual(len(resolver(D, MOV, EV)[0]), 1)

    def test_dos_peces_independientes(self):
        D = arma([(k, det(*p)) for k, p in enumerate(pez(200, 150))],
                 [(k, det(*p)) for k, p in enumerate(pez(120, 550))])
        self.assertEqual(len(resolver(D, MOV, EV)[0]), 2)

    def test_pez_nuevo_no_absorbe_al_que_salio(self):
        D = arma([(k, det(*p)) for k, p in enumerate(pez(200, 300))],
                 [(k + 4, det(*p)) for k, p in enumerate(pez(150, 300))])
        self.assertEqual(len(resolver(D, MOV, EV)[0]), 2)

    def test_grupo_en_el_cuadro_central(self):
        """Dos peces separados, unidos en el cuadro 1 y separados otra vez: son 2."""
        a, b = pez(150, 280), pez(150, 340)
        u = det((a[1][0] + b[1][0]) / 2, (a[1][1] + b[1][1]) / 2, 125, 100, 6000, 0, .78)
        D = arma([(0, det(*a[0])), (0, det(*b[0])), (1, u), (2, det(*a[2])), (2, det(*b[2]))])
        t, un, _ = resolver(D, MOV, EV)
        self.assertEqual(len(t), 2)
        self.assertEqual(un[(1, 0)], 2)

    def test_grupo_en_el_primer_cuadro(self):
        a, b = pez(150, 280), pez(150, 340)
        u = det(150, 310, 125, 100, 6000, 0, .78)
        D = arma([(0, u), (1, det(*a[1])), (1, det(*b[1])), (2, det(*a[2])), (2, det(*b[2]))])
        self.assertEqual(len(resolver(D, MOV, EV)[0]), 2)

    def test_pieza_partida_por_error(self):
        """Un pez entero en 3 cuadros, partido en dos en el central: es 1."""
        a = pez(150, 300)
        x, y = a[1]
        D = arma([(0, det(*a[0])), (1, det(x + 10, y, 90, 40, 2200, .85, .85)),
                  (1, det(x - 45, y, 40, 25, 800, .85, .85)), (2, det(*a[2]))])
        t, un, _ = resolver(D, MOV, EV)
        self.assertEqual(len(t), 1)

    def test_pez_perdido_un_cuadro(self):
        a = pez(80, 300)
        D = arma([(k, det(*p)) for k, p in enumerate(a) if k != 1])
        self.assertEqual(len(resolver(D, MOV, EV)[0]), 1)


if __name__ == "__main__":
    unittest.main()
