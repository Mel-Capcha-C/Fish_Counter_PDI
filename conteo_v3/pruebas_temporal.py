"""Pruebas de tracking y técnica 15: python pruebas_temporal.py"""
import unittest
from pathlib import Path

import numpy as np

from temporal import contar_con_grupos
from tracking import Movimiento, Pista, Rastreador, contar

MOV = Movimiento.cargar(Path(__file__).with_name("movimiento_pixeles.json"))


def det(x, y, w=120, h=40, area=3000, razon=0.0):
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2, area, razon]


def arma(D):
    return {k: np.array(D.get(k, []), float).reshape(-1, 6) for k in range(0, max(D) + 2)}


class Prediccion(unittest.TestCase):
    def test_residuo_decae_por_paso(self):
        r = Rastreador(MOV)
        t = Pista(0, 0, det(150, 300)[:4])
        t.rx, t.ry = 60.0, 10.0
        base = Pista(1, 0, det(150, 300)[:4])
        base.rx, base.ry = 0.0, 0.0
        dy = r._prediccion(t, 2)[1] - r._prediccion(base, 2)[1]
        self.assertAlmostEqual(dy, 10 * (MOV.rhoy + MOV.rhoy ** 2), delta=1e-6)


class Tracking(unittest.TestCase):
    def test_un_pez_perdido_un_cuadro_sigue_siendo_uno(self):
        x = 100.0
        D = {}
        for k in range(3):
            if k != 1:
                D[k] = [det(x, 300 + MOV.my * k)]
            x += MOV.salto(x, 300)
        r = contar({k: v[:, :4] for k, v in arma(D).items()}, MOV)
        self.assertEqual(r.conteo, 1)


class T15(unittest.TestCase):
    def grupo(self, primera):
        """A se ve en los cuadros 0, 1 y 2; en el 1 aparece además una pieza suelta que sólo
        coincide con A en el 2. ``primera`` es la vista 0 de A."""
        x0 = 100.0
        x1 = x0 + MOV.salto(x0, 300)
        x2 = x1 + MOV.salto(x1, 300)
        return {0: [primera], 1: [det(x1, 300), det(x1, 352, h=30)], 2: [det(x2, 320, w=150, h=110)]}

    def test_vista_cortada_no_hace_grande_a_un_pez_normal(self):
        """Las vistas de A son 3000 px, salvo una cortada de 900. La vista 2 (3400 px) no es anormal."""
        D = self.grupo([0, 280, 90, 320, 900, 0.0])
        D[2][0][4] = 3400
        r, total, rec = contar_con_grupos(arma(D), MOV)
        self.assertEqual(rec, 0)

    def test_grupo_verdadero_se_recupera(self):
        """La vista 2 sí es anormal (6200 px frente a 3000): la pieza suelta estaba dentro."""
        D = self.grupo(det(100, 300))
        D[2][0][4] = 6200
        r, total, rec = contar_con_grupos(arma(D), MOV)
        self.assertEqual(rec, 1)


if __name__ == "__main__":
    unittest.main()
