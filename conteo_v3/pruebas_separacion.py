"""Pruebas del separador: comprueban las CAJAS de los peces, no sólo cuántos son.

    python pruebas_separacion.py
"""
import unittest

import cv2
import numpy as np

from segmentacion import manchas
from separacion import cimas, separar


def lienzo():
    return np.full((684, 1214), 200, np.uint8)


def pez(g, x, y, gris=70, a=70, b=20):
    cv2.ellipse(g, (int(x), int(y)), (a, b), 0, 0, 360, gris, -1)


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    i = max(ix, 0) * max(iy, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


class Separacion(unittest.TestCase):
    def dos(self, dy=40):
        g = lienzo()
        pez(g, 400, 300), pez(g, 415, 300 + dy)
        return manchas(g), [(330, 280, 470, 320), (345, 280 + dy, 485, 320 + dy)]

    def test_cajas_cubren_el_cuerpo(self):
        ms, real = self.dos()                        # se tocan de lado
        self.assertEqual(len(ms), 1)
        p = separar(ms[0])
        self.assertEqual(len(p), 2)
        cajas = [(ms[0].x + x, ms[0].y + y, ms[0].x + x + w, ms[0].y + y + h) for x, y, w, h, _, _ in p]
        for r in real:
            self.assertGreater(max(iou(c, r) for c in cajas), .7)

    def test_reparte_toda_el_area(self):
        ms, _ = self.dos()
        p = separar(ms[0])
        self.assertEqual(len(p), 2)
        self.assertAlmostEqual(sum(q[4] for q in p) / ms[0].area, 1, delta=.02)

    def test_cuello_muy_fino(self):
        """Dos cuerpos unidos por un paso de campo 0,35: el barrido termina en 0,5 y nunca los
        funde. Antes se perdía el segundo."""
        f = np.zeros((60, 160), np.float32)
        yy, xx = np.mgrid[0:60, 0:160]
        for cx, pico in ((40, 12.1), (120, 11.0)):
            f = np.maximum(f, (pico - np.hypot(xx - cx, yy - 30) * .4).clip(0).astype(np.float32))
        f[29:32, 68:92] = np.maximum(f[29:32, 68:92], .35)
        self.assertEqual(len(cimas(f)), 2)

    def test_tres_peces(self):
        g = lienzo()
        for i in range(3):
            pez(g, 400 + 10 * i, 250 + 41 * i)
        self.assertEqual([len(separar(m)) for m in manchas(g)], [3])

    def test_pez_curvado_es_uno(self):
        g = lienzo()
        cv2.ellipse(g, (400, 300), (80, 60), 0, 200, 340, 70, 22)
        self.assertEqual([len(separar(m)) for m in manchas(g)], [1])

    def test_pez_cortado_por_el_borde_es_uno(self):
        g = lienzo()
        pez(g, 20, 300)
        self.assertEqual([len(separar(m)) for m in manchas(g)], [1])

    def test_coordenadas_de_caja(self):
        """Una mancha entera debe devolver exactamente su caja."""
        g = lienzo()
        pez(g, 400, 300)
        m, = manchas(g)
        x, y, w, h, a, r = separar(m)[0]
        self.assertEqual((m.x + x, m.y + y), (m.x, m.y))
        self.assertEqual((w, h), m.mask.shape[::-1])
        self.assertEqual((a, r), (m.area, 0.0))


if __name__ == "__main__":
    unittest.main()
