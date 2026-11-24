"""Pruebas sintéticas de los dos cambios: python pruebas_segmentacion.py"""
import unittest

import cv2
import numpy as np

from segmentacion import Quietos, manchas
from separacion import rellenar_huecos, separar


def lienzo():
    return np.full((684, 1214), 200, np.uint8)


def pez(g, x, y, gris=70, a=70, b=20):
    cv2.ellipse(g, (int(x), int(y)), (a, b), 0, 0, 360, gris, -1)


class Huecos(unittest.TestCase):
    def test_hueco_pequeno_se_tapa(self):
        g = np.full((60, 160), 255, np.uint8)
        cv2.ellipse(g, (80, 30), (70, 20), 0, 0, 360, 70, -1)
        cv2.circle(g, (80, 30), 3, 190, -1)
        self.assertTrue((rellenar_huecos(g)[27:34, 77:84] < 146).all())

    def test_hueco_grande_no_se_tapa(self):
        g = np.full((80, 160), 255, np.uint8)
        cv2.ellipse(g, (80, 40), (70, 35), 0, 0, 360, 70, -1)
        cv2.circle(g, (80, 40), 10, 190, -1)
        self.assertTrue((rellenar_huecos(g)[35:45, 75:85] >= 146).all())

    def test_rendija_abierta_no_se_tapa(self):
        g = np.full((90, 160), 255, np.uint8)
        pez(g, 80, 25), pez(g, 80, 65)
        cv2.rectangle(g, (30, 42), (32, 48), 70, -1)
        r = rellenar_huecos(g)
        self.assertTrue((r == g).all())

    def test_brillo_no_parte_al_pez(self):
        g = lienzo()
        pez(g, 400, 300, a=80, b=24)
        for x in (385, 400, 415):
            cv2.circle(g, (x, 300), 4, 190, -1)
        cv2.ellipse(g, (480, 300), (30, 8), 25, 0, 360, 70, -1)
        self.assertEqual([len(separar(m)) for m in manchas(g)], [1])


class ObjetosQuietos(unittest.TestCase):
    def correr(self, n, dibujar):
        q, out = Quietos(), []
        for k in range(n):
            g = lienzo()
            dibujar(g, k)
            out.append(len(q.filtrar(g, manchas(g))))
        return out

    def test_gota_fija_se_descarta(self):
        o = self.correr(30, lambda g, k: pez(g, 900, 300) if k >= 5 else None)
        self.assertEqual(sum(o[12:]), 0)
        self.assertLessEqual(sum(o), 6)                    # sólo los cuadros de aprendizaje

    def test_peces_que_pasan_se_conservan(self):
        def d(g, k):
            for i in range(4):
                pez(g, (k * 440 + i * 330) % 1500 - 100, 100 + 150 * i)
        total = self.correr(40, d)
        sin = []
        for k in range(40):
            g = lienzo(); d(g, k); sin.append(len(manchas(g)))
        self.assertEqual(total, sin)

    def test_pez_sobre_gota_se_conserva(self):
        def d(g, k):
            pez(g, 900, 300, a=25, b=18)
            if k == 20:
                pez(g, 930, 300)
        o = self.correr(25, d)
        self.assertEqual(o[20], 1)
        self.assertEqual(o[19], 0)

    def test_gota_retirada_se_olvida(self):
        o = self.correr(60, lambda g, k: pez(g, 900, 300) if k < 25 or k == 50 else None)
        self.assertEqual(o[50], 1)


if __name__ == "__main__":
    unittest.main()
