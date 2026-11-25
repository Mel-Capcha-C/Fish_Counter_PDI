"""Cada rutina optimizada debe dar EXACTAMENTE lo mismo que su definición simple (algoritmo-v1)."""
from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishcounter.core import perfil as pf, segmentacion as sg, separacion as sp  # noqa: E402

P = pf.cargar("lubina")


def escena(rng, n, h=684, w=1214):
    """Imagen corregida con n elipses oscuras, algunas en contacto y otras cortadas por el borde."""
    g = np.clip(rng.normal(200, 6, (h, w)), 0, 255).astype(np.uint8)
    for _ in range(n):
        cv2.ellipse(g, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                    (int(rng.integers(25, 80)), int(rng.integers(8, 24))), float(rng.uniform(0, 180)), 0, 360,
                    int(rng.integers(40, 130)), -1)
    return g


# ------------------------------------------------------------ definiciones simples
def componentes_simple(m, g, p):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    return [(*(int(v) for v in st[i]), lab == i) for i in range(1, n)]


class EstaticosSimple:
    def __init__(self, p):
        self.p, self.hist, self.suma = p, deque(), None

    def filtrar(self, g, ms):
        H, W = g.shape
        ch = cv2.resize((g < self.p.umbral).astype(np.uint8) * 255, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        osc, mov, est = (ch >= 128).astype(np.int16), ms, []
        if self.suma is not None and self.suma.max() >= self.p.quietos_min:
            per = cv2.resize((self.suma >= self.p.quietos_min).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            mov = []
            for m in ms:
                h, w = m.mask.shape
                q = per[m.y:m.y + h, m.x:m.x + w][m.mask > 0].mean() >= self.p.quietos_fraccion
                (est if q else mov).append(m)
        self.suma = osc.copy() if self.suma is None else self.suma + osc
        self.hist.append(osc)
        if len(self.hist) > self.p.quietos_ventana:
            self.suma -= self.hist.popleft()
        return mov, est


def repartir_simple(f, s, mask, paso=0.8):
    k, lab = np.ones((3, 3), np.uint8), np.zeros(f.shape, np.uint8)
    for i, (r, c, _) in enumerate(s):
        cv2.circle(lab, (c, r), 2, i + 1, -1)
    lab[mask == 0] = 0
    for t in [*np.arange(float(f.max()) - paso, 0, -paso), -1.0]:
        region = (f > t) & (mask > 0)
        while True:
            crece = cv2.dilate(lab, k)
            nuevo = region & (lab == 0) & (crece > 0)
            if not nuevo.any():
                break
            lab[nuevo] = crece[nuevo]
    return lab


class Equivalencia(unittest.TestCase):
    def test_campo_plano(self):
        rng = np.random.default_rng(0)
        for h in (684, 701, 63, 5, 1):
            roi = rng.integers(0, 256, (h, 1214), dtype=np.uint8)
            gan = rng.uniform(.5, 6, (h, 1214)).astype(np.float32)
            self.assertTrue(np.array_equal(sg.campo_plano(roi, gan), np.clip(roi * gan, 0, 255).astype(np.uint8)))

    def test_umbral_y_recorte(self):
        g = np.arange(256, dtype=np.uint8).reshape(16, 16)
        self.assertTrue(np.array_equal(sg.umbralizar(g, P), (g < P.umbral).astype(np.uint8)))
        b = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(sg.recortar(b, P), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)[P.roi_y0:, P.roi_x0:]))

    def test_apertura_y_componentes(self):
        rng = np.random.default_rng(2)
        for n in (0, 1, 3, 12, 40):
            g = escena(rng, n)
            b = sg.umbralizar(g, P)
            ref = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            a = sg.abrir(b, P)
            self.assertTrue(np.array_equal(a, ref))
            ok, chicas = sg.componentes(a, g, P)
            simple = componentes_simple(ref, g, P)
            self.assertEqual([(m.x, m.y, *m.mask.shape[::-1], m.area) for m in ok],
                             [c[:5] for c in simple if c[4] >= P.area_min])
            self.assertEqual(sorted(chicas), sorted(c[:5] for c in simple if c[4] < P.area_min))
            for m, c in zip(ok, [c for c in simple if c[4] >= P.area_min]):
                h, w = m.mask.shape
                self.assertTrue(np.array_equal(m.mask > 0, c[5][m.y:m.y + h, m.x:m.x + w]))

    def test_estaticos(self):
        for semilla in range(6):
            rng = np.random.default_rng(semilla)
            rapido, simple, hubo = sg.Estaticos(P), EstaticosSimple(P), 0
            fijas = [(int(rng.integers(0, 1214)), int(rng.integers(0, 684))) for _ in range(3)] + [(1212, 682), (2, 2)]
            for k in range(30):
                g = escena(rng, int(rng.integers(0, 30)))
                for i, (x, y) in enumerate(fijas):
                    if k > 3 + 2 * i and (k + i) % 9:          # aparecen a destiempo y parpadean
                        cv2.ellipse(g, (x, y), (40, 18), 0, 0, 360, 60, -1)
                ms, _ = sg.componentes(sg.abrir(sg.umbralizar(g, P), P), g, P)
                a, b = rapido.filtrar(g, ms), simple.filtrar(g, ms)
                self.assertEqual([[id(m) for m in x] for x in a], [[id(m) for m in x] for x in b])
                hubo += len(b[1])
            self.assertGreater(hubo, 0)                        # la prueba debe ejercitar el descarte

    def test_cimas(self):
        rng = np.random.default_rng(3)
        V = [P.con(razon=r, cima_min=c, area_cima=a) for r, c, a in
             ((.88, 7.5, 60), (.95, 3., 10), (.6, 5., 30), (.8, 1., 1), (.99, .5, 1))]
        varios = 0
        for it in range(600):
            h, w = rng.integers(20, 120, 2)
            z = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), float(rng.uniform(1.5, 6)))
            z = (z - z.min()) / (z.max() - z.min() + 1e-9) * float(rng.uniform(4, 40))
            if it % 3 == 0:                                    # mesetas y empates
                z = np.round(z * int(rng.integers(1, 4))) / 2
            z = z.astype(np.float32)
            a, b = sp.cimas(z, V[it % 5]), sp.cimas_completo(z, V[it % 5])
            self.assertEqual(len(a), len(b))
            if len(b) > 1:
                self.assertEqual(a, b)
                varios += 1
            elif b:                                            # un cuerpo: la cima sólo se dibuja; puede
                r, c, _ = a[0]                                 # ser otro píxel con el mismo máximo
                self.assertEqual(z[r, c], z[b[0][0], b[0][1]])
        self.assertGreater(varios, 200)

    def test_reparto(self):
        rng = np.random.default_rng(4)
        n = 0
        for _ in range(40):
            g = escena(rng, 45)
            ms, _ = sg.componentes(sg.abrir(sg.umbralizar(g, P), P), g, P)
            for m in ms:
                a = sp.analizar(m, P)
                if a.etiquetas is not None:
                    self.assertTrue(np.array_equal(a.etiquetas, repartir_simple(a.campo, a.cimas, a.rellena.mask)))
                    n += 1
        self.assertGreater(n, 20)


if __name__ == "__main__":
    unittest.main()
