"""Pruebas de la interfaz sin pantalla, con un video sintético (no necesitan el dataset)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    from PySide6.QtWidgets import QApplication
except ImportError as e:                                     # pragma: no cover
    raise unittest.SkipTest("PySide6 no está instalado") from e

from fishcounter.core import perfil as pf, vistas  # noqa: E402
from fishcounter.gui import tema  # noqa: E402
from fishcounter.gui.ajustes import Ajustes  # noqa: E402
from fishcounter.gui.ventana import Ventana  # noqa: E402

app = QApplication.instance() or QApplication([])
app.setStyleSheet(tema.QSS)


def video_sintetico(ruta, w=1280, h=720, n=40):
    """4 peces, cada uno visible en 3 cuadros, con el movimiento del perfil de lubina."""
    p = pf.cargar("lubina")
    pos = {}
    for i, (k0, y) in enumerate(((5, 200), (9, 400), (13, 300), (20, 550))):
        x = 120.0
        for k in range(k0, k0 + 5):
            if x < 1230:
                pos.setdefault(k, []).append((x, y))
            x, y = x + p.salto(x, y), y + p.my
    s = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for k in range(n):
        f = np.full((h, w, 3), 220, np.uint8)
        for x, y in pos.get(k, []):
            cv2.ellipse(f, (int(x / p.fx + p.roi_x0), int(y / p.fy + p.roi_y0)), (62, 17), 0, 0, 360, (50, 50, 50), -1)
        s.write(f)
    s.release()


def esperar(cond, seg=40):
    t = time.time()
    while time.time() - t < seg:
        app.processEvents()
        if cond():
            return True
        time.sleep(.005)
    return False


class Interfaz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.ruta = Path(cls.dir.name) / "s.avi"
        cls.chico = Path(cls.dir.name) / "chico.avi"
        video_sintetico(cls.ruta)
        video_sintetico(cls.chico, 960, 540, 12)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def setUp(self):
        self.errores = []
        self.parche = mock.patch("fishcounter.gui.ventana.QMessageBox.critical",
                                 side_effect=lambda *a: self.errores.append(a[-1]))
        self.parche.start()
        self.v = Ventana()
        self.v.resize(1480, 900)
        self.v.show()

    def tearDown(self):
        self.v.close()
        self.parche.stop()
        self.assertEqual(self.errores, [])

    def abrir(self, ruta=None):
        self.v.abrir(str(ruta or self.ruta))
        self.assertTrue(esperar(lambda: self.v.b_play.isEnabled() and self.v.t._k == 0))

    def final(self):
        self.v.b_fin.click()
        self.assertTrue(esperar(lambda: self.v.l_fase.text() == "Conteo final"))
        return int(self.v.n_total.text())

    def test_cuenta_hasta_el_final(self):
        self.abrir()
        self.assertEqual(self.final(), 4)
        self.assertEqual(self.v.n["pendientes"].text(), "0")

    def test_reproducir_da_lo_mismo_que_saltar(self):
        self.abrir()
        self.v.c_vel.setCurrentIndex(4)                      # Máx
        self.v.b_play.click()
        self.assertTrue(esperar(lambda: self.v.l_fase.text() == "Conteo final"))
        self.assertEqual(int(self.v.n_total.text()), 4)

    def test_retroceder_y_volver_no_recuenta(self):
        self.abrir()
        n = self.final()
        self.v.t.ir(7)
        self.assertTrue(esperar(lambda: self.v.l_k.text().startswith("8 ")))
        self.assertLess(int(self.v.n_total.text()), n)
        self.v.c_vel.setCurrentIndex(4)
        self.v.b_play.click()
        self.assertTrue(esperar(lambda: self.v.l_fase.text() == "Conteo final"))
        self.assertEqual(int(self.v.n_total.text()), n)

    def test_todas_las_pestanas_dibujan_todas_las_tecnicas(self):
        self.abrir()
        self.v.t.ir(10)
        for i, g in enumerate(vistas.GRUPOS):
            self.v.tabs.setCurrentIndex(i)
            pg = self.v.paginas[g.id]
            self.assertTrue(esperar(lambda: all(c.dato is not None for c in pg.cuad.tarjetas)), g.id)
            self.assertEqual(len(pg.cuad.tarjetas), len(g.tecnicas))
            for j in range(1, len(g.tecnicas)):              # vista ampliada, ambos modos
                pg.abrir_ampliado(j)
                for b in (pg.amp.lado, pg.amp.desl):
                    b.click()
                    pg.amp.dif.toggle()
                    self.v.grab()
            pg.pila.setCurrentIndex(0)

    def test_cambiar_de_pestana_no_altera_el_conteo(self):
        self.abrir()
        self.v.c_vel.setCurrentIndex(4)
        self.v.b_play.click()
        i = 0
        while self.v.l_fase.text() != "Conteo final":
            self.v.tabs.setCurrentIndex(i % 5)
            i += 1
            esperar(lambda: False, .03)
            self.assertLess(i, 600)
        self.assertEqual(int(self.v.n_total.text()), 4)

    def test_otra_resolucion_bloquea_el_conteo(self):
        self.abrir(self.chico)
        self.assertTrue(self.v.aviso.isVisible())
        self.assertIn("recalibrar", self.v.aviso.text())
        self.assertEqual(self.v.n_total.text(), "—")
        self.v.b_fin.click()
        self.assertTrue(esperar(lambda: self.v.l_k.text().startswith("12 ")))
        for i in range(5):
            self.v.tabs.setCurrentIndex(i)
            esperar(lambda: False, .1)
        self.assertEqual(self.v.n_total.text(), "—")
        self.abrir()                                          # al abrir uno correcto se desbloquea
        self.assertFalse(self.v.aviso.isVisible())
        self.assertEqual(self.final(), 4)

    def test_cambiar_de_pez_reinicia(self):
        self.abrir()
        self.final()
        self.v.c_pez.setCurrentIndex(self.v.c_pez.findData("trucha"))
        self.assertTrue(esperar(lambda: self.v.l_k.text().startswith("1 ") and self.v.n_total.text() == "0"))
        self.assertIn("Sin calibrar", self.v.l_estado.text())
        self.assertEqual(self.final(), 4)

    def test_aplicar_ajustes(self):
        self.abrir()
        self.v.abrir_ajustes(pf.SEGMENTACION)
        a = self.v.ajustes
        a.campos["area_min"].setValue(9000)                   # más grande que cualquier pez
        a.aplicar()
        self.assertEqual(a.error.text(), "")
        self.assertTrue(esperar(lambda: self.v.l_k.text().startswith("1 ")))
        self.assertIn("Editado", self.v.l_estado.text())
        self.assertEqual(self.final(), 0)
        a.restaurar()
        self.assertTrue(esperar(lambda: self.v.l_k.text().startswith("1 ")))
        self.assertEqual(self.final(), 4)
        a.close()

    def test_ajustes_incoherentes_se_rechazan(self):
        a = Ajustes(pf.cargar("lubina"))
        a.campos["umbral2"].setValue(200)
        a.aplicar()
        self.assertIn("decrecer", a.error.text())

    def test_abrir_ajustes_no_cambia_nada(self):
        """Los campos muestran valores redondeados: aplicar sin tocar debe conservar los exactos."""
        p = pf.cargar("lubina")
        a = Ajustes(p)
        L = []
        a.aplicado.connect(L.append)
        a.aplicar()
        self.assertEqual(L[0].valores, p.valores)
        self.assertEqual(L[0].estado, "calibrado")

    def test_ir_al_siguiente_caso(self):
        self.abrir()
        self.v.t.buscar("grupo")                              # no hay grupos en el video sintético
        self.assertTrue(esperar(lambda: "No hay más casos" in self.v.statusBar().currentMessage()))


class ArchivoMalo(unittest.TestCase):
    def test_avisa_si_no_es_un_video(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("fishcounter.gui.ventana.QMessageBox.critical") as m:
            r = Path(d) / "x.mp4"
            r.write_bytes(b"no soy un video")
            v = Ventana()
            v.abrir(str(r))
            self.assertTrue(esperar(lambda: m.called))
            self.assertIn("No se pudo abrir", m.call_args[0][-1])
            v.close()


if __name__ == "__main__":
    unittest.main()
