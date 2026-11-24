"""Pruebas del núcleo: perfiles, lector de video, motor y equivalencia con conteo_v3."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from fishcounter.core import perfil as pf, vistas  # noqa: E402
from fishcounter.core.motor import Motor  # noqa: E402
from fishcounter.core.video import LectorVideo, VideoInvalido, estimar_fondo  # noqa: E402

DATASET = RAIZ / "21601548" / "FCDataZJU"


class Perfiles(unittest.TestCase):
    def test_todos_cargan_y_son_validos(self):
        ids = [i for i, _ in pf.disponibles()]
        self.assertTrue({"lubina", "carassius", "trucha"} <= set(ids))
        for i in ids:
            self.assertEqual(set(pf.cargar(i).valores), {p.clave for p in pf.PARAMETROS})

    def test_estados_honestos(self):
        self.assertEqual(pf.cargar("trucha").estado, "sin_calibrar")
        self.assertEqual(pf.cargar("carassius").estado, "parcial")

    def test_carassius_solo_cambia_el_movimiento(self):
        a, b = pf.cargar("lubina").valores, pf.cargar("carassius").valores
        self.assertEqual({pf.POR_CLAVE[k].grupo for k in a if a[k] != b[k]}, {pf.TRACKING})

    def test_editar_marca_sin_validar(self):
        p = pf.cargar("lubina")
        self.assertEqual(p.con(area_min=400).estado, "editado")
        self.assertEqual(p.con(area_min=p.area_min).estado, "calibrado")
        self.assertEqual(p.area_min, 350)                      # el original no cambia

    def test_rechaza_valores_incoherentes(self):
        p = pf.cargar("lubina")
        for c in (dict(umbral2=150), dict(umbral=300), dict(gauss=4), dict(quietos_min=50), dict(roi_x0=1279)):
            with self.assertRaises(pf.PerfilInvalido, msg=c):
                p.con(**c)

    def test_guardar_no_sobrescribe(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for f in pf.CARPETA.glob("*.json"):
                (d / f.name).write_bytes(f.read_bytes())
            p = pf.cargar("lubina", d).con(area_min=500, roi_x0=70)
            pf.guardar_como(p, "nueva", "Nueva", carpeta=d)
            q = pf.cargar("nueva", d)
            self.assertEqual(q.valores, p.valores)
            self.assertEqual(pf.cargar("lubina", d).roi_x0, 66)    # no toca el canal común
            with self.assertRaises(pf.PerfilInvalido):
                pf.guardar_como(p, "lubina", "Otra", carpeta=d)
            with self.assertRaises(pf.PerfilInvalido):
                pf.guardar_como(p, "../fuera", "Otra", carpeta=d)

    def test_esquema_futuro_se_rechaza(self):
        with tempfile.TemporaryDirectory() as d:
            for f in pf.CARPETA.glob("*.json"):
                x = json.loads(f.read_text(encoding="utf8"))
                x["version_esquema"] = 99
                (Path(d) / f.name).write_text(json.dumps(x), encoding="utf8")
            with self.assertRaises(pf.PerfilInvalido):
                pf.cargar("lubina", Path(d))


class Arquitectura(unittest.TestCase):
    def test_el_nucleo_no_depende_de_qt(self):
        for f in (RAIZ / "fishcounter" / "core").glob("*.py"):
            for n in ast.walk(ast.parse(f.read_text(encoding="utf8"))):
                m = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                    [n.module or ""] if isinstance(n, ast.ImportFrom) else []
                self.assertFalse(any("PySide" in x or "gui" in x.split(".") for x in m), f.name)

    def test_cada_grupo_de_vistas_tiene_parametros_y_una_entrada(self):
        self.assertEqual([g.id for g in vistas.GRUPOS], list(pf.NOMBRES_GRUPO))
        for g in vistas.GRUPOS:
            self.assertEqual(g.tecnicas[0].tipo, "entrada")
            self.assertTrue(any(p.grupo == g.id for p in pf.PARAMETROS))


class Video(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.ruta = str(Path(cls.dir.name) / "v.mp4")
        s = cv2.VideoWriter(cls.ruta, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        rng = np.random.default_rng(0)
        for k in range(120):
            f = np.full((240, 320, 3), 200, np.uint8)
            cv2.putText(f, str(k), (60, 150), 0, 3, tuple(int(x) for x in rng.integers(0, 120, 3)), 6)
            s.write(f)
        s.release()

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_acceso_aleatorio_es_exacto(self):
        v = LectorVideo(self.ruta)
        ref = [v.leer(k).copy() for k in range(120)]
        for k in (119, 3, 60, 59, 2, 118, 0, 100, 99, 1):
            self.assertTrue(np.array_equal(v.leer(k), ref[k]), k)
        self.assertIsNone(v.leer(500))
        v.cerrar()

    def test_no_existe(self):
        with self.assertRaises(VideoInvalido):
            LectorVideo("no_existe.mp4")

    def test_cancelar_la_estimacion_de_fondo(self):
        v = LectorVideo(self.ruta)
        p = pf.cargar("lubina").con(roi_x0=0, roi_y0=0)
        self.assertIsNone(estimar_fondo(v, p, lambda f: False))
        v.cerrar()


class MotorBasico(unittest.TestCase):
    def test_exige_orden(self):
        p = pf.cargar("lubina")
        m = Motor(p, np.full((p.alto - p.roi_y0, p.ancho - p.roi_x0), 200, np.float32))
        f = np.full((p.alto, p.ancho, 3), 200, np.uint8)
        m.procesar(0, f)
        with self.assertRaises(ValueError):
            m.procesar(2, f)
        self.assertEqual(m.finalizar().total, 0)

    def test_dibujar_con_o_sin_intermedios_es_igual(self):
        """Reutilizar los intermedios del último cuadro no debe cambiar lo que se ve."""
        p = pf.cargar("lubina")
        m = Motor(p, np.full((p.alto - p.roi_y0, p.ancho - p.roi_x0), 220, np.float32))
        f = np.full((p.alto, p.ancho, 3), 220, np.uint8)
        for y in (300, 342):
            cv2.ellipse(f, (500, y), (70, 21), 0, 0, 360, (60, 60, 60), -1)
        r = m.procesar(0, f)
        a = vistas.Contexto(m, r, f)
        ult, m.ultimo = m.ultimo, None
        b = vistas.Contexto(m, r, f)
        m.ultimo = ult
        for g in vistas.GRUPOS:
            for (t, x), (_, y) in zip(vistas.dibujar(g.id, a), vistas.dibujar(g.id, b)):
                self.assertTrue(np.array_equal(x.rgb, y.rgb), t.id)


@unittest.skipUnless(DATASET.exists(), "requiere el dataset FCDataZJU")
class Regresion(unittest.TestCase):
    """Conteos de conteo_v3 (tracking + recuperación) que no deben cambiar sin querer."""
    CASOS = (("lubina", "MicropterusSalmoides/200/videos/m4.mp4", 198),
             ("lubina", "MicropterusSalmoides/500/videos/m8.mp4", 495),
             ("lubina", "MicropterusSalmoides/1000/videos/m21.mp4", 986))

    def test_conteos(self):
        from fishcounter.cli import contar
        for pez, r, n in self.CASOS:
            with self.subTest(r):
                self.assertEqual(contar(str(DATASET / r), pf.cargar(pez)).conteo.total, n)


if __name__ == "__main__":
    unittest.main()
