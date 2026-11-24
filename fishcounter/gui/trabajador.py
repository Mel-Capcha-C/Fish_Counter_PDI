"""Hilo de procesamiento. La ventana le manda órdenes; él emite señales.

Regla central: NUNCA se salta un cuadro al contar; sólo se saltan dibujos cuando la ventana va atrasada.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback

from PySide6.QtCore import QThread, Signal

from ..core import vistas
from ..core.motor import Motor, es_caso
from ..core.perfil import Perfil
from ..core.video import LectorVideo, VideoInvalido, estimar_fondo


DIBUJO_MIN = 1 / 12            # s entre dibujos al reproducir: el conteo tiene prioridad


class Trabajador(QThread):
    video_abierto = Signal(dict)
    progreso = Signal(float, str)
    preparado = Signal(bool)               # True: cuenta; False: conteo bloqueado
    numeros = Signal(int, object)          # cuadro, Conteo
    pintura = Signal(dict)                 # imágenes del grupo visible
    reproduciendo = Signal(bool)
    terminado = Signal(object)             # Conteo final
    sin_caso = Signal(str)
    error = Signal(str)

    def __init__(self, perfil: Perfil, parent=None):
        super().__init__(parent)
        self.perfil = perfil
        self._q: queue.Queue = queue.Queue()
        self._gui_lista = threading.Event()
        self._gui_lista.set()
        self._fin = False
        self._lector: LectorVideo | None = None
        self._motor: Motor | None = None
        self._fondo = None
        self._clave_fondo = None
        self._k = -1
        self._grupo = vistas.GRUPOS[0].id
        self._jugando = False
        self._vel = 1.0
        self._proximo = 0.0
        self._ultimo_dibujo = 0.0

    # ------------------------------------------------- órdenes (desde la ventana)
    def abrir(self, ruta):
        self._q.put(("abrir", ruta))

    def cambiar_perfil(self, p):
        self._q.put(("perfil", p))

    def reproducir(self, si):
        self._q.put(("jugar", si))

    def avanzar(self, n):
        self._q.put(("mover", n))

    def ir(self, k):
        self._q.put(("ir", k))

    def velocidad(self, v):
        self._q.put(("vel", v))

    def grupo(self, g):
        self._q.put(("grupo", g))

    def buscar(self, caso):
        self._q.put(("buscar", caso))

    def reiniciar(self):
        self._q.put(("reiniciar", None))

    def gui_lista(self):
        self._gui_lista.set()

    def detener(self):
        self._fin = True
        self._q.put(("salir", None))
        self.wait(8000)

    # ------------------------------------------------------------------- bucle
    def run(self):
        while not self._fin:
            try:
                espera = None
                if self._jugando:
                    espera = max(self._proximo - time.perf_counter(), 0)
                orden = self._q.get(timeout=espera) if espera is None or espera > 0 else self._q.get_nowait()
                self._atender(*orden)
                # Una ráfaga de órdenes no debe dejar el video parado
                if self._jugando and time.perf_counter() >= self._proximo:
                    self._tic()
            except queue.Empty:
                try:
                    self._tic()
                except Exception:
                    self._fallo()
            except Exception:
                self._fallo()
        if self._lector:
            self._lector.cerrar()

    def _fallo(self):
        self._poner_jugando(False)
        self.error.emit(traceback.format_exc(limit=4))

    def _atender(self, nombre, dato):
        if nombre == "salir":
            return
        if nombre == "abrir":
            self._abrir(dato)
        elif nombre == "perfil":
            self.perfil = dato
            if self._lector:
                self._preparar()
        elif nombre == "vel":
            self._vel = dato
            self._proximo = time.perf_counter()
        elif nombre == "grupo":
            self._grupo = dato
            if not self._jugando:                          # en marcha, lo dibuja el siguiente cuadro
                self._pintar(forzar=True)
        elif self._motor is None:
            return
        elif nombre == "jugar":
            if dato and self._k >= self._lector.total - 1:
                self._k = -1                              # al final: vuelve a empezar (sin recontar)
            self._proximo = time.perf_counter()
            self._poner_jugando(dato)
            if not dato:
                self._pintar(forzar=True)
        elif nombre == "mover":
            self._poner_jugando(False)
            self._ir(self._k + dato)
        elif nombre == "ir":
            self._poner_jugando(False)
            self._ir(dato)
        elif nombre == "buscar":
            self._poner_jugando(False)
            self._buscar(dato)
        elif nombre == "reiniciar":
            self._poner_jugando(False)
            self._nuevo_motor()

    def _poner_jugando(self, si):
        if si != self._jugando:
            self._jugando = si
            self.reproduciendo.emit(si)

    # ------------------------------------------------------------- preparación
    def _abrir(self, ruta):
        self._poner_jugando(False)
        try:
            nuevo = LectorVideo(ruta)
        except VideoInvalido as e:
            self.error.emit(str(e))
            return
        if self._lector:
            self._lector.cerrar()
        self._lector, self._motor, self._fondo, self._clave_fondo, self._k = nuevo, None, None, None, -1
        self._preparar()

    def _preparar(self):
        p, v = self.perfil, self._lector
        self._poner_jugando(False)
        if p.roi_x0 >= v.ancho - 16 or p.roi_y0 >= v.alto - 16:
            self._motor = None
            self.error.emit(f"La ROI del perfil no cabe en este video ({v.ancho}x{v.alto}).")
            return
        clave = (v.ruta, p.roi_x0, p.roi_y0, p.fondo_cuadros)
        if clave != self._clave_fondo:
            self._fondo = None

            def avance(f):
                self.progreso.emit(f, "Estimando el fondo por mediana temporal…")
                return not self._fin
            fondo = estimar_fondo(v, p, avance)
            if fondo is None:
                return
            self._fondo, self._clave_fondo = fondo, clave
        self.video_abierto.emit(dict(ruta=v.ruta, ancho=v.ancho, alto=v.alto, fps=v.fps, total=v.total))
        self._nuevo_motor()

    def _nuevo_motor(self):
        v, p = self._lector, self.perfil
        contar = (v.ancho, v.alto) == (p.ancho, p.alto)
        self._motor = Motor(p, self._fondo, contar)
        self._k = -1
        self.preparado.emit(contar)
        self._ir(0)

    # ------------------------------------------------------------------ avance
    def _procesar_hasta(self, k) -> bool:
        """Procesa (sin saltar) hasta el cuadro k. False si se acabó el video."""
        m = self._motor
        while m.siguiente <= k:
            f = self._lector.leer(m.siguiente)
            if f is None:
                return False
            m.procesar(m.siguiente, f)
        if m.siguiente >= self._lector.total and not m.terminado:
            m.finalizar()
            m.resultados[-1].conteo = m.conteo
            self.terminado.emit(m.conteo)
        return True

    def _tic(self):
        sig = self._k + 1
        if sig >= self._lector.total or not self._procesar_hasta(sig):
            self._poner_jugando(False)
            self._pintar(forzar=True)
            return
        self._k = sig
        self.numeros.emit(self._k, self._motor.resultados[self._k].conteo)
        self._pintar()
        ahora = time.perf_counter()
        if self._vel <= 0:
            self._proximo = ahora
        else:
            self._proximo = max(self._proximo + 1.0 / (self._lector.fps * self._vel), ahora - .1)

    def _ir(self, k):
        k = max(0, min(k, self._lector.total - 1))
        m = self._motor
        desde = m.siguiente
        while m.siguiente <= k:
            if not self._procesar_hasta(min(m.siguiente + 9, k)):
                k = m.siguiente - 1
                break
            if k - desde > 30:
                self.progreso.emit((m.siguiente - desde) / (k - desde + 1), "Contando los cuadros intermedios…")
            if not self._q.empty() and k - m.siguiente > 10:       # otra orden: se atiende desde aquí
                k = m.siguiente - 1
                break
        self._k = k
        self.progreso.emit(1.0, "")
        if k >= 0:
            self.numeros.emit(k, m.resultados[k].conteo)
            self._pintar(forzar=True)

    def _buscar(self, caso):
        m = self._motor
        k = self._k + 1
        while k < self._lector.total:
            if k >= m.siguiente and not self._procesar_hasta(k):
                break
            if es_caso(m.resultados[k], caso):
                self._ir(k)
                return
            if not self._q.empty():
                self._ir(k)
                return
            if k % 20 == 0:
                self.progreso.emit(k / self._lector.total, "Buscando el siguiente caso…")
            k += 1
        self.progreso.emit(1.0, "")
        self.sin_caso.emit(caso)

    # ------------------------------------------------------------------ dibujo
    def _pintar(self, forzar=False):
        if self._motor is None or not 0 <= self._k < self._motor.siguiente:
            return
        ahora = time.perf_counter()
        if not forzar and (not self._gui_lista.is_set() or ahora - self._ultimo_dibujo < DIBUJO_MIN):
            return
        self._ultimo_dibujo = ahora
        bgr = self._lector.leer(self._k)
        if bgr is None:
            return
        previa = self._lector.leer(self._k - 1) if self._grupo == "tracking" and self._k > 0 else None
        c = vistas.Contexto(self._motor, self._motor.resultados[self._k], bgr, previa)
        self._gui_lista.clear()
        self.pintura.emit(dict(
            k=self._k, grupo=self._grupo, vistas=vistas.dibujar(self._grupo, c),
            contadores={g.id: g.cuenta(c) for g in vistas.GRUPOS},
            conteo=c.r.conteo, roi=(self.perfil.roi_x0, self.perfil.roi_y0)))
