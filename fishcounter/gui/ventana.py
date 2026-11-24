"""Ventana principal."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSlider, QSplitter, QStackedWidget, QTabBar,
                               QVBoxLayout, QWidget)

from .. import __version__
from ..core import perfil as pf, vistas
from ..core.motor import CASOS, Conteo
from . import tema
from .ajustes import Ajustes
from .trabajador import Trabajador
from .visor import Camara, PaginaGrupo

EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
VELOCIDADES = [("0,25×", .25), ("0,5×", .5), ("1×", 1.0), ("2×", 2.0), ("Máx", 0.0)]


class Ventana(QMainWindow):
    def __init__(self, video: str | None = None):
        super().__init__()
        self.setWindowTitle(f"Contador de peces {__version__}")
        self.resize(1480, 900)
        self.setMinimumSize(1080, 680)
        self.setAcceptDrops(True)
        self.conf = QSettings("FishCounter", "FishCounter")
        self.perfil = pf.cargar(pf.disponibles()[0][0])
        self.info: dict | None = None
        self.cuenta = False
        self.cam = Camara()
        self._moviendo = False
        self.ajustes: Ajustes | None = None
        self._construir()
        self.t = Trabajador(self.perfil, self)
        self._conectar()
        self.t.start()
        self._habilitar(False)
        self._mostrar_estado_pez()
        if video:
            self.abrir(video)

    # -------------------------------------------------------------- construcción
    def _construir(self):
        raiz = QWidget()
        raiz.setObjectName("raiz")
        self.setCentralWidget(raiz)
        lay = QVBoxLayout(raiz)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.addLayout(self._barra())
        self.aviso = QLabel()
        self.aviso.setObjectName("aviso")
        self.aviso.setWordWrap(True)
        self.aviso.hide()
        lay.addWidget(self.aviso)
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        for i, g in enumerate(vistas.GRUPOS):
            self.tabs.addTab("")
            self.tabs.setTabToolTip(i, f"{g.nombre} (tecla {i + 1})")
        lay.addWidget(self.tabs)
        self._contadores({})
        div = QSplitter(Qt.Orientation.Horizontal)
        self.pila = QStackedWidget()
        self.paginas: dict[str, PaginaGrupo] = {}
        for g in vistas.GRUPOS:
            pg = PaginaGrupo(g, self.cam)
            pg.ajustes.connect(self.abrir_ajustes)
            self.paginas[g.id] = pg
            self.pila.addWidget(pg)
        self.vacio = QLabel("Arrastra un video aquí o pulsa «Abrir video»")
        self.vacio.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vacio.setObjectName("tenue")
        self.marco = QStackedWidget()
        self.marco.addWidget(self.vacio)
        self.marco.addWidget(self.pila)
        div.addWidget(self.marco)
        div.addWidget(self._panel_conteo())
        div.setStretchFactor(0, 1)
        div.setSizes([1180, 250])
        lay.addWidget(div, 1)
        lay.addLayout(self._reproduccion())
        self.barra = QProgressBar()
        self.barra.setRange(0, 1000)
        self.barra.setMaximumWidth(340)
        self.barra.hide()
        self.statusBar().addPermanentWidget(self.barra)
        self.statusBar().showMessage("Listo. Abre un video para empezar.")

    def _barra(self):
        self.b_abrir = QPushButton("📂 Abrir video")
        self.b_abrir.setObjectName("principal")
        self.b_abrir.setToolTip("Seleccionar el video de entrada (Ctrl+O)")
        self.l_video = QLabel("Sin video")
        self.l_video.setObjectName("tenue")
        self.c_pez = QComboBox()
        self.c_pez.setToolTip("Tipo de pez. Cambiarlo carga sus parámetros y reinicia el conteo.")
        self._llenar_peces(self.perfil.id)
        self.l_estado = QLabel()
        self.b_aj = QPushButton("⚙ Ajustes")
        h = QHBoxLayout()
        h.addWidget(self.b_abrir)
        h.addWidget(self.l_video, 1)
        h.addWidget(QLabel("Pez:"))
        h.addWidget(self.c_pez)
        h.addWidget(self.l_estado)
        h.addWidget(self.b_aj)
        return h

    def _llenar_peces(self, id_):
        self.c_pez.blockSignals(True)
        self.c_pez.clear()
        for i, n in pf.disponibles():
            self.c_pez.addItem(n, i)
        self.c_pez.setCurrentIndex(max(self.c_pez.findData(id_), 0))
        self.c_pez.blockSignals(False)

    def _panel_conteo(self):
        p = QFrame()
        p.setObjectName("panel")
        lay = QVBoxLayout(p)
        lay.setContentsMargins(16, 14, 16, 14)

        def et(t, n=""):
            x = QLabel(t)
            if n:
                x.setObjectName(n)
            return x
        lay.addWidget(et("CONTEO", "tenue"))
        self.n_total = et("0", "grande")
        self.n_total.setAccessibleName("Conteo total")
        lay.addWidget(self.n_total)
        self.l_fase = et("", "tenue")
        self.l_fase.setWordWrap(True)
        lay.addWidget(self.l_fase)
        lay.addSpacing(10)
        self.n = {}
        for k, t, col, tip in (
                ("confirmados", "Confirmados", tema.VERDE, "Trayectorias vistas en 2 o más cuadros"),
                ("recuperados", "Recuperados", tema.AZUL,
                 "Trayectorias ocluidas recuperadas por consistencia temporal de áreas"),
                ("pendientes", "Pendientes", tema.NARANJA,
                 "Vistos una sola vez: aún sin decidir. No forman parte del conteo")):
            fila = QHBoxLayout()
            a, b = QLabel(f"<span style='color:{col}'>■</span> {t}"), QLabel("0")
            a.setToolTip(tip)
            b.setObjectName("titulo")
            fila.addWidget(a)
            fila.addStretch(1)
            fila.addWidget(b)
            lay.addLayout(fila)
            self.n[k] = b
        lay.addStretch(1)
        self.b_rein = QPushButton("Reiniciar conteo")
        lay.addWidget(self.b_rein)
        p.setMinimumWidth(210)
        return p

    def _reproduccion(self):
        def b(t, tip):
            x = QPushButton(t)
            x.setToolTip(tip)
            x.setFixedWidth(44)
            return x
        self.b_ini, self.b_ant, self.b_play = b("⏮", "Ir al inicio (Inicio)"), b("◀", "Cuadro anterior (←)"), b("▶", "Reproducir / pausar (Espacio)")
        self.b_sig, self.b_fin = b("▶|", "Cuadro siguiente (→)"), b("⏭", "Ir al final, contando todo (Fin)")
        self.linea = QSlider(Qt.Orientation.Horizontal)
        self.linea.setAccessibleName("Posición del video")
        self.l_k = QLabel("0 / 0")
        self.l_k.setMinimumWidth(96)
        self.c_caso = QComboBox()
        for k, n in CASOS.items():
            self.c_caso.addItem(n, k)
        self.b_caso = QPushButton("Ir al siguiente")
        self.b_caso.setToolTip("Avanza (contando) hasta el siguiente caso del tipo elegido (N)")
        self.c_vel = QComboBox()
        for n, v in VELOCIDADES:
            self.c_vel.addItem(n, v)
        self.c_vel.setCurrentIndex(2)
        self.c_vel.setToolTip("Velocidad respecto a los fps del archivo. Si el PC no alcanza, el video va más "
                              "lento, pero no se salta ningún cuadro al contar.")
        h = QHBoxLayout()
        for w in (self.b_ini, self.b_ant, self.b_play, self.b_sig, self.b_fin):
            h.addWidget(w)
        h.addWidget(self.linea, 1)
        for w in (self.l_k, QLabel("Ir a:"), self.c_caso, self.b_caso, self.c_vel):
            h.addWidget(w)
        return h

    def _conectar(self):
        t = self.t
        self.b_abrir.clicked.connect(self.elegir)
        self.b_aj.clicked.connect(lambda: self.abrir_ajustes(vistas.GRUPOS[self.tabs.currentIndex()].id))
        self.c_pez.currentIndexChanged.connect(self._cambiar_pez)
        self.tabs.currentChanged.connect(self._cambiar_tab)
        self.b_play.clicked.connect(self._play)
        self.b_ini.clicked.connect(lambda: t.ir(0))
        self.b_fin.clicked.connect(lambda: t.ir(10 ** 9))
        self.b_ant.clicked.connect(lambda: t.avanzar(-1))
        self.b_sig.clicked.connect(lambda: t.avanzar(1))
        self.b_rein.clicked.connect(t.reiniciar)
        self.b_caso.clicked.connect(lambda: t.buscar(self.c_caso.currentData()))
        self.c_vel.currentIndexChanged.connect(lambda _: t.velocidad(self.c_vel.currentData()))
        self.linea.sliderPressed.connect(lambda: setattr(self, "_moviendo", True))
        self.linea.sliderReleased.connect(self._soltar)
        t.video_abierto.connect(self._video)
        t.progreso.connect(self._progreso)
        t.preparado.connect(self._preparado)
        t.numeros.connect(self._numeros)
        t.pintura.connect(self._pintura)
        t.reproduciendo.connect(self._reproduciendo)
        t.terminado.connect(self._terminado)
        t.sin_caso.connect(lambda c: self.statusBar().showMessage(f"No hay más casos de «{CASOS[c]}» en el video.", 8000))
        t.error.connect(self._error)
        for s, f in ((QKeySequence.StandardKey.Open, self.elegir), ("Space", self._play),
                     ("Left", lambda: self.b_ant.isEnabled() and t.avanzar(-1)),
                     ("Right", lambda: self.b_sig.isEnabled() and t.avanzar(1)),
                     ("Home", lambda: self.b_ini.isEnabled() and t.ir(0)),
                     ("End", lambda: self.b_fin.isEnabled() and t.ir(10 ** 9)),
                     ("Up", lambda: self._pagina().mover_tecnica(-1)), ("Down", lambda: self._pagina().mover_tecnica(1)),
                     ("N", lambda: self.b_caso.isEnabled() and self.b_caso.click()), ("0", self.cam.reiniciar)):
            QShortcut(QKeySequence(s), self, activated=f)
        for i in range(len(vistas.GRUPOS)):
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda i=i: self.tabs.setCurrentIndex(i))

    # ------------------------------------------------------------------- acciones
    def _pagina(self) -> PaginaGrupo:
        return self.paginas[vistas.GRUPOS[self.tabs.currentIndex()].id]

    def elegir(self):
        d = self.conf.value("carpeta", str(Path.home()))
        r, _ = QFileDialog.getOpenFileName(self, "Abrir video", d, "Videos (" + " ".join("*" + e for e in EXT) + ")")
        if r:
            self.abrir(r)

    def abrir(self, ruta: str):
        self.conf.setValue("carpeta", str(Path(ruta).parent))
        self._habilitar(False)
        self.statusBar().showMessage(f"Preparando {Path(ruta).name}…")
        self.t.abrir(ruta)

    def _play(self):
        if self.b_play.isEnabled():
            self.t.reproducir(self.b_play.text() == "▶")

    def _soltar(self):
        self._moviendo = False
        self.t.ir(self.linea.value())

    def _cambiar_tab(self, i):
        self.pila.setCurrentIndex(i)
        self.t.grupo(vistas.GRUPOS[i].id)

    def _cambiar_pez(self):
        try:
            p = pf.cargar(self.c_pez.currentData())
        except pf.PerfilInvalido as e:
            self._error(str(e))
            self._llenar_peces(self.perfil.id)
            return
        self._usar(p)

    def _usar(self, p: pf.Perfil):
        self.perfil = p
        self._mostrar_estado_pez()
        if self.ajustes:
            self.ajustes.cargar(p)
        self.t.cambiar_perfil(p)

    def _mostrar_estado_pez(self):
        col = {"calibrado": tema.VERDE, "parcial": tema.NARANJA}.get(self.perfil.estado, tema.ROJO)
        self.l_estado.setText(f"<span style='color:{col}'>●</span> {pf.ESTADOS[self.perfil.estado]}")
        self.l_estado.setToolTip(self.perfil.nota)

    def abrir_ajustes(self, grupo):
        if self.ajustes is None:
            self.ajustes = Ajustes(self.perfil, self)
            self.ajustes.aplicado.connect(self._aplicado)
            self.ajustes.guardado.connect(self._guardado)
        else:
            self.ajustes.cargar(self.perfil)
        self.ajustes.abrir_en(grupo)

    def _aplicado(self, p):
        self.perfil = p
        self._mostrar_estado_pez()
        self.t.cambiar_perfil(p)

    def _guardado(self, id_):
        self._llenar_peces(id_)
        self._usar(pf.cargar(id_))

    # -------------------------------------------------------------------- señales
    def _habilitar(self, si):
        for w in (self.b_ini, self.b_ant, self.b_play, self.b_sig, self.b_fin, self.linea, self.b_caso,
                  self.c_caso, self.b_rein):
            w.setEnabled(si)

    def _video(self, d):
        self.info = d
        self.l_video.setText(f"{Path(d['ruta']).name}  ·  {d['ancho']}×{d['alto']}  ·  {d['fps']:.0f} fps  ·  {d['total']} cuadros")
        self.l_video.setToolTip(d["ruta"])
        self.linea.setRange(0, max(d["total"] - 1, 0))
        self.marco.setCurrentIndex(1)

    def _preparado(self, cuenta):
        self.cuenta = cuenta
        self.barra.hide()
        self._habilitar(True)
        self._conteo(Conteo())
        self.n_total.setText("0" if cuenta else "—")
        if cuenta:
            self.aviso.hide()
            self.l_fase.setText("En curso")
            self.statusBar().showMessage("Listo. Espacio: reproducir.")
        else:
            i, p = self.info, self.perfil
            self.aviso.setText(f"⚠ Este video es {i['ancho']}×{i['alto']} y los parámetros se calibraron para "
                               f"{p.ancho}×{p.alto}: <b>hay que recalibrar</b>. El conteo está bloqueado, pero puedes "
                               "inspeccionar el preprocesamiento, la segmentación y la separación.")
            self.aviso.show()
            self.l_fase.setText("Conteo bloqueado")
            self.statusBar().showMessage("Conteo bloqueado: resolución distinta a la calibrada.")

    def _progreso(self, f, texto):
        self.barra.setVisible(bool(texto) and f < 1)
        self.barra.setValue(int(f * 1000))
        if texto:
            self.statusBar().showMessage(texto)
        elif f >= 1:
            self.statusBar().clearMessage()

    def _conteo(self, c: Conteo):
        self.n_total.setText(str(c.total) if self.cuenta else "—")
        for k, w in self.n.items():
            w.setText(str(getattr(c, k)))

    def _posicion(self, k):
        if not self._moviendo:
            self.linea.blockSignals(True)
            self.linea.setValue(k)
            self.linea.blockSignals(False)
        self.l_k.setText(f"{k + 1} / {self.info['total'] if self.info else 0}")

    def _numeros(self, k, c):
        self._posicion(k)
        self._conteo(c)
        if self.cuenta and self.info:
            self.l_fase.setText("Conteo final" if k >= self.info["total"] - 1 else "En curso")

    def _contadores(self, d):
        for i, g in enumerate(vistas.GRUPOS):
            t = f"{i + 1}  {g.corto}"
            self.tabs.setTabText(i, t + (f"   ·   {d[g.id]}" if g.id in d else ""))

    def _pintura(self, d):
        try:
            if d["grupo"] == vistas.GRUPOS[self.tabs.currentIndex()].id:
                self.paginas[d["grupo"]].poner(d["vistas"], d["roi"])
            self._contadores(d["contadores"])
        finally:
            self.t.gui_lista()

    def _reproduciendo(self, si):
        self.b_play.setText("⏸" if si else "▶")

    def _terminado(self, c: Conteo):
        if self.cuenta:
            self._conteo(c)                       # antes del rótulo: nunca «final» con un número viejo
            self.l_fase.setText("Conteo final")
            self.statusBar().showMessage(f"Video completo: {c.total} peces "
                                         f"({c.confirmados} confirmados y {c.recuperados} recuperados).")

    def _error(self, t):
        self.barra.hide()
        self._habilitar(self.t._motor is not None)
        QMessageBox.critical(self, "Ocurrió un error", t)

    # ----------------------------------------------------------------------- Qt
    def dragEnterEvent(self, e):
        u = e.mimeData().urls()
        if u and u[0].toLocalFile().lower().endswith(EXT):
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.abrir(e.mimeData().urls()[0].toLocalFile())

    def closeEvent(self, e):
        self.t.detener()
        e.accept()
