"""Panel de Ajustes: se genera solo a partir de core.perfil.PARAMETROS."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QGridLayout, QHBoxLayout, QInputDialog, QLabel,
                               QMessageBox, QPushButton, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ..core import perfil as pf
from . import tema

AMBITO = {pf.CANAL: ("🎥", "Depende del canal (cámara, luz, agua)"),
          pf.PEZ: ("🐟", "Depende del pez (o del pez en este canal)")}


class Ajustes(QDialog):
    aplicado = Signal(object)           # Perfil
    guardado = Signal(str)              # id del perfil nuevo

    def __init__(self, perfil: pf.Perfil, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajustes por técnica")
        self.resize(820, 620)
        self.campos: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self._avanzados: list[QWidget] = []
        self.cab = QLabel()
        self.cab.setWordWrap(True)
        self.tabs = QTabWidget()
        self.saltos = QLabel()
        self.saltos.setObjectName("tenue")
        for g, n in pf.NOMBRES_GRUPO.items():
            self.tabs.addTab(self._pagina(g), n)
        av = QCheckBox("Mostrar parámetros avanzados")
        av.toggled.connect(self._ver_avanzados)
        self.error = QLabel("")
        self.error.setStyleSheet(f"color: {tema.ROJO};")
        self.error.setWordWrap(True)
        rest, guar = QPushButton("Restaurar perfil"), QPushButton("Guardar como nuevo pez…")
        rest.setToolTip("Vuelve a los valores guardados del perfil seleccionado")
        canc, apl = QPushButton("Cerrar"), QPushButton("Aplicar")
        apl.setObjectName("principal")
        apl.setToolTip("Aplica los valores y reinicia el conteo")
        rest.clicked.connect(self.restaurar)
        guar.clicked.connect(self.guardar)
        canc.clicked.connect(self.close)
        apl.clicked.connect(self.aplicar)
        pie = QHBoxLayout()
        for w in (rest, guar):
            pie.addWidget(w)
        pie.addStretch(1)
        for w in (canc, apl):
            pie.addWidget(w)
        lay = QVBoxLayout(self)
        lay.addWidget(self.cab)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(av)
        lay.addWidget(self.error)
        lay.addLayout(pie)
        self._ver_avanzados(False)
        self.cargar(perfil)

    def _pagina(self, grupo):
        w = QWidget()
        gr = QGridLayout(w)
        gr.setHorizontalSpacing(14)
        gr.setColumnStretch(4, 1)
        fila = 0
        for p in pf.PARAMETROS:
            if p.grupo != grupo:
                continue
            c = QSpinBox() if p.tipo is int else QDoubleSpinBox()
            if p.tipo is float:
                c.setDecimals(4 if p.paso < .01 else 3 if p.paso < .1 else 2)
            c.setRange(p.minimo, p.maximo)
            c.setSingleStep(p.paso)
            c.setToolTip(p.ayuda)
            c.setAccessibleName(p.nombre)
            c.setMinimumWidth(110)
            c.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            c.wheelEvent = lambda e: e.ignore()             # la rueda no debe cambiar valores por accidente
            if grupo == pf.TRACKING:
                c.valueChanged.connect(lambda _: self._mostrar_saltos())
            self.campos[p.clave] = c
            ic, tip = AMBITO[p.ambito]
            et = QLabel(p.nombre)
            am = QLabel(ic)
            am.setToolTip(tip)
            un = QLabel(p.unidad)
            un.setObjectName("tenue")
            ay = QLabel(p.ayuda)
            ay.setObjectName("tenue")
            ay.setWordWrap(True)
            fila_w = (et, c, un, am, ay)
            for col, x in enumerate(fila_w):
                gr.addWidget(x, fila, col, Qt.AlignmentFlag.AlignTop)
            if p.avanzado:
                self._avanzados += fila_w
            fila += 1
        if grupo == pf.TRACKING:
            gr.addWidget(self.saltos, fila, 0, 1, 5)
            fila += 1
        gr.setRowStretch(fila, 1)
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setWidget(w)
        return s

    def _ver_avanzados(self, si):
        for w in self._avanzados:
            w.setVisible(si)

    def abrir_en(self, grupo):
        self.tabs.setCurrentIndex(list(pf.NOMBRES_GRUPO).index(grupo))
        self.show()
        self.raise_()
        self.activateWindow()

    def cargar(self, perfil: pf.Perfil):
        self.perfil = perfil
        for k, c in self.campos.items():
            c.blockSignals(True)
            c.setValue(perfil.valores[k])
            c.blockSignals(False)
        self.cab.setText(f"<b>{perfil.nombre}</b> — {pf.ESTADOS[perfil.estado]}<br>"
                         f"<span style='color:{tema.TENUE}'>{perfil.nota}</span>")
        self.error.setText("")
        self._mostrar_saltos()

    def _valores(self) -> dict:
        # Un campo no tocado conserva el valor exacto del perfil (el campo lo muestra redondeado)
        v = {}
        for k, c in self.campos.items():
            orig = self.perfil.valores[k]
            dec = c.decimals() if isinstance(c, QDoubleSpinBox) else 0
            v[k] = orig if abs(c.value() - orig) <= .5 * 10 ** -dec + 1e-12 else c.value()
        return v

    def _mostrar_saltos(self):
        try:
            v = self._valores()
            p = pf.Perfil("", "", "editado", v)
            self.saltos.setText("Salto esperado (normalizado a 1280×720):  " + "   ".join(
                f"x={x}: {p.salto(x, 360):.0f} px" for x in (0, 300, 600)))
        except (KeyError, AttributeError):
            pass

    def aplicar(self):
        try:
            nuevo = self.perfil.con(**self._valores())
        except pf.PerfilInvalido as e:
            self.error.setText(str(e))
            return
        self.error.setText("")
        self.cargar(nuevo)
        self.aplicado.emit(nuevo)

    def restaurar(self):
        try:
            p = pf.cargar(self.perfil.id)
        except pf.PerfilInvalido as e:
            self.error.setText(str(e))
            return
        self.cargar(p)
        self.aplicado.emit(p)

    def guardar(self):
        try:
            nuevo = self.perfil.con(**self._valores())
        except pf.PerfilInvalido as e:
            self.error.setText(str(e))
            return
        nombre, ok = QInputDialog.getText(self, "Nuevo pez", "Nombre del perfil:")
        if not ok or not nombre.strip():
            return
        id_ = "".join(c if c.isascii() and c.isalnum() else "_" for c in nombre.strip().lower()).strip("_")
        try:
            pf.guardar_como(nuevo, id_, nombre.strip())
        except (pf.PerfilInvalido, OSError) as e:
            QMessageBox.warning(self, "No se pudo guardar", str(e))
            return
        self.guardado.emit(id_)
