"""Visor: entrada del grupo seguida de la salida de cada técnica, con zoom compartido."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from ..core import vistas
from . import tema


def a_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class Camara(QObject):
    """Zoom, centro y cursor compartidos por todas las imágenes (en píxeles del cuadro completo)."""
    cambio = Signal()
    cursor = Signal(object)

    def __init__(self):
        super().__init__()
        self.zoom, self.centro, self.punto = 1.0, None, None

    def reiniciar(self):
        self.zoom, self.centro = 1.0, None
        self.cambio.emit()

    def poner(self, zoom, centro):
        self.zoom = max(1.0, min(zoom, 24.0))
        self.centro = centro if self.zoom > 1.0 else None
        self.cambio.emit()

    def apuntar(self, p):
        self.punto = p
        self.cursor.emit(p)


class Lienzo(QWidget):
    doble_clic = Signal()

    def __init__(self, camara: Camara):
        super().__init__()
        self.cam = camara
        self.imagen: QImage | None = None
        self.segunda: QImage | None = None        # modo deslizador
        self.corte = .5
        self.origen = (0, 0)                      # esquina de la imagen en el cuadro completo
        self.mensaje = ""
        self._arrastre = None
        self._moviendo_corte = False
        self.setMouseTracking(True)
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        camara.cambio.connect(self.update)
        camara.cursor.connect(lambda _: self.update())

    def poner(self, im: QImage, origen, mensaje="", segunda=None):
        self.imagen, self.origen, self.mensaje, self.segunda = im, origen, mensaje, segunda
        self.update()

    # --------------------------------------------------------------- geometría
    def _transf(self):
        """(escala, x0, y0): pantalla = x0 + escala * píxel de la imagen."""
        w, h = self.imagen.width(), self.imagen.height()
        s = min(self.width() / w, self.height() / h) * self.cam.zoom
        if self.cam.centro is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = self.cam.centro[0] - self.origen[0], self.cam.centro[1] - self.origen[1]
        return s, self.width() / 2 - s * cx, self.height() / 2 - s * cy

    def a_cuadro(self, pos: QPointF):
        s, x0, y0 = self._transf()
        return (pos.x() - x0) / s + self.origen[0], (pos.y() - y0) / s + self.origen[1]

    # ------------------------------------------------------------------ dibujo
    def paintEvent(self, _):
        q = QPainter(self)
        q.fillRect(self.rect(), QColor("#0c0e11"))
        if self.imagen is None:
            return
        s, x0, y0 = self._transf()
        q.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < 2)
        r = QRectF(x0, y0, s * self.imagen.width(), s * self.imagen.height())
        q.drawImage(r, self.imagen)
        if self.segunda is not None:
            xc = self.corte * self.width()
            q.save()
            q.setClipRect(QRectF(xc, 0, self.width() - xc, self.height()))
            q.drawImage(r, self.segunda)
            q.restore()
            q.setPen(QPen(QColor(tema.ACENTO), 2))
            q.drawLine(QPointF(xc, 0), QPointF(xc, self.height()))
            q.setBrush(QColor(tema.ACENTO))
            q.drawEllipse(QPointF(xc, self.height() / 2), 8, 8)
        if self.cam.punto is not None:
            px = x0 + s * (self.cam.punto[0] - self.origen[0])
            py = y0 + s * (self.cam.punto[1] - self.origen[1])
            q.setPen(QPen(QColor(230, 30, 110, 190), 1))
            q.drawLine(QPointF(px, 0), QPointF(px, self.height()))
            q.drawLine(QPointF(0, py), QPointF(self.width(), py))
        if self.mensaje:
            caja = q.fontMetrics().boundingRect(self.mensaje).adjusted(-14, -8, 14, 8)
            caja.moveCenter(self.rect().center())
            q.setPen(QPen(QColor(tema.BORDE)))
            q.setBrush(QColor(22, 25, 29, 225))
            q.drawRoundedRect(caja, 6, 6)
            q.setPen(QColor(tema.TEXTO))
            q.drawText(caja, Qt.AlignmentFlag.AlignCenter, self.mensaje)

    # ------------------------------------------------------------------- mouse
    def wheelEvent(self, e):
        if self.imagen is None:
            return
        f = 1.25 if e.angleDelta().y() > 0 else .8
        p = self.a_cuadro(e.position())
        z = max(1.0, min(self.cam.zoom * f, 24.0))
        s = min(self.width() / self.imagen.width(), self.height() / self.imagen.height()) * z
        c = (p[0] - (e.position().x() - self.width() / 2) / s, p[1] - (e.position().y() - self.height() / 2) / s)
        self.cam.poner(z, c)
        e.accept()

    def mousePressEvent(self, e):
        if self.imagen is None or e.button() != Qt.MouseButton.LeftButton:
            return
        if self.segunda is not None and abs(e.position().x() - self.corte * self.width()) < 14:
            self._moviendo_corte = True
        else:
            self._arrastre = (e.position(), self.cam.centro or self.a_cuadro(QPointF(self.width() / 2, self.height() / 2)))

    def mouseMoveEvent(self, e):
        if self.imagen is None:
            return
        if self._moviendo_corte:
            self.corte = max(.02, min(e.position().x() / max(self.width(), 1), .98))
            self.update()
            return
        if self._arrastre and self.cam.zoom > 1:
            s = self._transf()[0]
            p0, c = self._arrastre
            d = e.position() - p0
            self.cam.poner(self.cam.zoom, (c[0] - d.x() / s, c[1] - d.y() / s))
        self.cam.apuntar(self.a_cuadro(e.position()))

    def mouseReleaseEvent(self, _):
        self._arrastre, self._moviendo_corte = None, False

    def mouseDoubleClickEvent(self, _):
        self.doble_clic.emit()

    def leaveEvent(self, _):
        self.cam.apuntar(None)


class Tarjeta(QFrame):
    """Título, imagen y valor bajo el cursor."""
    ampliar = Signal(int)

    def __init__(self, indice, tecnica: vistas.Tecnica, numero: str, camara: Camara):
        super().__init__()
        self.setObjectName("tarjeta")
        self.indice, self.tecnica, self.cam = indice, tecnica, camara
        self.dato: vistas.Imagen | None = None
        t = QLabel(f"{numero}  {tecnica.nombre}")
        t.setObjectName("titulo")
        t.setToolTip(f"{tecnica.nombre}\n{tecnica.ayuda}\n\nDoble clic para ampliar y comparar con la entrada.")
        t.setWordWrap(True)
        self.valor = QLabel("")
        self.valor.setObjectName("tenue")
        self.valor.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cab = QHBoxLayout()
        cab.addWidget(t, 1)
        cab.addWidget(self.valor)
        self.lienzo = Lienzo(camara)
        self.lienzo.setAccessibleName(tecnica.nombre)
        self.lienzo.setToolTip(tecnica.ayuda)
        self.lienzo.doble_clic.connect(lambda: self.ampliar.emit(self.indice))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(4)
        lay.addLayout(cab)
        lay.addWidget(self.lienzo, 1)
        camara.cursor.connect(self._leer)

    def poner(self, dato: vistas.Imagen, origen):
        self.dato = dato
        self.lienzo.poner(a_qimage(dato.rgb), origen, dato.mensaje)
        self._leer(self.cam.punto)

    def _leer(self, p):
        self.valor.setText(leer_valor(self.dato, p, self.lienzo.origen))


def leer_valor(dato, p, origen) -> str:
    if dato is None or dato.valor is None or p is None:
        return ""
    x, y = int(p[0] - origen[0]), int(p[1] - origen[1])
    h, w = dato.valor.shape[:2]
    return dato.formato.format(float(dato.valor[y, x])) if 0 <= x < w and 0 <= y < h else ""


FLECHA = {"entrada": "ENTRADA", "auxiliar": "REF."}


class Cuadricula(QScrollArea):
    """Tarjetas en orden; las columnas se adaptan al ancho sin perder el orden."""
    ampliar = Signal(int)

    def __init__(self, grupo: vistas.Grupo, camara: Camara):
        super().__init__()
        self.setWidgetResizable(True)
        cont = QWidget()
        self.grid = QGridLayout(cont)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setSpacing(8)
        self.setWidget(cont)
        self.tarjetas, n = [], 0
        for i, t in enumerate(grupo.tecnicas):
            n += t.tipo == "tecnica"
            c = Tarjeta(i, t, FLECHA.get(t.tipo) or f"{n} →", camara)
            c.ampliar.connect(self.ampliar)
            self.tarjetas.append(c)
        self._cols = 0
        self._ordenar(3)

    def _ordenar(self, cols):
        if cols == self._cols:
            return
        self._cols = cols
        for c in self.tarjetas:
            self.grid.removeWidget(c)
        for i, c in enumerate(self.tarjetas):
            self.grid.addWidget(c, i // cols, i % cols)
        for i in range(4):
            self.grid.setColumnStretch(i, 1 if i < cols else 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        w, h = self.viewport().width(), self.viewport().height()
        self._ordenar(1 if w < 520 else 2 if w < 900 or len(self.tarjetas) <= 4 else 3)
        # Cada imagen conserva la proporción, pero todas caben sin desplazarse si hay alto suficiente
        filas = -(-len(self.tarjetas) // self._cols)
        alto = min(int(max(w / self._cols - 28, 160) * 9 / 16), max((h - 8 * filas) // filas - 48, 130))
        for c in self.tarjetas:
            c.lienzo.setMinimumHeight(alto)


class Ampliado(QWidget):
    """Una técnica grande junto a la entrada de su grupo (lado a lado o con deslizador)."""
    volver = Signal()

    def __init__(self, grupo: vistas.Grupo, camara: Camara):
        super().__init__()
        self.grupo, self.cam, self.i = grupo, camara, 1
        self.datos: list[vistas.Imagen] = []
        self.roi = (0, 0)
        self.titulo = QLabel()
        self.titulo.setObjectName("titulo")
        self.valor = QLabel()
        self.valor.setObjectName("tenue")
        b = QPushButton("← Cuadrícula")
        b.clicked.connect(self.volver)
        ant, sig = QPushButton("◀ Anterior"), QPushButton("Siguiente ▶")
        ant.setToolTip("Técnica anterior (↑)")
        sig.setToolTip("Técnica siguiente (↓)")
        ant.clicked.connect(lambda: self.mover(-1))
        sig.clicked.connect(lambda: self.mover(1))
        self.lado, self.desl = QPushButton("Lado a lado"), QPushButton("Deslizador")
        g = QButtonGroup(self)
        for x in (self.lado, self.desl):
            x.setCheckable(True)
            g.addButton(x)
        self.lado.setChecked(True)
        g.buttonClicked.connect(lambda _: self.refrescar())
        self.dif = QCheckBox("Resaltar diferencias")
        self.dif.setToolTip("Rojo: la técnica oscureció. Verde: aclaró. Respecto a la entrada del grupo.")
        self.dif.toggled.connect(lambda _: self.refrescar())
        barra = QHBoxLayout()
        for w in (b, ant, sig, self.titulo):
            barra.addWidget(w)
        barra.addStretch(1)
        for w in (self.valor, self.dif, self.lado, self.desl):
            barra.addWidget(w)
        self.a, self.b = Lienzo(camara), Lienzo(camara)
        self.a.doble_clic.connect(self.volver)
        self.b.doble_clic.connect(self.volver)
        cuerpo = QHBoxLayout()
        cuerpo.addWidget(self.a)
        cuerpo.addWidget(self.b)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(barra)
        lay.addLayout(cuerpo, 1)
        camara.cursor.connect(self._leer)

    def mover(self, d):
        self.i = max(0, min(self.i + d, len(self.grupo.tecnicas) - 1))
        self.refrescar()

    def poner(self, datos, roi):
        self.datos, self.roi = datos, roi
        self.refrescar()

    def _origen(self, t):
        return self.roi if t.roi else (0, 0)

    def _igualar(self, dato, t):
        """La entrada del preprocesamiento es el cuadro completo: se recorta a la ROI para comparar."""
        if t.roi or not self.grupo.tecnicas[self.i].roi:
            return dato.rgb, self._origen(t)
        x, y = self.roi
        return dato.rgb[y:, x:], self.roi

    def refrescar(self):
        if not self.datos:
            return
        t, te = self.grupo.tecnicas[self.i], self.grupo.tecnicas[0]
        d = self.datos[self.i]
        self.titulo.setText(t.nombre)
        self.titulo.setToolTip(t.ayuda)
        ent, o = self._igualar(self.datos[0], te)
        sal = d.rgb
        if self.dif.isChecked() and self.i:
            r = vistas.diferencia(vistas.Imagen(ent), d)
            sal = r if r is not None else sal
        if self.desl.isChecked() and ent.shape == sal.shape:
            self.b.hide()
            self.a.poner(a_qimage(ent), o, d.mensaje, a_qimage(sal))
        else:
            self.b.show()
            self.a.poner(a_qimage(ent), o, self.datos[0].mensaje)
            self.b.poner(a_qimage(sal), self._origen(t), d.mensaje)
        self._leer(self.cam.punto)

    def _leer(self, p):
        if self.datos:
            t = self.grupo.tecnicas[self.i]
            self.valor.setText(leer_valor(self.datos[self.i], p, self._origen(t)))


class PaginaGrupo(QWidget):
    """Un grupo de técnicas: cuadrícula o vista ampliada."""
    ajustes = Signal(str)

    def __init__(self, grupo: vistas.Grupo, camara: Camara):
        super().__init__()
        self.grupo, self.cam = grupo, camara
        self.cuad, self.amp = Cuadricula(grupo, camara), Ampliado(grupo, camara)
        self.pila = QStackedWidget()
        self.pila.addWidget(self.cuad)
        self.pila.addWidget(self.amp)
        self.cuad.ampliar.connect(self.abrir_ampliado)
        self.amp.volver.connect(lambda: self.pila.setCurrentIndex(0))
        t = QLabel(grupo.nombre)
        t.setObjectName("titulo")
        z = QPushButton("Ajustar zoom")
        z.setToolTip("Restablecer zoom (0). Rueda: zoom en todas las imágenes. Arrastrar: desplazar.")
        z.clicked.connect(camara.reiniciar)
        aj = QPushButton("⚙ Parámetros")
        aj.setToolTip(f"Abrir los parámetros de {grupo.nombre}")
        aj.clicked.connect(lambda: self.ajustes.emit(grupo.id))
        cab = QHBoxLayout()
        cab.addWidget(t)
        cab.addStretch(1)
        cab.addWidget(z)
        cab.addWidget(aj)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(cab)
        lay.addWidget(self.pila, 1)

    def abrir_ampliado(self, i):
        self.amp.i = max(i, 1) if len(self.grupo.tecnicas) > 1 else 0
        self.pila.setCurrentIndex(1)
        self.amp.refrescar()

    def mover_tecnica(self, d):
        if self.pila.currentIndex() == 0:
            self.abrir_ampliado(1)
        else:
            self.amp.mover(d)

    def poner(self, vs, roi):
        datos = [d for _, d in vs]
        for c, d in zip(self.cuad.tarjetas, datos):
            c.poner(d, roi if c.tecnica.roi else (0, 0))
        self.amp.poner(datos, roi)
