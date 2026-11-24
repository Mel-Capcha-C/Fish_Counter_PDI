"""Imágenes de cada técnica, agrupadas. Sin Qt: devuelven arrays RGB.

GRUPOS es el registro que usa la interfaz para crear las pestañas. Para añadir una técnica basta
con escribir su función y agregarla al grupo: no hay que tocar la interfaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import cv2
import numpy as np

from . import perfil as pf, segmentacion as sg, separacion as sp
from .motor import Motor, ResultadoCuadro
from .tracking import centro

# Paleta de Okabe-Ito (apta para daltonismo), RGB
AZUL, NARANJA, VERDE, ROJO, MORADO, CELESTE, AMARILLO = (
    (0, 114, 178), (230, 159, 0), (0, 158, 115), (213, 94, 0), (204, 121, 167), (86, 180, 233), (240, 228, 66))
CICLO = (CELESTE, NARANJA, VERDE, AMARILLO, MORADO, AZUL, ROJO)
GRIS = (150, 150, 150)


@dataclass
class Imagen:
    """Salida de una técnica. ``valor`` (opcional) es lo que se lee al pasar el mouse."""
    rgb: np.ndarray
    valor: np.ndarray | None = None
    formato: str = "gris {:.0f}"
    mensaje: str = ""               # si la técnica no tiene nada que mostrar en este cuadro


@dataclass(frozen=True)
class Tecnica:
    id: str
    nombre: str
    funcion: Callable[["Contexto"], Imagen]
    ayuda: str
    tipo: str = "tecnica"           # entrada | tecnica | auxiliar
    roi: bool = True                # False: usa el cuadro completo, no la ROI


@dataclass(frozen=True)
class Grupo:
    id: str
    nombre: str
    corto: str
    tecnicas: tuple[Tecnica, ...]
    cuenta: Callable[["Contexto"], str]      # texto del contador de la pestaña


def _rgb(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def _tenue(g):
    """Fondo aclarado para que destaquen las marcas."""
    return _rgb((g.astype(np.float32) * .55 + 255 * .45).astype(np.uint8))


def _texto(im, t, x, y, color=(0, 0, 0), esc=.7):
    x, y = int(min(max(x, 2), im.shape[1] - 10)), int(min(max(y, 14), im.shape[0] - 4))
    cv2.putText(im, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(im, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, color, 2, cv2.LINE_AA)


class Contexto:
    """Todo lo que necesita una vista del cuadro ``k``. Cada paso se calcula una vez y bajo demanda."""

    def __init__(self, motor: Motor, r: ResultadoCuadro, bgr: np.ndarray, previa: np.ndarray | None = None):
        self.motor, self.p, self.r, self.bgr, self._previa = motor, motor.p, r, bgr, previa
        if motor.ultimo and motor.ultimo[0] == r.k:          # ya calculado al contar este cuadro
            self.__dict__.update(motor.ultimo[1])

    @cached_property
    def roi(self):
        return sg.recortar(self.bgr, self.p)

    @cached_property
    def plano(self):
        return sg.campo_plano(self.roi, self.motor.gan)

    @cached_property
    def g(self):
        return sg.suavizar(self.plano, self.p)

    @cached_property
    def binaria(self):
        return sg.umbralizar(self.g, self.p)

    @cached_property
    def abierta(self):
        return sg.abrir(self.binaria, self.p)

    @cached_property
    def comps(self):
        return sg.componentes(self.abierta, self.g, self.p)

    @cached_property
    def moviles(self):
        est = {tuple(e) for e in self.r.estaticas}
        return [m for m in self.comps[0] if (m.x, m.y, *m.mask.shape[::-1]) not in est]

    @cached_property
    def analisis(self):
        return [sp.analizar(m, self.p) for m in self.moviles]

    @cached_property
    def g_previa(self):
        if self._previa is None:
            return None
        return sg.suavizar(sg.campo_plano(sg.recortar(self._previa, self.p), self.motor.gan), self.p)

    # caja en 1280x720 -> píxeles de la ROI
    def px(self, d):
        return (int(round(d[0] / self.p.fx)), int(round(d[1] / self.p.fy)),
                int(round(d[2] / self.p.fx)), int(round(d[3] / self.p.fy)))

    def pt(self, x, y):
        return int(round(x / self.p.fx)), int(round(y / self.p.fy))

    def lienzo_gris(self):
        im = np.full(self.g.shape, 255, np.uint8)
        for m in self.moviles:
            h, w = m.mask.shape
            im[m.y:m.y + h, m.x:m.x + w][m.mask > 0] = m.gris[m.mask > 0]
        return im


# ---------------------------------------------------------------- preprocesamiento
def v_original(c):
    im = cv2.cvtColor(c.bgr, cv2.COLOR_BGR2RGB)
    cv2.rectangle(im, (c.p.roi_x0, c.p.roi_y0), (im.shape[1] - 2, im.shape[0] - 2), NARANJA, 2)
    return Imagen(im, cv2.cvtColor(c.bgr, cv2.COLOR_BGR2GRAY))


def v_roi(c):
    return Imagen(_rgb(c.roi), c.roi)


def v_fondo(c):
    f = c.motor.fondo.astype(np.uint8)
    return Imagen(_rgb(f), f)


def v_plano(c):
    return Imagen(_rgb(c.plano), c.plano)


def v_gauss(c):
    return Imagen(_rgb(c.g), c.g)


# -------------------------------------------------------------------- segmentación
def v_umbral(c):
    return Imagen(_rgb(c.binaria * 255), c.g)


def v_apertura(c):
    im = _rgb(c.abierta * 255)
    im[(c.binaria > 0) & (c.abierta == 0)] = ROJO
    return Imagen(im, c.g)


def v_componentes(c):
    im = np.full((*c.g.shape, 3), 255, np.uint8)
    area = np.zeros(c.g.shape, np.int32)
    ok, chicas = c.comps
    for i, m in enumerate(ok):
        h, w = m.mask.shape
        im[m.y:m.y + h, m.x:m.x + w][m.mask > 0] = CICLO[i % len(CICLO)]
        area[m.y:m.y + h, m.x:m.x + w][m.mask > 0] = m.area
    for x, y, w, h, a in chicas:
        cv2.rectangle(im, (x, y), (x + w, y + h), GRIS, 1)
        area[y:y + h, x:x + w] = a
    return Imagen(im, area, "área {:.0f} px", "" if ok or chicas else "Sin manchas en este cuadro")


def v_estaticos(c):
    im = _tenue(c.g)
    for m in c.moviles:
        h, w = m.mask.shape
        im[m.y:m.y + h, m.x:m.x + w][m.mask > 0] = VERDE
    for x, y, w, h in c.r.estaticas:
        cv2.rectangle(im, (x, y), (x + w, y + h), ROJO, 3)
        cv2.line(im, (x, y), (x + w, y + h), ROJO, 2)
        _texto(im, "estatico", x, y - 5, ROJO)
    return Imagen(im, c.g, mensaje="" if c.moviles or c.r.estaticas else "Sin manchas en este cuadro")


# ---------------------------------------------------------------------- separación
def v_manchas(c):
    g = c.lienzo_gris()
    return Imagen(_rgb(g), g, mensaje="" if c.moviles else "Sin manchas en este cuadro")


SIN_SOSPECHOSAS = "Sin manchas sospechosas en este cuadro"


def _hay(c):
    return any(a.sospechosa for a in c.analisis)


def v_solidez(c):
    im = _tenue(c.lienzo_gris())
    s = np.zeros(c.g.shape, np.float32)
    for m, a in zip(c.moviles, c.analisis):
        h, w = m.mask.shape
        s[m.y:m.y + h, m.x:m.x + w][m.mask > 0] = a.solidez
        cont = max(cv2.findContours(m.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        casco = cv2.convexHull(cont) + np.array([m.x, m.y])
        col = ROJO if a.sospechosa else VERDE
        cv2.polylines(im, [casco], True, col, 2, cv2.LINE_AA)
        _texto(im, f"{a.solidez:.2f}" + (" ?" if a.sospechosa else ""), m.x, m.y - 4, col)
    return Imagen(im, s, "solidez {:.2f}", "" if c.moviles else "Sin manchas en este cuadro")


def v_huecos(c):
    im = _tenue(c.lienzo_gris())
    n = 0
    for m, a in zip(c.moviles, c.analisis):
        if not a.sospechosa:
            continue
        h, w = m.mask.shape
        z = im[m.y:m.y + h, m.x:m.x + w]
        z[a.rellena.mask > 0] = _rgb(a.rellena.gris)[a.rellena.mask > 0]
        rell = a.rellena.gris != m.gris
        z[rell] = NARANJA
        n += int(rell.sum())
    if not _hay(c):
        return Imagen(im, c.g, mensaje=SIN_SOSPECHOSAS)
    _texto(im, f"{n} px rellenados", 8, 20)
    return Imagen(im, c.g)


def _campo(c):
    im = np.full((*c.g.shape, 3), 255, np.uint8)
    val = np.zeros(c.g.shape, np.float32)
    mx = max([float(a.campo.max()) for a in c.analisis if a.sospechosa], default=1.0)
    for m, a in zip(c.moviles, c.analisis):
        if not a.sospechosa:
            continue
        h, w = m.mask.shape
        z = cv2.cvtColor(cv2.applyColorMap((255 * a.campo / max(mx, 1e-6)).astype(np.uint8),
                                           cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        k = a.rellena.mask > 0
        im[m.y:m.y + h, m.x:m.x + w][k] = z[k]
        val[m.y:m.y + h, m.x:m.x + w][k] = a.campo[k]
    return im, val


def v_distancia(c):
    im, val = _campo(c)
    return Imagen(im, val, "distancia {:.1f} px", "" if _hay(c) else SIN_SOSPECHOSAS)


def v_cimas(c):
    im, val = _campo(c)
    for m, a in zip(c.moviles, c.analisis):
        if not a.sospechosa:
            continue
        for i, (r, q, raz) in enumerate(a.cimas or []):
            x, y = m.x + q, m.y + r
            cv2.circle(im, (x, y), 6, (255, 255, 255), -1)
            cv2.circle(im, (x, y), 6, ROJO, 2)
            if i:
                _texto(im, f"cuello {raz:.2f}", x + 8, y, ROJO)
    return Imagen(im, val, "distancia {:.1f} px", "" if _hay(c) else SIN_SOSPECHOSAS)


def v_reparto(c):
    im = _tenue(c.lienzo_gris())
    n = 0
    for m, a in zip(c.moviles, c.analisis):
        h, w = m.mask.shape
        if a.etiquetas is not None:
            z = im[m.y:m.y + h, m.x:m.x + w]
            for i in range(len(a.piezas)):
                z[a.etiquetas == i + 1] = CICLO[(n + i) % len(CICLO)]
        for x, y, ww, hh, _, _ in a.piezas:
            col = ROJO if len(a.piezas) > 1 else VERDE
            cv2.rectangle(im, (m.x + x, m.y + y), (m.x + x + ww, m.y + y + hh), col, 2)
        if len(a.piezas) > 1:
            _texto(im, f"{len(a.piezas)} peces", m.x, m.y - 4, ROJO)
        n += len(a.piezas)
    return Imagen(im, c.g, mensaje="" if c.moviles else "Sin manchas en este cuadro")


# ------------------------------------------------------------------------ tracking
NO_CUENTA = "Conteo bloqueado: el video no coincide con la resolución calibrada"


def _bloqueado(c):
    if c.motor.contar:
        return None
    return Imagen(_tenue(c.g), c.g, mensaje=NO_CUENTA)


def v_previos(c):
    b = _bloqueado(c)
    if b:
        return b
    k = c.r.k
    if k == 0 or c.g_previa is None:
        return Imagen(_tenue(c.g), c.g, mensaje="Es el primer cuadro: no hay anterior")
    im = _tenue(c.g_previa)
    for j, d in enumerate(c.motor.dets[k - 1]):
        x0, y0, x1, y1 = c.px(d)
        cv2.rectangle(im, (x0, y0), (x1, y1), AZUL, 2)
        _texto(im, str(c.motor.ids[k - 1][j]), x0, y0 - 4, AZUL)
    _texto(im, f"cuadro {k - 1}", 8, 20)
    return Imagen(im, c.g_previa)


def v_prediccion(c):
    b = _bloqueado(c)
    if b:
        return b
    im = _tenue(c.g)
    for i, x0, y0, x, y, sx, sy in c.r.paso.predicciones:
        a, z = c.pt(x0, y0), c.pt(x, y)
        cv2.arrowedLine(im, a, z, AZUL, 1, cv2.LINE_AA, tipLength=.03)
        ej = (max(int(2 * sx / c.p.fx), 1), max(int(2 * sy / c.p.fy), 1))
        cv2.ellipse(im, z, ej, 0, 0, 360, AZUL, 1, cv2.LINE_AA)
        cv2.drawMarker(im, z, AZUL, cv2.MARKER_CROSS, 14, 2)
        _texto(im, str(i), z[0] + 6, z[1] - 6, AZUL)
    return Imagen(im, c.g, mensaje="" if c.r.paso.predicciones else "No hay trayectorias activas")


def v_asignacion(c):
    b = _bloqueado(c)
    if b:
        return b
    im = _tenue(c.g)
    pred = {q[0]: q for q in c.r.paso.predicciones}
    for j, d in enumerate(c.r.dets):
        x0, y0, x1, y1 = c.px(d)
        nueva = c.r.paso.ids[j] in c.r.paso.nuevas
        cv2.rectangle(im, (x0, y0), (x1, y1), NARANJA if nueva else VERDE, 2)
        if nueva:
            _texto(im, "nuevo", x0, y0 - 4, NARANJA)
    for i, j, s in c.r.paso.parejas:
        z, o = c.pt(*pred[i][3:5]), c.pt(*centro(c.r.dets[j]))
        cv2.line(im, z, o, VERDE, 2, cv2.LINE_AA)
        cv2.drawMarker(im, z, AZUL, cv2.MARKER_CROSS, 12, 2)
        _texto(im, f"{i}: {s:.1f}", o[0] - 10, o[1] - 12, VERDE, .6)
    for i in c.r.paso.perdidas:
        z = c.pt(*pred[i][3:5])
        cv2.drawMarker(im, z, ROJO, cv2.MARKER_TILTED_CROSS, 18, 2)
        _texto(im, f"{i} sin deteccion", z[0] + 8, z[1], ROJO, .6)
    return Imagen(im, c.g, mensaje="" if len(c.r.dets) or c.r.paso.predicciones else "Sin peces en este cuadro")


def v_trayectorias(c):
    b = _bloqueado(c)
    if b:
        return b
    im = _tenue(c.g)
    for j, d in enumerate(c.r.dets):
        i = c.r.paso.ids[j]
        col = CICLO[i % len(CICLO)]
        P = [c.pt(*centro(c.motor.dets[k][q])) for k, q in c.motor.obs[i] if k <= c.r.k]
        for a, z in zip(P, P[1:]):
            cv2.line(im, a, z, col, 2, cv2.LINE_AA)
        for z in P:
            cv2.circle(im, z, 4, col, -1)
        x0, y0, x1, y1 = c.px(d)
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
        _texto(im, f"ID {i}", x0, y0 - 4, col)
    return Imagen(im, c.g, mensaje="" if len(c.r.dets) else "Sin peces en este cuadro")


# ---------------------------------------------------------------------- oclusiones
COLOR_ESTADO = {"confirmada": VERDE, "recuperada": AZUL, "pendiente": NARANJA, "descartada": ROJO}


def v_confirmacion(c):
    b = _bloqueado(c)
    if b:
        return b
    im = _tenue(c.g)
    for j, d in enumerate(c.r.dets):
        i = c.r.paso.ids[j]
        n = len([1 for k, _ in c.motor.obs[i] if k <= c.r.k])
        x0, y0, x1, y1 = c.px(d)
        col = VERDE if n > 1 else NARANJA
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
        _texto(im, f"{n} vista" + ("s" if n > 1 else ""), x0, y0 - 4, col)
    return Imagen(im, c.g, mensaje="" if len(c.r.dets) else "Sin peces en este cuadro")


def v_recuperacion(c):
    b = _bloqueado(c)
    if b:
        return b
    im = _tenue(c.g)
    hay = False
    for j, d in enumerate(c.r.dets):
        i = c.r.paso.ids[j]
        x0, y0, x1, y1 = c.px(d)
        m = c.motor.motivos.get(i)
        if m:
            hay = True
            cv2.rectangle(im, (x0, y0), (x1, y1), AZUL, 3)
            _texto(im, f"ID {i} recuperado: oculto en {m.cuadro} (area x{m.razon_area:.2f})", x0, y0 - 5, AZUL, .6)
        elif len(c.motor.obs[i]) == 1:
            e = c.motor.estado(i)
            cv2.rectangle(im, (x0, y0), (x1, y1), COLOR_ESTADO[e], 2)
            _texto(im, f"ID {i} {e}", x0, y0 - 5, COLOR_ESTADO[e], .6)
            hay = True
    for i, m in c.motor.motivos.items():                # manchas de este cuadro que ocultan a un pez
        if m.cuadro == c.r.k and m.deteccion < len(c.r.dets):
            x0, y0, x1, y1 = c.px(c.r.dets[m.deteccion])
            cv2.rectangle(im, (x0, y0), (x1, y1), MORADO, 2)
            _texto(im, f"oculta al ID {i}", x0, y1 + 14, MORADO, .6)
            hay = True
    return Imagen(im, c.g, mensaje="" if hay else "Sin trayectorias de una sola vista en este cuadro")


def v_conteo(c):
    b = _bloqueado(c)
    if b:
        return b
    im = _tenue(c.g)
    for j, d in enumerate(c.r.dets):
        i = c.r.paso.ids[j]
        e = c.motor.estado(i)
        x0, y0, x1, y1 = c.px(d)
        col = COLOR_ESTADO[e]
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 3 if e in ("confirmada", "recuperada") else 1)
        _texto(im, {"confirmada": "contado", "recuperada": "contado (recuperado)", "pendiente": "pendiente",
                    "descartada": "no contado"}[e], x0, y0 - 4, col, .6)
    q = c.r.conteo
    _texto(im, f"{q.total} peces", 10, 30, VERDE, 1.0)
    return Imagen(im, c.g)


T = Tecnica
GRUPOS: tuple[Grupo, ...] = (
    Grupo(pf.PREPROCESAMIENTO, "Preprocesamiento", "Preproc.", (
        T("original", "Cuadro original (RGB)", v_original, "Imagen de la cámara. El marco es la ROI.", "entrada",
          roi=False),
        T("roi", "Recorte de ROI y luminancia", v_roi, "Sólo la zona del canal, en escala de grises."),
        T("fondo", "Estimación de fondo (mediana temporal)", v_fondo,
          "Referencia del video, no del cuadro: la usa el campo plano.", "auxiliar"),
        T("plano", "Corrección por campo plano (flat-field)", v_plano,
          "Divide por el fondo para que el canal vacío tenga el mismo gris en todas partes."),
        T("gauss", "Suavizado gaussiano", v_gauss, "Reduce el ruido sin borrar la cola de los peces pequeños."),
    ), lambda c: f"gris medio {c.g.mean():.0f}"),
    Grupo(pf.SEGMENTACION, "Segmentación", "Segmentación", (
        T("in_seg", "Imagen corregida", v_gauss, "Salida del preprocesamiento.", "entrada"),
        T("umbral", "Umbralización global fija", v_umbral, "Blanco = más oscuro que el umbral."),
        T("apertura", "Apertura morfológica", v_apertura, "Rojo = píxeles eliminados."),
        T("comp", "Componentes conexos y filtro por área", v_componentes,
          "Un color por mancha (4-conectividad). Marco gris = descartada por pequeña."),
        T("estatico", "Supresión de objetos estáticos", v_estaticos,
          "Verde = se conserva. Rojo = lleva oscuro casi todo el último tramo."),
    ), lambda c: f"{c.r.n_manchas} manchas"),
    Grupo(pf.SEPARACION, "Separación", "Separación", (
        T("in_sep", "Manchas segmentadas", v_manchas, "Salida de la segmentación, en intensidad.", "entrada"),
        T("solidez", "Análisis de convexidad (solidez)", v_solidez,
          "Rojo = poco convexa, posible grupo: sólo estas se analizan."),
        T("huecos", "Relleno de huecos acotado por área", v_huecos, "Naranja = brillos internos rellenados."),
        T("distancia", "Transformada de distancia multiumbral", v_distancia,
          "Amarillo = lejos del borde. Dos cuerpos dejan dos zonas amarillas."),
        T("cimas", "Máximos por persistencia topológica", v_cimas,
          "Círculos = cuerpos aceptados. El cuello es paso/cima: bajo = separación clara."),
        T("reparto", "Crecimiento de regiones por niveles", v_reparto, "Cada color es un pez; rojo = grupo separado."),
    ), lambda c: f"{len(c.r.dets)} peces"),
    Grupo(pf.TRACKING, "Tracking", "Tracking", (
        T("in_trk", "Peces del cuadro anterior", v_previos, "Cajas e identidades en el cuadro k-1.", "entrada"),
        T("pred", "Predicción AR(1)", v_prediccion, "Cruz = posición esperada; elipse = incertidumbre (2 sigma)."),
        T("asig", "Asignación óptima (húngaro)", v_asignacion,
          "Verde = emparejado con su puntaje; naranja = pez nuevo; X roja = sin detección."),
        T("tray", "Gestión de trayectorias", v_trayectorias, "Recorrido e identidad de cada pez."),
    ), lambda c: f"{len(c.r.paso.parejas)} enlaces" if c.motor.contar else "bloqueado"),
    Grupo(pf.OCLUSIONES, "Oclusiones y conteo", "Oclusiones y conteo", (
        T("in_ocl", "Trayectorias", v_trayectorias, "Salida del tracking.", "entrada"),
        T("conf", "Confirmación por observaciones (>= 2)", v_confirmacion,
          "Verde = visto 2 o más veces; naranja = visto una vez."),
        T("recu", "Recuperación de trayectorias ocluidas", v_recuperacion,
          "Azul = recuperado por consistencia temporal de áreas; morado = mancha que lo ocultaba."),
        T("conteo", "Conteo de trayectorias", v_conteo, "Verde/azul = contado; naranja = pendiente; rojo = no contado."),
    ), lambda c: f"{c.r.conteo.total} contados" if c.motor.contar else "bloqueado"),
)
POR_ID = {g.id: g for g in GRUPOS}


def dibujar(grupo: str, c: Contexto) -> list[tuple[Tecnica, Imagen]]:
    return [(t, t.funcion(c)) for t in POR_ID[grupo].tecnicas]


def diferencia(entrada: Imagen, salida: Imagen) -> np.ndarray | None:
    """Resalta en rojo lo que se oscureció y en verde lo que se aclaró respecto de la entrada."""
    if entrada.rgb.shape != salida.rgb.shape:
        return None
    a = cv2.cvtColor(entrada.rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    b = cv2.cvtColor(salida.rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    im = (salida.rgb * .5 + 127).astype(np.uint8)
    im[b - a < -40] = ROJO
    im[b - a > 40] = VERDE
    return im
