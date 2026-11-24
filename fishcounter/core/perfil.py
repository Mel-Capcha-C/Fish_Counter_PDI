"""Parámetros del algoritmo y perfiles por tipo de pez.

PARAMETROS es la ÚNICA lista de parámetros: de ella salen el panel de Ajustes, la validación y
los archivos JSON. Para añadir uno basta con agregar una entrada aquí y su valor en el perfil.

Un perfil combina dos archivos de ``perfiles/``:
    canal_<id>.json   lo que depende de la cámara, la luz y el agua
    pez_<id>.json     lo que depende del pez (o del pez en ese canal)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

VERSION_ESQUEMA = 1
CARPETA = Path(__file__).resolve().parents[2] / "perfiles"

PREPROCESAMIENTO, SEGMENTACION, SEPARACION, TRACKING, OCLUSIONES = (
    "preprocesamiento", "segmentacion", "separacion", "tracking", "oclusiones")
NOMBRES_GRUPO = {
    PREPROCESAMIENTO: "Preprocesamiento",
    SEGMENTACION: "Segmentación",
    SEPARACION: "Separación",
    TRACKING: "Tracking",
    OCLUSIONES: "Oclusiones y conteo",
}
CANAL, PEZ = "canal", "pez"


@dataclass(frozen=True)
class Parametro:
    clave: str
    grupo: str
    nombre: str
    tipo: type
    minimo: float
    maximo: float
    paso: float
    ambito: str                  # CANAL o PEZ: en qué archivo se guarda
    ayuda: str
    unidad: str = ""
    avanzado: bool = False


def _p(*a, **k):
    return Parametro(*a, **k)


PARAMETROS: tuple[Parametro, ...] = (
    # ---------------------------------------------------------- preprocesamiento
    _p("ancho", PREPROCESAMIENTO, "Ancho calibrado", int, 160, 8192, 2, CANAL,
       "Ancho del video con el que se calibró. Si el video no coincide, el conteo se bloquea.", "px"),
    _p("alto", PREPROCESAMIENTO, "Alto calibrado", int, 120, 8192, 2, CANAL,
       "Alto del video con el que se calibró.", "px"),
    _p("roi_x0", PREPROCESAMIENTO, "ROI: margen izquierdo", int, 0, 2000, 1, CANAL,
       "Píxeles que se descartan a la izquierda (pared del canal).", "px"),
    _p("roi_y0", PREPROCESAMIENTO, "ROI: margen superior", int, 0, 2000, 1, CANAL,
       "Píxeles que se descartan arriba (riel del canal).", "px"),
    _p("fondo_cuadros", PREPROCESAMIENTO, "Cuadros para el fondo", int, 5, 300, 1, CANAL,
       "Cuadros repartidos por el video cuya mediana temporal estima el fondo vacío."),
    _p("nivel", PREPROCESAMIENTO, "Nivel del fondo corregido", float, 100, 250, 1, CANAL,
       "Gris al que se lleva el fondo tras el campo plano (flat-field).", "gris"),
    _p("gauss", PREPROCESAMIENTO, "Núcleo gaussiano", int, 1, 9, 2, CANAL,
       "Tamaño (impar) del suavizado. 1 = sin suavizado.", "px"),
    # -------------------------------------------------------------- segmentación
    _p("umbral", SEGMENTACION, "Umbral global", int, 1, 254, 1, PEZ,
       "Un píxel más oscuro que este gris es pez. Es también el primer umbral de la distancia.", "gris"),
    _p("apertura", SEGMENTACION, "Apertura morfológica", int, 1, 9, 2, PEZ,
       "Elemento estructurante; elimina puntos sueltos. 1 = sin apertura.", "px"),
    _p("area_min", SEGMENTACION, "Área mínima", int, 1, 20000, 10, PEZ,
       "Las manchas más pequeñas se descartan.", "px"),
    _p("quietos_ventana", SEGMENTACION, "Estáticos: ventana", int, 2, 120, 1, PEZ,
       "Cuadros anteriores que se miran para reconocer un objeto estático.", "cuadros"),
    _p("quietos_min", SEGMENTACION, "Estáticos: cuadros oscuros", int, 1, 120, 1, PEZ,
       "Veces que un píxel debe estar oscuro en la ventana para ser estático. Supone que los peces "
       "no se detienen.", "cuadros"),
    _p("quietos_fraccion", SEGMENTACION, "Estáticos: fracción", float, .05, 1, .05, PEZ,
       "Parte de la mancha que debe ser estática para descartarla."),
    # ---------------------------------------------------------------- separación
    _p("solidez_libre", SEPARACION, "Solidez para no analizar", float, .5, 1, .01, PEZ,
       "Por encima de esta solidez la mancha se considera un solo pez y no se analiza."),
    _p("hueco_max", SEPARACION, "Hueco máximo", int, 0, 2000, 5, PEZ,
       "Se rellenan los huecos claros y cerrados (brillos) de hasta este tamaño. 0 = no rellenar.", "px"),
    _p("umbral2", SEPARACION, "2.º umbral de distancia", int, 1, 254, 1, PEZ,
       "La distancia se promedia sobre tres umbrales; este debe ser menor que el global.", "gris"),
    _p("umbral3", SEPARACION, "3.er umbral de distancia", int, 1, 254, 1, PEZ,
       "Debe ser menor que el segundo.", "gris"),
    _p("razon", SEPARACION, "Razón del cuello", float, .3, .99, .01, PEZ,
       "Dos cimas son dos peces si el paso entre ellas es menor que esta fracción de la menor."),
    _p("cima_min", SEPARACION, "Cima mínima", float, .5, 100, .5, PEZ,
       "Semiancho mínimo para que una cima cuente como un cuerpo.", "px"),
    _p("area_cima", SEPARACION, "Área mínima de la cima", int, 1, 5000, 5, PEZ,
       "Área de la cima al fundirse para que cuente como un cuerpo.", "px"),
    # ------------------------------------------------------------------ tracking
    _p("c0", TRACKING, "Salto: término constante", float, -5000, 5000, 1, PEZ,
       "Salto esperado por cuadro, en coordenadas normalizadas a 1280x720. Se aprende con "
       "peces aislados (Calibrar movimiento).", "px", True),
    _p("c1", TRACKING, "Salto: lineal en x", float, -5000, 5000, 1, PEZ, "Variación del salto con x.", "", True),
    _p("c2", TRACKING, "Salto: cuadrático en x", float, -5000, 5000, 1, PEZ, "Variación del salto con x².", "", True),
    _p("c3", TRACKING, "Salto: lineal en y", float, -5000, 5000, 1, PEZ, "Variación del salto con y.", "", True),
    _p("c4", TRACKING, "Salto: cuadrático en y", float, -5000, 5000, 1, PEZ, "Variación del salto con y².", "", True),
    _p("sx", TRACKING, "Variación del primer salto", float, 1, 1000, 1, PEZ,
       "Desviación horizontal antes de conocer la velocidad del pez.", "px"),
    _p("sx2", TRACKING, "Variación de los siguientes", float, 1, 1000, 1, PEZ,
       "Desviación horizontal cuando ya se conoce un salto.", "px"),
    _p("rho", TRACKING, "Memoria de velocidad (x)", float, 0, 1, .01, PEZ,
       "Coeficiente AR(1): cuánto conserva el pez su desvío de velocidad."),
    _p("my", TRACKING, "Desvío vertical medio", float, -500, 500, 1, PEZ, "Deriva vertical por cuadro.", "px"),
    _p("sy", TRACKING, "Variación vertical", float, 1, 500, 1, PEZ, "Desviación vertical del primer salto.", "px"),
    _p("sy2", TRACKING, "Variación vertical siguiente", float, 1, 500, 1, PEZ,
       "Desviación vertical con un salto conocido.", "px"),
    _p("rhoy", TRACKING, "Memoria de velocidad (y)", float, 0, 1, .01, PEZ, "Coeficiente AR(1) vertical."),
    _p("slw", TRACKING, "Variación del ancho", float, .01, 2, .01, PEZ,
       "Cambio del logaritmo del ancho entre cuadros."),
    _p("slh", TRACKING, "Variación del alto", float, .01, 2, .01, PEZ,
       "Cambio del logaritmo del alto entre cuadros."),
    _p("pob_w", TRACKING, "Población: log-ancho medio", float, 0, 10, .01, PEZ, "Talla de los peces.", "", True),
    _p("pob_sw", TRACKING, "Población: variación del ancho", float, .01, 3, .01, PEZ, "Talla de los peces.", "", True),
    _p("pob_h", TRACKING, "Población: log-alto medio", float, 0, 10, .01, PEZ, "Talla de los peces.", "", True),
    _p("pob_sh", TRACKING, "Población: variación del alto", float, .01, 3, .01, PEZ, "Talla de los peces.", "", True),
    _p("p_perdido", TRACKING, "Probabilidad de no detección", float, .0005, .5, .005, PEZ,
       "Probabilidad de que un pez visible no se detecte en un cuadro."),
    _p("p_medio", TRACKING, "Apariciones fuera de la entrada", float, .0001, .5, .001, PEZ,
       "Densidad relativa de peces que aparecen lejos de la entrada.", "", True),
    _p("max_perdidos", TRACKING, "Cuadros perdidos permitidos", int, 0, 5, 1, CANAL,
       "Cuadros seguidos sin detectar antes de cerrar la trayectoria.", "cuadros"),
    _p("vistas_por_pez", TRACKING, "Vistas por pez", float, 1, 100, .1, CANAL,
       "Cuadros que un pez permanece visible, en promedio. Depende de la velocidad del agua.", "", True),
    _p("margen_entrada", TRACKING, "Margen de entrada", float, 0, 500, 5, CANAL,
       "Distancia antes del borde desde la que entra un pez.", "px", True),
    _p("margen_salida", TRACKING, "Margen de salida", float, 0, 500, 5, CANAL,
       "Distancia tras el borde hasta la que un pez sigue a la vista.", "px", True),
    _p("borde", TRACKING, "Borde de corte", float, 0, 100, 1, CANAL,
       "Una caja a menos de esta distancia del borde se considera cortada.", "px", True),
    # ---------------------------------------------------------------- oclusiones
    _p("vistas_min", OCLUSIONES, "Observaciones mínimas", int, 2, 2, 1, CANAL,
       "Cuadros en que debe verse un pez para confirmarlo. Fijo en 2 en esta versión.", "cuadros"),
    _p("razon_clara", OCLUSIONES, "Razón de observación clara", float, 0, 1, .05, PEZ,
       "Una trayectoria de una vista solo se recupera si su separación fue al menos así de clara."),
    _p("exceso", OCLUSIONES, "Exceso de área", float, 1, 5, .05, PEZ,
       "Cuánto mayor debe ser la mancha que oculta al pez, respecto a las otras vistas de quien la ocupa."),
    _p("margen_oclusion", OCLUSIONES, "Margen vertical", float, 0, 200, 1, PEZ,
       "Tolerancia vertical al buscar la mancha que oculta al pez.", "px"),
    _p("retro_salto", OCLUSIONES, "Salto hacia atrás", float, 1, 5000, 10, PEZ,
       "Retroceso aproximado para estimar la velocidad en el cuadro anterior.", "px", True),
)
POR_CLAVE = {p.clave: p for p in PARAMETROS}

ESTADOS = {
    "calibrado": "Calibrado",
    "parcial": "Calibración parcial",
    "sin_calibrar": "Sin calibrar (provisional)",
    "editado": "Editado, sin validar",
}


class PerfilInvalido(ValueError):
    pass


@dataclass(frozen=True)
class Perfil:
    """Valores del algoritmo. Se accede como atributos: ``p.umbral``."""
    id: str
    nombre: str
    estado: str
    valores: dict = field(repr=False)
    nota: str = ""

    def __getattr__(self, clave):
        if clave.startswith("__") or clave == "valores":
            raise AttributeError(clave)
        try:
            return self.valores[clave]
        except KeyError:
            raise AttributeError(clave) from None

    # ------------------------------------------------------------- geometría
    @property
    def fx(self) -> float:
        return 1280 / (self.ancho - self.roi_x0)

    @property
    def fy(self) -> float:
        return 720 / (self.alto - self.roi_y0)

    @property
    def umbrales(self) -> tuple[int, int, int]:
        return self.umbral, self.umbral2, self.umbral3

    def salto(self, x, y):
        u, w = min(max(x, 0), 700) / 1000.0 - .3, y / 720.0 - .5
        return self.c0 + self.c1 * u + self.c2 * u * u + self.c3 * w + self.c4 * w * w

    # ------------------------------------------------------------- edición
    def con(self, **cambios) -> "Perfil":
        """Copia con valores cambiados; queda marcada como editada y sin validar."""
        v = {**self.valores, **cambios}
        validar(v)
        distinto = any(v[k] != self.valores[k] for k in v)
        return replace(self, valores=v, estado="editado" if distinto else self.estado)


def validar(v: dict) -> None:
    """Lanza PerfilInvalido con un mensaje entendible si algo no es coherente."""
    faltan = [p.clave for p in PARAMETROS if p.clave not in v]
    if faltan:
        raise PerfilInvalido("Faltan parámetros: " + ", ".join(faltan))
    for p in PARAMETROS:
        x = v[p.clave]
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not p.minimo <= x <= p.maximo:
            raise PerfilInvalido(f"{p.nombre}: {x} está fuera de [{p.minimo}, {p.maximo}]")
        if p.tipo is int and int(x) != x:
            raise PerfilInvalido(f"{p.nombre}: debe ser entero")
    for k in ("gauss", "apertura"):
        if v[k] % 2 == 0:
            raise PerfilInvalido(f"{POR_CLAVE[k].nombre}: debe ser impar")
    if not v["umbral3"] < v["umbral2"] < v["umbral"]:
        raise PerfilInvalido("Los umbrales de distancia deben decrecer: global > 2.º > 3.º")
    if v["quietos_min"] > v["quietos_ventana"]:
        raise PerfilInvalido("Estáticos: los cuadros oscuros no pueden superar la ventana")
    if v["roi_x0"] >= v["ancho"] - 16 or v["roi_y0"] >= v["alto"] - 16:
        raise PerfilInvalido("La ROI no deja imagen que analizar")


# ------------------------------------------------------------------ archivos
def _migrar(d: dict) -> dict:
    """Actualiza un JSON de una versión anterior del esquema. Añadir aquí cada migración."""
    ver = d.get("version_esquema", 0)
    if ver > VERSION_ESQUEMA:
        raise PerfilInvalido(f"El perfil usa el esquema {ver}; esta versión sólo entiende hasta {VERSION_ESQUEMA}")
    return d


def _leer(ruta: Path) -> dict:
    with open(ruta, encoding="utf8") as fh:
        return _migrar(json.load(fh))


def disponibles(carpeta: Path = CARPETA) -> list[tuple[str, str]]:
    """[(id, nombre)] de los perfiles de pez, en el orden declarado."""
    L = [_leer(r) for r in sorted(carpeta.glob("pez_*.json"))]
    L.sort(key=lambda d: (d.get("orden", 99), d["nombre"]))
    return [(d["id"], d["nombre"]) for d in L]


def cargar(id_: str, carpeta: Path = CARPETA) -> Perfil:
    ruta = carpeta / f"pez_{id_}.json"
    if not ruta.exists():
        raise PerfilInvalido(f"No existe el perfil «{id_}»")
    pez = _leer(ruta)
    canal = _leer(carpeta / f"canal_{pez['canal']}.json")
    v = {**canal["valores"], **pez["valores"]}
    v = {p.clave: p.tipo(v[p.clave]) for p in PARAMETROS if p.clave in v}
    validar(v)
    return Perfil(pez["id"], pez["nombre"], pez["estado"], v, pez.get("nota", ""))


def guardar_como(p: Perfil, id_: str, nombre: str, canal: str = "fcdatazju", carpeta: Path = CARPETA) -> Path:
    """Guarda los parámetros de pez con otro id. Nunca sobrescribe un perfil existente."""
    if not id_.replace("_", "").isalnum() or not id_.isascii():
        raise PerfilInvalido("El id sólo puede tener letras sin acento, números y _")
    ruta = carpeta / f"pez_{id_}.json"
    if ruta.exists():
        raise PerfilInvalido(f"Ya existe un perfil «{id_}»")
    base = _leer(carpeta / f"canal_{canal}.json")["valores"]
    # Los valores del canal que se hayan editado se guardan con el pez para no tocar el canal común
    v = {q.clave: p.valores[q.clave] for q in PARAMETROS
         if q.ambito == PEZ or p.valores[q.clave] != base.get(q.clave)}
    d = dict(version_esquema=VERSION_ESQUEMA, tipo=PEZ, id=id_, nombre=nombre, canal=canal,
             estado="editado", nota="Creado desde la interfaz; sin validar.", orden=90, valores=v)
    with open(ruta, "w", encoding="utf8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
    return ruta
