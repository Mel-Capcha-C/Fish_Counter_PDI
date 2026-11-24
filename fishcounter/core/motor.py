"""Motor: procesa un video cuadro a cuadro y guarda un resultado LIGERO de cada uno.

No guarda imágenes; las vistas las recalculan (ver vistas.py). Es el único lugar que conoce el
orden de las técnicas, así que la interfaz y la línea de comandos cuentan exactamente igual.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import oclusiones, segmentacion as sg, separacion as sp
from .perfil import Perfil
from .tracking import Paso, Rastreador

# Pasados estos cuadros, una trayectoria de una vista ya no puede cambiar: las pistas que la
# explican han terminado (un pez es visible pocos cuadros). Se comprueba en las pruebas.
HORIZONTE = 24


@dataclass
class Conteo:
    confirmados: int = 0         # trayectorias con 2 o más vistas
    recuperados: int = 0         # trayectorias ocluidas recuperadas
    pendientes: int = 0          # trayectorias de una vista aún sin decidir

    @property
    def total(self) -> int:
        return self.confirmados + self.recuperados


@dataclass
class ResultadoCuadro:
    k: int
    dets: np.ndarray                                  # (n, 6): caja, área, razón
    estaticas: list = field(default_factory=list)     # [(x, y, w, h)] en el ROI
    n_manchas: int = 0
    n_sospechosas: int = 0
    n_grupos: int = 0                                 # manchas separadas en 2 o más peces
    paso: Paso = field(default_factory=Paso)
    conteo: Conteo = field(default_factory=Conteo)    # tal como se veía en vivo
    recuperadas: list = field(default_factory=list)   # ids recuperados al procesar este cuadro


class Motor:
    """Uso: m = Motor(perfil, fondo); m.procesar(k, bgr) en orden; m.finalizar()."""

    def __init__(self, perfil: Perfil, fondo: np.ndarray, contar: bool = True):
        self.p, self.fondo, self.contar = perfil, fondo, contar
        self.gan = sg.ganancia(fondo, perfil)
        self.estaticos = sg.Estaticos(perfil)
        self.rastreador = Rastreador(perfil)
        self.resultados: list[ResultadoCuadro] = []
        self.dets: dict[int, np.ndarray] = {}
        self.ids: dict[int, list] = {}
        self.obs: dict[int, list] = {}                # id -> [(cuadro, detección)]
        self.motivos: dict[int, oclusiones.Motivo] = {}
        self._abiertas: set[int] = set()              # trayectorias de una vista sin decidir
        self.terminado = False
        self.ultimo: tuple[int, dict] | None = None

    @property
    def siguiente(self) -> int:
        return len(self.resultados)

    @property
    def conteo(self) -> Conteo:
        return Conteo(self.rastreador.confirmadas, len(self.motivos), self._pendientes())

    def _pendientes(self) -> int:
        if self.terminado:
            return 0
        k = self.siguiente - 1
        return sum(1 for i in self._abiertas
                   if i not in self.motivos and self.obs[i][0][0] >= k - 3)

    def _revisar_oclusiones(self, k: int) -> list[int]:
        """Reevalúa las trayectorias de una vista recientes; fija las antiguas."""
        nuevas = []
        for i in list(self._abiertas):
            if len(self.obs[i]) > 1:                   # ya se confirmó por tracking
                self._abiertas.discard(i)
                self.motivos.pop(i, None)
                continue
            m = oclusiones.buscar_oclusion(i, self.obs, self.dets, self.ids, self.p)
            if m is None:
                self.motivos.pop(i, None)
            else:
                if i not in self.motivos:
                    nuevas.append(i)
                self.motivos[i] = m
            if self.obs[i][0][0] < k - HORIZONTE:
                self._abiertas.discard(i)
        return nuevas

    def procesar(self, k: int, bgr: np.ndarray) -> ResultadoCuadro:
        if k != self.siguiente:
            raise ValueError(f"Los cuadros se procesan en orden: se esperaba {self.siguiente} y llegó {k}")
        p = self.p
        roi = sg.recortar(bgr, p)
        plano = sg.campo_plano(roi, self.gan)
        g = sg.suavizar(plano, p)
        binaria = sg.umbralizar(g, p)
        abierta = sg.abrir(binaria, p)
        comps = sg.componentes(abierta, g, p)
        mov, est = self.estaticos.filtrar(g, comps[0])
        dets, an = sp.detecciones(mov, p)
        # Intermedios SÓLO del último cuadro: evitan recalcular al dibujarlo
        self.ultimo = (k, dict(roi=roi, plano=plano, g=g, binaria=binaria, abierta=abierta, comps=comps,
                               moviles=mov, analisis=an))
        r = ResultadoCuadro(k, dets, [(m.x, m.y, *m.mask.shape[::-1]) for m in est], len(mov),
                            sum(a.sospechosa for a in an), sum(len(a.piezas) > 1 for a in an))
        self.dets[k] = dets
        if self.contar:
            r.paso = self.rastreador.paso(k, dets[:, :4])
            self.ids[k] = r.paso.ids
            for j, i in enumerate(r.paso.ids):
                self.obs.setdefault(i, []).append((k, j))
                if len(self.obs[i]) == 1:
                    self._abiertas.add(i)
            r.recuperadas = self._revisar_oclusiones(k)
        self.resultados.append(r)
        r.conteo = self.conteo
        return r

    def finalizar(self) -> Conteo:
        """Decisión definitiva con TODO el video. Coincide con conteo_v3/contar_video.py."""
        if self.contar:
            self.motivos = {}
            for i, o in self.obs.items():
                if len(o) == 1:
                    m = oclusiones.buscar_oclusion(i, self.obs, self.dets, self.ids, self.p)
                    if m:
                        self.motivos[i] = m
        self._abiertas = set()
        self.terminado = True
        return self.conteo

    # ------------------------------------------------------------- búsquedas
    def estado(self, i: int) -> str:
        """'confirmada' | 'recuperada' | 'pendiente' | 'descartada'."""
        if len(self.obs[i]) > 1:
            return "confirmada"
        if i in self.motivos:
            return "recuperada"
        reciente = self.obs[i][0][0] >= self.siguiente - 4
        return "pendiente" if reciente and not self.terminado else "descartada"


CASOS = {"grupo": "Grupo separado", "recuperado": "Pez recuperado", "estatico": "Objeto estático"}


def es_caso(r: ResultadoCuadro, caso: str) -> bool:
    return {"grupo": r.n_grupos > 0, "recuperado": bool(r.recuperadas), "estatico": bool(r.estaticas)}[caso]
