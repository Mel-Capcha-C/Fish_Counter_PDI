"""T12 - Estimación por área calibrada por lote.

Los peces llegan clasificados por talla, así que dentro de un lote el área de
un pez individual (vista superior, ya normalizada por velocidad) es bastante
estable. Si un blob contiene varios peces agrupados que ni el Watershed ni la
relación de aspecto pudieron resolver, su número se estima como:

        n = round( area_blob / area_individual_del_lote )

Los valores de referencia NO son fijos: se inicializan con el rango de tallas
del lote (config) y luego se reemplazan por la MEDIANA MÓVIL de los blobs que la
validación (T10) clasificó como "un solo pez". La mediana es robusta a los
blobs dobles que se cuelen. Se aprenden tres referencias:
    área individual, ancho individual y relación ancho/largo individual.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..config import BatchCfg


class BatchAreaModel:
    def __init__(self, cfg: BatchCfg, mm_per_px: float):
        lo, hi = cfg.length_range_mm
        length_px = 0.5 * (lo + hi) / mm_per_px
        width_px = cfg.width_ratio * length_px
        self._area0 = cfg.shape_factor * length_px * width_px
        self._width0 = width_px
        self._aspect0 = cfg.width_ratio
        self.areas: deque[float] = deque(maxlen=cfg.history)
        self.widths: deque[float] = deque(maxlen=cfg.history)
        self.aspects: deque[float] = deque(maxlen=cfg.history)
        self.min_samples = 5

    def _median(self, values: deque, default: float) -> float:
        return float(np.median(values)) if len(values) >= self.min_samples else default

    @property
    def single_area(self) -> float:
        return self._median(self.areas, self._area0)

    @property
    def single_width(self) -> float:
        return self._median(self.widths, self._width0)

    @property
    def single_aspect(self) -> float:
        return self._median(self.aspects, self._aspect0)

    def add_single(self, area_px2: float, width_px: float, aspect: float) -> None:
        # Se acepta sólo si es plausible respecto del modelo actual: evita que
        # un arranque con blobs dobles contamine la referencia.
        if 0.33 * self.single_area <= area_px2 <= 3.0 * self.single_area:
            self.areas.append(area_px2)
            self.widths.append(width_px)
            self.aspects.append(aspect)

    def estimate_count(self, area_px2: float) -> int:
        return max(1, int(round(area_px2 / self.single_area)))
