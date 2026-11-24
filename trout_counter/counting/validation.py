"""T10 - Ancho transversal del blob (validación), normalizado por el largo.

En la imagen espacio-temporal cada FILA es la línea de barrido en un instante;
el número de píxeles de pez en esa fila es el ancho del cuerpo cortado por la
línea. Se usa el percentil 90 de esos anchos (no la caja envolvente) porque:
  * la caja de un pez que cruza en diagonal es ancha aunque el cuerpo no lo sea
    (la línea corta al cuerpo inclinado en W / cos(theta) ~ W);
  * el P90 ignora una fila aislada con una aleta abierta.

MEJORA respecto de la tabla original: comparar el ancho ABSOLUTO con el de un
pez típico falla cuando el lote es heterogéneo (4-7 cm => el ancho varía 1.75x
y el área 3x). Se usa en cambio la RELACIÓN DE ASPECTO

        r = (ancho / largo) / (ancho / largo de un pez individual)

que no depende de la talla:
        r ~ 1    -> un pez
        r ~ 2    -> dos peces LADO A LADO (mismo largo, doble ancho)
        r ~ 0.5  -> dos peces EN FILA, cabeza con cola (doble largo, mismo ancho)

El largo se obtiene gracias a la velocidad de la doble línea (velocity.py).

Clases resultantes:
  "noise"  demasiado pequeño para ser un pez
  "wide"   r alto: peces lado a lado
  "long"   r bajo: peces en fila
  "large"  r normal pero área grande: grupo compacto o pez grande
  "single" un pez (alimenta el modelo del lote, T12)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ValidationCfg
from .components import Blob


@dataclass
class BlobMeasures:
    area_px2: float     # área normalizada por velocidad (px de canal^2)
    length_px: float    # largo a lo largo del flujo
    width_px: float     # ancho transversal robusto (P90 por fila)
    aspect: float       # width_px / length_px

    def aspect_ratio_to(self, single_aspect: float) -> float:
        return self.aspect / max(single_aspect, 1e-6)


def measure_blob(blob: Blob, v: float) -> BlobMeasures:
    row_widths = np.count_nonzero(blob.mask, axis=1)
    length = blob.n_frames * v
    width = float(np.percentile(row_widths, 90))
    return BlobMeasures(blob.area_st * v, length, width, width / max(length, 1e-6))


def classify_blob(m: BlobMeasures, single_area: float, single_aspect: float,
                  min_area_frac: float, cfg: ValidationCfg) -> str:
    if m.area_px2 < min_area_frac * single_area:
        return "noise"
    r = m.aspect_ratio_to(single_aspect)
    if r > cfg.aspect_wide_factor:
        return "wide"
    if r < cfg.aspect_long_factor:
        return "long"
    if m.area_px2 > cfg.area_factor * single_area:
        return "large"
    return "single"
