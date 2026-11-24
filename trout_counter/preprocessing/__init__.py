"""Etapa de PREPROCESAMIENTO: técnicas 4 (flat-field) y 5 (suavizado).

La técnica 3 (canal Y) se resuelve en la adquisición: la franja ya llega en Y.
"""
from .denoise import denoise
from .flat_field import FlatFieldCorrector

__all__ = ["denoise", "FlatFieldCorrector"]
