"""Contador de truchas por procesamiento digital de imágenes clásico (sin IA).

Paquetes:
    acquisition    T1 ROI de captura, T2 controles fijos, T3 canal Y
    preprocessing  T4 flat-field, T5 suavizado
    segmentation   T6 umbral Otsu fijo, T7 morfología
    counting       T8 imagen espacio-temporal, T9 componentes conexos,
                   T10 ancho transversal, T11 watershed, T12 área por lote,
                   + doble línea (velocidad y sentido)
"""
__version__ = "0.2.0"
