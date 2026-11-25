# Cambios

Versionado semántico (`MAYOR.MENOR.PARCHE`). La versión vive en `fishcounter/__init__.py`.
- **MAYOR**: cambia el formato de los perfiles o el conteo de un mismo video.
- **MENOR**: funciones nuevas que no cambian el conteo.
- **PARCHE**: correcciones.

## 1.2.0 - 2026-11-24
### Rendimiento
- En una Raspberry Pi 5, de 19,4 a 9,4 ms por cuadro de media; el más lento, de 202 a 108 ms.
- Máximos por persistencia: sólo se evalúan los niveles que pueden cambiar el resultado.
- Crecimiento de regiones: por lista de píxeles pendientes.
- Objetos estáticos: la reducción exacta se calcula sólo si una cota barata dice que puede haber alguno.
- Campo plano por bloques en dos hilos; apertura y componentes sólo donde hay píxeles oscuros.
### Comprobado
- Detecciones, identidades y conteo idénticos a `algoritmo-v1` en los 52 videos (PC) y en 4 videos en la
  Raspberry Pi. `tests/test_optimizacion.py` compara cada rutina con su definición simple.

## 1.1.0 - 2026-11-24
### Añadido
- `fishcounter/bench.py`: tiempo por cuadro sin interfaz (PC o Raspberry Pi), con desglose por técnica.
- `benchmarks/algoritmo-v1_pi5_4hilos.json`: medición de referencia en una Raspberry Pi 5.
### Sin cambios
- El algoritmo es idéntico al de 1.0.0. Queda congelado con la etiqueta `algoritmo-v1`, como punto de
  retorno antes de optimizar.

## 1.0.0 - 2026-11-23
### Añadido
- Interfaz PySide6: abrir video, cinco grupos de técnicas con la entrada y la salida de cada una,
  vista ampliada (lado a lado, deslizador, diferencias), zoom compartido, reproducción cuadro a
  cuadro, velocidades e «ir al siguiente caso».
- Conteo confirmado, recuperado y pendiente. El conteo se bloquea si el video no tiene la resolución
  calibrada.
- Perfiles por pez (lubina, carpín, trucha) y panel de ajustes por técnica, generados a partir de una
  única lista de parámetros.
- Núcleo sin dependencias de Qt y conteo por línea de comandos.
### Comprobado
- Detecciones, identidades y conteo idénticos a `conteo_v3` (tracking + recuperación) en los 52 videos
  de FCDataZJU.
