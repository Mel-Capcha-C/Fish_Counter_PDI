# Contador de peces

Cuenta peces que pasan por un canal retroiluminado, con **procesamiento digital de imágenes clásico
(sin inteligencia artificial)**, y muestra en una interfaz la salida de cada técnica.

## Instalar y abrir

Python 3.13 en Windows:

```bash
py -3.13 -m venv fcounter
fcounter\Scripts\python -m pip install -r requirements.txt
fcounter\Scripts\python -m fishcounter
```

Sin interfaz:

```bash
fcounter\Scripts\python -m fishcounter.cli video.mp4 --pez lubina
```

Pruebas (las que usan el dataset se omiten si no está):

```bash
fcounter\Scripts\python -m unittest tests.test_nucleo tests.test_optimizacion tests.test_interfaz
```

## Estructura

| Carpeta | Contenido |
|---|---|
| `fishcounter/core/` | Algoritmo y vistas. **No depende de Qt** (se puede usar en una Raspberry Pi) |
| `fishcounter/gui/` | Interfaz PySide6 |
| `perfiles/` | Parámetros por canal (`canal_*.json`) y por pez (`pez_*.json`) |
| `tests/` | Pruebas |
| `conteo_v3/` | Investigación: algoritmo de referencia, evaluación con etiquetas, flujo de costo mínimo |
| `tools/crear_perfiles.py` | Genera los perfiles base |
| `trout_counter/`, `main.py`, `config.toml` | **Primer prototipo, descartado.** Su separación usa `cv2.watershed` con un defecto conocido |

## Técnicas

| Grupo | Técnicas |
|---|---|
| Preprocesamiento | ROI y luminancia, campo plano (fondo por mediana temporal), suavizado gaussiano |
| Segmentación | Umbralización global, apertura morfológica, componentes conexos, supresión de objetos estáticos |
| Separación | Solidez, relleno de huecos, distancia multiumbral, persistencia topológica, crecimiento de regiones |
| Tracking | Predicción AR(1), asignación húngara por verosimilitud, gestión de trayectorias |
| Oclusiones y conteo | Confirmación (2 o más vistas), recuperación de trayectorias ocluidas, conteo |

## Cómo actualizarlo

- **Nuevo parámetro**: una entrada en `PARAMETROS` (`core/perfil.py`) y su valor en los JSON. El panel de
  Ajustes y la validación lo toman solos.
- **Nueva técnica**: su cálculo en `core/`, su vista en `core/vistas.py` y una línea en `GRUPOS`. Las
  pestañas se crean a partir de ese registro.
- **Nuevo pez**: desde Ajustes, «Guardar como nuevo pez», o un `perfiles/pez_<id>.json`.
- **Cambio de formato de perfil**: subir `VERSION_ESQUEMA` y añadir el paso en `_migrar`.
- **Publicar versión**: pruebas en verde, `__version__`, `CHANGELOG.md` y etiqueta `git tag vX.Y.Z`.
  Si un conteo de `Regresion` (en `tests/test_nucleo.py`) cambia, debe explicarse en el CHANGELOG.

## Medir el tiempo por cuadro

```bash
python3 -m fishcounter.bench video.mp4 --fps 30 --json salida.json
```

Referencia del algoritmo v1 en una Raspberry Pi 5 (OpenCV 4.10, 4 hilos), en `benchmarks/`:

| Video | Media | Mediana | p95 | Máximo | Conteo |
|---|---|---|---|---|---|
| m4 | 13,5 ms | 11,4 | 23,8 | 60 | 198 |
| m8 | 19,4 ms | 12,2 | 48,6 | 118 | 494 |
| m21 | 21,1 ms | 15,1 | 51,4 | 110 | 986 |
| m26 | 21,9 ms | 14,3 | 54,9 | 201 | 985 |

Versión 1.2.0 (optimizada), con los mismos conteos:

| Video | Media | Mediana | p95 | Máximo | Conteo |
|---|---|---|---|---|---|
| m4 | 4,8 ms | 3,4 | 12,0 | 28 | 198 |
| m8 | 9,3 ms | 5,3 | 26,5 | 78 | 494 |
| m21 | 11,2 ms | 7,7 | 32,4 | 82 | 986 |
| m26 | 10,9 ms | 6,7 | 31,6 | 108 | 985 |

## Volver al algoritmo original

La etiqueta `algoritmo-v1` no se mueve. Para probarlo sin perder el trabajo actual:

```bash
git worktree add ../algoritmo-v1 algoritmo-v1
```

Para continuar a partir de él: `git switch -c vuelta-a-v1 algoritmo-v1`. Una optimización sólo se acepta si
los conteos de `Regresion` (`tests/test_nucleo.py`) no cambian, o si el cambio se justifica en el CHANGELOG.

## Resultados y límites

Sobre el dataset FCDataZJU (26 videos de lubina y 26 de carpín), error medio del conteo de 0,7 a 1,3 %
según el grupo de videos. Hay que leerlo con cautela:

- **Los 52 videos ya se usaron para desarrollar**: no es una validación independiente.
- El total de cada video se infirió del nombre de su carpeta.
- Los perfiles dependen de esa cámara, canal y velocidad. **El perfil de trucha es provisional.**
- Parte del error se compensa (peces de más y de menos).

El dataset no se incluye: <https://figshare.com/articles/dataset/FCDataZJU/21601548>
