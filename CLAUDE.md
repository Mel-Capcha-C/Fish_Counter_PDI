# Contexto del proyecto para Claude

Contador de peces (objetivo final: **truchas**) para un canal retroiluminado, con procesamiento digital
de imágenes clásico. Instalación, estructura, técnicas y benchmarks: ver `README.md` y `CHANGELOG.md`.
Este archivo guarda lo que NO está en el código: reglas, decisiones, estado y próximos pasos.

## Cómo trabajar con el usuario

- Estudiante de procesamiento digital de imágenes. Escribe en español o inglés: responder en el idioma
  de su último mensaje. El código, los comentarios y la documentación del repositorio van en español.
- Prefiere explicaciones simples, para no especialistas, con tablas cuando ayudan.
- Cuando pide explicar un resultado, explicarlo con lo que ya se sabe; no lanzar análisis o simulaciones
  nuevas sin que lo pida.
- Commits sólo cuando lo pide. Nunca `push --force`. El push lo hace el usuario en su propia terminal
  (el acceso SSH a GitHub puede no estar disponible para Claude).
- No escribir contraseñas por el usuario: para la Raspberry Pi se usa una llave SSH que él instala.

## Reglas del proyecto

1. **Sin inteligencia artificial**: nada de CNN, YOLO ni aprendizaje profundo. Sólo técnicas clásicas
   (umbralización, morfología, distancia, tracking, etc.). Se evaluó una CNN + tracking y se descartó
   ("Solución A" = sin CNN).
2. **Una optimización debe dar resultados idénticos**: mismas detecciones, identidades y conteos que
   `algoritmo-v1`, comprobado en los 52 videos. Si un conteo de `Regresion` (`tests/test_nucleo.py`)
   cambia, es un cambio de algoritmo y debe justificarse en `CHANGELOG.md` (versión MAYOR).
3. **El algoritmo está congelado** hasta tener un dataset de truchas: sin recalibrar ni cambiar lógica
   sólo para mejorar FCDataZJU (sus 52 videos ya se usaron para desarrollar; sería sobreajuste).
4. El núcleo (`fishcounter/core/`) no depende de Qt, para que corra en la Raspberry Pi.
5. No subir a GitHub el dataset, PDFs, Excel, CSV ni imágenes (licencias; ver `.gitignore`).

## Historia (por qué el código es así)

- `trout_counter/` + `main.py`: primer prototipo (flujo de área, pensado para Raspberry Pi Zero 2 W).
  Falló en FCDataZJU y se descartó. Su separación con `cv2.watershed` tiene un defecto conocido (asigna
  sólo ~22 % del área); se documentó y NO se corrigió.
- `conteo_v3/`: rediseño desde cero sobre el dataset. Investigación y evaluación con etiquetas, incluido un
  flujo de costo mínimo para el tracking que no mejoró y se dejó como experimento.
- La recuperación de trayectorias ocluidas ("T15", `core/oclusiones.py`) se mantuvo, ya corregida
  (vistas cortadas por el borde, predicción AR(1) a 2 pasos).
- `fishcounter/` 1.0.0: núcleo + interfaz PySide6, idéntico a `conteo_v3`. 1.1.0: benchmark.
  1.2.0: optimización (la mitad de tiempo, resultados idénticos).

## Versiones y ramas

| Referencia | Qué es |
|---|---|
| `algoritmo-v1` (= `v1.1.0`) | Algoritmo original. **Punto de retorno; no se mueve** |
| `v1.2.0` = `main` | Optimizado: cimas por niveles relevantes, reparto por píxeles pendientes, estáticos bajo demanda, campo plano por bloques |
| rama `optimizacion` | Ya fusionada en `main` |

Publicar una versión: pruebas en verde, `__version__` en `fishcounter/__init__.py`, `CHANGELOG.md`,
`git tag -a vX.Y.Z`.

## Entorno

- PC (Windows): Python 3.13, entorno `fcounter/` (no está en git; recrearlo con `requirements.txt`).
- Dataset FCDataZJU (no está en git) en `21601548/FCDataZJU/` con `MicropterusSalmoides/` (lubina, 26
  videos) y `CarassiusAuratus/` (carpín, 26 videos). Sin él, las pruebas de regresión se omiten.
- Pruebas: `fcounter\Scripts\python -m unittest tests.test_nucleo tests.test_optimizacion tests.test_interfaz`
  (33 pruebas; `test_pipeline.py` es del prototipo viejo). En consola sin pantalla usar
  `QT_QPA_PLATFORM=offscreen`. El aviso "moov atom not found" de una prueba es esperado.
- Interfaz: `fcounter\Scripts\python -m fishcounter [video]`.
- Raspberry Pi 5 (8 GB, ventilador): `fish@192.168.18.20`, Python 3.13.5, OpenCV 4.10 del sistema.
  `~/Fish_Counter_PDI` tiene el código original y los videos m4, m8, m21, m26; `~/Fish_Counter_opt` el
  optimizado (con enlace a esos videos). La llave SSH es propia de cada PC: desde otra PC hay que crear
  e instalar una nueva. El código se copia a la Pi con scp (no hay git allí).
- Benchmark: `python3 -m fishcounter.bench video.mp4 --fps 30 --hilos 4 --json salida.json`.

## Resultados que importan

- Error medio de conteo en FCDataZJU: 0,7–1,3 % según el grupo (no es validación independiente).
- m4 cuenta 198 de 200: peces que nunca se separan de un grupo durante todo su paso.
- m8 cuenta 495 en la PC y 494 en la Pi (decodificación y versión de OpenCV distintas; igual antes y
  después de optimizar).
- Pi 5, 1.2.0: 9,4 ms por cuadro de media (antes 19,4), p95 28,5, máx. 108. Cuadros con 10 o más peces:
  28–35 ms. Con una cola pequeña aguanta 30–40 fps; 60 fps necesita ~60 cuadros de cola en videos
  densos; 90 fps no.
- Descartado al optimizar: analizar manchas en paralelo (~3 %), campo plano con 4 hilos (la memoria de la
  Pi es el límite).

## Límites conocidos

- Sólo 1280x720 (el conteo se bloquea con otra resolución).
- El fondo se estima con el video entero: no sirve tal cual en vivo.
- No se ha medido la captura de cámara; los fps reales de FCDataZJU no se confirmaron (30 en el archivo,
  posiblemente 10).
- El perfil de trucha (`perfiles/pez_trucha.json`) no está calibrado; el de carpín está calibrado sólo en parte.
- La memoria crece lentamente en ejecuciones largas (trayectorias antiguas no se liberan).

## Próximos pasos (en este orden)

1. **Prueba con cámara en vivo en la Pi 5**: medir captura + procesamiento y su retraso; fondo en vivo
   (canal vacío o primeros segundos, actualizado en el tiempo); liberar trayectorias viejas. Falta saber
   el modelo de cámara y cómo se conecta (USB o CSI).
2. **Dataset de truchas y calibración**: grabar con la cámara, iluminación y fps definitivos; varias
   tallas y densidades; conteo manual de cada video; separar videos sólo para prueba.
3. **Interfaz v2**: lista de eventos, gráfico, diagnóstico, exportar y asistente de recalibración (el
   asistente ayuda en el paso 2).
