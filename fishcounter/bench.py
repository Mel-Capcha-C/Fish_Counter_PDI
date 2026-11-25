"""Tiempo por cuadro del algoritmo, sin interfaz (PC o Raspberry Pi).

    python3 -m fishcounter.bench video.mp4 [otro.mp4 ...] --pez lubina --fps 30 --json salida.json

Mide lo MISMO que ejecuta el conteo (Motor.procesar), cuadro a cuadro y en un solo hilo de Python:
  * total por cuadro: media, mediana, p95, p99 y máximo;
  * desglose por técnica (una segunda pasada, para que cronometrar no altere el total);
  * retraso acumulado si los cuadros llegan a --fps y hay un solo proceso atendiéndolos.
La lectura del archivo y la estimación del fondo se informan aparte: con una cámara en vivo no
existen o se hacen de otra forma.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from . import __version__
from .core import motor as mt, oclusiones, perfil as pf, segmentacion as sg, separacion as sp
from .core.motor import Motor
from .core.tracking import Rastreador
from .core.video import LectorVideo, estimar_fondo

ETAPAS = (                                     # (nombre, dónde, función)
    ("ROI y luminancia", sg, "recortar"),
    ("campo plano", sg, "campo_plano"),
    ("gaussiano", sg, "suavizar"),
    ("umbral", sg, "umbralizar"),
    ("apertura", sg, "abrir"),
    ("componentes y área", sg, "componentes"),
    ("objetos estáticos", sg.Estaticos, "filtrar"),
    ("separación", sp, "detecciones"),
    ("tracking", Rastreador, "paso"),
    ("oclusiones", mt.Motor, "_revisar_oclusiones"),
)


def _orden(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def equipo() -> dict:
    d = dict(maquina=platform.machine(), python=platform.python_version(), opencv=cv2.__version__,
             numpy=np.__version__, hilos_opencv=cv2.getNumThreads())
    try:
        with open("/proc/device-tree/model") as fh:
            d["modelo"] = fh.read().strip("\0\n ")
    except OSError:
        d["modelo"] = platform.processor()
    return d


def termico() -> str:
    t, h = _orden("vcgencmd", "measure_temp"), _orden("vcgencmd", "get_throttled")
    return f"{t} {h}".strip() or "no disponible"


def memoria_mb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:                        # Windows
        return None


def pasada(cuadros, p, fondo):
    m = Motor(p, fondo)
    T = np.empty(len(cuadros))
    for k, f in enumerate(cuadros):
        t = time.perf_counter()
        m.procesar(k, f)
        T[k] = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    c = m.finalizar()
    return m, T, c, (time.perf_counter() - t) * 1e3


def desglose(cuadros, p, fondo) -> dict[str, float]:
    """ms medios por cuadro de cada técnica."""
    ac, orig = defaultdict(float), []

    def envolver(nombre, f):
        def g(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                ac[nombre] += time.perf_counter() - t
        return g
    try:
        for nombre, donde, fn in ETAPAS:
            f = getattr(donde, fn)
            orig.append((donde, fn, f))
            setattr(donde, fn, envolver(nombre, f))
        pasada(cuadros, p, fondo)
    finally:
        for donde, fn, f in orig:
            setattr(donde, fn, f)
    d = {n: ac[n] / len(cuadros) * 1e3 for n, _, _ in ETAPAS}
    # ``umbralizar`` se llama dos veces por cuadro (también dentro de ``filtrar``, que ya la incluye)
    d["umbral"] /= 2
    return d


def retraso(T, fps) -> tuple[float, int]:
    """(retraso máximo en ms, cuadros acumulados) si llega un cuadro cada 1/fps y se atienden en orden."""
    periodo, libre, peor = 1e3 / fps, 0.0, 0.0
    for k, t in enumerate(T):
        llega = k * periodo
        libre = max(libre, llega) + t
        peor = max(peor, libre - llega)
    return peor, int(np.ceil(peor / periodo))


def medir(ruta, p, fps) -> dict:
    v = LectorVideo(ruta)
    if (v.ancho, v.alto) != (p.ancho, p.alto):
        raise SystemExit(f"{ruta}: es {v.ancho}x{v.alto} y el perfil es {p.ancho}x{p.alto}")
    t = time.perf_counter()
    fondo = estimar_fondo(v, p)
    t_fondo = time.perf_counter() - t
    # Se decodifica antes para no mezclarlo con el algoritmo (2,8 MB por cuadro)
    t = time.perf_counter()
    cuadros = []
    for k in range(v.total):
        f = v.leer(k)
        if f is None:
            break
        cuadros.append(f)
    t_leer = (time.perf_counter() - t) / len(cuadros) * 1e3
    fps_archivo = v.fps
    v.cerrar()
    pasada(cuadros[:30], p, fondo)             # calentamiento
    antes = termico()
    m, T, c, t_fin = pasada(cuadros, p, fondo)
    despues = termico()
    des = desglose(cuadros, p, fondo)
    con_peces = np.array([len(r.dets) > 0 for r in m.resultados])
    densos = np.array([len(r.dets) >= 10 for r in m.resultados])
    peor, cola = retraso(T, fps)
    return dict(
        video=ruta, cuadros=len(cuadros), fps_archivo=fps_archivo, conteo=c.total, confirmados=c.confirmados,
        recuperados=c.recuperados, media=float(T.mean()), p50=float(np.percentile(T, 50)),
        p95=float(np.percentile(T, 95)), p99=float(np.percentile(T, 99)), maximo=float(T.max()),
        media_con_peces=float(T[con_peces].mean()) if con_peces.any() else None,
        media_densos=float(T[densos].mean()) if densos.any() else None, n_densos=int(densos.sum()),
        pct_sobre_presupuesto=float(100 * (T > 1e3 / fps).mean()), retraso_max_ms=float(peor), cola_max=cola,
        fps_sostenible=float(1e3 / T.mean()), finalizar_ms=t_fin, leer_ms=t_leer, fondo_s=t_fondo,
        desglose=des, termico_antes=antes, termico_despues=despues, tiempos=[round(float(x), 2) for x in T])


def imprimir(r, fps):
    print(f"\n=== {r['video']}  ({r['cuadros']} cuadros)")
    print(f"Conteo: {r['conteo']} ({r['confirmados']} confirmados + {r['recuperados']} recuperados)")
    print(f"Por cuadro: media {r['media']:.1f} ms | mediana {r['p50']:.1f} | p95 {r['p95']:.1f} | "
          f"p99 {r['p99']:.1f} | máx {r['maximo']:.1f}")
    if r["media_con_peces"]:
        print(f"Con peces: {r['media_con_peces']:.1f} ms" +
              (f" | con 10 o más peces ({r['n_densos']} cuadros): {r['media_densos']:.1f} ms" if r["media_densos"] else ""))
    print(f"Sostenible: {r['fps_sostenible']:.0f} cuadros/s. A {fps:g} fps ({1e3 / fps:.1f} ms): "
          f"{r['pct_sobre_presupuesto']:.1f} % de los cuadros se pasan; retraso máximo {r['retraso_max_ms']:.0f} ms "
          f"({r['cola_max']} cuadros en cola)")
    print("Desglose (ms): " + ", ".join(f"{k} {v:.2f}" for k, v in r["desglose"].items()))
    print(f"Aparte: leer el archivo {r['leer_ms']:.1f} ms/cuadro | fondo {r['fondo_s']:.1f} s | "
          f"cierre {r['finalizar_ms']:.0f} ms")
    print(f"Temperatura: {r['termico_antes']} -> {r['termico_despues']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--pez", default="lubina", choices=[i for i, _ in pf.disponibles()])
    ap.add_argument("--fps", type=float, default=30.0, help="cuadros por segundo de la cámara")
    ap.add_argument("--hilos", type=int, default=0, help="hilos de OpenCV (0 = los que elija)")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.hilos:
        cv2.setNumThreads(a.hilos)
    e = equipo()
    print(f"fishcounter {__version__} | " + " | ".join(f"{k}: {v}" for k, v in e.items()))
    p, R = pf.cargar(a.pez), []
    for ruta in a.videos:
        R.append(medir(ruta, p, a.fps))
        imprimir(R[-1], a.fps)
    mem = memoria_mb()
    if mem:
        print(f"\nMemoria máxima del proceso: {mem:.0f} MB (incluye los cuadros precargados para medir)")
    if a.json:
        with open(a.json, "w", encoding="utf8") as fh:
            json.dump(dict(equipo=e, pez=a.pez, fps=a.fps, resultados=R), fh, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
