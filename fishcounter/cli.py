"""Conteo por línea de comandos, sin interfaz:  python -m fishcounter.cli video.mp4 --pez lubina"""
from __future__ import annotations

import argparse
import time

from .core import perfil as pf
from .core.motor import Motor
from .core.video import LectorVideo, estimar_fondo


def contar(ruta: str, p: pf.Perfil) -> Motor:
    v = LectorVideo(ruta)
    if (v.ancho, v.alto) != (p.ancho, p.alto):
        raise SystemExit(f"El video es {v.ancho}x{v.alto} y el perfil se calibró para {p.ancho}x{p.alto}: "
                         "hay que recalibrar.")
    m = Motor(p, estimar_fondo(v, p))
    for k in range(v.total):
        f = v.leer(k)
        if f is None:
            break
        m.procesar(k, f)
    m.finalizar()
    v.cerrar()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--pez", default="lubina", choices=[i for i, _ in pf.disponibles()])
    a = ap.parse_args()
    p = pf.cargar(a.pez)
    for r in a.videos:
        t = time.perf_counter()
        m = contar(r, p)
        c = m.conteo
        print(f"{r}: {c.total} peces ({c.confirmados} confirmados + {c.recuperados} recuperados), "
              f"{time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
