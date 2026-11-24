"""Conteo completo desde un video del dataset, sólo con píxeles (sin etiquetas ni IA).

    python contar_video.py m21 [m22 ...]               # tracking + técnica 15 (por defecto)
    python contar_video.py --metodo flujo m21          # flujo de costo mínimo
    python contar_video.py --calibrar                  # reaprende el movimiento (sin etiquetas)

evidencia.json (P de pieza falsa/doble, sólo la usa el flujo) se aprendió con etiquetas de los
16 videos de ajuste de lubina. El movimiento se guarda por especie (FC_ESPECIE).
"""
import argparse
import time
from pathlib import Path

from calibrar_movimiento import aislados
from dataset import ESPECIE, division, todos
from flujo import Evidencia, detecciones_flujo, resolver
from segmentacion import procesar
from temporal import contar_con_grupos
from tracking import Movimiento, ajustar_movimiento

_DIR = Path(__file__).parent
MOV_LUBINA = _DIR / "movimiento_pixeles.json"
MOV_PROPIO = MOV_LUBINA if ESPECIE == "MicropterusSalmoides" else _DIR / f"movimiento_{ESPECIE}.json"
# Sin calibración propia se usa la de lubina (el conteo empeora un poco, no se rompe)
MOV = MOV_PROPIO if MOV_PROPIO.exists() else MOV_LUBINA
EVID = _DIR / "evidencia.json"


def detectar(video):
    return {k: detecciones_flujo(m) for k, m in enumerate(procesar(video))}


def contar_video(video, mov, ev, metodo="t15"):
    """-> (conteo, trayectorias [[(cuadro, índice de detección), ...]])."""
    d = detectar(video)
    if metodo == "t15":
        r, _, _ = contar_con_grupos(d, mov)
        obs = {}
        for k, ids in r.ids.items():
            for j, i in enumerate(ids):
                obs.setdefault(i, []).append((k, j))
        tr = [obs[i] for i in r.contadas]
        return len(tr), tr
    tr, unidades, estado = resolver(d, mov, ev)
    if not unidades and any(len(x) for x in d.values()):
        raise RuntimeError(f"El optimizador no encontró solución: {estado}")
    return len(tr), tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--metodo", choices=("t15", "flujo"), default="t15")
    ap.add_argument("--calibrar", action="store_true")
    a = ap.parse_args()
    if a.calibrar:
        D = [detectar(v) for v in division()[0]]
        ajustar_movimiento(*aislados([{k: b[:, :4] for k, b in d.items()} for d in D])).guardar(MOV_PROPIO)
        print("Guardado", MOV_PROPIO)
    mov, ev = Movimiento.cargar(MOV_PROPIO if a.calibrar else MOV), Evidencia.cargar(EVID)
    for v in a.videos:
        t = time.perf_counter()
        n, tr = contar_video(v, mov, ev, a.metodo)
        print(f"{v}: conteo {n} (real {todos()[v]}); vistas 2+: {sum(len(q) > 1 for q in tr)}; "
              f"{a.metodo}, {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    main()
