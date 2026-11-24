"""Genera los perfiles base a partir de los valores medidos en conteo_v3 (se ejecuta una vez).

    fcounter/Scripts/python tools/crear_perfiles.py [movimiento_carassius.json]
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from fishcounter.core import perfil as pf  # noqa: E402

CANAL = dict(ancho=1280, alto=720, roi_x0=66, roi_y0=36, fondo_cuadros=60, nivel=200.0, gauss=3,
             max_perdidos=1, vistas_por_pez=2.8, margen_entrada=30.0, margen_salida=40.0, borde=4.0,
             vistas_min=2)
PEZ = dict(umbral=146, apertura=3, area_min=350, quietos_ventana=10, quietos_min=6, quietos_fraccion=.6,
           solidez_libre=.94, hueco_max=60, umbral2=125, umbral3=105, razon=.88, cima_min=7.5, area_cima=60,
           razon_clara=.4, exceso=1.25, margen_oclusion=10.0, retro_salto=450.0)


def movimiento(d: dict) -> dict:
    c = d["coef"]
    q = d["pob"]
    return dict(c0=c[0], c1=c[1], c2=c[2], c3=c[3], c4=c[4],
                **{k: d[k] for k in ("sx", "sx2", "rho", "my", "sy", "sy2", "rhoy", "slw", "slh",
                                     "p_perdido", "p_medio")},
                pob_w=q[0], pob_sw=q[1], pob_h=q[2], pob_sh=q[3])


def escribir(nombre, d):
    pf.CARPETA.mkdir(exist_ok=True)
    with open(pf.CARPETA / nombre, "w", encoding="utf8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def main():
    lub = movimiento(json.load(open(RAIZ / "conteo_v3" / "movimiento_pixeles.json")))
    escribir("canal_fcdatazju.json", dict(version_esquema=1, tipo="canal", id="fcdatazju",
                                          nombre="Canal FCDataZJU", valores=CANAL))
    escribir("pez_lubina.json", dict(
        version_esquema=1, tipo="pez", id="lubina", nombre="Lubina negra (Micropterus salmoides)", canal="fcdatazju",
        orden=1, estado="calibrado", nota="Umbrales ajustados con las cajas anotadas de 16 videos; movimiento "
        "aprendido de peces aislados.", valores={**PEZ, **lub}))
    if len(sys.argv) > 1:
        car = movimiento(json.load(open(sys.argv[1])))
        escribir("pez_carassius.json", dict(
            version_esquema=1, tipo="pez", id="carassius", nombre="Carpín (Carassius auratus)", canal="fcdatazju",
            orden=2, estado="parcial", nota="Sólo el movimiento se aprendió con esta especie; el resto son "
            "los valores de lubina.", valores={**PEZ, **car}))
    escribir("pez_trucha.json", dict(
        version_esquema=1, tipo="pez", id="trucha", nombre="Trucha (Oncorhynchus mykiss)", canal="fcdatazju",
        orden=3, estado="sin_calibrar", nota="Copia provisional de lubina: falta el dataset de truchas.",
        valores={**PEZ, **lub}))
    for i, n in pf.disponibles():
        print(i, n, pf.cargar(i).estado)


if __name__ == "__main__":
    main()
