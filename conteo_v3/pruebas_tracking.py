"""Pruebas del tracking: (A) identidades conocidas y (B) cajas reales dañadas a propósito.

A. Cada nacimiento real (cuadro, posición, talla) se propaga con movimiento simulado
   (memoria de velocidad y colas pesadas, t de Student con 5 g.l.). Al conocer la identidad:
       conteo = reales + rotos - absorbidos
   rotos      = un pez ya contado que se cuenta otra vez (emparejamiento perdido)
   absorbidos = un pez nuevo tomado por otro anterior    (emparejamiento falso)
   Es optimista: el simulador comparte la forma del modelo de movimiento.
B. Errores de segmentación sobre las cajas reales de los videos de ajuste.
"""
import sys
from pathlib import Path

import numpy as np

from dataset import cajas, division, todos
from tracking import H, W, Movimiento, centro, contar

MOV = Movimiento.cargar(Path(__file__).with_name("movimiento.json"))


def simular(video, rng, mov, escala=1.0, densidad=1.0):
    """(cajas por cuadro, identidad por cuadro)"""
    r = contar(cajas(video), mov)
    nac = []
    vistos = set()
    for k in sorted(r.ids):
        for j, i in enumerate(r.ids[k]):
            if i not in vistos:
                vistos.add(i)
                if centro(cajas(video)[k][j])[0] < 560:
                    nac.append((k, cajas(video)[k][j]))
    if densidad > 1:                        # añade copias desplazadas en y y en el tiempo
        extra = [nac[i] for i in rng.choice(len(nac), int((densidad - 1) * len(nac)))]
        nac = nac + [(k + int(rng.integers(-2, 3)), b + [0, dy, 0, dy]) for k, b in extra
                     for dy in [rng.uniform(-60, 60)]]
    C, I = {}, {}

    def t5(s):
        return s * float(np.clip(rng.standard_t(5), -5, 5)) / 1.29

    for idn, (k, b) in enumerate(nac):
        b = np.array(b, float)
        if b[0] < 4:                        # entró cortado: se reconstruye la caja completa
            b[0] = b[2] - max(b[2] - b[0], np.exp(rng.normal(MOV.pob[0], MOV.pob[1])))
        rx, ry = t5(mov.sx * escala), t5(mov.sy * escala)
        for _ in range(8):
            v = np.clip(b, 0, [W, H, W, H])
            if v[2] - v[0] < 10 or v[3] - v[1] < 8:
                break
            C.setdefault(k, []).append(v)
            I.setdefault(k, []).append(idn)
            x, y = centro(b)
            dx, dy = mov.salto(x, y) + rx, mov.my + ry
            fw, fh = np.exp(rng.normal(0, mov.slw)), np.exp(rng.normal(0, mov.slh))
            w2, h2 = (b[2] - b[0]) * fw / 2, (b[3] - b[1]) * fh / 2
            w2, h2 = np.clip(w2, 15, 130), np.clip(h2, 8, 110)
            x, y = x + dx, np.clip(y + dy, 5, H - 5)
            b = np.array([x - w2, y - h2, x + w2, y + h2])
            rx = mov.rho * rx + t5(mov.sx2 * escala)
            ry = mov.rhoy * ry + t5(mov.sy2 * escala)
            k += 1
    lo, hi = min(C), max(C)
    return ({k: np.array(C.get(k, [])).reshape(-1, 4) for k in range(lo, hi + 1)},
            {k: I.get(k, []) for k in range(lo, hi + 1)}, len(nac))


def metricas(C, I, mov):
    r = contar(C, mov)
    pista_de, reales, rotos, absorb = {}, set(), 0, 0
    pista_usada = set()
    for k in sorted(C):
        for j, (idr, idp) in enumerate(zip(I[k], r.ids[k])):
            nueva = idp not in pista_usada
            pista_usada.add(idp)
            if idr not in reales:                  # primera vista real
                reales.add(idr)
                absorb += not nueva
            else:
                rotos += nueva
    return len(reales), r.conteo, rotos, absorb


def parte_a():
    print("A. IDENTIDADES CONOCIDAS (16 videos de ajuste simulados)")
    for nom, esc, den in (("como el dataset", 1.0, 1.0), ("velocidad 1,3x más variable", 1.3, 1.0),
                          ("2x más peces", 1.0, 2.0)):
        N = c = ro = ab = 0
        for v in division()[0]:
            rng = np.random.default_rng(hash_estable(v))
            n, k, a, b = metricas(*simular(v, rng, MOV, esc, den)[:2], MOV)
            N, c, ro, ab = N + n, c + k, ro + a, ab + b
        print(f"  {nom:28s} reales {N} conteo {c} ({100 * (c / N - 1):+.1f} %)  "
              f"contados de más {ro} ({100 * ro / N:.1f} %)  de menos {ab} ({100 * ab / N:.1f} %)")


def hash_estable(v):
    return int("".join(ch for ch in v if ch.isdigit())) + 1000


# ------------------------------------------------------------------ B. daños
def perder(c, p, rng):
    return {k: b[rng.random(len(b)) >= p] for k, b in c.items()}


def unir(c, p, rng):
    """Cajas que se solapan pasan a ser una sola con probabilidad p (por cuadro)."""
    out = {}
    for k, b in c.items():
        b = [x.copy() for x in b]
        i = 0
        while i < len(b):
            j = i + 1
            while j < len(b):
                ix = min(b[i][2], b[j][2]) - max(b[i][0], b[j][0])
                iy = min(b[i][3], b[j][3]) - max(b[i][1], b[j][1])
                amin = min((b[i][2] - b[i][0]) * (b[i][3] - b[i][1]), (b[j][2] - b[j][0]) * (b[j][3] - b[j][1]))
                if ix > 0 and iy > 0 and ix * iy > .25 * amin and rng.random() < p:
                    b[i] = np.array([min(b[i][0], b[j][0]), min(b[i][1], b[j][1]),
                                     max(b[i][2], b[j][2]), max(b[i][3], b[j][3])])
                    b.pop(j)
                    continue
                j += 1
            i += 1
        out[k] = np.array(b).reshape(-1, 4)
    return out


def partir(c, p, rng):
    """Un pez se separa en dos mitades (error de sobre-separación)."""
    out = {}
    for k, b in c.items():
        n = []
        for x in b:
            if rng.random() < p:
                if x[2] - x[0] > x[3] - x[1]:
                    xm = (x[0] + x[2]) / 2
                    n += [[x[0], x[1], xm, x[3]], [xm, x[1], x[2], x[3]]]
                else:
                    ym = (x[1] + x[3]) / 2
                    n += [[x[0], x[1], x[2], ym], [x[0], ym, x[2], x[3]]]
            else:
                n.append(x)
        out[k] = np.array(n).reshape(-1, 4)
    return out


def ruido(c, s, rng):
    out = {}
    for k, b in c.items():
        d = (b[:, 2:] - b[:, :2]) if len(b) else np.zeros((0, 2))
        e = rng.normal(0, s, (len(b), 4)) * np.hstack([d, d])
        out[k] = np.clip(b + e, 0, [W, H, W, H])
        out[k][:, 2:] = np.maximum(out[k][:, 2:], out[k][:, :2] + 6)
    return out


def falsas(c, p, rng):
    out = {}
    for k, b in c.items():
        n = rng.poisson(p * max(len(b), .0))
        f = np.column_stack([rng.uniform(0, W - 80, n), rng.uniform(0, H - 50, n)]) if n else np.zeros((0, 2))
        out[k] = np.vstack([b, np.hstack([f, f + [70, 30]])]) if n else b
    return out


def parte_b():
    print("B. CAJAS REALES DAÑADAS (16 videos de ajuste, 2 semillas): error medio por video / sesgo")
    casos = [("sin daño", lambda c, r: c),
             ("pierde 1 % de detecciones", lambda c, r: perder(c, .01, r)),
             ("pierde 3 %", lambda c, r: perder(c, .03, r)),
             ("pierde 8 %", lambda c, r: perder(c, .08, r)),
             ("cajas ruidosas (10 % del tamaño)", lambda c, r: ruido(c, .10, r)),
             ("falsas 0,5 %", lambda c, r: falsas(c, .005, r)),
             ("falsas 2 %", lambda c, r: falsas(c, .02, r)),
             ("parte 2 % de los peces en dos", lambda c, r: partir(c, .02, r)),
             ("parte 6 %", lambda c, r: partir(c, .06, r)),
             ("une 50 % de los solapados", lambda c, r: unir(c, .5, r)),
             ("une 100 % de los solapados", lambda c, r: unir(c, 1., r)),
             ("combinado: pierde 3 %, ruido 10 %, parte 2 %, une 50 %",
              lambda c, r: partir(ruido(perder(unir(c, .5, r), .03, r), .1, r), .02, r))]
    for nom, f in casos:
        e, e2, quita = [], [], []
        for s in (0, 1):
            for v in division()[0]:
                rng = np.random.default_rng(7 * hash_estable(v) + s)
                d = f(cajas(v), rng)
                r = contar(d, MOV)
                e.append(r.conteo / todos()[v] - 1)
                e2.append(r.confirmadas / todos()[v] - 1)
                quita.append(sum(map(len, d.values())) / sum(map(len, cajas(v).values())) - 1)
        e, e2 = 100 * np.array(e), 100 * np.array(e2)
        print(f"  {nom:58s} todas {abs(e).mean():5.1f} / {e.mean():+5.1f} %   vistas 2+ {abs(e2).mean():5.1f} / {e2.mean():+5.1f} %   detecciones {100 * np.mean(quita):+.1f} %")


if __name__ == "__main__":
    if "a" in sys.argv[1:] or len(sys.argv) == 1:
        parte_a()
    if "b" in sys.argv[1:] or len(sys.argv) == 1:
        parte_b()
