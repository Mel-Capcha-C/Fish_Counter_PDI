"""Demostración de la segmentación sobre una imagen fija (p. ej. Fish_Image_Test.png).

Aplica a una imagen completa las técnicas que no dependen del tiempo:
    T3 canal Y -> T5 suavizado -> T6 Otsu -> T7 cierre/apertura
    -> T9 componentes conexos -> T10 ancho/aspecto -> T11 watershed (si hace falta)
y guarda un panel con cada etapa. Las técnicas 1, 2, 4, 8 y 12 necesitan la
cámara, el canal vacío o una secuencia de video, por eso no aparecen aquí.

Uso:
    python tools/demo_imagen_estatica.py Fish_Image_Test.png --out demo_etapas.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trout_counter.acquisition import bgr_to_y                       # noqa: E402
from trout_counter.counting.separation import watershed_split        # noqa: E402
from trout_counter.preprocessing import denoise                      # noqa: E402
from trout_counter.segmentation import (MorphologyCleaner, apply_threshold,  # noqa: E402
                                        otsu_threshold)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (40, 40, 40), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="demo_etapas.png")
    ap.add_argument("--close", type=int, default=9)
    ap.add_argument("--open", type=int, default=5)
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"No se pudo leer {args.image}")

    y = bgr_to_y(bgr)                                           # T3
    smooth = denoise(y, "gaussian", 3)                          # T5
    otsu = otsu_threshold(smooth)                               # T6
    mask = apply_threshold(smooth, otsu.threshold)
    clean = MorphologyCleaner(args.close, args.open).apply(mask)  # T7
    # Relleno de huecos interiores (vientre claro con luz frontal)
    filled = clean.copy()
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)  # T9
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = 0.25 * np.median(areas) if len(areas) else 0
    result = bgr.copy()
    count = 0
    for i in range(1, n):
        x, yb, w, h, area = stats[i]
        if area < min_area:
            continue
        roi = (labels[yb:yb + h, x:x + w] == i).astype(np.uint8) * 255
        # T10: en esta vista lateral el "ancho" es la altura del blob
        aspect = h / max(w, 1)
        pieces = 1
        if area > 1.6 * np.median(areas):                       # T11 selectivo
            pieces, _ = watershed_split(roi, 1.0, float(np.median(areas)), 0.6, 0.35)
        count += pieces
        cv2.rectangle(result, (x, yb), (x + w, yb + h), (0, 0, 255), 2)
        cv2.putText(result, f"{pieces}  A={area}  asp={aspect:.2f}", (x, yb - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    print(f"Otsu: umbral={otsu.threshold}  eta={otsu.effectiveness:.3f}")
    print(f"Peces detectados: {count}")

    tiles = [label(y, "T3 canal Y"), label(smooth, "T5 gaussiano 3x3"),
             label(mask, f"T6 Otsu (k={otsu.threshold}, eta={otsu.effectiveness:.2f})"),
             label(filled, "T7 cierre+apertura (+ relleno)"),
             label(result, f"T9-T11 conteo = {count}")]
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    blank = np.full_like(tiles[0], 255)
    panel = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4]), np.hstack([tiles[4], blank])])
    panel = cv2.resize(panel, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.out, panel)
    print(f"Panel guardado en {args.out}")


if __name__ == "__main__":
    main()
