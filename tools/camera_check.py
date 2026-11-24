"""Prueba de la cámara en la Raspberry Pi (sin contar peces).

1. Guarda una foto del campo de visión completo del modo configurado, con la
   franja (T1) y las líneas A/B dibujadas: sirve para apuntar la cámara.
2. Abre la franja tal como la usa el pipeline (ScalerCrop, controles fijos,
   enfoque manual, canal Y) y mide los fps reales de captura, con y sin
   procesamiento.
3. Guarda una muestra de la franja y su nivel de brillo (para ajustar
   exposure_us / analogue_gain: el fondo debe quedar en ~200-230 sin saturar).

Uso:  python3 tools/camera_check.py --frames 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trout_counter.acquisition.picamera_source import PiCameraSource   # noqa: E402
from trout_counter.config import load_config                           # noqa: E402
from trout_counter.pipeline import TroutCountingPipeline               # noqa: E402


def full_view(cfg, out: Path) -> None:
    from picamera2 import Picamera2
    cam = cfg.camera
    picam2 = Picamera2()
    w, h = cam.sensor_mode_size
    conf = picam2.create_still_configuration(main={"size": (w, h), "format": "RGB888"},
                                             sensor={"output_size": (w, h)})
    picam2.configure(conf)
    if "AfMode" in picam2.camera_controls:
        from libcamera import controls as lc
        picam2.set_controls({"AfMode": lc.AfModeEnum.Manual,
                             "LensPosition": float(cam.lens_position)})
    picam2.start()
    time.sleep(1.5)
    img = picam2.capture_array("main")
    picam2.stop()
    picam2.close()
    # Franja y líneas en coordenadas de esta imagen (misma geometría que ScalerCrop)
    if cam.flow_axis == "y":
        rows = cam.strip_rows * w / cam.output_width
        y0 = int(cam.strip_center_frac * h - rows / 2)
        cv2.rectangle(img, (0, y0), (w - 1, int(y0 + rows)), (0, 255, 255), 2)
        for r, col in ((cfg.geometry.line_a_row, (0, 255, 0)), (cfg.geometry.line_b_row, (255, 0, 0))):
            yy = cam.flow_reversed and int(y0 + rows - r * w / cam.output_width) or int(y0 + r * w / cam.output_width)
            cv2.line(img, (0, yy), (w - 1, yy), col, 2)
    else:
        cols = cam.strip_rows * h / cam.output_width
        x0 = int(cam.strip_center_frac * w - cols / 2)
        cv2.rectangle(img, (x0, 0), (int(x0 + cols), h - 1), (0, 255, 255), 2)
    cv2.imwrite(str(out), img)
    print(f"Vista completa guardada en {out}  ({w}x{h}); amarillo = franja, verde = línea A, azul = línea B")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    ap.add_argument("--out-dir", default=str(ROOT / "camera_check"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    full_view(cfg, out / "vista_completa.jpg")

    src = PiCameraSource(cfg.camera)
    print(f"Franja de salida: {src.size[0]}x{src.size[1]}  ScalerCrop={src.crop}")
    pipe = None
    stamps, proc, last_idx, dropped = [], [], -1, 0
    sample = None
    for k, (idx, ts, strip) in enumerate(src.frames()):
        if pipe is None:
            pipe = TroutCountingPipeline(cfg, None, strip.shape)
            sample = strip.copy()
        if last_idx >= 0:
            dropped += max(0, idx - last_idx - 1)
        last_idx = idx
        t = time.perf_counter()
        pipe.process(idx, strip)
        proc.append(time.perf_counter() - t)
        stamps.append(ts)
        if k + 1 >= args.frames:
            break
    src.close()

    span = stamps[-1] - stamps[0]
    fps = (len(stamps) - 1) / span if span > 0 else 0
    print(f"Captura: {len(stamps)} cuadros en {span:.2f} s -> {fps:.1f} fps reales "
          f"(configurado {cfg.camera.fps}); cuadros perdidos: {dropped}")
    print(f"Procesamiento medio: {1000 * np.mean(proc):.2f} ms/cuadro "
          f"(máx {1000 * np.max(proc):.2f} ms); presupuesto: {1000 / cfg.camera.fps:.2f} ms")
    cv2.imwrite(str(out / "franja.png"), cv2.resize(sample, None, fx=2, fy=2,
                                                    interpolation=cv2.INTER_NEAREST))
    print(f"Brillo de la franja: media={sample.mean():.0f} mín={sample.min()} máx={sample.max()} "
          f"(saturados: {100 * np.mean(sample >= 250):.1f} %)")
    print(f"Muestra de la franja guardada en {out / 'franja.png'}")


if __name__ == "__main__":
    main()
