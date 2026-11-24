"""Punto de entrada del contador de truchas.

Uso:
    python main.py calibrate --source pi            # en la Raspberry Pi
    python main.py calibrate --source video.mp4 --bg-video vacio.mp4
    python main.py run --source pi
    python main.py run --source video.mp4 --show
    python main.py simulate --fish 200 --seed 1     # prueba completa en PC
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from trout_counter.calibration import Calibration, calibrate_background, calibrate_threshold
from trout_counter.config import Config, load_config
from trout_counter.pipeline import TroutCountingPipeline
from trout_counter.preprocessing import FlatFieldCorrector
from trout_counter.results import CsvLogger

log = logging.getLogger("trout_counter")


# ---------------------------------------------------------------------------
def open_source(spec: str, cfg: Config):
    if spec == "pi":
        from trout_counter.acquisition.picamera_source import PiCameraSource
        return PiCameraSource(cfg.camera)
    from trout_counter.acquisition import VideoFileSource
    return VideoFileSource(spec, cfg.camera)


def strips_of(source):
    for _, _, strip in source.frames():
        yield strip


# ---------------------------------------------------------------------------
def cmd_calibrate(args, cfg: Config) -> None:
    bg_src = open_source(args.bg_video or args.source, cfg)
    if args.source == "pi":
        input("1) Deja el canal VACÍO con los LED encendidos y presiona Enter...")
    background = calibrate_background(strips_of(bg_src), args.bg_frames)
    dark = None
    if args.dark and args.source == "pi":
        input("   Apaga los LED (cuadro oscuro) y presiona Enter...")
        dark = calibrate_background(strips_of(bg_src), 30)
        input("   Vuelve a encender los LED y presiona Enter...")
    bg_src.close()
    flat = FlatFieldCorrector(background, dark, cfg.preprocess.target_level)

    fish_src = open_source(args.source, cfg)
    if args.source == "pi":
        input("2) Deja pasar peces por el canal y presiona Enter...")
    res = calibrate_threshold(strips_of(fish_src), flat, cfg, args.fish_frames)
    fish_src.close()

    calib = Calibration(background, dark, res.threshold, res.effectiveness)
    calib.save(args.calib)
    print(f"Calibración guardada en {args.calib}: umbral={res.threshold} eta={res.effectiveness:.3f}")


def cmd_run(args, cfg: Config) -> None:
    calib = Calibration.load(args.calib) if Path(args.calib).exists() else None
    source = open_source(args.source, cfg)
    csv_log = CsvLogger(cfg.output.csv_path)
    pipe = None
    t_report = time.monotonic()
    n_frames = 0
    t0 = time.monotonic()
    try:
        for idx, _, strip in source.frames():
            if pipe is None:
                pipe = TroutCountingPipeline(cfg, calib, strip.shape)
            recs = pipe.process(idx, strip)
            csv_log.write(recs)
            n_frames += 1
            if args.show:
                vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
                vis[pipe.last_mask > 0] = (0, 0, 255)
                for r in (cfg.geometry.line_a_row, cfg.geometry.line_b_row):
                    cv2.line(vis, (0, r), (vis.shape[1], r), (0, 255, 0), 1)
                cv2.putText(vis, f"{pipe.count}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)
                cv2.imshow("franja", cv2.resize(vis, None, fx=2, fy=2))
                if cv2.waitKey(1) == 27:
                    break
            if time.monotonic() - t_report > cfg.output.report_every_s:
                fps = n_frames / (time.monotonic() - t0)
                print(f"conteo={pipe.count}  fps_proc={fps:.1f}  "
                      f"v_tipica={pipe.counter.velocity.typical:.1f}px/cuadro  "
                      f"area_ind={pipe.counter.batch.single_area:.0f}px2")
                t_report = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if pipe is not None:
            csv_log.write(pipe.flush())
            print(f"TOTAL = {pipe.count}  (adelante={pipe.counter.forward}, "
                  f"retroceso={pipe.counter.reverse})")
        csv_log.close()
        source.close()


def cmd_simulate(args, cfg: Config) -> None:
    from trout_counter.acquisition import crop_strip
    from trout_counter.simulation import ChannelSimulator

    sim = ChannelSimulator(width=cfg.camera.output_width, height=480,
                           mm_per_px=cfg.geometry.mm_per_px,
                           length_range_mm=cfg.batch.length_range_mm, seed=args.seed)
    n_total = sim.build_schedule(args.fish)
    truth = len(sim.fish)

    # Calibración con el simulador (mismo procedimiento que en campo)
    background = calibrate_background((crop_strip(f, cfg.camera)
                                       for f in sim.frames(60, with_fish=False)), 60)
    flat = FlatFieldCorrector(background, None, cfg.preprocess.target_level)
    res = calibrate_threshold((crop_strip(f, cfg.camera) for f in sim.frames(n_total)),
                              flat, cfg, n_with_fish=80)
    calib = Calibration(background, None, res.threshold, res.effectiveness)

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 cfg.camera.fps, (sim.W, sim.H), False)
    pipe = TroutCountingPipeline(cfg, calib, background.shape)
    methods: dict[str, int] = {}
    t0 = time.perf_counter()
    for idx, frame in enumerate(sim.frames(n_total)):
        if writer is not None:
            writer.write(frame)
        for r in pipe.process(idx, crop_strip(frame, cfg.camera)):
            methods[r.method] = methods.get(r.method, 0) + r.count
    for r in pipe.flush():
        methods[r.method] = methods.get(r.method, 0) + r.count
    dt = time.perf_counter() - t0
    if writer is not None:
        writer.release()

    err = pipe.count - truth
    print(f"Cuadros={n_total}  umbral Otsu={res.threshold} (eta={res.effectiveness:.3f})")
    print(f"Verdad={truth}  Conteo={pipe.count}  Error={err:+d} ({100 * err / truth:+.1f} %)")
    print(f"Conteo por método: {methods}")
    print(f"Velocidad típica={pipe.counter.velocity.typical:.2f} px/cuadro  "
          f"área individual={pipe.counter.batch.single_area:.0f} px2")
    print(f"Rendimiento en este PC: {n_total / dt:.0f} cuadros/s (incluye el render del simulador)")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Contador de truchas (visión clásica, sin IA)")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="calibra fondo (flat-field) y umbral (Otsu)")
    c.add_argument("--source", default="pi", help="'pi' o ruta a video con peces")
    c.add_argument("--bg-video", help="video del canal vacío (si la fuente no es 'pi')")
    c.add_argument("--bg-frames", type=int, default=100)
    c.add_argument("--fish-frames", type=int, default=60)
    c.add_argument("--dark", action="store_true", help="medir también cuadro oscuro")
    c.add_argument("--calib", default="calibration.npz")

    r = sub.add_parser("run", help="cuenta peces en vivo o desde video")
    r.add_argument("--source", default="pi")
    r.add_argument("--calib", default="calibration.npz")
    r.add_argument("--show", action="store_true", help="ventana de depuración (no en la Pi)")

    s = sub.add_parser("simulate", help="prueba de punta a punta con video sintético")
    s.add_argument("--fish", type=int, default=150)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--save-video")

    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    {"calibrate": cmd_calibrate, "run": cmd_run, "simulate": cmd_simulate}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
