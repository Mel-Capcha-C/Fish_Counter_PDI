"""Mide cuántos cuadros por segundo procesa el pipeline en este equipo.

Primero genera franjas sintéticas (no se mide) y luego cronometra sólo el
pipeline (T4-T12). Debe superar los fps de la cámara (config: camera.fps).

Uso:  python3 tools/benchmark.py --fish 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trout_counter.acquisition import crop_strip                       # noqa: E402
from trout_counter.calibration import (Calibration, calibrate_background,  # noqa: E402
                                       calibrate_threshold)
from trout_counter.config import load_config                           # noqa: E402
from trout_counter.pipeline import TroutCountingPipeline               # noqa: E402
from trout_counter.preprocessing import FlatFieldCorrector             # noqa: E402
from trout_counter.simulation import ChannelSimulator                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fish", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    sim = ChannelSimulator(width=cfg.camera.output_width, height=480,
                           mm_per_px=cfg.geometry.mm_per_px,
                           length_range_mm=cfg.batch.length_range_mm, seed=args.seed)
    n = sim.build_schedule(args.fish)
    t = time.perf_counter()
    bg = calibrate_background((crop_strip(f, cfg.camera) for f in sim.frames(40, False)), 40)
    strips = [crop_strip(f, cfg.camera) for f in sim.frames(n)]
    print(f"Generadas {n} franjas sintéticas en {time.perf_counter() - t:.1f} s")

    flat = FlatFieldCorrector(bg, None, cfg.preprocess.target_level)
    res = calibrate_threshold(iter(strips), flat, cfg, 60)
    pipe = TroutCountingPipeline(cfg, Calibration(bg, None, res.threshold, res.effectiveness),
                                 bg.shape)
    t = time.perf_counter()
    for i, s in enumerate(strips):
        pipe.process(i, s)
    pipe.flush()
    dt = time.perf_counter() - t
    fps = n / dt
    print(f"Verdad={len(sim.fish)}  Conteo={pipe.count}")
    print(f"Pipeline: {fps:.0f} cuadros/s  ({1000 / fps:.2f} ms por cuadro)  "
          f"cámara configurada a {cfg.camera.fps} fps -> "
          f"{'OK' if fps > cfg.camera.fps else 'INSUFICIENTE'}")


if __name__ == "__main__":
    main()
