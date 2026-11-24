"""Pruebas (sólo biblioteca estándar): python -m unittest discover -s tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trout_counter.acquisition import crop_strip                       # noqa: E402
from trout_counter.calibration import (Calibration, calibrate_background,  # noqa: E402
                                       calibrate_threshold)
from trout_counter.config import load_config                           # noqa: E402
from trout_counter.counting.scanline import SpatioTemporalAccumulator  # noqa: E402
from trout_counter.counting.separation import watershed_split          # noqa: E402
from trout_counter.pipeline import TroutCountingPipeline               # noqa: E402
from trout_counter.preprocessing import FlatFieldCorrector             # noqa: E402
from trout_counter.segmentation import otsu_threshold                  # noqa: E402
from trout_counter.simulation import ChannelSimulator                  # noqa: E402


class TestTechniques(unittest.TestCase):
    def test_otsu_bimodal(self):
        rng = np.random.default_rng(0)
        px = np.concatenate([rng.normal(40, 5, 2000), rng.normal(200, 5, 8000)])
        res = otsu_threshold(np.clip(px, 0, 255).astype(np.uint8))
        self.assertTrue(60 < res.threshold < 180)
        self.assertGreater(res.effectiveness, 0.9)

    def test_flat_field_equalizes(self):
        bg = np.tile(np.linspace(140, 240, 100, dtype=np.float32), (10, 1))
        ff = FlatFieldCorrector(bg, None, 200)
        out = ff.apply(bg.astype(np.uint8))
        self.assertLessEqual(int(out.max()) - int(out.min()), 2)

    def test_scanline_event(self):
        acc = SpatioTemporalAccumulator(gap_frames=3, max_event_frames=100)
        empty, full = np.zeros(10, np.uint8), np.full(10, 255, np.uint8)
        events = [acc.push(full if 5 <= t < 10 else empty, empty, t) for t in range(20)]
        ev = [e for e in events if e is not None]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].start_frame, 5)
        self.assertEqual(int(ev[0].st_a.any(axis=1).sum()), 5)

    def test_watershed_two_touching_vs_one(self):
        two = np.zeros((80, 140), np.uint8)
        cv2.circle(two, (45, 40), 28, 255, -1)
        cv2.circle(two, (98, 40), 28, 255, -1)
        n, _ = watershed_split(two, 1.0, float(np.pi * 28 ** 2), 0.6, 0.35)
        self.assertEqual(n, 2)
        one = np.zeros((80, 200), np.uint8)
        cv2.ellipse(one, (100, 40), (90, 22), 0, 0, 360, 255, -1)
        n, _ = watershed_split(one, 1.0, float(np.pi * 90 * 22), 0.6, 0.35)
        self.assertEqual(n, 1)


def run_sim(cfg, n_fish, seed, transform=lambda f: f):
    sim = ChannelSimulator(width=640, height=480, mm_per_px=cfg.geometry.mm_per_px,
                           length_range_mm=cfg.batch.length_range_mm, seed=seed)
    n_frames = sim.build_schedule(n_fish)
    bg = calibrate_background((crop_strip(transform(f), cfg.camera)
                               for f in sim.frames(40, False)), 40)
    flat = FlatFieldCorrector(bg, None, cfg.preprocess.target_level)
    strips = [crop_strip(transform(f), cfg.camera) for f in sim.frames(n_frames)]
    res = calibrate_threshold(iter(strips), flat, cfg, 60)
    pipe = TroutCountingPipeline(cfg, Calibration(bg, None, res.threshold, res.effectiveness),
                                 bg.shape)
    for i, s in enumerate(strips):
        pipe.process(i, s)
    pipe.flush()
    return len(sim.fish), pipe.count


class TestEndToEnd(unittest.TestCase):
    def test_synthetic_count(self):
        cfg = load_config(ROOT / "config.toml")
        truth, count = run_sim(cfg, 60, seed=3)
        self.assertLessEqual(abs(count - truth), 1)

    def test_flow_along_x_reversed(self):
        """Cámara girada 90° y flujo invertido: debe contar lo mismo."""
        cfg = load_config(ROOT / "config.toml")
        cfg.camera.flow_axis = "x"
        cfg.camera.flow_reversed = True
        truth, count = run_sim(cfg, 40, seed=5, transform=lambda f: np.ascontiguousarray(f.T[:, ::-1]))
        self.assertLessEqual(abs(count - truth), 1)


if __name__ == "__main__":
    unittest.main()
