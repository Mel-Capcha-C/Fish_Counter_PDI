"""Registro de resultados en CSV (un renglón por blob contado)."""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, fields
from pathlib import Path

from .counting import CountRecord


class CsvLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        new = not self.path.exists()
        self.fh = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fh)
        if new:
            self.writer.writerow(["timestamp"] + [f.name for f in fields(CountRecord)])

    def write(self, records: list[CountRecord]) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in records:
            row = asdict(r)
            self.writer.writerow([now] + [round(v, 2) if isinstance(v, float) else v
                                          for v in row.values()])
        if records:
            self.fh.flush()

    def close(self) -> None:
        self.fh.close()
