"""Arranque de la interfaz."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from . import tema
from .ventana import Ventana


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    app.setStyle("Fusion")
    app.setStyleSheet(tema.QSS)
    v = Ventana(argv[1] if len(argv) > 1 else None)
    v.show()
    return app.exec()
