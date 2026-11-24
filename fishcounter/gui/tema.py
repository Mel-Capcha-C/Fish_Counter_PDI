"""Colores y estilo (tema oscuro)."""

FONDO, PANEL, BORDE, TEXTO, TENUE, ACENTO = "#16191d", "#1f242a", "#343c45", "#e6e9ec", "#9aa5b1", "#56b4e9"
VERDE, NARANJA, ROJO, AZUL = "#22b58a", "#e69f00", "#e0703a", "#3aa0e0"

QSS = f"""
* {{ font-family: 'Segoe UI', 'Noto Sans', sans-serif; font-size: 13px; color: {TEXTO}; }}
QMainWindow, QDialog, QWidget#raiz {{ background: {FONDO}; }}
QWidget#panel, QFrame#tarjeta {{ background: {PANEL}; border: 1px solid {BORDE}; border-radius: 8px; }}
QLabel {{ background: transparent; border: none; }}
QLabel#tenue {{ color: {TENUE}; }}
QLabel#titulo {{ font-weight: 600; }}
QLabel#grande {{ font-size: 60px; font-weight: 700; color: {VERDE}; }}
QLabel#aviso {{ background: #4a2a16; border: 1px solid {ROJO}; border-radius: 6px; padding: 8px 10px; }}
QPushButton, QToolButton {{ background: {PANEL}; border: 1px solid {BORDE}; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover, QToolButton:hover {{ border-color: {ACENTO}; }}
QPushButton:focus, QToolButton:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border: 1px solid {ACENTO}; }}
QPushButton:disabled, QToolButton:disabled {{ color: #5c6670; }}
QPushButton#principal {{ background: #15496b; border-color: {ACENTO}; font-weight: 600; }}
QPushButton:checked, QToolButton:checked {{ background: #15496b; border-color: {ACENTO}; }}
QComboBox, QAbstractSpinBox, QLineEdit {{ background: {FONDO}; border: 1px solid {BORDE}; border-radius: 5px; padding: 4px 8px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: #15496b; }}
QTabBar::tab {{ background: {PANEL}; border: 1px solid {BORDE}; padding: 7px 16px; margin-right: 3px; border-radius: 6px; }}
QTabBar::tab:selected {{ background: #15496b; border-color: {ACENTO}; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {BORDE}; border-radius: 6px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 6px; background: {BORDE}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACENTO}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -5px 0; border-radius: 7px; background: {TEXTO}; }}
QProgressBar {{ background: {FONDO}; border: 1px solid {BORDE}; border-radius: 6px; text-align: center; min-height: 18px; }}
QProgressBar::chunk {{ background: #1d6fa5; border-radius: 5px; }}
QToolTip {{ background: {PANEL}; color: {TEXTO}; border: 1px solid {ACENTO}; padding: 4px; }}
QStatusBar {{ background: {PANEL}; color: {TENUE}; }}
"""
