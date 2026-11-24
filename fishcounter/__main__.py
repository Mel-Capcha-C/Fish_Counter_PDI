"""python -m fishcounter [video]  -> abre la interfaz."""
import sys

from .gui.app import main

if __name__ == "__main__":
    sys.exit(main())
