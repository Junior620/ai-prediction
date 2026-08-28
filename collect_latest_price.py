"""
Collecte prix cacao — delegue a ICE London (remplace Yahoo CC=F).

Conserve ce fichier pour compatibilite; preferer collect_ice_london_cocoa.py.
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "collect_ice_london_cocoa.py"
    raise SystemExit(subprocess.call([sys.executable, str(script)]))
