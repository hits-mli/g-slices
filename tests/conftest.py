import sys
from pathlib import Path

# gslice and bin are plain top-level packages; everything runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
