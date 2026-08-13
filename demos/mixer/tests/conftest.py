import sys
from pathlib import Path


MIXER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MIXER_DIR))
