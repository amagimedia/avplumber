import sys
from pathlib import Path


REPLAY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_DIR))
