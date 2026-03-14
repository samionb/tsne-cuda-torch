"""Pytest configuration for the standalone TorchTSNE repo."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
TESTS = ROOT / 'tests'

for path in (ROOT, SRC, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
