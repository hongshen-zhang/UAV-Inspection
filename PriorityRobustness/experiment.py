#!/usr/bin/env python3
"""Generate fresh results for paper Table4; no bundled data are required."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from simulation.current.experiment import main

if __name__ == '__main__':
    main('Table4', HERE)
