#!/usr/bin/env python3
"""Plot current paper Figure6 from locally generated data/results.csv.

Run experiment.py first, then: python plot.py --input data/results.csv
No experiment data or stored outcomes are included in this plotting code.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / "simulation"))
from current.plotting import main

if __name__ == "__main__":
    main('Figure6', HERE)
