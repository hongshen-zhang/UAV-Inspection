#!/usr/bin/env python3
"""Generate Figure 7: default (2200 s, 460 kJ), lower time (1800 s, 460 kJ),
and lower energy (2200 s, 380 kJ), with 300 paired missions per method/setting.

Run from the repository root: python Figure7/experiment.py --workers 8
Use a new --output-dir for the revised settings if old results already exist.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from simulation.current.experiment import main

if __name__ == '__main__':
    main('Figure7', HERE)
