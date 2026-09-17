#!/usr/bin/env python3
"""Run the additional NumericalAccuracy study from source parameters."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from simulation.current.experiment import main

if __name__ == '__main__':
    if '--output-dir' not in sys.argv and not any(x.startswith('--output-dir=') for x in sys.argv):
        sys.argv.extend(['--output-dir', str(HERE/'data'/'numerical_accuracy')])
    main('NumericalAccuracy', HERE)
