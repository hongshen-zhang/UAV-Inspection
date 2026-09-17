#!/usr/bin/env python3
"""Run the nominal mission for paper Figure 3's Macau route illustration.

Dependencies: numpy, scipy, numba (see ../simulation for readable algorithm code).
Run: python experiment.py
Then: python plot.py
Fresh results are generated locally in data/ from public model parameters.
The default seed is the first formal nominal seed, 2026092000.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from simulation.current.inputs import SEED_START
from simulation.current.route import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=SEED_START)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'data')
    args = parser.parse_args()
    result = run(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'replay.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    row = result['mission_result']
    print(json.dumps({'output': str(output), 'seed': args.seed, 'route': row['route'],
                      'weighted_completed': row['weighted_completed'],
                      'final_time_s': row['final_time_s'],
                      'final_energy_kj': row['final_energy_kj'],
                      'validation': result['validation']}, indent=2))


if __name__ == '__main__':
    main()
