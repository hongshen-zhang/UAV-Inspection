#!/usr/bin/env python3
"""Run the original six planners on the paired Figure 11 inputs.

Default: all 15 settings and 300 saved workload seeds per method.
Smoke example: python experiment.py --seeds 1 --workers 1 --points 0.7
Outputs raw mission records and full action traces; no saved outcome is reused.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "simulation"))
from sensitivity_cases import SEEDS, run_sensitivity

POINTS = [0.7, 1.05, 1.4, 1.75, 2.1, 2.45, 2.8, 3.15, 3.5, 3.85, 4.2, 4.55, 4.9, 5.25, 5.6]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=300, choices=range(1, 301), metavar="1..300",
                        help="number of original paired seeds, starting at 2026092000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--points", type=float, nargs="+", default=POINTS, choices=POINTS,
                        help="subset of the original measured parameter settings")
    parser.add_argument("--output", type=Path, default=HERE / "results.csv")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if len(set(args.points)) != len(args.points):
        parser.error("--points must not contain duplicates")
    run_sensitivity('mec', sorted(args.points), SEEDS[:args.seeds], args.workers, args.output)


if __name__ == "__main__":
    main()
