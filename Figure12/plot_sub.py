#!/usr/bin/env python3
"""Plot the execution-price ablation from locally generated paired outcomes."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / "simulation"))
from current.plotting import main

if __name__ == "__main__":
    main("ExecutionAblation", HERE,
         default_input=HERE/"data"/"execution_ablation"/"results.csv",
         default_output=HERE/"figure12_sub.pdf")
