# UAV Inspection

Source code for **Online Task Selection and Execution for UAV Inspection under Workload Uncertainty**.

Each original figure directory contains its experiment and plotting scripts.
Shared algorithms remain under `simulation/`; the current nine-method kernels
are in `simulation/current/`. The default mission budgets are **2200 s and 460 kJ**.

This repository contains source code only. Experiment inputs are generated from
model parameters and random seeds in the code. Result tables, mission traces,
input archives, and exported figures are created locally and excluded by
`.gitignore`.

## Setup

Use Python 3.12 and install the dependencies from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run an experiment and draw its figure

For the main comparison, first run a small example:

```sh
python Figure5/experiment.py --method Proposed --count 3
python Figure5/plot.py
```

The experiment creates `Figure5/data/results.csv`. The plot reads that local
CSV and writes `Figure5/figure5.pdf`. A subset run produces a plot of the methods
and points actually simulated, not the full paper experiment.

For all nine methods and 300 paired workload realizations:

```sh
python Figure5/experiment.py --count 300 --workers 8 --resume
python Figure5/plot.py
```

By default each study runs all its methods, all sweep points, and 300 paired
missions per method/point (Table IV: 10 missions per priority configuration).
Use repeated `--method` and `--point` options to select a subset. `--output-dir`
changes the local data directory. `--resume` continues matching-code results;
`--overwrite` explicitly replaces an existing local CSV. The plot accepts
`--input` and `--output` to select other local files. Run each script with
`--help` for its options.

## Directories and paper numbering

The original directory names are preserved, including historical figure numbers.

| Directory | Paper item | Experiment |
|---|---|---|
| `Figure4/` | Figure 3 | Nominal mission route |
| `Figure5/` | Figure 4 | Nine-method comparison |
| `Figure6/` | Figure 5 | Time-budget sweep |
| `Figure7/` | Figure 6 | Energy-budget sweep |
| `Figure12/` | Figure 7 | Policy ablations |
| `Figure8/` | Figure 8 | Candidate limit and decision time |
| `Figure9/` | Figure 9 | Workload uncertainty |
| `Figure10/` | Figure 10 | Workload scale |
| `Figure11/` | Figure 11 | MEC CPU frequency |
| `PriorityRobustness/` | Table IV | Priority robustness |

The original illustrations remain in `Figure1/`, `Figure2/`, and `Figure3/`.
`Figure4-v2/` contains the Top/Sub decision illustration. Run `experiment.py`
then `plot.py` in either nominal route/decision directory; both generate their
inputs locally. The embedded map/artwork in the illustration source is retained.

Figure 7 uses default `(2200 s, 460 kJ)`, larger time `(2600 s, 460 kJ)`, and
lower energy `(2200 s, 380 kJ)`, consistent with the supplied paper. The energy
sweep spans 140–800 kJ, including 620 kJ. Table IV uses 30 priority configurations,
10 paired missions each, and confidence intervals across configuration means.

Additional studies are kept inside the original `Figure12/` directory:

```sh
python Figure12/experiment_sub.py --count 300 --workers 8
python Figure12/plot_sub.py
python Figure12/experiment_numerical.py --count 300 --workers 8
python Figure12/plot_numerical.py
```

These are execution-price and numerical-accuracy studies, respectively; the
current paper does not number them as Figure 12 or later.

## Figure 8 timing

Decision timing is measured separately, serially, with warmed kernels. Do not
run other experiments concurrently with the timing command:

```sh
python Figure8/experiment.py --count 300 --workers 8
python Figure8/experiment.py --timing --workers 1 --warmups 3 --timing-count 30
python Figure8/plot.py
```

The local `timing.csv` includes Top and Sub time, including the final return
decision, and excludes construction/serialization. Without actual timing data,
the plot contains only the performance panel. Ordinary simulation runtimes
include possible JIT startup costs and are not used as paper decision timings.

## Model and implementation

The nine comparison methods are Proposed, Weight Greedy, Mean-workload Greedy,
Mean-workload DP, Distribution-aware Myopic, Nearest Neighbor, IO, Rollout, and
ADAPT. Mission budgets include the return to base. Deadlines are absolute
completion times from mission start. Workload is revealed only after a task
has been selected and reached. The same seeds pair methods and conditions.

IO uses wall-clock-limited MILP solves, so its fresh solutions can vary across
machines. The current numerical kernels are copied from the revised source;
the older simulation modules remain to support the original route/decision
illustrations and historical ablation helpers.
