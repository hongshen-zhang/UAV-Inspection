# Online Task Selection and Execution for UAV Inspection under Workload Uncertainty

Simulation and figure sources for the paper. The default mission has 20 tasks,
3 MEC servers, a time budget of 2200 s, and an onboard energy budget of 460 kJ.
Inputs and workloads are generated from the public parameters and fixed seeds;
no external dataset or stored result archive is needed.

## Install

Tested with Python 3.12.14 on macOS ARM64. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce

From the repository root, check all entries for Figures 4–11 and Table IV with
one seed per method and point:

```sh
for folder in Figure4 Figure5 Figure6 Figure7 Figure8 Figure9 Figure10 Figure11 Table
do
  python "$folder/experiment.py" --count 1 --workers 8 --output-dir "reproduced/quick/$folder" || break
  if [ "$folder" = Figure8 ]; then
    python Figure8/experiment.py --timing --workers 1 --warmups 3 --timing-count 2 --overwrite --output-dir reproduced/quick/Figure8 || break
  fi
  python "$folder/plot.py" --input "reproduced/quick/$folder/results.csv" --output "reproduced/quick/$folder.pdf" || break
done
```

This runs 991 missions and separately measures Figure 8 decision times. It is
an execution check; its sample means are not the paper's statistics.

For the complete experiment matrix, use the same loop with `--count 1`
removed, `--timing-count 30`, and `reproduced/full` in place of
`reproduced/quick`. Keep the two output directories separate.

The full matrix contains 219,000 mission outcomes. Each method and sweep point
uses 300 paired workloads; Table IV uses 30 priority configurations with 10
workloads each. Figure 8 additionally uses 3 warmups and 30 measured paired
missions per method and candidate cap, timed serially. The full run can take
many hours. Reduce `--workers` to fit the machine.

For just the default comparison, use `Figure4` alone in the loop. Resume
mission generation by adding `--resume` to the first command, using the same
sample count and unchanged simulation code. The timing command remeasures
Figure 8 and replaces its local timing CSV on each run.

Outputs are `reproduced/<mode>/<folder>/results.csv` and
`reproduced/<mode>/<folder>.pdf`. These commands preserve published PDFs.

Figures 1–2 are artwork: run their `plot.py`. For Figure 3, run
`python Figure3/experiment.py` followed by `python Figure3/plot.py`.

## Verified numbers

The 2026-10-16 default rerun used all 300 paired workloads for all nine methods:
Proposed **39.0429%**, Mean-workload Greedy **35.0595%**, a gain of
**3.9833 percentage points** (**11.3616%** relative).
Figure 7 settings are Default `(2200 s, 460 kJ)`, Lower T `(1800 s, 460 kJ)`,
and Lower E `(2200 s, 380 kJ)`. Baseline citations follow the current manuscript:
Rollout [25], IO [26], and ADAPT [27].

See [VALIDATION.md](VALIDATION.md) for the exact rerun coverage. IO uses
time-limited HiGHS solves (5 s initially, 0.25 s online), so solver timing and
hardware can slightly change its feasible solutions. Decision times also vary
by machine. Generated CSVs, traces, caches, and reproduction outputs stay local
and are excluded from the source repository.
