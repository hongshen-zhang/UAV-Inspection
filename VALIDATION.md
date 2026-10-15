# Reproduction check — 2026-10-16

Starting commit: `3d97bd526a1a3d8122d545c2b45ac092b2bf6401`.
Fresh Python 3.12.14 environment, macOS ARM64; every pinned requirement
installed successfully and `pip check` passed. Algorithm kernels, parameters,
and random seeds were preserved.

| Check | Coverage | Result |
|---|---|---|
| Input regeneration | 11 studies, 124 study points, 28,500 paired mission inputs | Exact match to the paper archives |
| Figure 4 | 9 methods × 300 workloads = 2,700 outcomes | Completed and plotted |
| Table IV | 9 methods × 30 priority configurations × 10 workloads = 2,700 outcomes | Completed and plotted |
| All Figures 4–11 and Table IV entries | Every method and point, one workload each = 991 outcomes | Completed and plotted |
| Figure 8 timing | All 11 candidate caps, 2 methods, 3 warmups and 2 measured seeds | Completed serially |

Across the 6,391 generated mission outcomes, all 5,685 non-IO records matched
archived WCR, routes, counts, final time, and energy within `1e-8`.
All records passed mission budgets, return feasibility after each visit,
completion deadlines, unique visits, and workload/reward/trace consistency.
There were no duplicate result keys within a run.

IO uses wall-clock-limited MILP solves: 69 of 706 checked records differed in
at least one route or resource metric, while remaining feasible. For Figure 4,
its mean WCR was 33.3905% (archive: 33.4262%); for Table IV, 36.3952%
(archive: 36.5762%). This solver variation does not affect the deterministic
Proposed results. Published result PDFs retain the manuscript's statistics.

The default comparison reproduced Proposed 39.0429% versus Mean-workload
Greedy 35.0595%: +3.9833 percentage points, or +11.3616% relative.
Table IV reproduced Proposed 40.4714%.

The complete 219,000-outcome matrix was not rerun during this check. The
smaller all-entry run verifies execution and plotting, not the full sweep
statistics or the paper's 30-seed timing estimates. Use the full command in
[README.md](README.md) to regenerate those statistics on the target machine.
