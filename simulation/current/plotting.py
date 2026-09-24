"""Plot locally generated mission CSV files; no stored results are embedded.

WCR is a fraction in the input. Confidence intervals resample paired mission
seeds 20,000 times. Table IV instead uses a t interval across priority groups.
Figure 8 timing requires measured decision totals/counts or decision events;
ordinary simulation runtime is deliberately not used as decision timing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from .inputs import ABLATION_SETTINGS, ABLATION_VARIANTS

METHODS = ("Proposed", "Weight Greedy", "Mean-workload Greedy", "Mean-workload DP",
           "Distribution-aware Myopic", "Nearest Neighbor", "IO", "Rollout", "ADAPT")
COLORS = dict(zip(METHODS, ("#EE2922", "#4F7CC7", "#F68C28", "#8673B5", "#4A7937",
                          "#B2B2B2", "#343576", "#269DA6", "#B65B9D")))
MARKERS = dict(zip(METHODS, ("o", "^", "D", "s", "P", "v", ">", "h", "X")))
LABELS = {"IO": "IO [11]", "Rollout": "Rollout [30]", "ADAPT": "ADAPT [13]"}
ALIASES = {"Priority Greedy": "Weight Greedy", "Shiri-IO": "IO",
           "Novoa-Rollout": "Rollout", "ADAPT-IACS": "ADAPT"}
VARIANTS = tuple(ABLATION_VARIANTS)
VARIANT_LABELS = ("Proposed", "Mean workload\nDP", "No priority\nfactor",
                  "Myopic\nTop", "No proactive\nskip")
SETTINGS = tuple(name for name, _, _ in ABLATION_SETTINGS)
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = 2026091510
FONT = "Times New Roman" if any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist) else "DejaVu Serif"


def configure():
    plt.rcParams.update({"font.family": FONT, "font.size": 13, "axes.labelsize": 15.5,
        "xtick.labelsize": 12, "ytick.labelsize": 13, "axes.unicode_minus": False,
        "mathtext.fontset": "custom", "mathtext.rm": FONT,
        "mathtext.it": FONT + ":italic", "mathtext.bf": FONT + ":bold",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def require_columns(frame, columns, description):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{description} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{description} is empty")


def load_results(path):
    frame = pd.read_csv(path)
    require_columns(frame, ("method", "seed", "wcr"), "Mission results")
    frame["method"] = frame.method.replace(ALIASES)
    if not np.isfinite(frame.wcr).all() or not frame.wcr.between(0, 1).all():
        raise ValueError("WCR must be a finite fraction in [0, 1]")
    return frame


def paired_values(frame, key="point", series="method", value="wcr"):
    """Require one observation for every series/setting/paired-seed cell."""
    require_columns(frame, (key, series, "seed", value), "Paired results")
    if frame[[key, series, "seed", value]].isna().any().any():
        raise ValueError("Paired results contain missing keys or values")
    if frame.duplicated([key, series, "seed"]).any():
        raise ValueError("Duplicate setting/method/seed in results")
    points = sorted(frame[key].unique())
    present = set(frame[series])
    order = VARIANTS if series == "variant_id" else METHODS
    names = [name for name in order if name in present]
    names += sorted(present - set(names))
    seeds = sorted(frame.seed.unique())
    index = pd.MultiIndex.from_product([seeds, points, names], names=["seed", key, series])
    column = frame.set_index(["seed", key, series])[value].reindex(index)
    if column.isna().any():
        raise ValueError("Every plotted setting and method must use the same paired seed set")
    values = column.to_numpy(float).reshape(len(seeds), len(points), len(names))
    if not np.isfinite(values).all():
        raise ValueError("Results contain nonfinite values")
    return points, names, seeds, values


def bootstrap(values, seed=BOOTSTRAP_SEED):
    mean = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAPS,) + mean.shape)
    for start in range(0, BOOTSTRAPS, 100):
        indices = rng.integers(0, len(values), size=(min(100, BOOTSTRAPS-start), len(values)))
        draws[start:start+len(indices)] = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    constant = np.all(values == values[:1], axis=0)
    mean[constant] = values[0][constant]
    low[constant] = high[constant] = mean[constant]
    return mean, np.minimum(low, mean), np.maximum(high, mean)


def summarize(frame, key="point", series="method"):
    points, methods, seeds, values = paired_values(frame, key, series)
    mean, low, high = bootstrap(values * 100)
    summary = pd.DataFrame([dict(point=point, method=method, mean=mean[i, j],
        ci_low=low[i, j], ci_high=high[i, j], n=len(seeds))
        for i, point in enumerate(points) for j, method in enumerate(methods)])
    return summary, methods


def style(ax):
    ax.grid(True, color="#ECECEC", linewidth=.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3.8, width=.8, pad=4)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.85)


def plot_series(ax, summary, methods):
    for j, method in enumerate(methods):
        q = summary[summary.method.eq(method)].sort_values("point")
        color = COLORS.get(method, f"C{j % 10}")
        ax.errorbar(q.point, q["mean"], yerr=np.maximum(0, np.stack((q["mean"]-q.ci_low, q.ci_high-q["mean"]))),
            label=LABELS.get(method, method), color=color,
            linestyle="-" if method == "Proposed" else "--", linewidth=2,
            marker=MARKERS.get(method, "o"), markersize=4.5 if method == "Mean-workload DP" else 7.3,
            markeredgewidth=.65, elinewidth=1.35, capsize=4.8, capthick=1, zorder=12-j)


def legend(fig, ax, methods, top=.985, *, display_order=None, display_labels=None):
    handles, labels = ax.get_legend_handles_labels()
    if display_order is not None:
        # Reorder only legend entries, never the curves or their style mappings.
        entries = dict(zip(methods, zip(handles, labels)))
        names = [name for name in display_order if name in entries]
        names += [name for name in methods if name not in names]
        handles = [entries[name][0] for name in names]
        labels = [(display_labels or {}).get(name, entries[name][1]) for name in names]
    # Matplotlib fills legend columns first; rearrange so readers see rows first.
    columns = min(3, len(methods))
    order = [i for col in range(columns) for i in range(col, len(handles), columns)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(.54, top), ncol=columns,
               fontsize=10.4, columnspacing=1.35, handlelength=2.2, frameon=False)


def fit_wcr(ax, summary, minimum=0, maximum=60):
    lo = min(minimum, 10*np.floor(summary.ci_low.min()/10))
    hi = min(100, max(maximum, 10*np.ceil((summary.ci_high.max()+3)/10)))
    ax.set_ylim(max(0, lo), hi)
    ax.set_ylabel("WCR (%)")


def comparison(frame):
    frame = frame.copy()
    frame["point"] = 1
    summary, methods = summarize(frame)
    # Figure 4 has its own display order and citations; other figures stay unchanged.
    jitter_indices = {method: index for index, method in enumerate(methods)}
    display_order = (*METHODS[:-3], "Rollout", "IO", "ADAPT")
    order = {method: index for index, method in enumerate(display_order)}
    methods.sort(key=lambda method: order.get(method, len(order)))
    labels = {"Rollout": "Rollout [22]", "IO": "IO [23]", "ADAPT": "ADAPT [24]"}
    subtitles = {"Proposed": "Shadow prices + stochastic DP", "Weight Greedy": "Min cost + max weight",
        "Mean-workload Greedy": "Min cost + max weight/cost", "Mean-workload DP": "Shadow prices + mean DP",
        "Distribution-aware Myopic": "Min cost + max expected reward", "Nearest Neighbor": "Min cost + nearest task",
        "IO": "Min cost + iterative orienteering", "Rollout": "Min cost + sampled rollout",
        "ADAPT": "Min cost + adaptive IACS"}
    from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
    height = 5.25 + max(0, len(methods)-6)*.58
    fig, ax = plt.subplots(figsize=(7.2, height))
    fig.subplots_adjust(left=.29, right=.985, bottom=.76/height, top=1-.315/height)
    for j, method in enumerate(methods):
        values = frame[frame.method.eq(method)].sort_values("seed").wcr.to_numpy()*100
        row = summary[summary.method.eq(method)].iloc[0]
        color = COLORS.get(method, f"C{j % 10}")
        jitter = np.random.default_rng(BOOTSTRAP_SEED+jitter_indices[method]).uniform(-.065, .065, len(values))
        ax.scatter(values, j+jitter, s=9, color=color, alpha=.05, linewidths=0)
        ax.errorbar(row["mean"], j, xerr=[[row["mean"]-row.ci_low], [row.ci_high-row["mean"]]],
                    fmt=MARKERS.get(method, "o"), color=color, capsize=4.8, elinewidth=1.35,
                    markersize=4.5 if method == "Mean-workload DP" else 6)
        ax.annotate(f'{row["mean"]:.1f}%', (row["mean"], j), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=12.5)
        block = VPacker(children=[TextArea(labels.get(method, method),
            textprops={"fontsize":12.0 if method in labels else 11.3}),
            TextArea(subtitles.get(method, ""), textprops={"fontsize":8.6, "color":"#555555"})],
            align="center", pad=0, sep=2.8)
        ax.add_artist(AnnotationBbox(block, (-.215, j), xycoords=ax.get_yaxis_transform(),
                      box_alignment=(.5,.5), frameon=False, annotation_clip=False))
    ax.set_xlim(0, 60)
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_ylim(len(methods)-.45, -1.65)
    ax.set_yticks([])
    ax.set_xlabel("WCR (%)")
    style(ax)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="y", visible=False)
    n = int(summary.n.iloc[0])
    ax.text(.29, .979, f"Same workload distributions ({n} samples)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.2)
    ax.text(.29, .93, "T = 2200 s; E = 460 kJ", transform=ax.transAxes, ha="center", va="top", fontsize=10.2)
    ax.text(.79, .97, "Mean and 95%\nconfidence interval", transform=ax.transAxes,
            ha="center", va="top", fontsize=9)
    return fig


def sweep(frame, study):
    summary, methods = summarize(frame)
    mec = study == "Figure11"
    if mec:
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.6))
        fig.subplots_adjust(left=.125, right=.985, bottom=.085, top=.84, hspace=.42)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.25))
        fig.subplots_adjust(left=.11, right=.985, bottom=.145, top=.78 if len(methods)>6 else .83)
    plot_series(ax, summary, methods)
    fit_wcr(ax, summary, minimum=20 if study in ("Figure9", "Figure11") else 0,
            maximum=100 if study == "Figure10" else 80 if study in ("Figure9", "Figure11") else 60)
    style(ax)
    if study in ("Figure5", "Figure9"):
        legend(fig, ax, methods,
               display_order=(*METHODS[:-3], "Rollout", "IO", "ADAPT"),
               display_labels={"Rollout": "Rollout [22]", "IO": "IO [23]", "ADAPT": "ADAPT [24]"})
    else:
        legend(fig, ax, methods)
    labels = {"Figure5": r"Mission time budget $T$ (s)", "Figure6": r"Onboard energy budget $E$ (kJ)",
              "Figure9": "Workload uncertainty multiplier", "Figure10": "Workload multiplier",
              "Figure11": r"MEC CPU frequency $F_m$ (GHz)"}
    points = sorted(summary.point.unique())
    if len(points) == 1:
        pad = max(.1, abs(float(points[0]))*.05)
        ax.set_xlim(float(points[0])-pad, float(points[0])+pad)
    if mec:
        ax.set_title("(a) Completion performance", fontsize=14, pad=12)
        require_columns(frame, ("local_actions", "mec_actions"), "MEC mode results")
        selected = frame[frame.method.eq("Proposed")]
        if selected.empty:
            raise ValueError("The MEC execution-mode panel requires Proposed outcomes")
        counts = selected.groupby("point")[["local_actions", "mec_actions"]].sum().sort_index()
        total = counts.sum(axis=1)
        if (counts < 0).any().any() or not np.isfinite(counts).all().all() or (total <= 0).any():
            raise ValueError("MEC mode shares require finite nonnegative counts and completed tasks")
        for name, column, color, marker in (("UAV local", "local_actions", "#4F7CC7", "^"),
                                            ("MEC", "mec_actions", "#F68C28", "D")):
            axes[1].plot(counts.index, 100*counts[column]/total, color=color, marker=marker,
                         markersize=7.3, linewidth=2, label=name)
        for boundary in (1., 2.5):
            axes[1].axvline(boundary, color="#777777", linewidth=1.1, linestyle=(0,(4,3)))
        axes[1].text(1.75, 81, "UAV local CPU range\n1.0-2.5 GHz", ha="center", fontsize=11.3, color="#555555")
        axes[1].set_ylim(-4, 108)
        axes[1].set_yticks([0,25,50,75,100])
        axes[1].set_ylabel("Execution share (%)")
        axes[1].set_xlabel(labels[study])
        axes[1].set_title("(b) Execution modes of Proposed", fontsize=14, pad=12)
        axes[1].legend(frameon=False, fontsize=11.3, loc="center right")
        style(axes[1])
    else:
        ax.set_xlabel(labels[study])
    return fig


def ablation(frame):
    require_columns(frame, ("setting", "variant_id", "method", "point", "t_max_s", "e_max_kj"),
                    "Ablation results")
    unknown = set(frame.setting)-set(SETTINGS)
    if unknown:
        raise ValueError(f"Unsupported Figure 7 settings: {sorted(map(str, unknown))}. "
                         "The current paper uses lower_time (1800 s), not larger_time (2600 s). "
                         "Rerun Figure7/experiment.py in a new --output-dir.")
    if set(frame.variant_id)-set(VARIANTS):
        raise ValueError("Unknown Figure 7 ablation variant")
    if not frame.method.eq(frame.variant_id.map(ABLATION_VARIANTS)).all():
        raise ValueError("Figure 7 method and variant_id do not match")
    for point, (setting, time_budget, energy_budget) in enumerate(ABLATION_SETTINGS):
        selected = frame[frame.setting.eq(setting)]
        for column, expected in (("point", point), ("t_max_s", time_budget),
                                 ("e_max_kj", energy_budget)):
            values = pd.to_numeric(selected[column], errors="coerce").to_numpy()
            if not np.isfinite(values).all() or not np.allclose(values, expected, rtol=0, atol=1e-9):
                raise ValueError(f"Figure 7 {setting} requires {column}={expected:g}; "
                                 "use newly simulated results rather than relabeling old data.")
    summary, variants = summarize(frame, key="setting", series="variant_id")
    settings = [s for s in SETTINGS if s in set(frame.setting)]
    fig, ax = plt.subplots(figsize=(9.6, 4.95))
    fig.subplots_adjust(left=.075, right=.99, bottom=.18, top=.79)
    x = np.arange(len(variants))*1.08
    hatches = dict(zip(SETTINGS, ("", "///", "\\\\\\")))
    shades = dict(zip(SETTINGS, (1., .68, .45)))
    colors = dict(zip(VARIANTS, ("#EE2922", "#8673B5", "#4F7CC7", "#4F7CC7", "#4F7CC7")))
    for j, setting in enumerate(settings):
        q = summary[summary.point.eq(setting)].set_index("method").loc[variants]
        pos = x+(j-(len(settings)-1)/2)*.235
        shade = shades[setting]
        ax.bar(pos, q["mean"], width=.21,
               color=[tuple(shade*c+1-shade for c in to_rgb(colors[v])) for v in variants],
               hatch=hatches[setting], edgecolor="#686868", linewidth=.5)
        ax.errorbar(pos, q["mean"], yerr=[q["mean"]-q.ci_low, q.ci_high-q["mean"]],
                    fmt="none", color="#303030", capsize=3.4, elinewidth=1.2)
        for xi, mean, hi in zip(pos, q["mean"], q.ci_high):
            ax.text(xi, hi+.9, f"{mean:.2f}", ha="center", fontsize=10.5)
    fit_wcr(ax, summary)
    ax.set_ylim(0, max(60, 10*np.ceil((summary.ci_high.max()+5)/10)))
    labels = dict(zip(VARIANTS, VARIANT_LABELS))
    ax.set_xticks(x, [labels[v] for v in variants])
    ax.tick_params(axis="x", length=0, pad=10, labelsize=14)
    style(ax)
    setting_labels = {"default":"Default", "lower_time":"Lower T", "lower_energy":"Lower E"}
    texts = {name: f"{setting_labels[name]}\nT = {t:g} s, E = {e:g} kJ"
             for name, t, e in ABLATION_SETTINGS}
    fig.legend(handles=[Patch(facecolor="#BBBBBB", edgecolor="#686868", hatch=hatches[s], label=texts[s]) for s in settings],
               loc="upper center", ncol=len(settings), frameon=False, fontsize=12.5, bbox_to_anchor=(.535,1.))
    return fig


def timing_summary(path):
    rows = pd.read_csv(path)
    require_columns(rows, ("seed", "method", "L"), "Measured timing")
    rows = rows.rename(columns={"total_decision_ms":"task_decision_total_ms", "n_decisions":"task_decision_count"})
    rows["method"] = rows.method.replace(ALIASES)
    totals = "task_decision_total_ms"
    counts = "task_decision_count"
    if totals not in rows or counts not in rows:
        require_columns(rows, ("decision_ms",), "Measured decision events")
        # Keep the terminal Top call: the current timing study includes it.
        rows = rows.groupby(["seed", "L", "method"]).decision_ms.agg(["sum", "count"]).reset_index().rename(columns={"sum":totals,"count":counts})
    points, methods, seeds, elapsed = paired_values(rows, "L", value=totals)
    _, _, _, number = paired_values(rows, "L", value=counts)
    if (elapsed <= 0).any() or (number <= 0).any():
        raise ValueError("Measured timing totals and decision counts must be positive")
    # Each mission has equal weight, matching measure_timing.py. Pooling all
    # decisions would give longer routes more influence and change the metric.
    mission_means = elapsed/number
    mean = mission_means.mean(axis=0)
    low, high = np.empty_like(mean), np.empty_like(mean)
    for i, cap in enumerate(points):
        rng = np.random.default_rng(np.random.SeedSequence([460, 2200, int(cap)]))
        draws = np.empty((BOOTSTRAPS, len(methods)))
        for start in range(0, BOOTSTRAPS, 100):
            indices = rng.integers(0, len(seeds), size=(100, len(seeds)))
            draws[start:start+100] = mission_means[indices,i,:].mean(axis=1)
        low[i], high[i] = np.quantile(draws, [.025,.975], axis=0)
    return pd.DataFrame([dict(point=p,method=m,mean=mean[i,j],ci_low=min(low[i,j],mean[i,j]),ci_high=max(high[i,j],mean[i,j]))
                         for i,p in enumerate(points) for j,m in enumerate(methods)]), methods


def candidate_cap(frame, timing_path):
    require_columns(frame, ("L",), "Candidate-cap performance")
    summary, methods = summarize(frame, key="L")
    if set(methods)-{"Proposed", "Mean-workload DP"}:
        raise ValueError("Candidate-cap study expects Proposed and Mean-workload DP")
    has_timing = timing_path is not None
    fig, axes = plt.subplots(2 if has_timing else 1, 1, figsize=(6.8,8.6 if has_timing else 4.7), squeeze=False)
    fig.subplots_adjust(left=.14, right=.985, bottom=.08 if has_timing else .17, top=.855, hspace=.46)
    ax = axes[0,0]
    plot_series(ax, summary, methods)
    fit_wcr(ax, summary, maximum=5*np.ceil((summary.ci_high.max()+1)/5))
    ax.set_title("(a) Completion performance" if has_timing else "Completion performance", fontsize=14, pad=16)
    legend(fig, ax, methods)
    if has_timing:
        timing, timed_methods = timing_summary(timing_path)
        if set(timed_methods) != set(methods) or set(timing.point) != set(summary.point):
            raise ValueError("Timing and performance must cover the same methods and candidate caps")
        plot_series(axes[1,0], timing, methods)
        axes[1,0].set_yscale("log")
        axes[1,0].set_ylabel("Mean decision time (ms)")
        axes[1,0].set_title("(b) Computation time", fontsize=14, pad=16)
    else:
        print("No measured timing.csv supplied; drawing completion performance only.")
    for ax in axes[:,0]:
        ax.set_xticks(sorted(summary.point.unique()))
        ax.set_xlabel("Candidate cap $L$")
        style(ax)
    return fig


def priority_table(frame):
    from scipy.stats import t
    field = "group_id" if "group_id" in frame and frame.group_id.notna().all() else "priority_config"
    require_columns(frame, (field,), "Priority-configuration results")
    if frame[field].isna().any() or frame.duplicated([field,"method","seed"]).any():
        raise ValueError("Missing priority group or duplicated group/method/seed")
    methods = [m for m in METHODS if m in set(frame.method)]
    if not methods or set(frame.method)-set(methods):
        raise ValueError("Unknown priority-robustness method")
    for _, group in frame.groupby(field):
        expected = {(m,s) for m in methods for s in group.seed.unique()}
        if set(zip(group.method,group.seed)) != expected:
            raise ValueError("Priority configurations need paired seeds for all methods")
    cells = []
    for method in methods:
        selected = frame[frame.method.eq(method)]
        groups = selected.groupby(field).wcr.mean()*100
        mean = groups.mean()
        if len(groups) < 2:
            interval = "Unavailable (1 configuration)"
        else:
            margin = t.ppf(.975,len(groups)-1)*groups.std(ddof=1)/np.sqrt(len(groups))
            interval = f"[{mean-margin:.2f}, {mean+margin:.2f}]"
        cells.append([LABELS.get(method,method), f"{mean:.2f}", interval])
    fig, ax = plt.subplots(figsize=(8.8,.55+len(cells)*.42))
    ax.axis("off")
    table = ax.table(cellText=cells,colLabels=["Method","Mean WCR (%)","95% CI (%)"],cellLoc="center",loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1,1.6)
    for (row,_), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_linewidth(.5)
        if row == 0:
            cell.set_text_props(weight="bold")
    print(f"Priority table: {frame[field].nunique()} configurations; {len(frame)} mission outcomes.")
    return fig


def plain_table(headers, cells, width=10.8):
    fig, ax = plt.subplots(figsize=(width, .55+len(cells)*.42))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_linewidth(.5)
        if row == 0:
            cell.set_text_props(weight="bold")
    return fig


def execution_ablation(frame):
    require_columns(frame, ("setting",), "Execution ablation")
    points, methods, _, values = paired_values(frame, key="setting")
    expected = ("Proposed", "Normalized resource cost")
    if set(methods) != set(expected):
        raise ValueError("Execution ablation requires paired Proposed and Normalized resource cost outcomes")
    means = values.mean(axis=0)*100
    difference = (values[:,:,methods.index(expected[0])]-values[:,:,methods.index(expected[1])])*100
    delta, low, high = bootstrap(difference)
    budgets = {"default":"2200 / 460", "lower_energy":"2200 / 380", "larger_time":"2600 / 460"}
    cells = [[budgets.get(p, p), f"{means[i,methods.index(expected[0])]:.2f}",
              f"{means[i,methods.index(expected[1])]:.2f}", f"{delta[i]:+.3f} [{low[i]:+.3f}, {high[i]:+.3f}]"]
             for i,p in enumerate(points)]
    return plain_table(["T (s) / E (kJ)", "Shadow WCR (%)", "Normalized WCR (%)", "Difference (pp), 95% CI"], cells)


def numerical_accuracy(frame):
    frame = frame.copy()
    frame["point"] = 1
    _, variants, _, values = paired_values(frame, series="variant_id")
    if "default" not in variants:
        raise ValueError("Numerical accuracy requires the default variant for paired differences")
    values = values[:,0,:]*100
    changes = values-values[:,[variants.index("default")]]
    delta, low, high = bootstrap(changes)
    means = values.mean(axis=0)
    order = ("default", "dp_g1", "dp_g5", "support_h3", "support_h15", "time_grid50", "energy_grid15")
    labels = {"default":"Default", "dp_g1":"G = 1", "dp_g5":"G = 5", "support_h3":"H = 3",
              "support_h15":"H = 15", "time_grid50":"Time grid = 50 s", "energy_grid15":"Energy grid = 15 kJ"}
    if set(variants)-set(order):
        raise ValueError("Unknown numerical-accuracy configuration")
    cells = []
    for variant in order:
        if variant not in variants:
            continue
        i = variants.index(variant)
        cells.append([labels[variant], f"{means[i]:.2f}", f"{delta[i]:+.3f}", f"[{low[i]:+.3f}, {high[i]:+.3f}]"])
    return plain_table(["Configuration", "Mean WCR (%)", "Paired change (pp)", "95% paired CI (pp)"], cells)


def main(study, directory, *, default_input=None, default_output=None):
    directory = Path(directory)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--results", dest="input", type=Path, default=default_input or directory/"data"/"results.csv")
    parser.add_argument("--output", type=Path, default=default_output or directory/(directory.name.lower()+".pdf"))
    parser.add_argument("--timing", type=Path, help="Figure8 measured mission timing or decision-event CSV")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"No local results: {args.input}. Run this directory's experiment.py first or pass --input.")
    if args.timing is not None and not args.timing.is_file():
        parser.error(f"Timing CSV does not exist: {args.timing}")
    configure()
    frame = load_results(args.input)
    if study == "Figure4":
        fig = comparison(frame)
    elif study in ("Figure5","Figure6","Figure9","Figure10","Figure11"):
        fig = sweep(frame, study)
    elif study == "Figure7":
        fig = ablation(frame)
    elif study == "Figure8":
        timing = args.timing
        if timing is None and args.input.with_name("timing.csv").is_file():
            timing = args.input.with_name("timing.csv")
        fig = candidate_cap(frame, timing)
    elif study == "Table4":
        fig = priority_table(frame)
    elif study == "ExecutionAblation":
        fig = execution_ablation(frame)
    elif study == "NumericalAccuracy":
        fig = numerical_accuracy(frame)
    else:
        raise ValueError(f"Unknown study: {study}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    print(f"Saved {args.output.resolve()} from {len(frame)} locally generated mission outcomes.")
