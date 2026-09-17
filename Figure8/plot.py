#!/usr/bin/env python3
"""Redraw Figure 8 using the original experiment results embedded below."""
# Dependencies: numpy, matplotlib.
from pathlib import Path
import csv
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT = Path(__file__).resolve().with_name("figure8.pdf")

# Prefer the publication font; use a portable serif font when unavailable.
try:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    FONT = "Times New Roman"
except ValueError:
    FONT = "DejaVu Serif"


def require(condition, message):
    if not condition:
        raise ValueError(message)


METHODS = ("Proposed", "Mean-workload DP")
CAPS = tuple(range(1, 12))
COLORS = {"Proposed": "#EE2922", "Mean-workload DP": "#8673B5"}
MARKERS = {"Proposed": "o", "Mean-workload DP": "s"}
MARKER_SIZES = {"Proposed": 7.3, "Mean-workload DP": 4.5}

# Original WCR means and pointwise 95% bootstrap confidence intervals (%).
WCR_CSV = """method,L,mean_wcr_percent,lower_95_wcr_percent,upper_95_wcr_percent
Proposed,1,40.05952380952375,38.91190476190472,41.192857142857115
Mean-workload DP,1,37.533333333333296,36.41428571428568,38.628571428571405
Proposed,2,40.68809523809522,39.749999999999986,41.6380952380952
Mean-workload DP,2,38.09761904761901,37.14523809523806,39.052380952380894
Proposed,3,41.63095238095239,40.6880357142857,42.57380952380953
Mean-workload DP,3,40.414285714285654,39.43571428571424,41.397619047619
Proposed,4,43.226190476190425,42.27142857142854,44.188095238095194
Mean-workload DP,4,39.13095238095233,38.24999999999997,40.030952380952314
Proposed,5,43.392857142857096,42.447619047619014,44.34761904761894
Mean-workload DP,5,39.0904761904761,38.21428571428565,39.97380952380946
Proposed,6,43.39999999999995,42.43571428571426,44.36904761904755
Mean-workload DP,6,35.85476190476187,35.04523809523805,36.68095238095236
Proposed,7,43.62142857142851,42.66904761904757,44.578571428571415
Mean-workload DP,7,35.21190476190473,34.44047619047618,35.98571428571427
Proposed,8,44.040476190476156,43.08333333333328,45.004821428571425
Mean-workload DP,8,34.96190476190472,34.211904761904755,35.719047619047586
Proposed,9,44.08095238095236,43.104761904761865,45.06904761904758
Mean-workload DP,9,35.03333333333331,34.27857142857143,35.80482142857145
Proposed,10,44.14523809523805,43.16428571428565,45.14047619047615
Mean-workload DP,10,34.5428571428571,33.76428571428573,35.328571428571415
Proposed,11,44.14523809523805,43.16428571428565,45.14047619047615
Mean-workload DP,11,34.44761904761901,33.66428571428571,35.240476190476194
"""

# Measured decision-time means and pointwise 95% bootstrap confidence intervals (ms).
TIMING_CSV = """method,L,mean_decision_ms,lower_95_mean_ms,upper_95_mean_ms
Proposed,1,0.2819849325153374,0.27033808089491757,0.29326593950803215
Mean-workload DP,1,0.27414637599999997,0.2650698559196499,0.28363915360817576
Proposed,2,0.2794135174418604,0.2682796786412609,0.29132074870201324
Mean-workload DP,2,0.2842253703703704,0.27086632402666105,0.2982999202895894
Proposed,3,0.365560755319149,0.3493982487571023,0.38250866435307657
Mean-workload DP,3,0.31801601257861645,0.3069446819444444,0.32933534506763157
Proposed,4,0.7948170579710147,0.7541683672820626,0.8334557540725472
Mean-workload DP,4,0.3986305755813954,0.3793902235052431,0.41917900512307354
Proposed,5,2.5539864396135266,2.3741832684572164,2.7304873766849083
Mean-workload DP,5,0.6711697919075144,0.6297964969780218,0.7133024830770449
Proposed,6,8.360635789719627,7.792912492779643,8.910986565834248
Mean-workload DP,6,1.5841174196891192,1.498763398236269,1.6723075402532794
Proposed,7,23.430320431818185,21.590244302918542,25.288434289354793
Mean-workload DP,7,3.8657081950000003,3.629351915899254,4.12104731504878
Proposed,8,61.96344900452487,57.647673974499014,66.35824480546988
Mean-workload DP,8,10.67622882653061,10.090258371949473,11.248210960828384
Proposed,9,185.27003859911895,170.23422271776752,200.61220937203953
Mean-workload DP,9,30.38008664361702,28.476766480492227,32.34872416169213
Proposed,10,539.3869942380952,501.8105453693231,577.5139777580025
Mean-workload DP,10,97.5089021128205,92.13028313198355,103.72654534299488
Proposed,11,1330.4904343766234,1240.6381206615008,1424.9188011394233
Mean-workload DP,11,260.36069808673477,246.67769435539768,275.96900396619543
"""

def add_method_legend(fig, colors):
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker
    from matplotlib.text import Text

    font = FontProperties(family=FONT, size=12.0)
    renderer = fig.canvas.get_renderer()
    cells = []
    for method in METHODS:
        width = renderer.get_text_width_height_descent(method, font, False)[0] * 72 / fig.dpi
        cell = DrawingArea(36 + width, 14, clip=False)
        cell.add_artist(Line2D([0, 26], [7, 7], color=colors[method], linewidth=2,
                              linestyle="-" if method == "Proposed" else "--"))
        cell.add_artist(Line2D([13], [7], color=colors[method], linestyle="none",
                              marker=MARKERS[method], markersize=MARKER_SIZES[method],
                              markeredgewidth=0.65))
        cell.add_artist(Text(36 + width / 2, 7, method, fontproperties=font,
                             ha="center", va="center"))
        cells.append(cell)
    box = AnchoredOffsetbox(loc="upper center",
                            child=HPacker(children=cells, pad=0, sep=25, align="center"),
                            bbox_to_anchor=(0.5625, 0.995), bbox_transform=fig.transFigure,
                            frameon=False, pad=0, borderpad=0)
    fig.add_artist(box)


def draw(wcr, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import LogLocator, LogFormatterMathtext

    font_path = font_manager.findfont(FONT)
    colors = COLORS
    require(colors == {"Proposed": "#EE2922", "Mean-workload DP": "#8673B5"},
            "Figure8 must preserve the approved palette")
    plt.rcParams.update({
        "font.family": FONT, "font.size": 13,
        "axes.labelsize": 15.5, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "mathtext.fontset": "custom", "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 8.6))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.08, top=0.855, hspace=0.46)
    series_metadata = []
    for metric, ax, data in zip(("wcr", "decision_time"), axes, (wcr, timing)):
        mean, lower, upper = data[:3]
        require(np.all(lower <= mean) and np.all(upper >= mean),
                "Confidence interval must contain its estimate")
        for m, method in enumerate(METHODS):
            ax.errorbar(CAPS, mean[:, m],
                        yerr=np.stack((mean[:, m] - lower[:, m], upper[:, m] - mean[:, m])),
                        color=colors[method], linestyle="-" if method == "Proposed" else "--",
                        marker=MARKERS[method], markersize=MARKER_SIZES[method],
                        markerfacecolor=colors[method], markeredgecolor=colors[method],
                        markeredgewidth=0.65, linewidth=2,
                        elinewidth=1.35, capsize=4.8, capthick=1,
                        zorder=5-m, clip_on=True)
            series_metadata.append(dict(metric=metric, method=method, L=list(CAPS),
                                        y=mean[:, m].tolist(), lower=lower[:, m].tolist(),
                                        upper=upper[:, m].tolist()))
        ax.set_xlim(0.6, 11.4)
        ax.set_xticks(CAPS)
        ax.set_xlabel("Candidate cap $L$", labelpad=5)
        ax.grid(True, color="#ECECEC", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.85)
            ax.spines[side].set_color("black")
        ax.tick_params(direction="out", length=3.8, width=0.8, pad=4)
    axes[0].set_ylim(30, 48)
    axes[0].set_yticks([30, 35, 40, 45])
    require(wcr[1].min() > 30 and wcr[2].max() < 48, "WCR bars outside the axes")
    axes[0].set_ylabel("WCR (%)", labelpad=5)
    axes[0].set_title("(a) Completion performance", fontsize=14, pad=26)
    axes[1].set_yscale("log")
    lower_limit = 10 ** (np.floor(np.log10(timing[1].min())) - 0.12)
    upper_limit = 10 ** (np.log10(timing[2].max()) + 0.16)
    axes[1].set_ylim(lower_limit, upper_limit)
    axes[1].yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axes[1].minorticks_off()
    axes[1].set_ylabel("Mean decision time (ms)", labelpad=5)
    axes[1].set_title("(b) Computation time", fontsize=14, pad=26)
    add_method_legend(fig, colors)
    fig.text(0.5625, 0.949, r"$T = 2200$ s; $E = 620$ kJ", ha="center", va="center",
             fontsize=10.2, color="#333333")
    paths = []
    for extension in ("pdf",):
        p = OUTPUT
        fig.savefig(p, dpi=400, bbox_inches="tight", pad_inches=0.06)
        paths.append(p)
    plt.close(fig)
    return paths, dict(colors=colors, marker_sizes_pt=MARKER_SIZES,
                       font_family=FONT, font_path=font_path,
                       figure_inches=[6.8, 8.6], layout="two vertically stacked panels",
                       series=series_metadata,
                       decision_time_axis="logarithmic", label_alignment="centered")


def read_summary(text, fields):
    rows = list(csv.DictReader(io.StringIO(text)))
    lookup = {(int(row["L"]), row["method"]): row for row in rows}
    return tuple(np.array([[float(lookup[cap, method][field]) for method in METHODS]
                           for cap in CAPS]) for field in fields)


def main():
    wcr = read_summary(WCR_CSV, ("mean_wcr_percent", "lower_95_wcr_percent", "upper_95_wcr_percent"))
    timing = read_summary(TIMING_CSV, ("mean_decision_ms", "lower_95_mean_ms", "upper_95_mean_ms"))
    draw(wcr, timing)


if __name__ == "__main__":
    main()
