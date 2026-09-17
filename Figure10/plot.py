#!/usr/bin/env python3
"""Redraw Figure 10 from the original, embedded result summaries.

Requires Python 3, NumPy and Matplotlib. Run: python plot.py
Writes figure10.pdf beside this script. No simulations or external data are used.
The embedded values and confidence intervals are preserved from the source tables.
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT = Path(__file__).resolve().with_name("figure10.pdf")
try:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    FONT_FAMILY = "Times New Roman"
except ValueError:
    FONT_FAMILY = "DejaVu Serif"


def require(condition, message):
    if not condition:
        raise ValueError(message)

METHODS = ("Proposed", "Weight Greedy", "Mean-workload Greedy", "Mean-workload DP",
           "Distribution-aware Myopic", "Nearest Neighbor")
MARKERS = dict(zip(METHODS, ("o", "^", "D", "s", "P", "v")))
SIZES = {m: 4.5 if m == "Mean-workload DP" else 7.3 for m in METHODS}
COLORS = dict(zip(METHODS, ("#EE2922", "#4F7CC7", "#F68C28", "#8673B5", "#4A7937", "#B2B2B2")))

# Original result summary; values are not recomputed.
WCR_CSV = """point,method,n,mean_wcr_percent,lower_95_percent,upper_95_percent
0.125,Proposed,300,84.71666666666657,84.21666666666658,85.21428571428551
0.125,Weight Greedy,300,68.19047619047603,67.837976190476,68.52380952380935
0.125,Mean-workload Greedy,300,77.28095238095258,76.6357142857145,77.92142857142878
0.125,Mean-workload DP,300,82.72142857142872,82.2095238095239,83.23095238095244
0.125,Distribution-aware Myopic,300,51.80714285714294,51.40000000000008,52.19761904761911
0.125,Nearest Neighbor,300,63.31666666666668,62.809523809523846,63.819047619047595
0.25,Proposed,300,72.94285714285708,72.26428571428562,73.6380952380952
0.25,Weight Greedy,300,62.80238095238094,61.933333333333366,63.609523809523814
0.25,Mean-workload Greedy,300,65.89761904761905,64.9809523809524,66.77857142857142
0.25,Mean-workload DP,300,70.58095238095231,69.84047619047608,71.32619047619043
0.25,Distribution-aware Myopic,300,47.62142857142857,47.195238095238096,48.030952380952385
0.25,Nearest Neighbor,300,52.91190476190478,52.02619047619046,53.77619047619051
0.375,Proposed,300,64.497619047619,63.659523809523755,65.34523809523799
0.375,Weight Greedy,300,56.64285714285723,55.676190476190556,57.607142857142904
0.375,Mean-workload Greedy,300,58.478571428571435,57.54994047619048,59.39285714285714
0.375,Mean-workload DP,300,61.63571428571426,60.74523809523814,62.52380952380955
0.375,Distribution-aware Myopic,300,44.507142857142895,43.91428571428574,45.0666666666667
0.375,Nearest Neighbor,300,43.31190476190464,42.47619047619035,44.173809523809396
0.5,Proposed,300,58.60714285714289,57.723809523809535,59.490476190476215
0.5,Weight Greedy,300,51.37380952380957,50.26660714285715,52.46904761904767
0.5,Mean-workload Greedy,300,51.607142857142875,50.52619047619049,52.66666666666667
0.5,Mean-workload DP,300,55.180952380952434,54.21666666666673,56.15714285714291
0.5,Distribution-aware Myopic,300,41.23571428571427,40.48089285714278,41.97380952380951
0.5,Nearest Neighbor,300,38.05238095238087,37.36190476190466,38.749999999999915
0.625,Proposed,300,53.900000000000105,52.95000000000003,54.857142857142904
0.625,Weight Greedy,300,46.828571428571415,45.67142857142858,47.9547619047619
0.625,Mean-workload Greedy,300,47.0023809523809,45.90952380952375,48.09761904761901
0.625,Mean-workload DP,300,49.09761904761902,48.1690476190476,50.03339285714284
0.625,Distribution-aware Myopic,300,36.985714285714224,36.145238095238035,37.81910714285709
0.625,Nearest Neighbor,300,34.39761904761903,33.74285714285711,35.03571428571423
0.75,Proposed,300,50.25952380952382,49.30238095238095,51.20714285714287
0.75,Weight Greedy,300,42.90952380952379,41.730952380952374,44.08333333333332
0.75,Mean-workload Greedy,300,43.24285714285715,42.17857142857139,44.297619047619044
0.75,Mean-workload DP,300,45.890476190476164,44.94279761904755,46.8523809523809
0.75,Distribution-aware Myopic,300,32.952380952380864,32.188095238095165,33.7142857142856
0.75,Nearest Neighbor,300,32.250000000000014,31.549999999999994,32.942857142857164
0.875,Proposed,300,46.96428571428567,45.99285714285711,47.94529761904763
0.875,Weight Greedy,300,40.292857142857095,39.10952380952377,41.485714285714245
0.875,Mean-workload Greedy,300,40.36666666666663,39.31898809523805,41.39285714285712
0.875,Mean-workload DP,300,40.4619047619047,39.54517857142848,41.38333333333327
0.875,Distribution-aware Myopic,300,31.25714285714278,30.521428571428494,31.985714285714177
0.875,Nearest Neighbor,300,30.12380952380954,29.3809523809524,30.859523809523843
1.0,Proposed,300,44.08095238095236,43.104761904761865,45.06904761904758
1.0,Weight Greedy,300,38.097619047619,36.94523809523804,39.23815476190473
1.0,Mean-workload Greedy,300,37.80714285714283,36.71184523809519,38.89047619047616
1.0,Mean-workload DP,300,35.03333333333331,34.27857142857143,35.80482142857145
1.0,Distribution-aware Myopic,300,29.43809523809518,28.685714285714226,30.19761904761898
1.0,Nearest Neighbor,300,28.102380952380972,27.330892857142857,28.87380952380954
1.125,Proposed,300,41.680952380952334,40.726190476190446,42.659583333333295
1.125,Weight Greedy,300,35.83095238095233,34.69047619047615,36.96666666666661
1.125,Mean-workload Greedy,300,35.4690476190476,34.40238095238093,36.51672619047618
1.125,Mean-workload DP,300,34.37142857142849,33.50946428571422,35.23571428571422
1.125,Distribution-aware Myopic,300,28.15476190476185,27.433273809523754,28.895238095238053
1.125,Nearest Neighbor,300,26.592857142857184,25.776130952380967,27.40952380952384
1.25,Proposed,300,39.77857142857137,38.802380952380894,40.773809523809454
1.25,Weight Greedy,300,34.27619047619041,33.166666666666636,35.404821428571395
1.25,Mean-workload Greedy,300,33.69047619047617,32.63571428571424,34.73095238095232
1.25,Mean-workload DP,300,32.29999999999994,31.538095238095202,33.06672619047616
1.25,Distribution-aware Myopic,300,26.819047619047602,26.076190476190465,27.576190476190444
1.25,Nearest Neighbor,300,24.45476190476194,23.628571428571462,25.280952380952428
1.375,Proposed,300,37.942857142857086,36.95714285714284,38.9309523809523
1.375,Weight Greedy,300,32.70952380952377,31.599999999999937,33.82380952380949
1.375,Mean-workload Greedy,300,31.75238095238091,30.718988095238053,32.77624999999999
1.375,Mean-workload DP,300,29.88095238095235,29.090476190476167,30.659523809523765
1.375,Distribution-aware Myopic,300,34.204761904761874,33.06428571428566,35.347619047619006
1.375,Nearest Neighbor,300,23.245238095238125,22.48333333333336,24.014285714285734
1.5,Proposed,300,36.39999999999995,35.47619047619045,37.361904761904704
1.5,Weight Greedy,300,31.14047619047615,30.042857142857095,32.23095238095232
1.5,Mean-workload Greedy,300,29.992857142857098,29.00714285714284,30.99047619047616
1.5,Mean-workload DP,300,28.549999999999976,27.828511904761864,29.2833333333333
1.5,Distribution-aware Myopic,300,32.983333333333306,31.876190476190448,34.090476190476124
1.5,Nearest Neighbor,300,22.064285714285745,21.309523809523846,22.82142857142859
1.625,Proposed,300,34.74999999999994,33.78333333333329,35.740535714285684
1.625,Weight Greedy,300,30.071428571428534,29.02142857142851,31.12619047619042
1.625,Mean-workload Greedy,300,28.64047619047615,27.69999999999998,29.578571428571394
1.625,Mean-workload DP,300,26.757142857142824,26.040476190476188,27.476249999999997
1.625,Distribution-aware Myopic,300,31.66904761904758,30.59285714285711,32.738095238095205
1.625,Nearest Neighbor,300,20.880952380952415,20.128571428571448,21.63809523809526
1.75,Proposed,300,33.47619047619046,32.549999999999976,34.40952380952376
1.75,Weight Greedy,300,28.961904761904723,27.940416666666586,29.992916666666602
1.75,Mean-workload Greedy,300,27.402380952380923,26.504761904761892,28.30952380952375
1.75,Mean-workload DP,300,25.761904761904763,25.08333333333333,26.440476190476186
1.75,Distribution-aware Myopic,300,30.245238095238058,29.21422619047615,31.29053571428568
1.75,Nearest Neighbor,300,19.74047619047622,19.028571428571453,20.46190476190478
1.875,Proposed,300,32.14285714285711,31.207142857142834,33.08571428571424
1.875,Weight Greedy,300,27.688095238095205,26.657142857142812,28.735714285714227
1.875,Mean-workload Greedy,300,25.9142857142857,25.02857142857143,26.81672619047618
1.875,Mean-workload DP,300,22.77142857142854,22.138095238095197,23.399999999999963
1.875,Distribution-aware Myopic,300,28.952380952380903,27.911904761904744,29.992857142857087
1.875,Nearest Neighbor,300,18.797619047619087,18.130952380952404,19.478571428571435
2.0,Proposed,300,31.097619047618995,30.15714285714286,32.02380952380952
2.0,Weight Greedy,300,26.773809523809483,25.75946428571424,27.792916666666624
2.0,Mean-workload Greedy,300,24.878571428571426,24.023809523809504,25.75952380952378
2.0,Mean-workload DP,300,22.907142857142848,22.25714285714286,23.550059523809523
2.0,Distribution-aware Myopic,300,27.99285714285711,26.96190476190476,29.028571428571393
2.0,Nearest Neighbor,300,18.092857142857167,17.46904761904764,18.72857142857144
"""

def load_summary(csv_text, point_field, series_field, series_names, mean_field="mean_wcr_percent"):
    rows = list(csv.DictReader(StringIO(csv_text)))
    points = tuple(sorted({float(row[point_field]) for row in rows}))
    lookup = {(float(row[point_field]), row[series_field]): row for row in rows}
    require(len(lookup) == len(rows), "Duplicate result rows")
    require(len(rows) == len(points) * len(series_names), "Incomplete result grid")
    matrices = tuple(np.array([[float(lookup[point, name][field]) for name in series_names]
                               for point in points])
                     for field in (mean_field, "lower_95_percent", "upper_95_percent"))
    return points, matrices

def legend(fig, ax, colors):
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker
    from matplotlib.text import Text
    font = FontProperties(family=FONT_FAMILY, size=11.3)
    renderer = fig.canvas.get_renderer()
    columns = []
    for names in (METHODS[:3], METHODS[3:]):
        width = max(renderer.get_text_width_height_descent(m, font, False)[0] for m in names)*72/fig.dpi
        cells = []
        for method in names:
            cell = DrawingArea(34+width, 12.5, clip=False)
            cell.add_artist(Line2D([0, 26], [6.25, 6.25], color=colors[method], linewidth=2,
                                  linestyle="-" if method == "Proposed" else "--"))
            cell.add_artist(Line2D([13], [6.25], color=colors[method], linestyle="none",
                                  marker=MARKERS[method], markersize=SIZES[method], markeredgewidth=0.65))
            cell.add_artist(Text(34+width/2, 6.25, method, fontproperties=font, ha="center", va="center"))
            cells.append(cell)
        columns.append(VPacker(children=cells, align="center", pad=0, sep=1.5))
    box = AnchoredOffsetbox(
        loc="upper center", child=HPacker(children=columns, align="center", pad=0, sep=18),
        bbox_to_anchor=(0.5, 0.986), bbox_transform=ax.transAxes,
        frameon=False, pad=0, borderpad=0.2)
    ax.add_artist(box)
    return box

def draw(wcr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    colors = COLORS
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 13,
                         "axes.labelsize": 15.5, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "mathtext.fontset": "custom", "mathtext.rm": FONT_FAMILY,
                         "mathtext.it": FONT_FAMILY + ":italic", "mathtext.bf": FONT_FAMILY + ":bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.2, 5.25))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.145, top=0.94)
    x = np.asarray(POINTS)
    mean, low, high = wcr
    require(np.all(low <= mean) and np.all(high >= mean), "CI excludes its mean")
    for m, method in enumerate(METHODS):
        plotted = ax.errorbar(x, mean[:, m], yerr=np.stack((mean[:, m]-low[:, m], high[:, m]-mean[:, m])),
                             color=colors[method], linestyle="-" if method == "Proposed" else "--",
                             marker=MARKERS[method], markersize=SIZES[method],
                             markerfacecolor=colors[method], markeredgecolor=colors[method],
                             markeredgewidth=0.65, linewidth=2, elinewidth=1.35, capsize=4.8, capthick=1,
                             zorder=10-m, clip_on=True)
        require(np.array_equal(plotted.lines[0].get_xdata(), x), "Missing real experiment point")
    # Padding keeps the measured endpoint markers and CI caps fully visible.
    ax.set_xlim(0, 2.05)
    major = np.arange(0, 2.001, .25)
    ax.set_xticks(major, [f"{p:g}" for p in major])
    ax.set_xticks([p for p in POINTS if p not in major], minor=True)
    ax.set_xlabel("Workload multiplier", labelpad=5)
    ax.set_ylabel("WCR (%)", labelpad=5)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    require(low.min() >= 0 and high.max() <= 100, "Invalid WCR confidence interval")
    ax.grid(True, color="#ECECEC", linewidth=.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(.85)
    ax.tick_params(direction="out", length=3.8, width=.8, pad=4)
    ax.tick_params(axis="x", which="minor", length=2.1, width=.65)
    legend_box = legend(fig, ax, colors)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    legend_bounds = legend_box.get_window_extent(renderer)
    require(axes_box.contains(legend_bounds.x0, legend_bounds.y0) and
            axes_box.contains(legend_bounds.x1, legend_bounds.y1), "Legend is outside the axes")
    from matplotlib.transforms import Bbox
    for m in range(len(METHODS)):
        for x_value, upper in zip(x, high[:, m]):
            px, py = ax.transData.transform((x_value, upper))
            require(not legend_bounds.overlaps(Bbox.from_extents(px-7, py-7, px+7, py+7)),
                    "Legend overlaps a measured confidence interval")
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def main():
    global POINTS
    POINTS, wcr = load_summary(WCR_CSV, "point", "method", METHODS)
    draw(wcr)
    print(OUTPUT.name)


if __name__ == "__main__":
    main()
