#!/usr/bin/env python3
"""Redraw Figure 9 from the original, embedded result summaries.

Requires Python 3, NumPy and Matplotlib. Run: python plot.py
Writes figure9.pdf beside this script. No simulations or external data are used.
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

OUTPUT = Path(__file__).resolve().with_name("figure9.pdf")
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
0.0,Proposed,300,32.857142857142854,32.857142857142854,32.857142857142854
0.0,Weight Greedy,300,32.142857142857146,32.142857142857146,32.142857142857146
0.0,Mean-workload Greedy,300,32.142857142857146,32.142857142857146,32.142857142857146
0.0,Mean-workload DP,300,32.857142857142854,32.857142857142854,32.857142857142854
0.0,Distribution-aware Myopic,300,32.142857142857146,32.142857142857146,32.142857142857146
0.0,Nearest Neighbor,300,22.857142857142858,22.857142857142858,22.857142857142858
0.125,Proposed,300,34.5119047619046,34.214285714285566,34.811904761904636
0.125,Weight Greedy,300,32.10714285714275,32.035714285714164,32.142857142857025
0.125,Mean-workload Greedy,300,34.04523809523801,33.77142857142849,34.31904761904753
0.125,Mean-workload DP,300,30.86666666666661,30.359523809523765,31.35238095238088
0.125,Distribution-aware Myopic,300,34.16428571428563,33.895238095238014,34.43333333333325
0.125,Nearest Neighbor,300,23.83571428571424,23.499999999999943,24.207142857142813
0.25,Proposed,300,35.209523809523716,34.78809523809514,35.62857142857133
0.25,Weight Greedy,300,31.754761904761793,31.433333333333223,32.06190476190465
0.25,Mean-workload Greedy,300,34.338095238095164,33.961904761904684,34.711904761904684
0.25,Mean-workload DP,300,30.0095238095238,29.447619047619046,30.557142857142832
0.25,Distribution-aware Myopic,300,26.37142857142858,26.18571428571428,26.571428571428584
0.25,Nearest Neighbor,300,24.559523809523792,24.102380952380926,25.033333333333324
0.375,Proposed,300,35.72619047619039,35.17619047619041,36.28571428571421
0.375,Weight Greedy,300,31.540476190476085,30.945238095237993,32.13333333333323
0.375,Mean-workload Greedy,300,34.36428571428565,33.83809523809519,34.87619047619042
0.375,Mean-workload DP,300,30.354761904761908,29.7952380952381,30.907142857142855
0.375,Distribution-aware Myopic,300,26.930952380952384,26.64761904761905,27.223809523809493
0.375,Nearest Neighbor,300,24.780952380952385,24.238095238095237,25.33571428571429
0.5,Proposed,300,36.661904761904665,36.02380952380943,37.297619047618966
0.5,Weight Greedy,300,32.52380952380944,31.79279761904753,33.261904761904695
0.5,Mean-workload Greedy,300,34.69047619047613,34.033333333333296,35.33339285714283
0.5,Mean-workload DP,300,30.98809523809526,30.41184523809522,31.545238095238098
0.5,Distribution-aware Myopic,300,27.385714285714275,26.990476190476166,27.785714285714256
0.5,Nearest Neighbor,300,25.166666666666682,24.566607142857137,25.76904761904764
0.625,Proposed,300,38.3261904761904,37.57857142857137,39.07857142857135
0.625,Weight Greedy,300,33.73571428571423,32.868988095238024,34.59761904761899
0.625,Mean-workload Greedy,300,35.159523809523755,34.37380952380948,35.926190476190406
0.625,Mean-workload DP,300,31.25000000000002,30.599940476190433,31.900000000000013
0.625,Distribution-aware Myopic,300,27.719047619047586,27.269047619047594,28.173809523809485
0.625,Nearest Neighbor,300,26.045238095238123,25.35714285714288,26.735714285714312
0.75,Proposed,300,40.197619047618964,39.373749999999895,41.033333333333225
0.75,Weight Greedy,300,34.78095238095232,33.82380952380948,35.747619047619025
0.75,Mean-workload Greedy,300,35.97380952380951,35.07142857142854,36.8690476190476
0.75,Mean-workload DP,300,32.4,31.71904761904764,33.08101190476192
0.75,Distribution-aware Myopic,300,28.028571428571382,27.509523809523767,28.549999999999947
0.75,Nearest Neighbor,300,26.585714285714314,25.861904761904793,27.300000000000036
0.875,Proposed,300,42.30952380952374,41.39285714285709,43.23333333333325
0.875,Weight Greedy,300,36.37857142857136,35.29523809523803,37.45952380952377
0.875,Mean-workload Greedy,300,36.65714285714282,35.66422619047615,37.63095238095235
0.875,Mean-workload DP,300,33.25714285714285,32.538095238095245,33.98095238095241
0.875,Distribution-aware Myopic,300,28.75714285714281,28.126190476190438,29.385714285714233
0.875,Nearest Neighbor,300,27.566666666666688,26.80714285714289,28.328630952380983
1.0,Proposed,300,44.08095238095236,43.104761904761865,45.06904761904758
1.0,Weight Greedy,300,38.097619047619,36.94523809523804,39.23815476190473
1.0,Mean-workload Greedy,300,37.80714285714283,36.71184523809519,38.89047619047616
1.0,Mean-workload DP,300,35.03333333333331,34.27857142857143,35.80482142857145
1.0,Distribution-aware Myopic,300,29.43809523809518,28.685714285714226,30.19761904761898
1.0,Nearest Neighbor,300,28.102380952380972,27.330892857142857,28.87380952380954
1.125,Proposed,300,46.34047619047613,45.280952380952364,47.41196428571428
1.125,Weight Greedy,300,39.52619047619042,38.2833333333333,40.78809523809521
1.125,Mean-workload Greedy,300,38.69999999999998,37.54279761904759,39.859523809523786
1.125,Mean-workload DP,300,36.50476190476187,35.69523809523811,37.323809523809516
1.125,Distribution-aware Myopic,300,30.56428571428565,29.76666666666663,31.359523809523722
1.125,Nearest Neighbor,300,28.90714285714287,28.11428571428573,29.704761904761934
1.25,Proposed,300,48.62142857142857,47.55714285714286,49.68095238095242
1.25,Weight Greedy,300,41.352380952380905,40.02136904761899,42.67380952380947
1.25,Mean-workload Greedy,300,40.23333333333331,39.00232142857138,41.45476190476187
1.25,Mean-workload DP,300,37.88333333333332,36.99285714285714,38.77380952380947
1.25,Distribution-aware Myopic,300,42.68571428571427,41.40476190476188,43.94047619047617
1.25,Nearest Neighbor,300,29.714285714285726,28.921369047619056,30.51666666666669
1.375,Proposed,300,50.759523809523834,49.66904761904761,51.84285714285711
1.375,Weight Greedy,300,43.419047619047575,42.054702380952314,44.778571428571404
1.375,Mean-workload Greedy,300,41.776190476190465,40.49999999999997,43.02380952380949
1.375,Mean-workload DP,300,39.67142857142855,38.74999999999999,40.599999999999945
1.375,Distribution-aware Myopic,300,44.47619047619044,43.185714285714255,45.72380952380951
1.375,Nearest Neighbor,300,30.871428571428584,30.02380952380954,31.714345238095266
1.5,Proposed,300,52.9214285714286,51.800000000000004,54.05714285714285
1.5,Weight Greedy,300,45.390476190476136,44.01666666666662,46.75238095238092
1.5,Mean-workload Greedy,300,43.31428571428569,41.99041666666663,44.60476190476186
1.5,Mean-workload DP,300,41.22619047619042,40.264285714285705,42.19291666666663
1.5,Distribution-aware Myopic,300,46.13809523809522,44.828571428571394,47.42142857142857
1.5,Nearest Neighbor,300,31.854761904761908,30.97142857142861,32.750000000000014
1.625,Proposed,300,55.214285714285694,54.10000000000002,56.32380952380953
1.625,Weight Greedy,300,47.65714285714284,46.21184523809519,49.07142857142857
1.625,Mean-workload Greedy,300,44.73333333333329,43.40714285714283,46.02624999999999
1.625,Mean-workload DP,300,42.58809523809518,41.61190476190468,43.56904761904754
1.625,Distribution-aware Myopic,300,47.70714285714286,46.361904761904746,49.009523809523856
1.625,Nearest Neighbor,300,33.423809523809524,32.4809523809524,34.37857142857143
1.75,Proposed,300,57.88809523809526,56.742857142857126,59.033333333333324
1.75,Weight Greedy,300,49.135714285714286,47.70952380952379,50.523869047619044
1.75,Mean-workload Greedy,300,46.37142857142856,45.00714285714283,47.71434523809524
1.75,Mean-workload DP,300,44.02380952380946,43.0428571428571,45.01904761904754
1.75,Distribution-aware Myopic,300,49.24523809523815,47.93089285714288,50.53339285714294
1.75,Nearest Neighbor,300,34.74285714285713,33.747619047619054,35.7404761904762
1.875,Proposed,300,60.54761904761904,59.40476190476184,61.685714285714276
1.875,Weight Greedy,300,51.18095238095239,49.788095238095245,52.554821428571465
1.875,Mean-workload Greedy,300,48.116666666666674,46.742857142857154,49.457202380952424
1.875,Mean-workload DP,300,45.288095238095174,44.297619047618966,46.29523809523806
1.875,Distribution-aware Myopic,300,50.785714285714356,49.50946428571432,52.038095238095295
1.875,Nearest Neighbor,300,36.37857142857142,35.35476190476193,37.42857142857143
2.0,Proposed,300,62.75714285714281,61.595178571428534,63.91672619047611
2.0,Weight Greedy,300,52.84761904761909,51.452380952380985,54.19767857142862
2.0,Mean-workload Greedy,300,49.74285714285716,48.357083333333335,51.10476190476192
2.0,Mean-workload DP,300,46.799999999999976,45.86428571428569,47.75244047619047
2.0,Distribution-aware Myopic,300,52.09761904761915,50.84047619047626,53.32380952380959
2.0,Nearest Neighbor,300,38.09999999999999,36.99994047619046,39.216726190476194
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
    ax.set_xlim(-0.05, 2.05)
    major = np.arange(0, 2.001, .25)
    ax.set_xticks(major, [f"{p:g}" for p in major])
    ax.set_xticks([p for p in POINTS if p not in major], minor=True)
    ax.set_xlabel("Workload uncertainty multiplier", labelpad=5)
    ax.set_ylabel("WCR (%)", labelpad=5)
    ax.set_ylim(20, 80)
    ax.set_yticks(np.arange(20, 81, 10))
    require(low.min() >= 20 and high.max() < 65, "Data encroaches on the internal annotations")
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
    require(legend_bounds.y0 > ax.transData.transform((0, high.max()))[1] + 5,
            "Legend overlaps the data")
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def main():
    global POINTS
    POINTS, wcr = load_summary(WCR_CSV, "point", "method", METHODS)
    draw(wcr)
    print(OUTPUT.name)


if __name__ == "__main__":
    main()
