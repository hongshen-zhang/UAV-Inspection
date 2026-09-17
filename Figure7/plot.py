#!/usr/bin/env python3
"""Redraw Figure 7 using the original experiment results embedded below."""
# Dependencies: numpy, matplotlib.
from pathlib import Path
import csv
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT = Path(__file__).resolve().with_name("figure7.pdf")

# Prefer the publication font; use a portable serif font when unavailable.
try:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    FONT = "Times New Roman"
except ValueError:
    FONT = "DejaVu Serif"


def require(condition, message):
    if not condition:
        raise ValueError(message)


METHODS = ('Proposed', 'Weight Greedy', 'Mean-workload Greedy', 'Mean-workload DP', 'Distribution-aware Myopic', 'Nearest Neighbor')
COLORS = {'Proposed': '#EE2922', 'Weight Greedy': '#4F7CC7', 'Mean-workload Greedy': '#F68C28', 'Mean-workload DP': '#8673B5', 'Distribution-aware Myopic': '#4A7937', 'Nearest Neighbor': '#B2B2B2'}
FIGURE = 7
SETTINGS = {'budgets': [140.0, 200.0, 260.0, 300.0, 340.0, 380.0, 420.0, 460.0, 500.0, 560.0, 620.0, 680.0, 740.0, 800.0], 'xlabel': 'Onboard energy budget $E$ (kJ)', 'plot_note': 'T = 2200 s; n = 300'}
MAJOR_TICKS = {7: (140.0, 260.0, 380.0, 500.0, 620.0, 800.0)}

# Pointwise 95% bootstrap confidence intervals of mean WCR (fractions).
SUMMARY_CSV = """budget_value,method,mean_wcr,lower_95,upper_95
140.0,Proposed,0.12792857142857114,0.12359523809523792,0.13226190476190436
140.0,Weight Greedy,0.12242857142857133,0.11788095238095238,0.12707142857142847
140.0,Mean-workload Greedy,0.12228571428571401,0.11833333333333314,0.12611904761904733
140.0,Mean-workload DP,0.08878571428571415,0.08380952380952371,0.09369047619047606
140.0,Distribution-aware Myopic,0.12735714285714267,0.12288095238095215,0.13183333333333305
140.0,Nearest Neighbor,0.09476190476190464,0.08947619047619035,0.10016666666666654
200.0,Proposed,0.1741666666666661,0.16938095238095188,0.17895238095238028
200.0,Weight Greedy,0.13578571428571395,0.13057142857142837,0.1408339285714283
200.0,Mean-workload Greedy,0.1663095238095234,0.16238095238095213,0.17021428571428524
200.0,Mean-workload DP,0.15611904761904735,0.15169047619047582,0.16045238095238074
200.0,Distribution-aware Myopic,0.16988095238095216,0.1657142857142853,0.1740952380952376
200.0,Nearest Neighbor,0.14340476190476215,0.1389047619047621,0.14785714285714308
260.0,Proposed,0.21104761904761857,0.2058571428571424,0.2162619047619044
260.0,Weight Greedy,0.1844761904761904,0.177738095238095,0.19111904761904755
260.0,Mean-workload Greedy,0.1890952380952378,0.1842851190476186,0.1937857142857141
260.0,Mean-workload DP,0.18535714285714208,0.1808333333333326,0.18985714285714206
260.0,Distribution-aware Myopic,0.1996666666666664,0.19416666666666635,0.20509523809523786
260.0,Nearest Neighbor,0.18509523809523806,0.1803095238095239,0.1899285714285714
300.0,Proposed,0.2392380952380954,0.23328571428571418,0.2451666666666665
300.0,Weight Greedy,0.20459523809523816,0.19799999999999993,0.21104821428571435
300.0,Mean-workload Greedy,0.2154285714285717,0.2100714285714287,0.2207619047619049
300.0,Mean-workload DP,0.19640476190476147,0.19173809523809474,0.20102380952380902
300.0,Distribution-aware Myopic,0.23333333333333342,0.22738095238095235,0.23933333333333323
300.0,Nearest Neighbor,0.20323809523809522,0.19864285714285712,0.20778571428571416
340.0,Proposed,0.2758571428571426,0.26980952380952333,0.2819999999999995
340.0,Weight Greedy,0.2355714285714285,0.22864285714285695,0.2423333333333332
340.0,Mean-workload Greedy,0.24016666666666683,0.234952380952381,0.24533333333333338
340.0,Mean-workload DP,0.2373095238095237,0.2313333333333331,0.24330952380952345
340.0,Distribution-aware Myopic,0.2620714285714286,0.2560238095238097,0.26804761904761903
340.0,Nearest Neighbor,0.21428571428571422,0.2102857142857142,0.2180476190476189
380.0,Proposed,0.2984285714285712,0.29099999999999976,0.3058095238095234
380.0,Weight Greedy,0.2716190476190472,0.26321428571428546,0.2801666666666662
380.0,Mean-workload Greedy,0.2735476190476186,0.26699999999999996,0.28009523809523806
380.0,Mean-workload DP,0.26157142857142845,0.2554285714285713,0.2678095238095236
380.0,Distribution-aware Myopic,0.29540476190476156,0.28761904761904733,0.30323809523809475
380.0,Nearest Neighbor,0.2149999999999999,0.21118988095238078,0.2186428571428569
420.0,Proposed,0.35109523809523774,0.34235654761904744,0.3599053571428571
420.0,Weight Greedy,0.30664285714285655,0.2972619047619041,0.31583333333333274
420.0,Mean-workload Greedy,0.3150476190476185,0.30659523809523753,0.32347619047618975
420.0,Mean-workload DP,0.28861904761904705,0.28240476190476194,0.29473809523809485
420.0,Distribution-aware Myopic,0.3289047619047613,0.319666666666666,0.33802380952380867
420.0,Nearest Neighbor,0.22149999999999986,0.2168809523809522,0.22607142857142837
460.0,Proposed,0.39042857142857107,0.3817619047619043,0.3991428571428564
460.0,Weight Greedy,0.3431904761904756,0.33290476190476154,0.3533815476190473
460.0,Mean-workload Greedy,0.35059523809523774,0.34099999999999964,0.3600238095238091
460.0,Mean-workload DP,0.3328571428571425,0.3244285714285711,0.34121428571428536
460.0,Distribution-aware Myopic,0.2814047619047613,0.2757136904761898,0.2871428571428565
460.0,Nearest Neighbor,0.27057142857142863,0.2628571428571428,0.2782857142857144
500.0,Proposed,0.43857142857142806,0.42902380952380903,0.44828571428571395
500.0,Weight Greedy,0.3776428571428567,0.3663095238095236,0.3889523809523806
500.0,Mean-workload Greedy,0.3775952380952378,0.36666666666666614,0.3884291666666665
500.0,Mean-workload DP,0.3480714285714283,0.3406190476190476,0.3554999999999998
500.0,Distribution-aware Myopic,0.28457142857142814,0.27816666666666623,0.2909999999999995
500.0,Nearest Neighbor,0.28102380952380973,0.2733089285714286,0.2887380952380954
560.0,Proposed,0.4408095238095236,0.4310476190476187,0.4506904761904758
560.0,Weight Greedy,0.38097619047618997,0.3694523809523804,0.3923815476190473
560.0,Mean-workload Greedy,0.3780714285714283,0.36711845238095187,0.3889047619047616
560.0,Mean-workload DP,0.3503333333333331,0.3427857142857143,0.35804821428571454
560.0,Distribution-aware Myopic,0.2943809523809518,0.28685714285714226,0.3019761904761898
560.0,Nearest Neighbor,0.28102380952380973,0.2733089285714286,0.2887380952380954
620.0,Proposed,0.4408095238095236,0.4310476190476187,0.4506904761904758
620.0,Weight Greedy,0.38097619047618997,0.3694523809523804,0.3923815476190473
620.0,Mean-workload Greedy,0.3780714285714283,0.36711845238095187,0.3889047619047616
620.0,Mean-workload DP,0.3503333333333331,0.3427857142857143,0.35804821428571454
620.0,Distribution-aware Myopic,0.2943809523809518,0.28685714285714226,0.3019761904761898
620.0,Nearest Neighbor,0.28102380952380973,0.2733089285714286,0.2887380952380954
680.0,Proposed,0.4408095238095236,0.4310476190476187,0.4506904761904758
680.0,Weight Greedy,0.38097619047618997,0.3694523809523804,0.3923815476190473
680.0,Mean-workload Greedy,0.3780714285714283,0.36711845238095187,0.3889047619047616
680.0,Mean-workload DP,0.3503333333333331,0.3427857142857143,0.35804821428571454
680.0,Distribution-aware Myopic,0.2943809523809518,0.28685714285714226,0.3019761904761898
680.0,Nearest Neighbor,0.28102380952380973,0.2733089285714286,0.2887380952380954
740.0,Proposed,0.4408095238095236,0.4310476190476187,0.4506904761904758
740.0,Weight Greedy,0.38097619047618997,0.3694523809523804,0.3923815476190473
740.0,Mean-workload Greedy,0.3780714285714283,0.36711845238095187,0.3889047619047616
740.0,Mean-workload DP,0.3503333333333331,0.3427857142857143,0.35804821428571454
740.0,Distribution-aware Myopic,0.2943809523809518,0.28685714285714226,0.3019761904761898
740.0,Nearest Neighbor,0.28102380952380973,0.2733089285714286,0.2887380952380954
800.0,Proposed,0.4408095238095236,0.4310476190476187,0.4506904761904758
800.0,Weight Greedy,0.38097619047618997,0.3694523809523804,0.3923815476190473
800.0,Mean-workload Greedy,0.3780714285714283,0.36711845238095187,0.3889047619047616
800.0,Mean-workload DP,0.3503333333333331,0.3427857142857143,0.35804821428571454
800.0,Distribution-aware Myopic,0.2943809523809518,0.28685714285714226,0.3019761904761898
800.0,Nearest Neighbor,0.28102380952380973,0.2733089285714286,0.2887380952380954
"""

def configure_fonts():
    matplotlib.rcParams.update({
        "font.family": FONT,
        "font.size": 13,
        "axes.labelsize": 15.5,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def draw_reference_style(figure: int, settings: dict, output: Path, mean, lower, upper, palette):
    """Use the shared budget-figure style and retain every measured data point."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker
    from matplotlib.text import Text
    colors = COLORS
    require(set(METHODS) <= set(colors), "The reference palette lacks a required method.")
    x = np.asarray(settings["budgets"], dtype=float)
    require(len(np.unique(x)) > 10, "Every curve must retain all measured budget points.")
    mean, lower, upper = 100 * mean, 100 * lower, 100 * upper
    markers = {
        "Proposed": "o", "Weight Greedy": "^", "Mean-workload Greedy": "D",
        "Mean-workload DP": "s", "Distribution-aware Myopic": "P", "Nearest Neighbor": "v",
    }
    legend_order = list(METHODS)
    marker_sizes = {method: 4.5 if method == "Mean-workload DP" else 7.3
                    for method in METHODS}
    settings["marker_sizes_pt"] = marker_sizes
    plt.rcParams.update({
        "mathtext.fontset": "custom", "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.25))
    handles, marker_counts = {}, {}
    for mi, method in enumerate(METHODS):
        series = ax.errorbar(
            x, mean[:, mi],
            yerr=np.vstack((mean[:, mi] - lower[:, mi], upper[:, mi] - mean[:, mi])),
            color=colors[method], linestyle="-" if method == "Proposed" else "--",
            linewidth=2.0, marker=markers[method], markersize=marker_sizes[method],
            markerfacecolor=colors[method], markeredgecolor=colors[method],
            markeredgewidth=0.65, elinewidth=1.35, capsize=4.8, capthick=1.0,
            zorder=10-mi, clip_on=True,
        )
        line = series.lines[0]
        require(np.array_equal(np.asarray(line.get_xdata()), x),
                f"Missing measured budget markers for {method}")
        handles[method] = line
        marker_counts[method] = len(x)
    ax.set_xlim(x.min()-0.035*np.ptp(x), x.max()+0.035*np.ptp(x))
    y_min = 10 if figure == 6 else 0
    ax.set_ylim(y_min, 60)
    require(lower.min() >= y_min and upper.max() < 60, "A confidence interval is outside the axes.")
    ax.set_yticks(np.arange(y_min, 61, 10))
    ax.set_xticks(MAJOR_TICKS[figure], [f"{value:g}" for value in MAJOR_TICKS[figure]])
    ax.set_xlabel(settings["xlabel"], fontsize=15.5, labelpad=5)
    ax.set_ylabel("WCR (%)", fontsize=15.5, labelpad=5)
    ax.grid(True, color="#ECECEC", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(direction="out", length=3.8, width=0.8, pad=4, labelsize=13)
    # Each fixed-width legend cell centers the text in its column and puts
    # the sample marker and the label on the same vertical midpoint.
    font = FontProperties(family=FONT, size=11.3)
    renderer = fig.canvas.get_renderer()
    columns = []
    for names in (legend_order[:3], legend_order[3:]):
        label_width = max(renderer.get_text_width_height_descent(name, font, False)[0]
                          for name in names) * 72 / fig.dpi
        cells = []
        for method in names:
            cell = DrawingArea(34 + label_width, 12.5, clip=False)
            cell.add_artist(Line2D([0, 26], [6.25, 6.25], color=colors[method],
                                   linewidth=2.0,
                                   linestyle="-" if method == "Proposed" else "--"))
            cell.add_artist(Line2D([13], [6.25], color=colors[method],
                                   linestyle="none", marker=markers[method],
                                   markersize=marker_sizes[method], markeredgewidth=0.65))
            cell.add_artist(Text(34 + label_width/2, 6.25, method,
                                 fontproperties=font, ha="center", va="center"))
            cells.append(cell)
        columns.append(VPacker(children=cells, align="center", pad=0, sep=1.5))
    legend = AnchoredOffsetbox(loc="upper center",
                              child=HPacker(children=columns, align="center", pad=0, sep=18),
                              bbox_to_anchor=(0.5, 0.986), bbox_transform=ax.transAxes,
                              frameon=False, pad=0, borderpad=0.2)
    ax.add_artist(legend)
    ax.text(0.5, 1.025, settings["plot_note"], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10.2, color="#333333")
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.145, top=0.94)
    paths = []
    for extension in ("pdf",):
        path = OUTPUT
        fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.05)
        paths.append(path)
    plt.close(fig)
    return paths, colors, marker_counts


def main():
    rows = list(csv.DictReader(io.StringIO(SUMMARY_CSV)))
    lookup = {(float(row["budget_value"]), row["method"]): row for row in rows}
    arrays = [np.array([[float(lookup[budget, method][field]) for method in METHODS]
                        for budget in SETTINGS["budgets"]])
              for field in ("mean_wcr", "lower_95", "upper_95")]
    configure_fonts()
    draw_reference_style(FIGURE, SETTINGS, OUTPUT.parent, *arrays, None)


if __name__ == "__main__":
    main()
