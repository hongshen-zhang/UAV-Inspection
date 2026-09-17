#!/usr/bin/env python3
"""Redraw Figure 6 using the original experiment results embedded below."""
# Dependencies: numpy, matplotlib.
from pathlib import Path
import csv
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT = Path(__file__).resolve().with_name("figure6.pdf")

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
FIGURE = 6
SETTINGS = {'budgets': [800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0], 'xlabel': 'Mission time budget $T$ (s)', 'plot_note': 'E = 620 kJ; n = 300'}
MAJOR_TICKS = {6: (800.0, 1200.0, 1600.0, 2000.0, 2400.0, 2800.0, 3200.0)}

# Pointwise 95% bootstrap confidence intervals of mean WCR (fractions).
SUMMARY_CSV = """budget_value,method,mean_wcr,lower_95,upper_95
800.0,Proposed,0.15352380952380915,0.1488095238095234,0.15823809523809468
800.0,Weight Greedy,0.12471428571428549,0.11923809523809514,0.13016666666666651
800.0,Mean-workload Greedy,0.1452142857142855,0.14104761904761878,0.14942857142857116
800.0,Mean-workload DP,0.13930952380952366,0.13476190476190447,0.14380952380952353
800.0,Distribution-aware Myopic,0.15295238095238073,0.14847619047619023,0.15747619047619008
800.0,Nearest Neighbor,0.12183333333333345,0.11692857142857148,0.12685714285714295
1000.0,Proposed,0.17954761904761857,0.1740952380952376,0.18495238095238029
1000.0,Weight Greedy,0.1554285714285713,0.14907142857142833,0.16192857142857123
1000.0,Mean-workload Greedy,0.16988095238095183,0.16504761904761872,0.17466666666666616
1000.0,Mean-workload DP,0.1530952380952379,0.14759523809523795,0.15866666666666643
1000.0,Distribution-aware Myopic,0.17447619047619015,0.1693809523809522,0.17957142857142838
1000.0,Nearest Neighbor,0.15240476190476204,0.1475000000000001,0.15742857142857156
1200.0,Proposed,0.21499999999999966,0.20864285714285694,0.2212624999999998
1200.0,Weight Greedy,0.18542857142857133,0.17871428571428558,0.19228571428571428
1200.0,Mean-workload Greedy,0.19273809523809496,0.1874761904761903,0.19802380952380935
1200.0,Mean-workload DP,0.16754761904761892,0.16252380952380938,0.17264285714285688
1200.0,Distribution-aware Myopic,0.20928571428571405,0.20311904761904756,0.21547619047619032
1200.0,Nearest Neighbor,0.17966666666666664,0.1745708333333332,0.1847619047619047
1400.0,Proposed,0.25054761904761913,0.24371428571428547,0.2574761904761904
1400.0,Weight Greedy,0.21838095238095234,0.21128571428571424,0.22542916666666676
1400.0,Mean-workload Greedy,0.22185714285714284,0.21657142857142858,0.22719047619047614
1400.0,Mean-workload DP,0.21485714285714272,0.20861904761904743,0.2209999999999999
1400.0,Distribution-aware Myopic,0.24126190476190496,0.23478571428571432,0.24771428571428566
1400.0,Nearest Neighbor,0.19878571428571423,0.19402380952380946,0.20349999999999993
1600.0,Proposed,0.2942142857142854,0.286285714285714,0.3020952380952377
1600.0,Weight Greedy,0.25907142857142823,0.25047619047619013,0.2675476190476185
1600.0,Mean-workload Greedy,0.25721428571428573,0.2509761904761905,0.2634529761904762
1600.0,Mean-workload DP,0.24035714285714269,0.2341660714285711,0.24666666666666656
1600.0,Distribution-aware Myopic,0.2829999999999998,0.275190476190476,0.2907619047619046
1600.0,Nearest Neighbor,0.21109523809523806,0.20688095238095225,0.21519047619047607
1800.0,Proposed,0.33766666666666634,0.3292613095238093,0.34614285714285686
1800.0,Weight Greedy,0.29373809523809463,0.28426190476190427,0.3030238095238089
1800.0,Mean-workload Greedy,0.30007142857142827,0.291428571428571,0.30861904761904724
1800.0,Mean-workload DP,0.2790238095238094,0.2715714285714287,0.2863095238095235
1800.0,Distribution-aware Myopic,0.3212142857142852,0.31169047619047596,0.3305714285714278
1800.0,Nearest Neighbor,0.21688095238095226,0.21285714285714283,0.22073809523809507
2000.0,Proposed,0.39476190476190437,0.3853327380952374,0.4042857142857136
2000.0,Weight Greedy,0.3390952380952375,0.32838095238095183,0.3497857142857139
2000.0,Mean-workload Greedy,0.34126190476190443,0.3314041666666663,0.3511666666666662
2000.0,Mean-workload DP,0.3189999999999997,0.3110952380952382,0.3267619047619046
2000.0,Distribution-aware Myopic,0.3636428571428569,0.35266607142857076,0.3745952380952377
2000.0,Nearest Neighbor,0.24035714285714269,0.2336666666666667,0.24711904761904738
2200.0,Proposed,0.4408095238095236,0.4310476190476187,0.4506904761904758
2200.0,Weight Greedy,0.38097619047618997,0.3694523809523804,0.3923815476190473
2200.0,Mean-workload Greedy,0.3780714285714283,0.36711845238095187,0.3889047619047616
2200.0,Mean-workload DP,0.3503333333333331,0.3427857142857143,0.35804821428571454
2200.0,Distribution-aware Myopic,0.2943809523809518,0.28685714285714226,0.3019761904761898
2200.0,Nearest Neighbor,0.28102380952380973,0.2733089285714286,0.2887380952380954
2400.0,Proposed,0.4644523809523807,0.45464226190476115,0.4742857142857141
2400.0,Weight Greedy,0.4047619047619046,0.39335654761904726,0.4159291666666668
2400.0,Mean-workload Greedy,0.3921190476190474,0.38107142857142834,0.40300059523809534
2400.0,Mean-workload DP,0.398166666666666,0.38897559523809433,0.4074053571428566
2400.0,Distribution-aware Myopic,0.36407142857142794,0.3556666666666659,0.37226190476190396
2400.0,Nearest Neighbor,0.2987142857142861,0.2903809523809528,0.3069761904761908
2600.0,Proposed,0.49569047619047635,0.4854517857142852,0.5059047619047619
2600.0,Weight Greedy,0.42683333333333306,0.4153571428571426,0.43814285714285706
2600.0,Mean-workload Greedy,0.41164285714285703,0.40007083333333293,0.4230958333333331
2600.0,Mean-workload DP,0.4318809523809517,0.4223571428571421,0.4413339285714283
2600.0,Distribution-aware Myopic,0.3762857142857135,0.3680470238095227,0.38428571428571323
2600.0,Nearest Neighbor,0.3322857142857147,0.3222851190476194,0.3421904761904765
2800.0,Proposed,0.498309523809524,0.48792797619047595,0.5088095238095237
2800.0,Weight Greedy,0.4314047619047616,0.4199285714285711,0.4427142857142856
2800.0,Mean-workload Greedy,0.4152142857142855,0.4035952380952377,0.42673809523809497
2800.0,Mean-workload DP,0.4330238095238089,0.4233571428571422,0.44259523809523754
2800.0,Distribution-aware Myopic,0.378214285714285,0.36999999999999916,0.386119642857142
2800.0,Nearest Neighbor,0.3322857142857147,0.3222851190476194,0.3421904761904765
3000.0,Proposed,0.498309523809524,0.48792797619047595,0.5088095238095237
3000.0,Weight Greedy,0.4314047619047616,0.4199285714285711,0.4427142857142856
3000.0,Mean-workload Greedy,0.4152142857142855,0.4035952380952377,0.42673809523809497
3000.0,Mean-workload DP,0.4330238095238089,0.4233571428571422,0.44259523809523754
3000.0,Distribution-aware Myopic,0.378214285714285,0.36999999999999916,0.386119642857142
3000.0,Nearest Neighbor,0.3322857142857147,0.3222851190476194,0.3421904761904765
3200.0,Proposed,0.498309523809524,0.48792797619047595,0.5088095238095237
3200.0,Weight Greedy,0.4314047619047616,0.4199285714285711,0.4427142857142856
3200.0,Mean-workload Greedy,0.4152142857142855,0.4035952380952377,0.42673809523809497
3200.0,Mean-workload DP,0.43316666666666603,0.42347619047618973,0.44276190476190436
3200.0,Distribution-aware Myopic,0.378214285714285,0.36999999999999916,0.386119642857142
3200.0,Nearest Neighbor,0.3322857142857147,0.3222851190476194,0.3421904761904765
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
