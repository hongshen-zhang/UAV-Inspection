#!/usr/bin/env python3
"""Redraw Figure 11 from the original, embedded result summaries.

Requires Python 3, NumPy and Matplotlib. Run: python plot.py
Uses results.csv when available; --archived selects the original saved results.
Use --results PATH for another experiment run, or --output PATH for a new PDF.
Writes figure11.pdf beside this script. No simulations or external data are used.
The embedded values and confidence intervals are preserved from the source tables.
"""
from __future__ import annotations

import argparse
import csv
import sys
sys.dont_write_bytecode = True
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

NEW_RESULTS = False
OUTPUT = Path(__file__).resolve().with_name("figure11.pdf")
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
MAX_MEC_GHZ = 5.6
LOCAL_CPU_RANGE = (1.0, 2.5)
MODES = ("UAV local", "MEC")
MODE_COLORS = {"UAV local": "#4F7CC7", "MEC": "#F68C28"}
MODE_MARKERS = {"UAV local": "^", "MEC": "D"}

# Original result summary; values are not recomputed.
WCR_CSV = """mec_cpu_ghz,method,n,mean_wcr_percent,lower_95_percent,upper_95_percent
0.7,Proposed,300,43.68333333333328,42.68333333333326,44.69291666666663
0.7,Weight Greedy,300,37.13571428571424,35.964226190476126,38.29761904761901
0.7,Mean-workload Greedy,300,36.69285714285711,35.573809523809494,37.79529761904759
0.7,Mean-workload DP,300,36.45238095238089,35.54761904761901,37.3571428571428
0.7,Distribution-aware Myopic,300,28.91904761904756,28.152380952380902,29.69047619047614
0.7,Nearest Neighbor,300,27.278571428571453,26.47380952380955,28.08333333333335
1.05,Proposed,300,43.68333333333328,42.68333333333326,44.69291666666663
1.05,Weight Greedy,300,37.13571428571424,35.964226190476126,38.29761904761901
1.05,Mean-workload Greedy,300,36.69285714285711,35.573809523809494,37.79529761904759
1.05,Mean-workload DP,300,36.45238095238089,35.54761904761901,37.3571428571428
1.05,Distribution-aware Myopic,300,28.91904761904756,28.152380952380902,29.69047619047614
1.05,Nearest Neighbor,300,27.278571428571453,26.47380952380955,28.08333333333335
1.4,Proposed,300,43.68333333333328,42.68333333333326,44.69291666666663
1.4,Weight Greedy,300,37.13571428571424,35.964226190476126,38.29761904761901
1.4,Mean-workload Greedy,300,36.69285714285711,35.573809523809494,37.79529761904759
1.4,Mean-workload DP,300,36.45238095238089,35.54761904761901,37.3571428571428
1.4,Distribution-aware Myopic,300,28.91904761904756,28.152380952380902,29.69047619047614
1.4,Nearest Neighbor,300,27.278571428571453,26.47380952380955,28.08333333333335
1.75,Proposed,300,43.68333333333328,42.68333333333326,44.69291666666663
1.75,Weight Greedy,300,37.13571428571424,35.964226190476126,38.29761904761901
1.75,Mean-workload Greedy,300,36.69285714285711,35.573809523809494,37.79529761904759
1.75,Mean-workload DP,300,36.45238095238089,35.54761904761901,37.3571428571428
1.75,Distribution-aware Myopic,300,28.91904761904756,28.152380952380902,29.69047619047614
1.75,Nearest Neighbor,300,27.278571428571453,26.47380952380955,28.08333333333335
2.1,Proposed,300,43.68333333333328,42.68333333333326,44.69291666666663
2.1,Weight Greedy,300,37.13571428571424,35.964226190476126,38.29761904761901
2.1,Mean-workload Greedy,300,36.69285714285711,35.573809523809494,37.79529761904759
2.1,Mean-workload DP,300,36.45238095238089,35.54761904761901,37.3571428571428
2.1,Distribution-aware Myopic,300,28.91904761904756,28.152380952380902,29.69047619047614
2.1,Nearest Neighbor,300,27.278571428571453,26.47380952380955,28.08333333333335
2.45,Proposed,300,43.68333333333328,42.68333333333326,44.69291666666663
2.45,Weight Greedy,300,37.13571428571424,35.964226190476126,38.29761904761901
2.45,Mean-workload Greedy,300,36.69285714285711,35.573809523809494,37.79529761904759
2.45,Mean-workload DP,300,36.45238095238089,35.54761904761901,37.3571428571428
2.45,Distribution-aware Myopic,300,28.91904761904756,28.152380952380902,29.69047619047614
2.45,Nearest Neighbor,300,27.278571428571453,26.47380952380955,28.08333333333335
2.8,Proposed,300,44.08095238095236,43.104761904761865,45.06904761904758
2.8,Weight Greedy,300,38.097619047619,36.94523809523804,39.23815476190473
2.8,Mean-workload Greedy,300,37.80714285714283,36.71184523809519,38.89047619047616
2.8,Mean-workload DP,300,35.03333333333331,34.27857142857143,35.80482142857145
2.8,Distribution-aware Myopic,300,29.43809523809518,28.685714285714226,30.19761904761898
2.8,Nearest Neighbor,300,28.102380952380972,27.330892857142857,28.87380952380954
3.15,Proposed,300,45.49999999999994,44.5523809523809,46.46190476190474
3.15,Weight Greedy,300,39.51904761904757,38.352321428571386,40.671428571428535
3.15,Mean-workload Greedy,300,39.695238095238054,38.62380952380951,40.761904761904745
3.15,Mean-workload DP,300,38.33809523809514,37.488095238095156,39.19285714285706
3.15,Distribution-aware Myopic,300,30.76190476190469,30.05476190476184,31.46904761904753
3.15,Nearest Neighbor,300,29.657142857142876,28.904761904761923,30.40952380952383
3.5,Proposed,300,47.509523809523785,46.550000000000004,48.46904761904761
3.5,Weight Greedy,300,40.9880952380952,39.84047619047618,42.13809523809522
3.5,Mean-workload Greedy,300,41.39999999999997,40.33089285714283,42.43571428571427
3.5,Mean-workload DP,300,39.96666666666657,39.13095238095233,40.780952380952286
3.5,Distribution-aware Myopic,300,31.899999999999913,31.19523809523802,32.604761904761816
3.5,Nearest Neighbor,300,31.13333333333335,30.41428571428574,31.828571428571458
3.85,Proposed,300,48.78333333333333,47.826190476190455,49.747678571428594
3.85,Weight Greedy,300,42.23571428571425,41.05952380952378,43.411904761904744
3.85,Mean-workload Greedy,300,40.619047619047436,39.70714285714273,41.51904761904747
3.85,Mean-workload DP,300,41.623809523809456,40.74523809523804,42.492916666666616
3.85,Distribution-aware Myopic,300,32.77619047619037,32.08809523809516,33.471428571428476
3.85,Nearest Neighbor,300,32.54285714285715,31.85714285714288,33.21666666666667
4.2,Proposed,300,50.23809523809524,49.31422619047618,51.17857142857144
4.2,Weight Greedy,300,43.33333333333331,42.140476190476164,44.49761904761903
4.2,Mean-workload Greedy,300,41.3761904761903,40.41904761904748,42.316726190476054
4.2,Mean-workload DP,300,42.85476190476183,41.947619047618964,43.773809523809454
4.2,Distribution-aware Myopic,300,34.119047619047535,33.35708333333323,34.878571428571334
4.2,Nearest Neighbor,300,33.25238095238094,32.57619047619047,33.91190476190475
4.55,Proposed,300,51.109523809523836,50.21428571428572,52.004761904761935
4.55,Weight Greedy,300,44.75714285714283,43.59999999999998,45.88571428571428
4.55,Mean-workload Greedy,300,43.73809523809525,42.82857142857143,44.626190476190516
4.55,Mean-workload DP,300,44.3214285714285,43.50232142857132,45.13333333333327
4.55,Distribution-aware Myopic,300,34.423809523809425,33.77857142857133,35.05476190476181
4.55,Nearest Neighbor,300,34.26666666666667,33.6142857142857,34.90476190476187
4.9,Proposed,300,51.85000000000007,50.90952380952386,52.790476190476255
4.9,Weight Greedy,300,45.75476190476189,44.54761904761902,46.91428571428571
4.9,Mean-workload Greedy,300,44.75238095238097,43.83803571428571,45.65952380952385
4.9,Mean-workload DP,300,45.85476190476187,44.98089285714278,46.74523809523804
4.9,Distribution-aware Myopic,300,34.583333333333236,33.98571428571422,35.16428571428564
4.9,Nearest Neighbor,300,35.24999999999997,34.626190476190466,35.85476190476186
5.25,Proposed,300,53.71904761904772,52.83571428571432,54.597619047619126
5.25,Weight Greedy,300,46.64285714285713,45.44523809523809,47.82142857142855
5.25,Mean-workload Greedy,300,45.783333333333346,44.85952380952383,46.68809523809526
5.25,Mean-workload DP,300,47.90714285714282,47.01184523809515,48.79999999999999
5.25,Distribution-aware Myopic,300,34.50952380952372,34.00232142857134,34.99761904761897
5.25,Nearest Neighbor,300,36.092857142857106,35.48333333333332,36.671428571428535
5.6,Proposed,300,54.61904761904771,53.71428571428579,55.51190476190484
5.6,Weight Greedy,300,47.52857142857143,46.37619047619047,48.66904761904764
5.6,Mean-workload Greedy,300,46.68571428571431,45.75952380952382,47.60476190476191
5.6,Mean-workload DP,300,48.78333333333331,47.90238095238092,49.67142857142859
5.6,Distribution-aware Myopic,300,34.55714285714277,34.08809523809518,35.007142857142796
5.6,Nearest Neighbor,300,36.959523809523766,36.3666666666666,37.53571428571422
"""

# Original result summary; values are not recomputed.
SHARES_CSV = """mec_cpu_ghz,method,execution_mode,n_missions,completed_tasks_in_mode,all_completed_tasks,share_percent,lower_95_percent,upper_95_percent
0.7,Proposed,UAV local,300,1508,1508,100.0,100.0,100.0
0.7,Proposed,MEC,300,0,1508,0.0,0.0,0.0
1.05,Proposed,UAV local,300,1508,1508,100.0,100.0,100.0
1.05,Proposed,MEC,300,0,1508,0.0,0.0,0.0
1.4,Proposed,UAV local,300,1508,1508,100.0,100.0,100.0
1.4,Proposed,MEC,300,0,1508,0.0,0.0,0.0
1.75,Proposed,UAV local,300,1508,1508,100.0,100.0,100.0
1.75,Proposed,MEC,300,0,1508,0.0,0.0,0.0
2.1,Proposed,UAV local,300,1508,1508,100.0,100.0,100.0
2.1,Proposed,MEC,300,0,1508,0.0,0.0,0.0
2.45,Proposed,UAV local,300,1508,1508,100.0,100.0,100.0
2.45,Proposed,MEC,300,0,1508,0.0,0.0,0.0
2.8,Proposed,UAV local,300,844,1499,56.30420280186791,54.12650248118698,58.4145708628699
2.8,Proposed,MEC,300,655,1499,43.69579719813209,41.5854291371301,45.87349751881301
3.15,Proposed,UAV local,300,539,1563,34.4849648112604,32.4889170360988,36.47435897435897
3.15,Proposed,MEC,300,1024,1563,65.5150351887396,63.52564102564103,67.5110829639012
3.5,Proposed,UAV local,300,430,1671,25.73309395571514,24.142083082480433,27.357392316647264
3.5,Proposed,MEC,300,1241,1671,74.26690604428487,72.64260768335274,75.85791691751956
3.85,Proposed,UAV local,300,424,1716,24.708624708624708,23.265275843274996,26.18079500707846
3.85,Proposed,MEC,300,1292,1716,75.29137529137529,73.81920499292154,76.73472415672501
4.2,Proposed,UAV local,300,354,1819,19.461242440901593,18.090390846137257,20.840298416741962
4.2,Proposed,MEC,300,1465,1819,80.5387575590984,79.15970158325804,81.90960915386273
4.55,Proposed,UAV local,300,304,1861,16.33530360021494,15.17094017094017,17.508055853920517
4.55,Proposed,MEC,300,1557,1861,83.66469639978506,82.49194414607949,84.82905982905983
4.9,Proposed,UAV local,300,266,1877,14.171550346297282,13.156498673740053,15.19193081141972
4.9,Proposed,MEC,300,1611,1877,85.82844965370272,84.80806918858028,86.84350132625995
5.25,Proposed,UAV local,300,297,1977,15.022761760242792,13.945202942982931,16.114494199472006
5.25,Proposed,MEC,300,1680,1977,84.97723823975721,83.885505800528,86.05479705701707
5.6,Proposed,UAV local,300,307,2016,15.228174603174603,14.13637320867286,16.305418719211822
5.6,Proposed,MEC,300,1709,2016,84.77182539682539,83.69458128078817,85.86362679132714
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

def mode_legend(fig,ax):
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox,DrawingArea,VPacker
    from matplotlib.text import Text
    font=FontProperties(family=FONT_FAMILY,size=11.3)
    renderer=fig.canvas.get_renderer()
    width=max(renderer.get_text_width_height_descent(m,font,False)[0] for m in MODES)*72/fig.dpi
    cells=[]
    for mode in MODES:
        cell=DrawingArea(34+width,12.5,clip=False)
        cell.add_artist(Line2D([0,26],[6.25,6.25],color=MODE_COLORS[mode],linewidth=2,linestyle="-"))
        cell.add_artist(Line2D([13],[6.25],color=MODE_COLORS[mode],linestyle="none",marker=MODE_MARKERS[mode],markersize=7.3,markeredgewidth=.65))
        cell.add_artist(Text(34+width/2,6.25,mode,fontproperties=font,ha="center",va="center"))
        cells.append(cell)
    box=AnchoredOffsetbox(loc="center right",child=VPacker(children=cells,align="center",pad=0,sep=1.5),
        bbox_to_anchor=(.965,.48),bbox_transform=ax.transAxes,frameon=False,pad=0,borderpad=.2)
    ax.add_artist(box)
    return box

def draw(wcr,shares):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.transforms import Bbox
    plt.rcParams.update({"font.family":FONT_FAMILY,"font.size":13,"axes.labelsize":15.5,
        "xtick.labelsize":13,"ytick.labelsize":13,"mathtext.fontset":"custom",
        "mathtext.rm":FONT_FAMILY,"mathtext.it":FONT_FAMILY + ":italic",
        "mathtext.bf":FONT_FAMILY + ":bold","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,1,figsize=(7.2,8.6),gridspec_kw={"height_ratios":[1.12,1]})
    fig.subplots_adjust(left=.125,right=.985,bottom=.085,top=.95,hspace=.36)
    x=np.asarray(POINTS)
    for ax,data,names,colors,markers,sizes,metric in (
        (axes[0],wcr,METHODS,COLORS,MARKERS,SIZES,"WCR percent"),
        (axes[1],shares,MODES,MODE_COLORS,MODE_MARKERS,{m:7.3 for m in MODES},"Share of Proposed completed tasks percent")):
        mean,low,high=data
        require(np.all(low<=mean+1e-10) and np.all(mean<=high+1e-10),"Invalid CI bounds")
        show_ci=metric=="WCR percent"
        for j,name in enumerate(names):
            yerr=np.maximum(0,np.stack((mean[:,j]-low[:,j],high[:,j]-mean[:,j]))) if show_ci else None
            series=ax.errorbar(x,mean[:,j],yerr=yerr,
                color=colors[name],linestyle="-" if name=="Proposed" or metric.startswith("Share") else "--",
                marker=markers[name],markersize=sizes[name],markerfacecolor=colors[name],markeredgecolor=colors[name],
                markeredgewidth=.65,linewidth=2,elinewidth=1.35,capsize=4.8,capthick=1,zorder=10-j,clip_on=True)
            require(np.array_equal(series.lines[0].get_xdata(),x),"Missing measured point")
        ax.set_xlim(.48,MAX_MEC_GHZ+.22)
        ax.set_xticks(MAJOR,[f"{v:g}" for v in MAJOR])
        ax.set_xticks([v for v in POINTS if v not in MAJOR],minor=True)
        ax.grid(True,color="#ECECEC",linewidth=.65);ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        for side in ("left","bottom"):ax.spines[side].set_color("black");ax.spines[side].set_linewidth(.85)
        ax.tick_params(direction="out",length=3.8,width=.8,pad=4)
        ax.tick_params(axis="x",which="minor",length=2.1,width=.65)
    axes[0].set_ylim(20,80);axes[0].set_yticks(np.arange(20,81,10))
    if NEW_RESULTS:
        top = max(80, 10 * np.ceil((wcr[2].max() + 20) / 10))
        bottom = min(20, 10 * np.floor(wcr[1].min() / 10))
        axes[0].set_ylim(bottom, top)
        axes[0].set_yticks(np.arange(bottom, top + 1, 10))
    else:
        require(wcr[1].min()>=20 and wcr[2].max()<65,"WCR interval outside plot/legend space")
    axes[0].set_ylabel("WCR (%)",labelpad=5)
    axes[0].set_title("(a) Completion performance",fontsize=14,pad=12)
    axes[1].set_ylim(-4,108);axes[1].set_yticks([0,25,50,75,100])
    axes[1].set_ylabel("Execution share (%)",labelpad=5)
    axes[1].set_xlabel(r"MEC CPU frequency $F_m$ (GHz)",labelpad=5)
    axes[1].set_title("(b) Execution modes of Proposed",fontsize=14,pad=12)
    for boundary in LOCAL_CPU_RANGE:
        axes[1].axvline(boundary,color="#777777",linewidth=1.1,linestyle=(0,(4,3)),zorder=2)
    local_note=axes[1].text(sum(LOCAL_CPU_RANGE)/2,81,"UAV local CPU range\n1.0-2.5 GHz",
        ha="center",va="center",fontsize=11.3,color="#555555",linespacing=1.3,zorder=12)
    boxes=[legend(fig,axes[0],COLORS),mode_legend(fig,axes[1])]
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in axes:
        ticks=[label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_visible()]
        require(all(left.x1+2<right.x0 for left,right in zip(ticks,ticks[1:])),"Frequency tick labels overlap")
    note_bounds=local_note.get_window_extent(renderer)
    reference_x=[axes[1].transData.transform((v,81))[0] for v in LOCAL_CPU_RANGE]
    require(reference_x[0]<note_bounds.x0 and note_bounds.x1<reference_x[1],"Local range label crosses reference lines")
    for ax,box,data in zip(axes,boxes,(wcr,shares)):
        bounds=box.get_window_extent(renderer);ab=ax.get_window_extent(renderer)
        require(ab.contains(bounds.x0,bounds.y0) and ab.contains(bounds.x1,bounds.y1),"Legend outside axes")
        for j in range(data[0].shape[1]):
            for xv,lo,hi in zip(x,data[1][:,j],data[2][:,j]):
                px,py0=ax.transData.transform((xv,lo));_,py1=ax.transData.transform((xv,hi))
                require(not bounds.overlaps(Bbox.from_extents(px-7,py0-7,px+7,py1+7)),f"Legend overlaps CI or marker: {ax.get_ylabel()}, x={xv}, series={j}, legend={bounds.bounds}, point={(px,py0,py1)}")
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)



def main():
    global OUTPUT, POINTS, MAJOR, NEW_RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archived", action="store_true", help="use the original embedded paper results")
    source.add_argument("--results", type=Path, help="CSV emitted by experiment.py")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result_path = args.results
    if result_path is None and not args.archived:
        candidate = Path(__file__).resolve().with_name("results.csv")
        if candidate.is_file():
            result_path = candidate
    if result_path is not None:
        NEW_RESULTS = True
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
        from sensitivity_cases import result_matrices, share_matrices
        POINTS, wcr, lookup, seeds = result_matrices(result_path, None, METHODS)
        shares = share_matrices(POINTS, lookup, seeds)
        print(f"Drawing new experiment results: {len(seeds)} paired seeds per frequency")
    else:
        POINTS, wcr = load_summary(WCR_CSV, "mec_cpu_ghz", "method", METHODS)
        share_points, shares = load_summary(SHARES_CSV, "mec_cpu_ghz", "execution_mode", MODES, "share_percent")
        require(share_points == POINTS, "MEC frequencies differ between panels")
    MAJOR = POINTS
    draw(wcr, shares)
    print(OUTPUT)


if __name__ == "__main__":
    main()
