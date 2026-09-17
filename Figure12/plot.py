#!/usr/bin/env python3
"""Redraw Figure 12 from the original, embedded result summaries.

Requires Python 3, NumPy and Matplotlib. Run: python plot.py
Writes figure12.pdf beside this script. No simulations or external data are used.
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

OUTPUT = Path(__file__).resolve().with_name("figure12.pdf")
try:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    FONT_FAMILY = "Times New Roman"
except ValueError:
    FONT_FAMILY = "DejaVu Serif"


def require(condition, message):
    if not condition:
        raise ValueError(message)

VARIANTS = ('proposed', 'mean_workload_dp', 'no_priority', 'top_myopic', 'no_proactive_skip')
SETTINGS = {'default': (2200, 620), 'larger_time': (2600, 620), 'lower_energy': (2200, 380)}

# Original result summary; values are not recomputed.
SUMMARY_CSV = """setting,t_s,e_kj,variant_id,method,n,mean_wcr_percent,lower_95_percent,upper_95_percent,delta_vs_proposed_pp,delta_lower_95_pp,delta_upper_95_pp,wins,ties,losses
default,2200,620,proposed,Proposed,300,44.08095238095238,43.10476190476196,45.06904761904762,0.0,0.0,0.0,0,300,0
default,2200,620,mean_workload_dp,Mean-workload DP,300,35.03333333333334,34.278571428571425,35.80482142857144,-9.04761904761905,-9.821428571428584,-8.276190476190493,20,12,268
default,2200,620,no_priority,No priority factor,300,41.40476190476195,40.42380952380957,42.40000000000003,-2.6761904761904756,-3.2785714285714462,-2.0761904761904617,78,38,184
default,2200,620,top_myopic,Myopic Top,300,29.823809523809505,29.073749999999958,30.569047619047602,-14.257142857142856,-14.902380952380962,-13.61904761904766,3,2,295
default,2200,620,no_proactive_skip,No proactive skip,300,40.31666666666667,39.11904761904763,41.521428571428544,-3.7642857142857133,-4.53333333333331,-3.0118452380952347,51,82,167
larger_time,2600,620,proposed,Proposed,300,49.56904761904765,48.54517857142854,50.59047619047618,0.0,0.0,0.0,0,300,0
larger_time,2600,620,mean_workload_dp,Mean-workload DP,300,43.18809523809525,42.235714285714266,44.13339285714288,-6.380952380952382,-7.026190476190486,-5.745238095238115,33,18,249
larger_time,2600,620,no_priority,No priority factor,300,45.94285714285716,44.7714285714286,47.123809523809555,-3.626190476190476,-4.330952380952361,-2.928571428571366,68,41,191
larger_time,2600,620,top_myopic,Myopic Top,300,37.55714285714285,36.776190476190536,38.31904761904763,-12.011904761904761,-12.738095238095248,-11.297619047619044,9,5,286
larger_time,2600,620,no_proactive_skip,No proactive skip,300,45.111904761904746,43.80714285714287,46.407202380952334,-4.457142857142856,-5.323809523809551,-3.626190476190565,55,61,184
lower_energy,2200,380,proposed,Proposed,300,29.842857142857145,29.100000000000012,30.580952380952393,0.0,0.0,0.0,0,300,0
lower_energy,2200,380,mean_workload_dp,Mean-workload DP,300,26.15714285714287,25.54285714285713,26.78095238095236,-3.685714285714286,-4.316726190476224,-3.0523809523809433,53,62,185
lower_energy,2200,380,no_priority,No priority factor,300,29.66904761904762,28.97619047619048,30.354761904761883,-0.1738095238095216,-0.5452380952381084,0.1976190476190391,33,222,45
lower_energy,2200,380,top_myopic,Myopic Top,300,30.054761904761914,29.304761904761897,30.80000000000003,0.21190476190476393,-0.3190476190476127,0.7523809523809463,108,79,113
lower_energy,2200,380,no_proactive_skip,No proactive skip,300,29.778571428571432,29.004761904761903,30.55244047619049,-0.06428571428571246,-0.6738095238095135,0.5428571428571303,106,90,104
"""


def configure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':13,'axes.labelsize':16,
        'xtick.labelsize':12,'ytick.labelsize':13,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'mathtext.fontset':'custom','mathtext.rm':FONT_FAMILY,'mathtext.it':FONT_FAMILY + ":italic"})
    return plt

def plot(stats):
    plt=configure()
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgb
    fig,ax=plt.subplots(figsize=(9.6,4.95))
    fig.subplots_adjust(left=.075,right=.99,bottom=.18,top=.79)
    x=np.arange(5)*1.08
    for j,(s,hatch,shade) in enumerate(zip(SETTINGS,['','///','\\\\\\'],[1,.68,.45])):
        records=[stats[s,v] for v in VARIANTS]
        means=np.array([r['mean_wcr_percent'] for r in records])
        lo=np.array([r['lower_95_percent'] for r in records]);hi=np.array([r['upper_95_percent'] for r in records])
        colors=[tuple(shade*c+(1-shade) for c in to_rgb(color))
                for color in ['#EE2922','#8673B5','#4F7CC7','#4F7CC7','#4F7CC7']]
        pos=x+(j-1)*.235
        ax.bar(pos,means,width=.21,color=colors,edgecolor='#686868' if hatch else 'none',
               linewidth=.5,hatch=hatch,zorder=3)
        ax.errorbar(pos,means,yerr=np.vstack((means-lo,hi-means)),fmt='none',
                    ecolor='#303030',capsize=3.4,elinewidth=1.2,zorder=4)
        for xi,m,u in zip(pos,means,hi):
            ax.text(xi,u+.9,f'{m:.2f}',ha='center',va='bottom',fontsize=10.5)
    ymax=max(60,int(np.ceil((max(r['upper_95_percent'] for r in stats.values())+4)/10))*10)
    ax.set_ylim(0,ymax);ax.set_yticks(np.arange(0,ymax+1,10));ax.set_ylabel('WCR (%)',fontsize=18)
    ax.set_xticks(x,['Proposed','Mean workload\nDP','No priority\nfactor','Myopic\nTop','No proactive\nskip'])
    ax.tick_params(axis='x',length=0,pad=10,labelsize=15)
    ax.tick_params(axis='y',labelsize=15)
    ax.grid(axis='y',color='#ECECEC',linewidth=.65);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    legend_labels=['Default\nT = 2200 s, E = 620 kJ','Larger T\nT = 2600 s, E = 620 kJ','Lower E\nT = 2200 s, E = 380 kJ']
    handles=[Patch(facecolor=c,edgecolor='#686868',hatch=h,label=l)
             for c,h,l in zip(['#999999','#BBBBBB','#DDDDDD'],['','///','\\\\\\'],legend_labels)]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,fontsize=13,
               bbox_to_anchor=(.535,1.0),handlelength=1.5,columnspacing=2.2)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    assert fig.legends[0].get_window_extent(renderer).y0 > ax.get_window_extent(renderer).y1
    ticks=[t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    assert all(a.x1+2<b.x0 for a,b in zip(ticks,ticks[1:]))
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


def main():
    rows = list(csv.DictReader(StringIO(SUMMARY_CSV)))
    stats = {(row["setting"], row["variant_id"]):
             {field: float(row[field]) for field in
              ("mean_wcr_percent", "lower_95_percent", "upper_95_percent")}
             for row in rows}
    require(len(stats) == len(SETTINGS) * len(VARIANTS), "Incomplete ablation result grid")
    plot(stats)
    print(OUTPUT.name)


if __name__ == "__main__":
    main()
