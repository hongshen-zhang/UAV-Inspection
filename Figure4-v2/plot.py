"""Draw the Top/Sub example from a fresh local mission.

Run experiment.py first to create data/replay.json, then run this script.
"""
from pathlib import Path
import argparse,json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
for name in ('Times New Roman.ttf', 'Times New Roman Bold.ttf', 'Times New Roman Italic.ttf'):
    font_path = FONT_DIR / name
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)

SERIF_FONT = ('Times New Roman' if any(font.name == 'Times New Roman'
              for font in font_manager.fontManager.ttflist) else 'DejaVu Serif')

plt.rcParams.update({
    'font.family': SERIF_FONT,
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

BLUE = '#235F8E'
GRAY = '#C8CDD2'
DARK = '#222222'


def clean_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E8E8E8', linewidth=.65)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#777777')
    ax.tick_params(axis='y', length=0, pad=8)
    ax.tick_params(axis='x', length=3)
    ax.xaxis.set_major_locator(MaxNLocator(4))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE / 'data' / 'replay.json')
    parser.add_argument('--output',type=Path,default=HERE / 'figure4-v2.pdf')
    args=parser.parse_args()
    if not args.input.is_file():
        parser.error(f'No experiment result at {args.input}. From the repository root, '
                     'run: python Figure4-v2/experiment.py && python Figure4-v2/plot.py. '
                     'For custom outputs, pass --input /path/to/replay.json.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data = json.loads(args.input.read_text(encoding='utf-8'))
    assert all(data.get('validation', {}).values())
    initial = data['top_decisions'][0]
    candidates = sorted([x for x in initial['candidate_values'] if x['expected_value'] is not None],
                        key=lambda c: c['expected_value'], reverse=True)[:3]
    examples = {x['task']: x for x in data['sub_decisions']}
    fig = plt.figure(figsize=(7.4, 3.0), facecolor='white')
    fig.text(.06, .96, '(a) MTE-Top: select a task', fontsize=12.5,
             fontweight='bold', va='top')
    fig.text(.57, .96, '(b) MTE-Sub: execute or skip', fontsize=12.5,
             fontweight='bold', va='top')
    fig.text(.06, .865, 'Before arrival: three leading candidates', fontsize=11, color='#555555')
    fig.text(.57, .865, 'After arrival', fontsize=11, color='#555555')

    ax = fig.add_axes([.10, .18, .35, .59])
    values = [c['expected_value'] for c in candidates]
    for j, c in enumerate(candidates):
        chosen = c['task'] == initial['selected_task']
        value = c['expected_value']
        ax.barh(j, value, height=.5, color=BLUE if chosen else GRAY,
                edgecolor='none')
        label = f"{value:.1f}"
        ax.text(value + max(values)*.025, j, label,
                va='center', ha='left', fontsize=10.5,
                color=BLUE if chosen else DARK,
                fontweight='bold' if chosen else 'normal')
        if chosen:
            ax.text(value*.5, j, 'Selected', color='white', fontsize=11,
                    ha='center', va='center')
    ax.set_yticks(range(len(candidates)), [f"T{c['task']}" for c in candidates])
    ax.set_ylim(len(candidates)-.5, -.5)
    ax.set_xlim(0, max(values)*1.16)
    ax.set_xlabel('Expected mission reward', labelpad=7)
    clean_axis(ax)

    complete=next(x['task'] for x in data['sub_decisions'] if x['selected_action']['mode']!='skip')
    skip=next((x['task'] for x in data['sub_decisions'] if x['selected_action']['mode']=='skip'),data['sub_decisions'][-1]['task'])
    for task, y in ((complete, .69), (skip, .35)):
        ex = examples[task]
        mode = ex['selected_action']['mode']
        decision = {'local':'Local','mec':'MEC','skip':'Skip'}[mode]
        fig.text(.57, y+.055, f'T{task}', fontsize=12, fontweight='bold')
        fig.text(.57, y-.025,
                 f"Workload: {ex['observed_workload_gcy']:.1f} Gcycles",
                 fontsize=11)
        fig.add_artist(FancyArrowPatch((.845, y+.015), (.902, y+.015),
                                      transform=fig.transFigure,
                                      arrowstyle='->', mutation_scale=12,
                                      linewidth=1.3, color=BLUE))
        fig.text(.925, y+.015, decision, color=BLUE, fontsize=12,
                 fontweight='bold', ha='center', va='center')
        if ex['completion_feasible']:
            total = max(a['total_value'] for a in ex['alternatives']
                        if a['mode'] != 'skip')
            skip = next(a['total_value'] for a in ex['alternatives']
                        if a['mode'] == 'skip')
            relation = '>' if total > skip else ('<' if total < skip else '=')
            reason = f'Expected reward: {total:.1f} (complete) {relation} {skip:.1f} (skip)'
        else:
            reason = f"Workload limit: {ex['maximum_feasible_workload_gcy']:.1f} Gcycles"
        fig.text(.57, y-.12, reason, fontsize=10, color='#555555')

    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(args.output.with_suffix('.'+suffix), dpi=300,
                    bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


if __name__ == '__main__':
    main()
