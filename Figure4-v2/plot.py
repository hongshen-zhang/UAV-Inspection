"""Redraw the Top/Sub decision example from recorded experiment values.

The data below are the original nominal-mission observations and DP values
(seed 2026092000), not a new simulation run.
Requires: pip install matplotlib
Run: python plot.py
Output: figure4-v2.pdf next to this script.
"""
from pathlib import Path
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent

# Register local Times New Roman fonts when available; otherwise use a serif
# fallback distributed with the system or Matplotlib.
for font_dir in (Path('/System/Library/Fonts/Supplemental'),
                 Path('/Library/Fonts'),
                 Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'):
    for filename in ('Times New Roman.ttf', 'Times New Roman Bold.ttf',
                     'Times New Roman Italic.ttf', 'times.ttf',
                     'timesbd.ttf', 'timesi.ttf'):
        font_path = font_dir / filename
        if font_path.is_file():
            font_manager.fontManager.addfont(font_path)

RECORDED_DECISIONS = {'description': 'Single recorded mission from Figure4, with audited root DP values and actual '
                'arrival decisions.',
 'seed': 2026092000,
 'value_definition': 'Expected sum of task priority; completion total = immediate reward + '
                     'continuation value; skip total = continuation value.',
 'feasibility_note': 'An infeasible completion has no comparison value and must not be shown '
                     'as a zero-value completion bar.',
 'validation_against_formal_result': {'route': True,
                                      'weighted_completed': True,
                                      'total_weight': True,
                                      'completed_tasks': True,
                                      'visited_tasks': True,
                                      'skipped_tasks': True,
                                      'local_actions': True,
                                      'mec_actions': True,
                                      'final_time_s': True,
                                      'final_energy_kj': True,
                                      'mcr': True},
 'top_decisions': [{'screening': [{'rank': 1,
                                   'task': 3,
                                   'score': 28.583404145174036,
                                   'retained': True},
                                  {'rank': 2,
                                   'task': 7,
                                   'score': 21.730137305224513,
                                   'retained': True},
                                  {'rank': 3,
                                   'task': 12,
                                   'score': 18.672630669050193,
                                   'retained': True},
                                  {'rank': 4,
                                   'task': 5,
                                   'score': 18.253514386673952,
                                   'retained': True},
                                  {'rank': 5,
                                   'task': 15,
                                   'score': 17.5241831592068,
                                   'retained': True},
                                  {'rank': 6,
                                   'task': 11,
                                   'score': 13.099776015984395,
                                   'retained': True},
                                  {'rank': 7,
                                   'task': 19,
                                   'score': 12.444304542050858,
                                   'retained': True},
                                  {'rank': 8,
                                   'task': 4,
                                   'score': 11.829221894271734,
                                   'retained': True},
                                  {'rank': 9,
                                   'task': 16,
                                   'score': 11.310051299185675,
                                   'retained': True},
                                  {'rank': 10,
                                   'task': 8,
                                   'score': 8.568319253398666,
                                   'retained': False},
                                  {'rank': 11,
                                   'task': 14,
                                   'score': 7.373541291895265,
                                   'retained': False},
                                  {'rank': 12,
                                   'task': 20,
                                   'score': 6.916746847964903,
                                   'retained': False},
                                  {'rank': 13,
                                   'task': 9,
                                   'score': 4.2428970741097025,
                                   'retained': False},
                                  {'rank': 14,
                                   'task': 10,
                                   'score': 4.125858385016809,
                                   'retained': False},
                                  {'rank': 15,
                                   'task': 18,
                                   'score': 3.5247539879156675,
                                   'retained': False},
                                  {'rank': 16,
                                   'task': 17,
                                   'score': 3.385993441848626,
                                   'retained': False},
                                  {'rank': 17,
                                   'task': 2,
                                   'score': 2.3660703775426377,
                                   'retained': False},
                                  {'rank': 18,
                                   'task': 1,
                                   'score': 2.1946345243963683,
                                   'retained': False},
                                  {'rank': 19,
                                   'task': 6,
                                   'score': 1.5406451872665075,
                                   'retained': False},
                                  {'rank': 20,
                                   'task': 13,
                                   'score': 1.2934045363425728,
                                   'retained': False}],
                    'retained_tasks': [3, 7, 12, 5, 15, 11, 19, 4, 16],
                    'candidate_values': [{'task': 3,
                                          'expected_value': 59.2275073296056,
                                          'rounded_state_feasible': True},
                                         {'task': 7,
                                          'expected_value': 52.21361605190924,
                                          'rounded_state_feasible': True},
                                         {'task': 12,
                                          'expected_value': 49.269965879196796,
                                          'rounded_state_feasible': True},
                                         {'task': 5,
                                          'expected_value': 56.719567969616676,
                                          'rounded_state_feasible': True},
                                         {'task': 15,
                                          'expected_value': 48.87870684924132,
                                          'rounded_state_feasible': True},
                                         {'task': 11,
                                          'expected_value': 49.18988334949243,
                                          'rounded_state_feasible': True},
                                         {'task': 19,
                                          'expected_value': 50.63435558692453,
                                          'rounded_state_feasible': True},
                                         {'task': 4,
                                          'expected_value': 57.023385367084146,
                                          'rounded_state_feasible': True},
                                         {'task': 16,
                                          'expected_value': 47.276114840475906,
                                          'rounded_state_feasible': True}],
                    'epoch': 0,
                    'current_node': 0,
                    'time_s': 0.0,
                    'energy_kj': 0.0,
                    'remaining_tasks': [1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        8,
                                        9,
                                        10,
                                        11,
                                        12,
                                        13,
                                        14,
                                        15,
                                        16,
                                        17,
                                        18,
                                        19,
                                        20],
                    'selected_task': 3,
                    'selected_value': 59.2275073296056}],
 'sub_decisions': [{'epoch': 0,
                    'task': 3,
                    'priority': 15.0,
                    'observed_workload_gcy': 81.65190054184228,
                    'workload_mean_gcy': 805.6860051297374,
                    'mu': 6.0921815948044875,
                    'sigma': 1.095,
                    'arrival_time_s': 61.733293476418964,
                    'arrival_energy_kj': 15.741989836486837,
                    'execution_time_available_s': 1808.85443596156,
                    'execution_energy_available_kj': 588.5160203270264,
                    'maximum_feasible_workload_gcy': 5009.260695121504,
                    'shadow_price_time': 0.02776728440011267,
                    'shadow_price_energy': 0.0,
                    'completion_feasible': True,
                    'alternatives': [{'mode': 'skip',
                                      'mec': -1,
                                      'frequency_ghz': 0.0,
                                      'service_time_s': 0.0,
                                      'service_energy_kj': 0.0,
                                      'occupation': 0.0,
                                      'immediate_reward': 0.0,
                                      'future_value': 53.68868448616486,
                                      'total_value': 53.68868448616486,
                                      'resource_cost': 0.0,
                                      'successor': {'node': 3,
                                                    'time_s': 61.733293476418964,
                                                    'energy_kj': 15.741989836486837,
                                                    'remaining_mask': 1048571}},
                                     {'mode': 'local',
                                      'mec': -1,
                                      'frequency_ghz': 2.5,
                                      'service_time_s': 32.66076021673691,
                                      'service_energy_kj': 6.12389254063817,
                                      'occupation': 0.028461698108934768,
                                      'immediate_reward': 15.0,
                                      'future_value': 52.21022761573143,
                                      'total_value': 67.21022761573143,
                                      'resource_cost': 0.9069006176620192,
                                      'successor': {'node': 3,
                                                    'time_s': 94.39405369315588,
                                                    'energy_kj': 21.865882377125008,
                                                    'remaining_mask': 1048571}}],
                    'selected_action': {'mode': 'local',
                                        'mec': -1,
                                        'frequency_ghz': 2.5,
                                        'service_time_s': 32.66076021673691,
                                        'service_energy_kj': 6.12389254063817,
                                        'occupation': 0.028461698108934768},
                    'selected_value': 67.21022761573143},
                   {'epoch': 6,
                    'task': 19,
                    'priority': 10.0,
                    'observed_workload_gcy': 975.9204749419978,
                    'workload_mean_gcy': 658.8470105442198,
                    'mu': 6.170103723610407,
                    'sigma': 0.8004843908360189,
                    'arrival_time_s': 1557.1310739493924,
                    'arrival_energy_kj': 349.8992548780802,
                    'execution_time_available_s': 136.123561264663,
                    'execution_energy_available_kj': 140.88067710150392,
                    'maximum_feasible_workload_gcy': 340.30890316165755,
                    'shadow_price_time': 0.0,
                    'shadow_price_energy': 0.0,
                    'completion_feasible': False,
                    'alternatives': [{'mode': 'skip',
                                      'mec': -1,
                                      'frequency_ghz': 0.0,
                                      'service_time_s': 0.0,
                                      'service_energy_kj': 0.0,
                                      'occupation': 0.0,
                                      'immediate_reward': 0.0,
                                      'future_value': 1.7464533558865465,
                                      'total_value': 1.7464533558865465,
                                      'resource_cost': 0.0,
                                      'successor': {'node': 19,
                                                    'time_s': 1557.1310739493924,
                                                    'energy_kj': 349.8992548780802,
                                                    'remaining_mask': 766899}}],
                    'selected_action': {'mode': 'skip',
                                        'mec': -1,
                                        'frequency_ghz': 0.0,
                                        'service_time_s': 0.0,
                                        'service_energy_kj': 0.0,
                                        'occupation': 0.0},
                    'selected_value': 1.7464533558865465}]}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
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
    data = RECORDED_DECISIONS
    assert all(data['validation_against_formal_result'].values())
    initial = data['top_decisions'][0]
    candidates = sorted(initial['candidate_values'],
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

    for task, y in ((3, .69), (19, .35)):
        ex = examples[task]
        mode = ex['selected_action']['mode']
        decision = 'Local' if mode == 'local' else 'skip'
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
            reason = f'Expected reward: {total:.1f} (complete) > {skip:.1f} (skip)'
        else:
            reason = f"Workload limit: {ex['maximum_feasible_workload_gcy']:.1f} Gcycles"
        fig.text(.57, y-.12, reason, fontsize=10, color='#555555')

    fig.savefig(HERE / 'figure4-v2.pdf', dpi=300,
                bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


if __name__ == '__main__':
    main()
