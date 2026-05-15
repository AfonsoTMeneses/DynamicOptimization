import argparse
import os
import re
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ALGORITHMS = ['NSGA2', 'SPEA2', 'SMS_EMOA', 'MOEAD_DE']

ALGORITHM_LABEL = {
    'NSGA2':    'NSGA-II',
    'SPEA2':    'SPEA2',
    'SMS_EMOA': 'SMS-EMOA',
    'MOEAD_DE': 'MOEA/D-DE',
}

ALGORITHM_COLORS = {
    'NSGA2':    '#378ADD',
    'SPEA2':    '#1D9E75',
    'SMS_EMOA': '#D85A30',
    'MOEAD_DE': '#7F77DD',
}

CASE_STUDY_SAMPLERS = ['GPSampler', 'TPESampler', 'CmaEsSampler']

SAMPLER_LABEL = {
    'GPSampler':    'GP',
    'TPESampler':   'TPE',
    'CmaEsSampler': 'CMA-ES',
}

SAMPLER_LINESTYLES = {
    'GPSampler':    '-',
    'TPESampler':   '--',
    'CmaEsSampler': ':',
}

DEFAULT_BASELINES = {
    'NSGA2':    0.9964,
    'SPEA2':    1.0043,
    'SMS_EMOA': 0.9993,
    'MOEAD_DE': 0.8894,
}

STUDY_FILENAME_REGEX = re.compile(
    r'^study_(.+)_([A-Za-z]+Sampler)_Problem_(\d+)\.db$'
)
TRIAL_WINDOW_LAST = 99


# DB I/O

def discover_studies(db_dir):
    studies = []
    for fname in sorted(os.listdir(db_dir)):
        m = STUDY_FILENAME_REGEX.match(fname)
        if not m:
            continue
        studies.append({
            'algorithm': m.group(1),
            'sampler':   m.group(2),
            'problem':   int(m.group(3)),
            'db_path':   os.path.join(db_dir, fname),
        })
    return studies


def read_trial_trajectory(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("""
            SELECT t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON tv.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE' AND t.number <= ?
            ORDER BY t.number
        """, (TRIAL_WINDOW_LAST,))
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return np.array([], dtype=int), np.array([], dtype=float)

    numbers = np.array([r[0] for r in rows], dtype=int)
    values  = np.array([r[1] for r in rows], dtype=float)
    finite  = np.isfinite(values)
    return numbers[finite], values[finite]


def best_so_far(numbers, values):
    if len(numbers) == 0:
        return np.array([]), np.array([])
    x = numbers + 1
    cummax = np.maximum.accumulate(values)
    return x, cummax


# Plotting

def plot_overlay(trajectories, baselines, output_path, title, mode='relative'):
    if mode not in ('absolute', 'relative'):
        raise ValueError(f"unknown overlay mode: {mode}")

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in ALGORITHMS:
        baseline_val = baselines.get(algo)
        if mode == 'relative' and (baseline_val is None or baseline_val <= 0):
            continue

        for sampler in CASE_STUDY_SAMPLERS:
            key = (algo, sampler)
            if key not in trajectories:
                continue
            x, y = trajectories[key]
            if len(x) == 0:
                continue

            y_plot = (y - baseline_val) / baseline_val * 100.0 if mode == 'relative' else y
            ax.plot(
                x, y_plot,
                color=ALGORITHM_COLORS[algo],
                linestyle=SAMPLER_LINESTYLES[sampler],
                linewidth=1.5,
                alpha=0.9,
            )

    if mode == 'relative':
        ax.axhline(0.0, color='gray', linestyle='-.', linewidth=1.0, alpha=0.7)
    elif baselines:
        for algo, bl in baselines.items():
            ax.axhline(
                bl,
                color=ALGORITHM_COLORS.get(algo, 'gray'),
                linestyle='-.', linewidth=0.8, alpha=0.45,
            )

    algo_handles = [
        Line2D([0], [0], color=ALGORITHM_COLORS[a], lw=2, label=ALGORITHM_LABEL[a])
        for a in ALGORITHMS
        if mode == 'absolute' or (baselines.get(a) is not None and baselines.get(a) > 0)
    ]
    sampler_handles = [
        Line2D([0], [0], color='black', linestyle=SAMPLER_LINESTYLES[s],
               lw=1.8, label=SAMPLER_LABEL[s])
        for s in CASE_STUDY_SAMPLERS
    ]
    leg1 = ax.legend(handles=algo_handles, title='Algorithm',
                     loc='upper right', fontsize=9, frameon=False,
                     bbox_to_anchor=(1.0, 0.62))
    ax.add_artist(leg1)
    ax.legend(handles=sampler_handles, title='Sampler',
              loc='upper right', fontsize=9, frameon=False,
              bbox_to_anchor=(1.0, 0.80))

    ax.set_xlabel('Trial')
    ax.set_xlim(1, 100)
    ax.grid(alpha=0.3)
    if mode == 'relative':
        ax.set_ylabel('Improvement over baseline (%)')
        ax.set_title(f'{title} (relative to baseline)')
    else:
        ax.set_ylabel('Best HV so far')
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# CLI

def parse_baseline_arg(s):
    out = {}
    if not s:
        return out
    for pair in s.split(','):
        pair = pair.strip()
        if '=' not in pair:
            continue
        k, v = pair.split('=', 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db-dir', required=True,
                    help='Directory containing study_*.db files.')
    ap.add_argument('--output-dir', required=True,
                    help='Directory for the output PDF.')
    ap.add_argument('--baseline', default='',
                    help='Inline baseline overrides, e.g. '
                         '"NSGA2=0.99,SPEA2=1.00,SMS_EMOA=0.99,MOEAD_DE=0.89".')
    ap.add_argument('--no-baseline', action='store_true',
                    help='Suppress baseline reference lines.')
    ap.add_argument('--title',
                    default='HPO convergence on the truss case study',
                    help='Figure title.')
    ap.add_argument('--overlay-mode', choices=('relative', 'absolute'),
                    default='relative',
                    help='Y-axis. "relative" plots (HV - baseline)/baseline in '
                         'percent so curves on different scales become '
                         'comparable. "absolute" plots raw best-HV-so-far. '
                         'Default: relative.')
    args = ap.parse_args()

    if not os.path.isdir(args.db_dir):
        sys.exit(f'--db-dir not found: {args.db_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size']   = 11
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

    if args.no_baseline:
        baselines = {}
    elif args.baseline:
        baselines = parse_baseline_arg(args.baseline)
    else:
        baselines = DEFAULT_BASELINES.copy()

    studies = discover_studies(args.db_dir)
    if not studies:
        sys.exit(f'No study .db files matched in {args.db_dir}')

    problems_seen = sorted({s['problem'] for s in studies})
    if len(problems_seen) > 1:
        print(f'WARNING: multiple problem ids found in --db-dir: {problems_seen}. '
              f'This script is intended for a single-problem case study. '
              f'All curves will be drawn anyway; later (algo, sampler) '
              f'duplicates will overwrite earlier ones.')

    trajectories = {}
    print(f'Found {len(studies)} study .db file(s).')
    for s in studies:
        numbers, values = read_trial_trajectory(s['db_path'])
        x, y = best_so_far(numbers, values)
        trajectories[(s['algorithm'], s['sampler'])] = (x, y)
        print(f"  {s['algorithm']:10s} {s['sampler']:15s} "
              f"problem {s['problem']:3d}: {len(x)} completed trials")

    overlay_path = os.path.join(args.output_dir, 'convergence_overlay.pdf')
    plot_overlay(trajectories, baselines, overlay_path, args.title,
                 mode=args.overlay_mode)
    print(f'\nWrote:\n  {overlay_path}')


if __name__ == '__main__':
    main()
