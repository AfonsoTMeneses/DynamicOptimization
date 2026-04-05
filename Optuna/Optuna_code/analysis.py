#!/usr/bin/env python3
"""Generates all thesis figures and tables from experiment results."""

import argparse, csv, json, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ALGORITHMS = ['NSGA2', 'SPEA2', 'SMS_EMOA', 'MOEAD_DE']

ALGORITHM_DISPLAY_LABELS = {
    'NSGA2': 'NSGA-II',
    'SPEA2': 'SPEA2',
    'SMS_EMOA': 'SMS-EMOA',
    'MOEAD_DE': 'MOEA/D-DE'
}

ALGORITHM_COLORS = {
    'NSGA2': '#378ADD',
    'SPEA2': '#1D9E75',
    'SMS_EMOA': '#D85A30',
    'MOEAD_DE': '#7F77DD'
}

SAMPLER_STYLES = {
    'TPESampler':        {'ls': '-',  'marker': 'o', 'color': '#E24B4A'},
    'NSGAIISampler':     {'ls': '--', 'marker': 's', 'color': '#EF9F27'},
    'CmaEsSampler':      {'ls': '-.', 'marker': '^', 'color': '#639922'},
    'RandomSampler':     {'ls': ':',  'marker': 'D', 'color': '#D4537E'},
    'QMCSampler':        {'ls': '-',  'marker': 'v', 'color': '#854F0B'},
    'NSGAIIISampler':    {'ls': '--', 'marker': 'P', 'color': '#534AB7'},
    'GPSampler':         {'ls': '-.', 'marker': 'X', 'color': '#0F6E56'},
    'BruteForceSampler': {'ls': ':',  'marker': '*', 'color': '#5F5E5A'},
    'GridSampler':       {'ls': '-',  'marker': 'h', 'color': '#993556'},
}

TOTAL_BENCHMARK_PROBLEMS = 50

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_baseline_csv_prefix(results_dir):
    """Returns the filename prefix for baseline CSVs, or None if not found."""
    for prefix in ['baseline_benchmark', 'Get_minimum_runs']:
        if os.path.isfile(os.path.join(results_dir, f'minimum_runs_{prefix}_NSGA2.csv')):
            return prefix
    return None


def parse_baseline_results_csv(filepath):
    """Parses a baseline minimum_runs CSV into a list of per-problem dicts."""
    problems = []

    with open(filepath) as f:
        header = next(csv.reader(f))
        all_rows = list(csv.reader(f))

    confidence_interval_column_index = None
    for col_index, col_name in enumerate(header):
        if 'confidence_interval_95' in col_name:
            confidence_interval_column_index = col_index
            break

    row_index = 0
    while row_index < len(all_rows):
        row = all_rows[row_index]

        if not row or not row[0] or not row[0].strip():
            row_index += 1
            continue

        if row[0].startswith('['):
            if problems:
                try:
                    hv_values = json.loads(row[0])
                    problems[-1]['hv_values'] = hv_values
                    problems[-1]['mean_hv'] = np.mean(hv_values)
                    problems[-1]['std_hv'] = np.std(hv_values, ddof=1) if len(hv_values) > 1 else 0
                    problems[-1]['pct_feasible'] = sum(1 for hv in hv_values if hv > 0) / len(hv_values) * 100
                except:
                    pass
            row_index += 1
            continue

        problem_name = row[0].lstrip('#')

        min_runs_value = None
        if confidence_interval_column_index and len(row) > confidence_interval_column_index and row[confidence_interval_column_index]:
            try:
                ci_array = json.loads(row[confidence_interval_column_index])
                min_runs_value = ci_array[-1]
            except:
                pass

        problem_instance = len(problems) + 1
        try:
            if len(row) > 3 and row[3]:
                problem_instance = int(row[3])
        except:
            pass

        problems.append({
            'name': problem_name,
            'instance': problem_instance,
            'min_runs': min_runs_value,
            'hv_values': [],
            'mean_hv': 0,
            'std_hv': 0,
            'pct_feasible': 0,
        })
        row_index += 1

    return problems


def load_hpo_results(results_dir, hpo_dir_name):
    """Loads HPO result CSVs into a nested dict: results[algorithm][sampler][problem_id]."""
    hpo_base_dir = os.path.join(results_dir, hpo_dir_name)
    all_results = defaultdict(lambda: defaultdict(dict))

    if not os.path.isdir(hpo_base_dir):
        return all_results

    for algorithm in ALGORITHMS:
        algorithm_dir = os.path.join(hpo_base_dir, algorithm)
        if not os.path.isdir(algorithm_dir):
            continue

        for iteration_dir_name in os.listdir(algorithm_dir):
            iteration_dir_path = os.path.join(algorithm_dir, iteration_dir_name)
            if not os.path.isdir(iteration_dir_path):
                continue

            for filename in os.listdir(iteration_dir_path):
                if not filename.endswith('.csv') or not filename.startswith(algorithm) or 'convergence' in filename:
                    continue

                try:
                    dataframe = pd.read_csv(os.path.join(iteration_dir_path, filename))

                    for _, row in dataframe.iterrows():
                        sampler_name = str(row.get('sampler', ''))
                        problem_id = int(row.get('problem_instance', 0))

                        result = {
                            'hv_value': float(row.get('hv_value', 0)),
                            'params': str(row.get('params', '')),
                            'All_HV': [],
                            'param_importances': {},
                            'elapsed_seconds': None,
                        }

                        if 'elapsed_seconds' in row and pd.notna(row.get('elapsed_seconds')):
                            result['elapsed_seconds'] = float(row['elapsed_seconds'])

                        if 'All_HV' in row and pd.notna(row.get('All_HV')):
                            try:
                                result['All_HV'] = json.loads(row['All_HV'])
                            except:
                                pass

                        if 'param_importances' in row and pd.notna(row.get('param_importances')):
                            try:
                                result['param_importances'] = json.loads(row['param_importances'])
                            except:
                                pass

                        all_results[algorithm][sampler_name][problem_id] = result

                except Exception as e:
                    print(f"  warning: couldn't parse {filename}: {e}")

    return all_results


def load_convergence_data(results_dir, hpo_dir_name):
    """Loads convergence CSVs into a dict keyed by (algorithm, sampler, problem_id)."""
    hpo_base_dir = os.path.join(results_dir, hpo_dir_name)
    convergence_data = {}

    if not os.path.isdir(hpo_base_dir):
        return convergence_data

    for algorithm in ALGORITHMS:
        algorithm_dir = os.path.join(hpo_base_dir, algorithm)
        if not os.path.isdir(algorithm_dir):
            continue

        for iteration_dir_name in os.listdir(algorithm_dir):
            iteration_dir_path = os.path.join(algorithm_dir, iteration_dir_name)
            if not os.path.isdir(iteration_dir_path):
                continue

            for filename in os.listdir(iteration_dir_path):
                if not filename.startswith('convergence_') or not filename.endswith('.csv'):
                    continue
                try:
                    tokens = filename.replace('convergence_', '').replace('.csv', '').split('_')
                    sampler_token_index = next(i for i, t in enumerate(tokens) if 'Sampler' in t)
                    sampler_name = tokens[sampler_token_index]
                    problem_id = int(tokens[-1])
                    convergence_data[(algorithm, sampler_name, problem_id)] = pd.read_csv(
                        os.path.join(iteration_dir_path, filename))
                except:
                    pass

    return convergence_data


def load_pareto_fronts(results_dir, hpo_dir_name):
    """Loads all_fronts CSVs into a dict keyed by (algorithm, sampler, problem_id)."""
    hpo_base_dir = os.path.join(results_dir, hpo_dir_name)
    fronts = {}

    if not os.path.isdir(hpo_base_dir):
        return fronts

    for root, _, files in os.walk(hpo_base_dir):
        for filename in files:
            if not filename.startswith('all_fronts_') or not filename.endswith('.csv'):
                continue
            try:
                dataframe = pd.read_csv(os.path.join(root, filename))
                if 'algorithm' not in dataframe.columns or 'sampler' not in dataframe.columns:
                    continue

                for (algorithm, sampler_name, problem_id), group in dataframe.groupby(
                        ['algorithm', 'sampler', 'problem_instance']):
                    fronts[(algorithm, sampler_name, int(problem_id))] = group
            except:
                pass

    return fronts


# ---------------------------------------------------------------------------
# Baseline outputs
# ---------------------------------------------------------------------------

def generate_baseline_outputs(baseline_data, output_dir):
    """Generates baseline summary CSV, boxplot, and feasibility heatmap."""
    summary_rows = []
    for problem_id in range(1, TOTAL_BENCHMARK_PROBLEMS + 1):
        row = {'problem': problem_id, 'name': '?'}
        for algorithm in ALGORITHMS:
            algorithm_problems = baseline_data.get(algorithm, [])
            if problem_id <= len(algorithm_problems):
                row['name'] = algorithm_problems[problem_id - 1]['name']
                row[f'{algorithm}_mean_hv'] = algorithm_problems[problem_id - 1]['mean_hv']
                row[f'{algorithm}_std_hv'] = algorithm_problems[problem_id - 1]['std_hv']
                row[f'{algorithm}_min_runs'] = algorithm_problems[problem_id - 1]['min_runs']
                row[f'{algorithm}_pct_feas'] = algorithm_problems[problem_id - 1]['pct_feasible']
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, 'baseline_summary.csv'), index=False)
    print("  baseline_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []
    for algorithm in ALGORITHMS:
        mean_hv_values = [entry['mean_hv'] for entry in baseline_data.get(algorithm, []) if entry['mean_hv'] > 0]
        boxplot_data.append(mean_hv_values)
        boxplot_labels.append(ALGORITHM_DISPLAY_LABELS[algorithm])
        boxplot_colors.append(ALGORITHM_COLORS[algorithm])

    box_artists = ax.boxplot(boxplot_data, tick_labels=boxplot_labels, patch_artist=True, widths=0.6,
                             medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(box_artists['boxes'], boxplot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Mean normalized HV')
    ax.set_title('Baseline HV distribution (solvable problems)')
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(os.path.join(output_dir, 'baseline_hv_boxplot.pdf'))
    plt.close(fig)
    print("  baseline_hv_boxplot.pdf")

    feasibility_matrix = np.zeros((TOTAL_BENCHMARK_PROBLEMS, len(ALGORITHMS)))
    for col_index, algorithm in enumerate(ALGORITHMS):
        for row_index in range(TOTAL_BENCHMARK_PROBLEMS):
            algorithm_problems = baseline_data.get(algorithm, [])
            if row_index < len(algorithm_problems):
                feasibility_matrix[row_index, col_index] = algorithm_problems[row_index]['pct_feasible']

    partially_or_fully_infeasible = (
        np.any((feasibility_matrix > 0) & (feasibility_matrix < 100), axis=1)
        | np.all(feasibility_matrix == 0, axis=1)
    )
    infeasible_problem_indices = np.where(partially_or_fully_infeasible)[0]

    if len(infeasible_problem_indices) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, max(4, len(infeasible_problem_indices) * 0.3)))
    feasibility_subset = feasibility_matrix[infeasible_problem_indices, :]

    im = ax.imshow(feasibility_subset, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGORITHM_DISPLAY_LABELS[a] for a in ALGORITHMS], rotation=45, ha='right')

    y_tick_labels = []
    for problem_index in infeasible_problem_indices:
        algorithm_problems = baseline_data.get(ALGORITHMS[0], [])
        name = algorithm_problems[problem_index]['name'][:25] if problem_index < len(algorithm_problems) else f'P{problem_index+1}'
        y_tick_labels.append(f"{problem_index+1}. {name}")
    ax.set_yticks(range(len(infeasible_problem_indices)))
    ax.set_yticklabels(y_tick_labels, fontsize=8)

    for row_index in range(len(infeasible_problem_indices)):
        for col_index in range(len(ALGORITHMS)):
            value = feasibility_subset[row_index, col_index]
            text_color = 'white' if value < 40 or value > 85 else 'black'
            ax.text(col_index, row_index, f'{value:.0f}%', ha='center', va='center', fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label='Feasibility (%)')
    ax.set_title('Feasibility rate (default params)')
    fig.savefig(os.path.join(output_dir, 'feasibility_heatmap.pdf'))
    plt.close(fig)
    print("  feasibility_heatmap.pdf")


# ---------------------------------------------------------------------------
# HPO vs baseline comparison
# ---------------------------------------------------------------------------

def generate_hpo_vs_baseline_comparison(baseline_data, hpo_results, output_dir):
    """Generates comparison tables, statistical tests, and improvement charts."""
    if not hpo_results:
        print("  No HPO results found, skipping comparison.")
        return None

    comparison_rows = []

    for algorithm in ALGORITHMS:
        if algorithm not in hpo_results:
            continue
        for sampler_name in hpo_results[algorithm]:
            for problem_id in sorted(hpo_results[algorithm][sampler_name].keys()):
                hpo_entry = hpo_results[algorithm][sampler_name][problem_id]
                algorithm_problems = baseline_data.get(algorithm, [])
                baseline_entry = algorithm_problems[problem_id - 1] if problem_id <= len(algorithm_problems) else None

                baseline_hv = baseline_entry['mean_hv'] if baseline_entry else 0
                hpo_hv = hpo_entry['hv_value']

                p_value = None
                if baseline_entry and len(baseline_entry.get('hv_values', [])) > 1 and len(hpo_entry.get('All_HV', [])) > 1:
                    baseline_hv_values = baseline_entry['hv_values']
                    hpo_hv_values = hpo_entry['All_HV']
                    if len(set(baseline_hv_values)) > 1 or len(set(hpo_hv_values)) > 1:
                        try:
                            _, p_value = stats.mannwhitneyu(hpo_hv_values, baseline_hv_values, alternative='greater')
                        except:
                            pass

                comparison_rows.append({
                    'algorithm': algorithm,
                    'sampler': sampler_name,
                    'problem': problem_id,
                    'problem_name': baseline_entry['name'] if baseline_entry else '?',
                    'baseline_hv': round(baseline_hv, 6),
                    'hpo_hv': round(hpo_hv, 6),
                    'delta_hv': round(hpo_hv - baseline_hv, 6),
                    'elapsed_s': hpo_entry.get('elapsed_seconds'),
                    'p_value': round(p_value, 6) if p_value is not None else None,
                    'significant': p_value < 0.05 if p_value is not None else None,
                })

    if not comparison_rows:
        print("  No comparison data generated.")
        return None

    comparison_dataframe = pd.DataFrame(comparison_rows)
    comparison_dataframe.to_csv(os.path.join(output_dir, 'hpo_vs_baseline_full.csv'), index=False)
    print(f"  hpo_vs_baseline_full.csv ({len(comparison_dataframe)} rows)")

    summary = comparison_dataframe.groupby(['algorithm', 'sampler']).agg(
        mean_delta_hv=('delta_hv', 'mean'),
        n_improved=('delta_hv', lambda x: (x > 0).sum()),
        n_significant=('significant', lambda x: x.sum() if x.notna().any() else 0),
        n_problems=('problem', 'count'),
        total_time=('elapsed_s', lambda x: x.dropna().sum()),
        mean_time=('elapsed_s', lambda x: x.dropna().mean() if x.notna().any() else None),
    ).reset_index()

    summary.to_csv(os.path.join(output_dir, 'hpo_summary_by_sampler.csv'), index=False)
    print("  hpo_summary_by_sampler.csv")

    tex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        r'\caption{HPO improvement over default parameters. $\Delta\overline{HV}$: mean gain. Sig.: significant wins ($p<0.05$, Mann-Whitney U).}',
        r'\label{tab:hpo_comparison}',
        r'\begin{tabular}{llcccc}',
        r'\toprule',
        r'\textbf{Alg.} & \textbf{Sampler} & $\Delta\overline{HV}$ & \textbf{\% impr.} & \textbf{Sig.} & $\overline{t}$ (s) \\',
        r'\midrule',
    ]
    for _, row in summary.iterrows():
        algorithm_label = ALGORITHM_DISPLAY_LABELS.get(row['algorithm'], row['algorithm'])
        pct_improved = row['n_improved'] / row['n_problems'] * 100 if row['n_problems'] > 0 else 0
        time_str = f"{row['mean_time']:.0f}" if pd.notna(row['mean_time']) else '--'
        tex.append(
            f"{algorithm_label} & {row['sampler']} & {row['mean_delta_hv']:.4f} & "
            f"{pct_improved:.0f}\\% & {int(row['n_significant'])}/{int(row['n_problems'])} & {time_str} \\\\"
        )
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(output_dir, 'comparison_table.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print("  comparison_table.tex")

    win_tie_loss_rows = []
    for algorithm in ALGORITHMS:
        if algorithm not in hpo_results:
            continue
        for sampler_name in hpo_results[algorithm]:
            wins, ties, losses, total_tested = 0, 0, 0, 0
            for problem_id in hpo_results[algorithm][sampler_name]:
                algorithm_problems = baseline_data.get(algorithm, [])
                baseline_entry = algorithm_problems[problem_id - 1] if problem_id <= len(algorithm_problems) else None
                if not baseline_entry:
                    continue

                baseline_hv_values = baseline_entry.get('hv_values', [])
                hpo_hv_values = hpo_results[algorithm][sampler_name][problem_id].get('All_HV', [])
                if len(baseline_hv_values) < 2 or len(hpo_hv_values) < 2:
                    continue

                total_tested += 1
                try:
                    _, two_sided_p_value = stats.mannwhitneyu(hpo_hv_values, baseline_hv_values, alternative='two-sided')
                    if two_sided_p_value < 0.05:
                        if np.mean(hpo_hv_values) > np.mean(baseline_hv_values):
                            wins += 1
                        else:
                            losses += 1
                    else:
                        ties += 1
                except:
                    ties += 1

            if total_tested > 0:
                win_tie_loss_rows.append({
                    'algorithm': algorithm, 'sampler': sampler_name,
                    'wins': wins, 'ties': ties, 'losses': losses, 'total': total_tested
                })

    if win_tie_loss_rows:
        win_tie_loss_dataframe = pd.DataFrame(win_tie_loss_rows)
        win_tie_loss_dataframe.to_csv(os.path.join(output_dir, 'win_tie_loss.csv'), index=False)

        tex = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\small',
            r'\caption{Win/tie/loss (Mann-Whitney U, $\alpha=0.05$).}',
            r'\label{tab:wtl}',
            r'\begin{tabular}{llcccc}',
            r'\toprule',
            r'\textbf{Alg.} & \textbf{Sampler} & \textbf{W} & \textbf{T} & \textbf{L} & \textbf{N} \\',
            r'\midrule',
        ]
        for _, row in win_tie_loss_dataframe.iterrows():
            tex.append(
                f"{ALGORITHM_DISPLAY_LABELS.get(row['algorithm'], row['algorithm'])} & {row['sampler']} & "
                f"{row['wins']} & {row['ties']} & {row['losses']} & {row['total']} \\\\"
            )
        tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        with open(os.path.join(output_dir, 'win_tie_loss_table.tex'), 'w') as f:
            f.write('\n'.join(tex))
        print("  win_tie_loss.csv + win_tie_loss_table.tex")

    all_samplers = sorted(summary['sampler'].unique())
    if not all_samplers:
        return comparison_dataframe

    fig, ax = plt.subplots(figsize=(max(8, len(all_samplers) * 2), 5))
    x_positions = np.arange(len(ALGORITHMS))
    bar_width = 0.8 / max(len(all_samplers), 1)

    for sampler_index, sampler_name in enumerate(all_samplers):
        delta_hv_per_algorithm = []
        for algorithm in ALGORITHMS:
            matching_rows = summary[(summary['algorithm'] == algorithm) & (summary['sampler'] == sampler_name)]
            delta_hv_per_algorithm.append(matching_rows['mean_delta_hv'].values[0] if len(matching_rows) > 0 else 0)

        style = SAMPLER_STYLES.get(sampler_name, {})
        offset = (sampler_index - len(all_samplers) / 2 + 0.5) * bar_width
        ax.bar(x_positions + offset, delta_hv_per_algorithm, bar_width * 0.9,
               label=sampler_name, alpha=0.8, color=style.get('color'))

    ax.set_xticks(x_positions)
    ax.set_xticklabels([ALGORITHM_DISPLAY_LABELS[a] for a in ALGORITHMS])
    ax.set_ylabel('Mean $\\Delta$HV')
    ax.set_title('HPO improvement')
    ax.legend(fontsize=8, ncol=min(3, len(all_samplers)))
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(os.path.join(output_dir, 'hpo_improvement_barchart.pdf'))
    plt.close(fig)
    print("  hpo_improvement_barchart.pdf")

    return comparison_dataframe


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------

def plot_convergence(convergence_data, output_dir, prefix='benchmark'):
    """Plots average best-so-far convergence curves per algorithm, one line per sampler."""
    if not convergence_data:
        print("  No convergence data.")
        print(f"  (looked in hpo_*_Results/{{ALG}}/{{iterations}}/ for convergence_*.csv)")
        return

    grouped_by_algorithm = defaultdict(lambda: defaultdict(list))
    for (algorithm, sampler_name, problem_id), dataframe in convergence_data.items():
        if 'best_so_far' in dataframe.columns:
            grouped_by_algorithm[algorithm][sampler_name].append(dataframe)

    for algorithm in ALGORITHMS:
        if algorithm not in grouped_by_algorithm:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for sampler_name, sampler_dataframes in sorted(grouped_by_algorithm[algorithm].items()):
            max_trials = max(len(df) for df in sampler_dataframes)
            average_best_so_far = np.zeros(max_trials)
            count_per_trial = np.zeros(max_trials)

            for dataframe in sampler_dataframes:
                best_so_far_values = dataframe['best_so_far'].values
                average_best_so_far[:len(best_so_far_values)] += best_so_far_values
                count_per_trial[:len(best_so_far_values)] += 1

            count_per_trial[count_per_trial == 0] = 1
            average_best_so_far /= count_per_trial

            style = SAMPLER_STYLES.get(sampler_name, {'ls': '-', 'marker': 'o', 'color': 'gray'})
            marker_interval = max(1, max_trials // 10)
            ax.plot(
                range(1, max_trials + 1), average_best_so_far,
                label=sampler_name, lw=1.5,
                ls=style['ls'], marker=style['marker'],
                ms=4, markevery=marker_interval, color=style['color']
            )

        ax.set_xlabel('Optuna trial')
        ax.set_ylabel('Best HV so far (avg. across problems)')
        ax.set_title(f'{ALGORITHM_DISPLAY_LABELS[algorithm]} — sampler convergence ({prefix})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.savefig(os.path.join(output_dir, f'convergence_{prefix}_{algorithm}.pdf'))
        plt.close(fig)
        print(f"  convergence_{prefix}_{algorithm}.pdf")


def plot_convergence_single(convergence_data, target_problem_id, output_dir, prefix='truss'):
    """Plots convergence for a single problem (e.g. the truss)."""
    relevant_entries = {
        (algorithm, sampler_name): dataframe
        for (algorithm, sampler_name, problem_id), dataframe in convergence_data.items()
        if problem_id == target_problem_id
    }

    if not relevant_entries:
        print(f"  No convergence data for problem {target_problem_id}.")
        return

    grouped_by_algorithm = defaultdict(dict)
    for (algorithm, sampler_name), dataframe in relevant_entries.items():
        grouped_by_algorithm[algorithm][sampler_name] = dataframe

    for algorithm, sampler_dataframes in grouped_by_algorithm.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for sampler_name, dataframe in sorted(sampler_dataframes.items()):
            if 'best_so_far' not in dataframe.columns:
                continue
            best_so_far_values = dataframe['best_so_far'].values
            style = SAMPLER_STYLES.get(sampler_name, {'ls': '-', 'marker': 'o', 'color': 'gray'})
            marker_interval = max(1, len(best_so_far_values) // 10)
            ax.plot(
                range(1, len(best_so_far_values) + 1), best_so_far_values,
                label=sampler_name, lw=1.5,
                ls=style['ls'], marker=style['marker'],
                ms=4, markevery=marker_interval, color=style['color']
            )

        ax.set_xlabel('Optuna trial')
        ax.set_ylabel('Best HV so far')
        ax.set_title(f'{ALGORITHM_DISPLAY_LABELS.get(algorithm, algorithm)} — convergence ({prefix}, prob {target_problem_id})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.savefig(os.path.join(output_dir, f'convergence_{prefix}_{algorithm}_prob{target_problem_id}.pdf'))
        plt.close(fig)
        print(f"  convergence_{prefix}_{algorithm}_prob{target_problem_id}.pdf")


# ---------------------------------------------------------------------------
# Pareto front scatter plots
# ---------------------------------------------------------------------------

def plot_fronts(pareto_fronts, output_dir, prefix=''):
    """Plots Pareto front scatter for each problem."""
    if not pareto_fronts:
        print("  No Pareto front data.")
        return

    grouped_by_problem = defaultdict(dict)
    for (algorithm, sampler_name, problem_id), dataframe in pareto_fronts.items():
        grouped_by_problem[problem_id][(algorithm, sampler_name)] = dataframe

    for problem_id, entries in sorted(grouped_by_problem.items()):
        fig, ax = plt.subplots(figsize=(8, 6))

        for (algorithm, sampler_name), dataframe in sorted(entries.items()):
            objective_columns = [col for col in dataframe.columns if col.startswith('obj_')]
            if len(objective_columns) < 2:
                continue

            style = SAMPLER_STYLES.get(sampler_name, {'marker': 'o'})
            label = f"{ALGORITHM_DISPLAY_LABELS.get(algorithm, algorithm)} ({sampler_name})"
            ax.scatter(
                dataframe[objective_columns[0]], dataframe[objective_columns[1]],
                c=ALGORITHM_COLORS.get(algorithm, 'gray'),
                marker=style['marker'], s=20, alpha=0.6,
                label=label, edgecolors='none'
            )

        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title(f'Pareto fronts — problem {problem_id}')
        ax.legend(fontsize=7, markerscale=1.5)
        ax.grid(alpha=0.2)

        tag = f'_{prefix}' if prefix else ''
        fig.savefig(os.path.join(output_dir, f'pareto_front{tag}_prob{problem_id}.pdf'))
        plt.close(fig)
        print(f"  pareto_front{tag}_prob{problem_id}.pdf")


# ---------------------------------------------------------------------------
# Hyperparameter importance
# ---------------------------------------------------------------------------

def analyze_importance(hpo_results, output_dir):
    """Generates importance bar charts and LaTeX table from fANOVA data."""
    if not hpo_results:
        print("  No HPO results, skipping importance analysis.")
        return

    importance_by_algorithm = defaultdict(lambda: defaultdict(list))

    for algorithm in ALGORITHMS:
        if algorithm not in hpo_results:
            continue
        for sampler_name in hpo_results[algorithm]:
            for problem_id, result in hpo_results[algorithm][sampler_name].items():
                for param_name, importance_value in result.get('param_importances', {}).items():
                    importance_by_algorithm[algorithm][param_name].append(importance_value)

    if not importance_by_algorithm:
        print("  No importance data in the CSVs.")
        return

    importance_rows = []
    for algorithm in ALGORITHMS:
        if algorithm not in importance_by_algorithm:
            continue
        for param_name, values in sorted(importance_by_algorithm[algorithm].items()):
            importance_rows.append({
                'algorithm': algorithm,
                'parameter': param_name,
                'mean_importance': np.mean(values),
                'std_importance': np.std(values, ddof=1) if len(values) > 1 else 0,
                'n': len(values),
            })

    importance_dataframe = pd.DataFrame(importance_rows)
    importance_dataframe.to_csv(os.path.join(output_dir, 'param_importance.csv'), index=False)
    print("  param_importance.csv")

    for algorithm in ALGORITHMS:
        subset = importance_dataframe[importance_dataframe['algorithm'] == algorithm].sort_values('mean_importance', ascending=True)
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, max(3, len(subset) * 0.4)))
        ax.barh(
            range(len(subset)), subset['mean_importance'],
            xerr=subset['std_importance'],
            color=ALGORITHM_COLORS.get(algorithm, 'gray'), alpha=0.7, capsize=3
        )
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(subset['parameter'])
        ax.set_xlabel('Mean importance (fANOVA)')
        ax.set_title(f'{ALGORITHM_DISPLAY_LABELS[algorithm]} — hyperparameter importance')
        ax.grid(axis='x', alpha=0.3)

        fig.savefig(os.path.join(output_dir, f'param_importance_{algorithm}.pdf'))
        plt.close(fig)
        print(f"  param_importance_{algorithm}.pdf")

    tex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        r'\caption{Mean hyperparameter importance (fANOVA) across all benchmark problems.}',
        r'\label{tab:importance}',
        r'\begin{tabular}{llcc}',
        r'\toprule',
        r'\textbf{Algorithm} & \textbf{Parameter} & \textbf{Importance} & \textbf{N} \\',
        r'\midrule',
    ]
    for _, row in importance_dataframe.sort_values(['algorithm', 'mean_importance'], ascending=[True, False]).iterrows():
        tex.append(
            f"{ALGORITHM_DISPLAY_LABELS.get(row['algorithm'], row['algorithm'])} & "
            f"{row['parameter']} & "
            f"{row['mean_importance']:.3f} $\\pm$ {row['std_importance']:.3f} & "
            f"{int(row['n'])} \\\\"
        )
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(output_dir, 'param_importance_table.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print("  param_importance_table.tex")


# ---------------------------------------------------------------------------
# Computational cost
# ---------------------------------------------------------------------------

def analyze_costs(hpo_results, output_dir):
    """Generates computational cost summary CSV and LaTeX table."""
    if not hpo_results:
        print("  No HPO results, skipping cost analysis.")
        return

    cost_rows = []
    for algorithm in ALGORITHMS:
        if algorithm not in hpo_results:
            continue
        for sampler_name in hpo_results[algorithm]:
            elapsed_times = [
                result['elapsed_seconds']
                for result in hpo_results[algorithm][sampler_name].values()
                if result.get('elapsed_seconds') is not None
            ]
            if not elapsed_times:
                continue
            cost_rows.append({
                'algorithm': algorithm,
                'sampler': sampler_name,
                'n': len(elapsed_times),
                'total_h': sum(elapsed_times) / 3600,
                'mean_s': np.mean(elapsed_times),
                'median_s': np.median(elapsed_times),
                'max_s': max(elapsed_times),
            })

    if not cost_rows:
        print("  No timing data found in the CSVs.")
        return

    cost_dataframe = pd.DataFrame(cost_rows)
    cost_dataframe.to_csv(os.path.join(output_dir, 'computational_cost.csv'), index=False)
    print("  computational_cost.csv")

    tex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        r'\caption{Computational cost of HPO per algorithm and sampler.}',
        r'\label{tab:cost}',
        r'\begin{tabular}{llcccc}',
        r'\toprule',
        r'\textbf{Alg.} & \textbf{Sampler} & \textbf{Total (h)} & $\overline{t}$ (s) & \textbf{Med.} (s) & \textbf{N} \\',
        r'\midrule',
    ]
    for _, row in cost_dataframe.iterrows():
        tex.append(
            f"{ALGORITHM_DISPLAY_LABELS.get(row['algorithm'], row['algorithm'])} & "
            f"{row['sampler']} & "
            f"{row['total_h']:.1f} & {row['mean_s']:.0f} & {row['median_s']:.0f} & {int(row['n'])} \\\\"
        )
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(output_dir, 'computational_cost_table.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print("  computational_cost_table.tex")


# ---------------------------------------------------------------------------
# Combined Pareto fronts
# ---------------------------------------------------------------------------

def plot_combined_fronts(results_dir, output_dir, combined_fronts_dir_name):
    """Plots combined Pareto front per problem: cloud + PF line + HPO markers."""
    combined_fronts_dir = os.path.join(results_dir, combined_fronts_dir_name)
    if not os.path.isdir(combined_fronts_dir):
        print(f"  {combined_fronts_dir_name}/ not found — run compute_cPF.jl first.")
        return

    for filename in sorted(os.listdir(combined_fronts_dir)):
        if not filename.startswith('combined_front_') or not filename.endswith('.csv'):
            continue

        filepath = os.path.join(combined_fronts_dir, filename)
        try:
            dataframe = pd.read_csv(filepath)
        except:
            continue

        objective_columns = [col for col in dataframe.columns if col.startswith('obj_')]
        if len(objective_columns) < 2:
            continue

        if 'is_nd' not in dataframe.columns:
            print(f"  skipping {filename} — no is_nd column (re-run compute_cPF.jl)")
            continue

        problem_label = filename.replace('combined_front_Problem_', '').replace('.csv', '')

        fig, ax = plt.subplots(figsize=(8, 6))

        dominated_solutions = dataframe[~dataframe['is_nd']]
        non_dominated_solutions = dataframe[dataframe['is_nd']]

        baseline_dominated = dominated_solutions[dominated_solutions['source'].str.startswith('baseline')]
        hpo_dominated = dominated_solutions[dominated_solutions['source'].str.startswith('hpo')]

        if len(baseline_dominated) > 0:
            ax.scatter(baseline_dominated[objective_columns[0]], baseline_dominated[objective_columns[1]],
                       c='#D3D1C7', marker='.', s=8, alpha=0.3,
                       label=f'Baseline dominated ({len(baseline_dominated)})', edgecolors='none')

        if len(hpo_dominated) > 0:
            ax.scatter(hpo_dominated[objective_columns[0]], hpo_dominated[objective_columns[1]],
                       c='#F5C4B3', marker='.', s=8, alpha=0.3,
                       label=f'HPO dominated ({len(hpo_dominated)})', edgecolors='none')

        if len(non_dominated_solutions) > 0:
            sorted_non_dominated = non_dominated_solutions.sort_values(objective_columns[0])
            ax.plot(sorted_non_dominated[objective_columns[0]], sorted_non_dominated[objective_columns[1]],
                    color='black', linewidth=1.5, zorder=5, label=f'Combined PF ({len(non_dominated_solutions)})')
            ax.scatter(sorted_non_dominated[objective_columns[0]], sorted_non_dominated[objective_columns[1]],
                       c='black', s=15, zorder=6, edgecolors='none')

            hpo_on_front = non_dominated_solutions[non_dominated_solutions['source'].str.startswith('hpo')]
            if len(hpo_on_front) > 0:
                ax.scatter(hpo_on_front[objective_columns[0]], hpo_on_front[objective_columns[1]],
                           facecolors='none', edgecolors='#E24B4A', s=40, linewidths=1.2,
                           zorder=7, label=f'HPO on front ({len(hpo_on_front)})')

        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title(f'Problem {problem_label} — combined Pareto front')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.15)

        fig.savefig(os.path.join(output_dir, f'combined_front_prob{problem_label}.pdf'))
        plt.close(fig)
        print(f"  combined_front_prob{problem_label}.pdf")


# ---------------------------------------------------------------------------
# IGD+ analysis
# ---------------------------------------------------------------------------

def analyze_igd(results_dir, output_dir, combined_fronts_dir_name):
    """Generates IGD+ summary tables, heatmaps, and comparison bar charts."""
    igd_filepath = os.path.join(results_dir, combined_fronts_dir_name, 'igd_results.csv')
    if not os.path.isfile(igd_filepath):
        print(f"  igd_results.csv not found in {combined_fronts_dir_name}/")
        return

    igd_dataframe = pd.read_csv(igd_filepath)
    print(f"  loaded {len(igd_dataframe)} IGD+ entries")

    tag = combined_fronts_dir_name.replace('combined_fronts_', '')

    summary = igd_dataframe.groupby(['source_type', 'algorithm', 'sampler']).agg(
        mean_igd=('igd_plus', 'mean'),
        std_igd=('igd_plus', 'std'),
        n=('problem', 'count'),
    ).reset_index()

    summary.to_csv(os.path.join(output_dir, f'igd_summary_{tag}.csv'), index=False)
    print(f"  igd_summary_{tag}.csv")

    tex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        r'\caption{IGD+ against the combined Pareto front (lower is better).}',
        r'\label{tab:igd_' + tag + r'}',
        r'\begin{tabular}{lllcc}',
        r'\toprule',
        r'\textbf{Type} & \textbf{Alg.} & \textbf{Sampler} & $\overline{IGD^+}$ & \textbf{N} \\',
        r'\midrule',
    ]

    for _, row in summary.sort_values(['algorithm', 'source_type', 'sampler']).iterrows():
        algorithm_label = ALGORITHM_DISPLAY_LABELS.get(row['algorithm'], row['algorithm'])
        sampler_label = row['sampler'] if row['sampler'] else '--'
        igd_str = f"{row['mean_igd']:.4f} $\\pm$ {row['std_igd']:.4f}" if pd.notna(row['std_igd']) else f"{row['mean_igd']:.4f}"
        tex.append(f"{row['source_type']} & {algorithm_label} & {sampler_label} & {igd_str} & {int(row['n'])} \\\\")

    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(output_dir, f'igd_table_{tag}.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print(f"  igd_table_{tag}.tex")

    hpo_only = igd_dataframe[igd_dataframe['source_type'] == 'hpo']
    if not hpo_only.empty:
        sampler_summary = hpo_only.groupby(['sampler', 'algorithm']).agg(
            mean_igd=('igd_plus', 'mean'),
            std_igd=('igd_plus', 'std'),
            n=('problem', 'count'),
        ).reset_index()

        sampler_summary.to_csv(os.path.join(output_dir, f'igd_per_sampler_{tag}.csv'), index=False)
        print(f"  igd_per_sampler_{tag}.csv")

        tex2 = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\small',
            r'\caption{IGD+ per sampler and algorithm (HPO configurations only).}',
            r'\label{tab:igd_sampler_' + tag + r'}',
            r'\begin{tabular}{llcc}',
            r'\toprule',
            r'\textbf{Sampler} & \textbf{Alg.} & $\overline{IGD^+}$ & \textbf{N} \\',
            r'\midrule',
        ]
        for _, row in sampler_summary.sort_values(['sampler', 'algorithm']).iterrows():
            algorithm_label = ALGORITHM_DISPLAY_LABELS.get(row['algorithm'], row['algorithm'])
            igd_str = f"{row['mean_igd']:.4f} $\\pm$ {row['std_igd']:.4f}" if pd.notna(row['std_igd']) else f"{row['mean_igd']:.4f}"
            tex2.append(f"{row['sampler']} & {algorithm_label} & {igd_str} & {int(row['n'])} \\\\")
        tex2 += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        with open(os.path.join(output_dir, f'igd_sampler_table_{tag}.tex'), 'w') as f:
            f.write('\n'.join(tex2))
        print(f"  igd_sampler_table_{tag}.tex")

        all_samplers = sorted(sampler_summary['sampler'].unique())
        all_algorithms = sorted(sampler_summary['algorithm'].unique())

        if len(all_samplers) > 1 and len(all_algorithms) > 1:
            heatmap_values = np.full((len(all_samplers), len(all_algorithms)), np.nan)
            for row_index, sampler_name in enumerate(all_samplers):
                for col_index, algorithm in enumerate(all_algorithms):
                    matching = sampler_summary[
                        (sampler_summary['sampler'] == sampler_name) & (sampler_summary['algorithm'] == algorithm)
                    ]
                    if len(matching) > 0:
                        heatmap_values[row_index, col_index] = matching['mean_igd'].values[0]

            fig, ax = plt.subplots(figsize=(max(6, len(all_algorithms) * 1.5), max(4, len(all_samplers) * 0.5)))
            im = ax.imshow(heatmap_values, cmap='RdYlGn_r', aspect='auto')
            ax.set_xticks(range(len(all_algorithms)))
            ax.set_xticklabels([ALGORITHM_DISPLAY_LABELS.get(a, a) for a in all_algorithms], rotation=45, ha='right')
            ax.set_yticks(range(len(all_samplers)))
            ax.set_yticklabels(all_samplers, fontsize=9)

            for row_index in range(len(all_samplers)):
                for col_index in range(len(all_algorithms)):
                    value = heatmap_values[row_index, col_index]
                    if not np.isnan(value):
                        text_color = 'white' if value > np.nanmedian(heatmap_values) else 'black'
                        ax.text(col_index, row_index, f'{value:.3f}', ha='center', va='center', fontsize=8, color=text_color)

            plt.colorbar(im, ax=ax, label='Mean IGD+ (lower = better)')
            ax.set_title(f'IGD+ per sampler × algorithm ({tag})')
            fig.savefig(os.path.join(output_dir, f'igd_heatmap_{tag}.pdf'))
            plt.close(fig)
            print(f"  igd_heatmap_{tag}.pdf")

    baseline_summary = summary[summary['source_type'] == 'baseline']
    hpo_summary = summary[summary['source_type'] == 'hpo']

    if not baseline_summary.empty and not hpo_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))

        algorithms_present = sorted(summary['algorithm'].unique())
        x_positions = np.arange(len(algorithms_present))

        baseline_igd_values = [
            baseline_summary[baseline_summary['algorithm'] == a]['mean_igd'].mean()
            if len(baseline_summary[baseline_summary['algorithm'] == a]) > 0 else 0
            for a in algorithms_present
        ]
        ax.bar(x_positions - 0.2, baseline_igd_values, 0.35, label='Baseline', color='#B4B2A9', alpha=0.7)

        hpo_igd_values = [
            hpo_summary[hpo_summary['algorithm'] == a]['mean_igd'].mean()
            if len(hpo_summary[hpo_summary['algorithm'] == a]) > 0 else 0
            for a in algorithms_present
        ]
        ax.bar(x_positions + 0.2, hpo_igd_values, 0.35, label='HPO (avg. samplers)', color='#E24B4A', alpha=0.7)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([ALGORITHM_DISPLAY_LABELS.get(a, a) for a in algorithms_present])
        ax.set_ylabel('Mean IGD+ (lower is better)')
        ax.set_title(f'IGD+ against combined Pareto front ({tag})')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        fig.savefig(os.path.join(output_dir, f'igd_comparison_{tag}.pdf'))
        plt.close(fig)
        print(f"  igd_comparison_{tag}.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    arg_parser = argparse.ArgumentParser(description='Generate thesis figures and tables')
    arg_parser.add_argument('--results-dir', default='./Results')
    arg_parser.add_argument('--output-dir', default='./thesis_figures')
    args = arg_parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print()

    print("=" * 50)
    print("1. Baseline")
    print("=" * 50)

    baseline_prefix = find_baseline_csv_prefix(results_dir)
    if not baseline_prefix:
        print("  ERROR: no baseline CSVs found. Run baseline_benchmark.jl first.")
        sys.exit(1)

    baseline_data = {}
    for algorithm in ALGORITHMS:
        filepath = os.path.join(results_dir, f'minimum_runs_{baseline_prefix}_{algorithm}.csv')
        if os.path.isfile(filepath):
            baseline_data[algorithm] = parse_baseline_results_csv(filepath)
            print(f"  {algorithm}: {len(baseline_data[algorithm])} problems")

    generate_baseline_outputs(baseline_data, output_dir)

    print(f"\n{'=' * 50}")
    print("2. HPO results")
    print("=" * 50)

    hpo_results = load_hpo_results(results_dir, 'hpo_benchmark_Results')
    total_entries = sum(len(samplers) for algorithm in hpo_results for samplers in hpo_results[algorithm].values())
    print(f"  {total_entries} result entries loaded")

    print(f"\n{'=' * 50}")
    print("3. HPO vs baseline comparison")
    print("=" * 50)

    generate_hpo_vs_baseline_comparison(baseline_data, hpo_results, output_dir)

    print(f"\n{'=' * 50}")
    print("4. Convergence plots (benchmarks)")
    print("=" * 50)

    benchmark_convergence_data = load_convergence_data(results_dir, 'hpo_benchmark_Results')
    print(f"  {len(benchmark_convergence_data)} convergence files")
    plot_convergence(benchmark_convergence_data, output_dir, 'benchmark')

    print(f"\n{'=' * 50}")
    print("5. Pareto fronts (benchmarks)")
    print("=" * 50)

    benchmark_pareto_fronts = load_pareto_fronts(results_dir, 'hpo_benchmark_Results')
    print(f"  {len(benchmark_pareto_fronts)} front files")
    plot_fronts(benchmark_pareto_fronts, output_dir, 'benchmark')

    print(f"\n{'=' * 50}")
    print("6. Hyperparameter importance")
    print("=" * 50)

    analyze_importance(hpo_results, output_dir)

    print(f"\n{'=' * 50}")
    print("7. Computational cost")
    print("=" * 50)

    analyze_costs(hpo_results, output_dir)

    print(f"\n{'=' * 50}")
    print("8. Truss case study")
    print("=" * 50)

    truss_convergence_data = load_convergence_data(results_dir, 'hpo_truss_Results')
    print(f"  {len(truss_convergence_data)} convergence files")
    plot_convergence_single(truss_convergence_data, 1, output_dir, 'truss')

    truss_pareto_fronts = load_pareto_fronts(results_dir, 'hpo_truss_Results')
    print(f"  {len(truss_pareto_fronts)} front files")
    plot_fronts(truss_pareto_fronts, output_dir, 'truss')

    print(f"\n{'=' * 50}")
    print("9. Combined Pareto fronts + IGD+ (benchmarks)")
    print("=" * 50)

    plot_combined_fronts(results_dir, output_dir, 'combined_fronts_benchmark')
    analyze_igd(results_dir, output_dir, 'combined_fronts_benchmark')

    print(f"\n{'=' * 50}")
    print("10. Combined Pareto fronts + IGD+ (truss)")
    print("=" * 50)

    plot_combined_fronts(results_dir, output_dir, 'combined_fronts_truss')
    analyze_igd(results_dir, output_dir, 'combined_fronts_truss')

    print(f"\n{'=' * 50}")
    print(f"Done. Outputs in {output_dir}/")
    print("=" * 50)


if __name__ == '__main__':
    main()
