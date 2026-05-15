import argparse, json, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = ['NSGA2', 'SPEA2', 'SMS_EMOA', 'MOEAD_DE']

ALGORITHM_DISPLAY_LABELS = {
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

SAMPLERS = ['CmaEsSampler', 'GPSampler', 'NSGAIISampler', 'NSGAIIISampler',
            'QMCSampler', 'RandomSampler', 'TPESampler']

SAMPLER_DISPLAY = {
    'CmaEsSampler':   'CMA-ES',
    'GPSampler':      'GP',
    'NSGAIISampler':  'NSGA-II',
    'NSGAIIISampler': 'NSGA-III',
    'QMCSampler':     'QMC',
    'RandomSampler':  'Random',
    'TPESampler':     'TPE',
}

BASELINE_FLOOR = 1e-2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size']   = 11
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# Helpers

def safe_json_load(s):
    if pd.isna(s) or s == '':
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


def algo_label(name):
    return ALGORITHM_DISPLAY_LABELS.get(name, name)


def sampler_label(name):
    return SAMPLER_DISPLAY.get(name, name)


# Analysis 1: HV (baseline mean vs HPO best-trial HV)

def analysis_hv_comparison(hv_per_task_df, baseline_hv_df, output_dir):
    if hv_per_task_df is None or hv_per_task_df.empty:
        print("  hv_per_task.csv missing or empty — skipping HV analysis")
        return

    if baseline_hv_df is None or baseline_hv_df.empty:
        print("  no baseline HV data — skipping HV analysis")
        return

    baseline_lookup = {}
    for _, row in baseline_hv_df.iterrows():
        key = (row['algorithm'], int(row['problem']))
        baseline_lookup[key] = {
            'mean_hv':     row.get('mean_hv'),
            'best_run_hv': row.get('best_run_hv'),
            'all_hv':      safe_json_load(row.get('all_hv_json', '')),
            'n_runs':      row.get('n_runs', 0),
        }

    rows = []
    for _, hpo_row in hv_per_task_df.iterrows():
        algorithm = hpo_row['algorithm']
        sampler   = hpo_row['sampler']
        problem   = int(hpo_row['problem'])
        baseline_info = baseline_lookup.get((algorithm, problem), {})

        baseline_best_hv = baseline_info.get('best_run_hv', np.nan)
        baseline_mean_hv = baseline_info.get('mean_hv', np.nan)

        hpo_best_hv = hpo_row.get('best_hv_value', np.nan)

        rows.append({
            'algorithm':            algorithm,
            'sampler':              sampler,
            'problem':              problem,
            'baseline_mean_hv':     baseline_mean_hv,
            'baseline_best_hv':     baseline_best_hv,
            'hpo_best_hv':          hpo_best_hv,
            'delta_mean':           (hpo_best_hv - baseline_mean_hv) if pd.notna(hpo_best_hv) and pd.notna(baseline_mean_hv) else np.nan,
            'delta_best':           (hpo_best_hv - baseline_best_hv) if pd.notna(hpo_best_hv) and pd.notna(baseline_best_hv) else np.nan,
            'best_trial_db_number': hpo_row.get('best_trial_db_number'),
            'n_complete_in_window': hpo_row.get('n_complete_in_window'),
        })

    full_df = pd.DataFrame(rows)
    full_df.to_csv(os.path.join(output_dir, 'hv_full.csv'), index=False)
    print(f"  hv_full.csv ({len(full_df)} rows)")

    for algorithm in ALGORITHMS:
        algo_df = full_df[full_df['algorithm'] == algorithm].copy()
        if algo_df.empty:
            continue

        pivot = algo_df.pivot(index='problem', columns='sampler', values='hpo_best_hv')
        pivot = pivot.reindex(columns=[s for s in SAMPLERS if s in pivot.columns])

        baseline_per_problem = (algo_df.groupby('problem')['baseline_mean_hv']
                                       .first()
                                       .reindex(pivot.index))
        pivot.insert(0, 'baseline (mean)', baseline_per_problem)

        out_csv = os.path.join(output_dir, f'hv_{algorithm}.csv')
        pivot.to_csv(out_csv)
        print(f"  hv_{algorithm}.csv")

        write_hv_latex_table(pivot, algorithm, output_dir)

    write_hv_summary_tables(full_df, output_dir)
    write_hv_combined_latex_table(full_df, output_dir)


def write_hv_latex_table(pivot, algorithm, output_dir):
    sampler_columns = [c for c in pivot.columns if c != 'baseline (mean)']
    n_cols = 1 + 1 + len(sampler_columns)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering', r'\small',
        r'\caption{HV results for ' + algo_label(algorithm) + r'.}',
        r'\label{tab:hv_' + algorithm + '}',
        r'\begin{tabular}{' + 'l' + 'c' * (n_cols - 1) + '}',
        r'\toprule',
    ]
    header = ['Problem', 'Baseline'] + [sampler_label(s) for s in sampler_columns]
    lines.append(' & '.join(rf'\textbf{{{h}}}' for h in header) + r' \\')
    lines.append(r'\midrule')

    rows_emitted = 0
    rows_dropped_infeasible = 0
    rows_dropped_low_baseline = 0
    rel_per_sampler = {s: [] for s in sampler_columns}

    for problem_id, row in pivot.iterrows():
        baseline_val = row.get('baseline (mean)')

        if pd.isna(baseline_val) or baseline_val < BASELINE_FLOOR:
            rows_dropped_low_baseline += 1
            continue

        rel_vals = {}
        sampler_abs_vals = {}
        for s in sampler_columns:
            v = row.get(s)
            if pd.notna(v) and v > 1e-9:
                rel_vals[s] = (v - baseline_val) / baseline_val * 100.0
                sampler_abs_vals[s] = v

        if not rel_vals:
            rows_dropped_infeasible += 1
            continue

        all_abs_vals = {'__baseline__': baseline_val, **sampler_abs_vals}
        max_abs = max(all_abs_vals.values())
        n_at_max = sum(1 for v_ in all_abs_vals.values() if abs(v_ - max_abs) < 1e-9)
        unique_max = (n_at_max == 1)
        baseline_wins = unique_max and abs(baseline_val - max_abs) < 1e-9

        baseline_cell = rf'\textbf{{{baseline_val:.4f}}}' if baseline_wins else f"{baseline_val:.4f}"
        row_cells = [str(problem_id), baseline_cell]
        for s in sampler_columns:
            if s not in rel_vals:
                row_cells.append('--')
            else:
                rel = rel_vals[s]
                rel_per_sampler[s].append(rel)
                txt = f"{rel:+.1f}\\%"
                if unique_max and abs(sampler_abs_vals[s] - max_abs) < 1e-9:
                    row_cells.append(rf'\textbf{{{txt}}}')
                else:
                    row_cells.append(txt)
        lines.append(' & '.join(row_cells) + r' \\')
        rows_emitted += 1

    if rows_emitted > 0:
        import statistics as _stats
        lines.append(r'\midrule')

        median_cells = [r'\textbf{Median}', '--']
        for s in sampler_columns:
            vals = rel_per_sampler[s]
            median_cells.append(f"{_stats.median(vals):+.1f}\\%" if vals else '--')
        lines.append(' & '.join(median_cells) + r' \\')

        mean_cells = [r'\textbf{Mean}', '--']
        for s in sampler_columns:
            vals = rel_per_sampler[s]
            mean_cells.append(f"{sum(vals) / len(vals):+.1f}\\%" if vals else '--')
        lines.append(' & '.join(mean_cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    out_tex = os.path.join(output_dir, f'hv_{algorithm}.tex')
    with open(out_tex, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  hv_{algorithm}.tex "
          f"({rows_emitted} rows emitted; "
          f"dropped {rows_dropped_low_baseline} low-baseline, "
          f"{rows_dropped_infeasible} all-infeasible)")


def write_hv_combined_latex_table(full_df, output_dir):
    if full_df is None or full_df.empty:
        return

    samplers_present = [s for s in SAMPLERS if s in full_df['sampler'].unique()]
    algorithms_present = [a for a in ALGORITHMS if a in full_df['algorithm'].unique()]
    if not samplers_present or not algorithms_present:
        return

    n_problems = full_df['problem'].nunique() if 'problem' in full_df.columns else 0
    single_problem = (n_problems == 1)

    rows_by_algorithm = {}
    for algorithm in algorithms_present:
        algo_df = full_df[full_df['algorithm'] == algorithm]
        baselines = algo_df.groupby('problem')['baseline_mean_hv'].first()
        baselines = baselines[baselines.notna() & (baselines >= BASELINE_FLOOR)]
        if baselines.empty:
            continue
        rel_per_sampler = {s: [] for s in samplers_present}
        for problem_id, baseline_val in baselines.items():
            for s in samplers_present:
                rec = algo_df[(algo_df['problem'] == problem_id) & (algo_df['sampler'] == s)]
                if rec.empty:
                    continue
                v = rec['hpo_best_hv'].iloc[0]
                if pd.notna(v) and v > 1e-9:
                    rel_per_sampler[s].append((v - baseline_val) / baseline_val * 100.0)
        rows_by_algorithm[algorithm] = (baselines, rel_per_sampler)

    if not rows_by_algorithm:
        return

    n_cols = 1 + 1 + len(samplers_present)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering', r'\small',
        r'\caption{HV summary across algorithms.}',
        r'\label{tab:hv_combined}',
        r'\begin{tabular}{' + 'l' + 'c' * (n_cols - 1) + '}',
        r'\toprule',
    ]
    header = ['Algorithm', 'Baseline'] + [sampler_label(s) for s in samplers_present]
    lines.append(' & '.join(rf'\textbf{{{h}}}' for h in header) + r' \\')
    lines.append(r'\midrule')

    summary_per_sampler = {s: [] for s in samplers_present}

    for algorithm in algorithms_present:
        if algorithm not in rows_by_algorithm:
            continue
        baselines, rel_per_sampler = rows_by_algorithm[algorithm]

        if single_problem:
            problem_id = baselines.index[0]
            baseline_val = baselines.iloc[0]
            sampler_abs_vals = {}
            for s in samplers_present:
                rec = full_df[(full_df['algorithm'] == algorithm) &
                              (full_df['sampler'] == s) &
                              (full_df['problem'] == problem_id)]
                if not rec.empty:
                    v = rec['hpo_best_hv'].iloc[0]
                    if pd.notna(v) and v > 1e-9:
                        sampler_abs_vals[s] = v
            all_abs = {'__baseline__': baseline_val, **sampler_abs_vals}
            max_abs = max(all_abs.values())
            n_at_max = sum(1 for v in all_abs.values() if abs(v - max_abs) < 1e-9)
            unique_max = (n_at_max == 1)
            baseline_wins = unique_max and abs(baseline_val - max_abs) < 1e-9

            baseline_cell = rf'\textbf{{{baseline_val:.4f}}}' if baseline_wins else f"{baseline_val:.4f}"
            row_cells = [algo_label(algorithm), baseline_cell]
            for s in samplers_present:
                if s not in sampler_abs_vals:
                    row_cells.append('--')
                else:
                    rel = (sampler_abs_vals[s] - baseline_val) / baseline_val * 100.0
                    summary_per_sampler[s].append(rel)
                    txt = f"{rel:+.1f}\\%"
                    if unique_max and abs(sampler_abs_vals[s] - max_abs) < 1e-9:
                        row_cells.append(rf'\textbf{{{txt}}}')
                    else:
                        row_cells.append(txt)
        else:
            import statistics as _stats
            baseline_avg = float(baselines.mean())
            row_cells = [algo_label(algorithm), f"{baseline_avg:.4f}"]
            sampler_medians = {}
            for s in samplers_present:
                vals = rel_per_sampler[s]
                if vals:
                    sampler_medians[s] = _stats.median(vals)
                    summary_per_sampler[s].append(sampler_medians[s])
            if sampler_medians:
                best_median = max(sampler_medians.values())
                winners = [s for s, m in sampler_medians.items() if abs(m - best_median) < 1e-9]
                unique_best = (len(winners) == 1)
            else:
                unique_best = False
                winners = []
            for s in samplers_present:
                if s not in sampler_medians:
                    row_cells.append('--')
                else:
                    txt = f"{sampler_medians[s]:+.1f}\\%"
                    row_cells.append(rf'\textbf{{{txt}}}' if unique_best and s in winners else txt)

        lines.append(' & '.join(row_cells) + r' \\')

    if any(summary_per_sampler[s] for s in samplers_present):
        import statistics as _stats
        lines.append(r'\midrule')

        median_cells = [r'\textbf{Median}', '--']
        for s in samplers_present:
            vals = summary_per_sampler[s]
            median_cells.append(f"{_stats.median(vals):+.1f}\\%" if vals else '--')
        lines.append(' & '.join(median_cells) + r' \\')

        mean_cells = [r'\textbf{Mean}', '--']
        for s in samplers_present:
            vals = summary_per_sampler[s]
            mean_cells.append(f"{sum(vals)/len(vals):+.1f}\\%" if vals else '--')
        lines.append(' & '.join(mean_cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    out_tex = os.path.join(output_dir, 'hv_combined.tex')
    with open(out_tex, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  hv_combined.tex ({len(algorithms_present)} algorithms)")


def write_hv_summary_tables(full_df, output_dir):
    if full_df is None or full_df.empty:
        return

    # Per-sampler win counts (unique max only, ties excluded)
    feasible = full_df[full_df['hpo_best_hv'] > 1e-9].copy()

    win_records = []
    for (algorithm, problem), group in feasible.groupby(['algorithm', 'problem']):
        max_val = group['hpo_best_hv'].max()
        winners = group[group['hpo_best_hv'].sub(max_val).abs() < 1e-9]
        if len(winners) == 1:
            win_records.append({
                'algorithm': algorithm,
                'problem':   problem,
                'sampler':   winners.iloc[0]['sampler'],
            })

    if win_records:
        wins_df = pd.DataFrame(win_records)
        wins_pivot = (wins_df.groupby(['sampler', 'algorithm']).size()
                            .unstack(fill_value=0)
                            .reindex(index=SAMPLERS, columns=ALGORITHMS, fill_value=0))
    else:
        wins_pivot = pd.DataFrame(0, index=SAMPLERS, columns=ALGORITHMS)

    wins_pivot['Total'] = wins_pivot.sum(axis=1)
    wins_pivot.to_csv(os.path.join(output_dir, 'wins_per_sampler.csv'))

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{HV wins per sampler.}',
        r'\label{tab:wins_per_sampler}',
        r'\begin{tabular}{l' + 'c' * (len(ALGORITHMS) + 1) + '}',
        r'\toprule',
    ]
    header = ['Sampler'] + [algo_label(a) for a in ALGORITHMS] + ['Total']
    lines.append(' & '.join(rf'\textbf{{{h}}}' for h in header) + r' \\')
    lines.append(r'\midrule')

    for sampler in SAMPLERS:
        cells = [sampler_label(sampler)]
        for algorithm in ALGORITHMS:
            cells.append(str(int(wins_pivot.loc[sampler, algorithm])))
        cells.append(rf"\textbf{{{int(wins_pivot.loc[sampler, 'Total'])}}}")
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(output_dir, 'wins_per_sampler.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print(f"  wins_per_sampler.tex / .csv")

    # Per-algorithm HPO-vs-baseline aggregate (best HPO across samplers vs baseline mean)
    summary_records = []
    for algorithm in ALGORITHMS:
        algo_df = full_df[full_df['algorithm'] == algorithm].copy()
        if algo_df.empty:
            continue

        per_problem = algo_df.groupby('problem').agg(
            best_hpo=('hpo_best_hv', 'max'),
            baseline=('baseline_mean_hv', 'first'),
        )
        per_problem = per_problem[
            (per_problem['best_hpo'] > 1e-9) |
            (per_problem['baseline'].notna() & (per_problem['baseline'] > 1e-9))
        ]
        per_problem['delta'] = per_problem['best_hpo'] - per_problem['baseline']

        n_total = per_problem['delta'].notna().sum()
        n_wins  = (per_problem['delta'] > 0).sum()
        mean_d  = per_problem['delta'].mean()
        med_d   = per_problem['delta'].median()

        summary_records.append({
            'algorithm':    algorithm,
            'n_total':      int(n_total),
            'n_wins':       int(n_wins),
            'mean_delta':   mean_d,
            'median_delta': med_d,
        })

    if summary_records:
        sdf = pd.DataFrame(summary_records)
        sdf.to_csv(os.path.join(output_dir, 'hpo_vs_baseline_per_algorithm.csv'), index=False)

        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r"\caption{HPO vs.\ baseline HV, per algorithm.}",
            r'\label{tab:hpo_vs_baseline_per_algorithm}',
            r'\begin{tabular}{lcccc}',
            r'\toprule',
            r'\textbf{Algorithm} & \textbf{Wins/Total} & \textbf{Win rate} & \textbf{Mean $\Delta$HV} & \textbf{Median $\Delta$HV} \\',
            r'\midrule',
        ]
        for r in summary_records:
            win_rate = (r['n_wins'] / r['n_total'] * 100) if r['n_total'] > 0 else 0.0
            lines.append(
                f"{algo_label(r['algorithm'])} & "
                f"{r['n_wins']}/{r['n_total']} & "
                f"{win_rate:.1f}\\% & "
                f"{r['mean_delta']:+.4f} & "
                f"{r['median_delta']:+.4f} \\\\"
            )
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        with open(os.path.join(output_dir, 'hpo_vs_baseline_per_algorithm.tex'), 'w') as f:
            f.write('\n'.join(lines))
        print(f"  hpo_vs_baseline_per_algorithm.tex / .csv")


# Analysis 2: IGD+ (baseline best-run vs HPO best-trial)

def analysis_igd_comparison(hpo_igd_df, baseline_igd_df, output_dir):
    if hpo_igd_df is None or hpo_igd_df.empty:
        print("  igd_results.csv missing or empty — skipping IGD+ analysis")
        return

    if baseline_igd_df is None or baseline_igd_df.empty:
        print("  baseline_igd_results.csv missing — skipping IGD+ analysis")
        return

    baseline_lookup = {}
    for _, row in baseline_igd_df.iterrows():
        key = (row['algorithm'], int(row['problem']))
        baseline_lookup[key] = row['igd_plus']

    rows = []
    for _, r in hpo_igd_df.iterrows():
        algorithm = r['algorithm']
        sampler   = r['sampler']
        problem   = int(r['problem'])
        hpo_igd   = r['igd_plus']
        baseline_igd = baseline_lookup.get((algorithm, problem), np.nan)

        rows.append({
            'algorithm':    algorithm,
            'sampler':      sampler,
            'problem':      problem,
            'baseline_igd': baseline_igd,
            'hpo_igd':      hpo_igd,
            'delta':        (hpo_igd - baseline_igd) if pd.notna(hpo_igd) and pd.notna(baseline_igd) else np.nan,
            'hpo_better':   pd.notna(hpo_igd) and pd.notna(baseline_igd) and hpo_igd < baseline_igd,
        })

    full_df = pd.DataFrame(rows)
    full_df.to_csv(os.path.join(output_dir, 'igd_full.csv'), index=False)
    print(f"  igd_full.csv ({len(full_df)} rows)")

    for algorithm in ALGORITHMS:
        algo_df = full_df[full_df['algorithm'] == algorithm].copy()
        if algo_df.empty:
            continue

        pivot = algo_df.pivot(index='problem', columns='sampler', values='hpo_igd')
        pivot = pivot.reindex(columns=[s for s in SAMPLERS if s in pivot.columns])
        baseline_per_problem = (algo_df.groupby('problem')['baseline_igd']
                                       .first()
                                       .reindex(pivot.index))
        pivot.insert(0, 'baseline', baseline_per_problem)

        pivot.to_csv(os.path.join(output_dir, f'igd_{algorithm}.csv'))
        print(f"  igd_{algorithm}.csv")
        write_igd_latex_table(pivot, algorithm, output_dir)

    write_igd_wins_table(full_df, output_dir)
    write_igd_combined_latex_table(full_df, output_dir)


def write_igd_wins_table(full_df, output_dir):
    if full_df is None or full_df.empty:
        return

    valid = full_df[full_df['hpo_igd'].notna()].copy()
    if valid.empty:
        return

    win_records = []
    for (algorithm, problem), group in valid.groupby(['algorithm', 'problem']):
        if len(group) < 2:
            continue
        min_val = group['hpo_igd'].min()
        winners = group[group['hpo_igd'].sub(min_val).abs() < 1e-9]
        if len(winners) == 1:
            win_records.append({
                'algorithm': algorithm,
                'problem':   problem,
                'sampler':   winners.iloc[0]['sampler'],
            })

    if win_records:
        wins_df = pd.DataFrame(win_records)
        wins_pivot = (wins_df.groupby(['sampler', 'algorithm']).size()
                            .unstack(fill_value=0)
                            .reindex(index=SAMPLERS, columns=ALGORITHMS, fill_value=0))
    else:
        wins_pivot = pd.DataFrame(0, index=SAMPLERS, columns=ALGORITHMS)

    wins_pivot['Total'] = wins_pivot.sum(axis=1)
    wins_pivot = wins_pivot.sort_values('Total', ascending=False)
    wins_pivot.to_csv(os.path.join(output_dir, 'wins_per_sampler_igd.csv'))

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{IGD+ wins per sampler.}',
        r'\label{tab:wins_per_sampler_igd}',
        r'\begin{tabular}{l' + 'c' * (len(ALGORITHMS) + 1) + '}',
        r'\toprule',
    ]
    header = ['Sampler'] + [algo_label(a) for a in ALGORITHMS] + ['Total']
    lines.append(' & '.join(rf'\textbf{{{h}}}' for h in header) + r' \\')
    lines.append(r'\midrule')

    for sampler in wins_pivot.index:
        cells = [sampler_label(sampler)]
        for algorithm in ALGORITHMS:
            cells.append(str(int(wins_pivot.loc[sampler, algorithm])))
        cells.append(rf"\textbf{{{int(wins_pivot.loc[sampler, 'Total'])}}}")
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(output_dir, 'wins_per_sampler_igd.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print("  wins_per_sampler_igd.tex / .csv")


def write_igd_latex_table(pivot, algorithm, output_dir):
    sampler_columns = [c for c in pivot.columns if c != 'baseline']
    n_cols = 1 + 1 + len(sampler_columns)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering', r'\small',
        r'\caption{IGD+ results for ' + algo_label(algorithm) + r'.}',
        r'\label{tab:igd_' + algorithm + '}',
        r'\begin{tabular}{' + 'l' + 'c' * (n_cols - 1) + '}',
        r'\toprule',
    ]
    header = ['Problem', 'Baseline'] + [sampler_label(s) for s in sampler_columns]
    lines.append(' & '.join(rf'\textbf{{{h}}}' for h in header) + r' \\')
    lines.append(r'\midrule')

    for problem_id, row in pivot.iterrows():
        row_cells = [str(problem_id)]
        baseline_val = row.get('baseline')

        sampler_vals = {s: row.get(s) for s in sampler_columns}
        valid_sampler_vals = {s: v for s, v in sampler_vals.items() if pd.notna(v)}

        all_candidates = dict(valid_sampler_vals)
        if pd.notna(baseline_val):
            all_candidates['__baseline__'] = baseline_val

        if all_candidates:
            min_val = min(all_candidates.values())
            n_at_min = sum(1 for v_ in all_candidates.values() if abs(v_ - min_val) < 1e-9)
            unique_min = (n_at_min == 1)
        else:
            min_val = None
            unique_min = False

        baseline_wins = (unique_min and pd.notna(baseline_val)
                         and abs(baseline_val - min_val) < 1e-9)
        if pd.isna(baseline_val):
            row_cells.append('--')
        elif baseline_wins:
            row_cells.append(rf'\textbf{{{baseline_val:.4f}}}')
        else:
            row_cells.append(f"{baseline_val:.4f}")

        for s in sampler_columns:
            v = row.get(s)
            if pd.isna(v):
                row_cells.append('--')
            elif unique_min and min_val is not None and abs(v - min_val) < 1e-9:
                row_cells.append(rf'\textbf{{{v:.4f}}}')
            else:
                row_cells.append(f"{v:.4f}")
        lines.append(' & '.join(row_cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    out_tex = os.path.join(output_dir, f'igd_{algorithm}.tex')
    with open(out_tex, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  igd_{algorithm}.tex")


def write_igd_combined_latex_table(full_df, output_dir):
    if full_df is None or full_df.empty:
        return

    samplers_present = [s for s in SAMPLERS if s in full_df['sampler'].unique()]
    algorithms_present = [a for a in ALGORITHMS if a in full_df['algorithm'].unique()]
    if not samplers_present or not algorithms_present:
        return

    n_problems = full_df['problem'].nunique() if 'problem' in full_df.columns else 0
    single_problem = (n_problems == 1)

    n_cols = 1 + 1 + len(samplers_present)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering', r'\small',
        r'\caption{IGD+ summary across algorithms.}',
        r'\label{tab:igd_combined}',
        r'\begin{tabular}{' + 'l' + 'c' * (n_cols - 1) + '}',
        r'\toprule',
    ]
    header = ['Algorithm', 'Baseline'] + [sampler_label(s) for s in samplers_present]
    lines.append(' & '.join(rf'\textbf{{{h}}}' for h in header) + r' \\')
    lines.append(r'\midrule')

    import statistics as _stats
    baseline_summary = []
    sampler_summary = {s: [] for s in samplers_present}

    for algorithm in algorithms_present:
        algo_df = full_df[full_df['algorithm'] == algorithm]
        baselines = algo_df.groupby('problem')['baseline_igd'].first().dropna()
        if single_problem:
            baseline_val = baselines.iloc[0] if not baselines.empty else None
            sampler_vals = {}
            for s in samplers_present:
                rec = algo_df[algo_df['sampler'] == s]
                if not rec.empty:
                    v = rec['hpo_igd'].iloc[0]
                    if pd.notna(v):
                        sampler_vals[s] = v
            all_candidates = {}
            if baseline_val is not None and pd.notna(baseline_val):
                all_candidates['__baseline__'] = float(baseline_val)
            all_candidates.update({s: v for s, v in sampler_vals.items()})
            if all_candidates:
                min_val = min(all_candidates.values())
                unique_min = sum(1 for v in all_candidates.values() if abs(v - min_val) < 1e-9) == 1
            else:
                min_val = None
                unique_min = False
        else:
            baseline_val = float(baselines.median()) if not baselines.empty else None
            sampler_vals = {}
            for s in samplers_present:
                vals = algo_df[algo_df['sampler'] == s]['hpo_igd'].dropna().values
                if len(vals) > 0:
                    sampler_vals[s] = float(_stats.median(vals))
            candidate_pool = dict(sampler_vals)
            if baseline_val is not None:
                candidate_pool['__baseline__'] = baseline_val
            if candidate_pool:
                min_val = min(candidate_pool.values())
                unique_min = sum(1 for v in candidate_pool.values() if abs(v - min_val) < 1e-9) == 1
            else:
                min_val = None
                unique_min = False

        row_cells = [algo_label(algorithm)]
        if baseline_val is None or pd.isna(baseline_val):
            row_cells.append('--')
        else:
            baseline_summary.append(float(baseline_val))
            baseline_wins = (unique_min and min_val is not None and abs(float(baseline_val) - min_val) < 1e-9)
            row_cells.append(rf'\textbf{{{baseline_val:.4f}}}' if baseline_wins else f"{baseline_val:.4f}")

        for s in samplers_present:
            if s not in sampler_vals:
                row_cells.append('--')
            else:
                v = sampler_vals[s]
                sampler_summary[s].append(v)
                if unique_min and min_val is not None and abs(v - min_val) < 1e-9:
                    row_cells.append(rf'\textbf{{{v:.4f}}}')
                else:
                    row_cells.append(f"{v:.4f}")
        lines.append(' & '.join(row_cells) + r' \\')

    if baseline_summary or any(sampler_summary[s] for s in samplers_present):
        lines.append(r'\midrule')

        median_cells = [r'\textbf{Median}']
        median_cells.append(f"{_stats.median(baseline_summary):.4f}" if baseline_summary else '--')
        for s in samplers_present:
            vals = sampler_summary[s]
            median_cells.append(f"{_stats.median(vals):.4f}" if vals else '--')
        lines.append(' & '.join(median_cells) + r' \\')

        mean_cells = [r'\textbf{Mean}']
        mean_cells.append(f"{sum(baseline_summary)/len(baseline_summary):.4f}" if baseline_summary else '--')
        for s in samplers_present:
            vals = sampler_summary[s]
            mean_cells.append(f"{sum(vals)/len(vals):.4f}" if vals else '--')
        lines.append(' & '.join(mean_cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    out_tex = os.path.join(output_dir, 'igd_combined.tex')
    with open(out_tex, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  igd_combined.tex ({len(algorithms_present)} algorithms)")


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',      required=True, help='Path to Results/')
    parser.add_argument('--db-metrics-dir',   required=True, help='Dir with hv_per_task.csv from extract_db_metrics.jl')
    parser.add_argument('--baseline-igd-dir', required=True, help='Dir with baseline_igd_results.csv from compute_baseline_igd.jl')
    parser.add_argument('--hpo-igd-csv',      default=None,
                        help='Path to igd_results.csv from compute_igd.jl. '
                             'Default: <results-dir>/combined_fronts_<experiment>/igd_results.csv')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment', choices=['benchmark', 'truss'], default='benchmark',
                        help='Which experiment is being analyzed (default: benchmark)')
    args = parser.parse_args()

    if os.path.abspath(args.output_dir).startswith(os.path.abspath(args.results_dir) + os.sep):
        print(f"Refusing to write inside Results/ ({args.results_dir})")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    cpf_dir_name = f'combined_fronts_{args.experiment}'
    hpo_igd_path = args.hpo_igd_csv or os.path.join(args.results_dir, cpf_dir_name, 'igd_results.csv')

    print(f"Experiment:     {args.experiment}")
    print(f"Results dir:    {args.results_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"hpo-igd-csv:    {hpo_igd_path}")
    print()

    print("Loading inputs...")
    hv_per_task_csv = os.path.join(args.db_metrics_dir, 'hv_per_task.csv')
    hv_per_task_df = pd.read_csv(hv_per_task_csv) if os.path.isfile(hv_per_task_csv) else None
    print(f"  hv_per_task.csv: {0 if hv_per_task_df is None else len(hv_per_task_df)} rows")

    baseline_igd_csv = os.path.join(args.baseline_igd_dir, 'baseline_igd_results.csv')
    baseline_igd_df = pd.read_csv(baseline_igd_csv) if os.path.isfile(baseline_igd_csv) else None
    print(f"  baseline_igd_results.csv: {0 if baseline_igd_df is None else len(baseline_igd_df)} rows")

    hpo_igd_df = pd.read_csv(hpo_igd_path) if os.path.isfile(hpo_igd_path) else None
    print(f"  igd_results.csv: {0 if hpo_igd_df is None else len(hpo_igd_df)} rows")

    print(f"\n{'=' * 60}")
    print("Analysis 1: HV — baseline vs HPO best trial")
    print(f"{'=' * 60}")
    analysis_hv_comparison(hv_per_task_df, baseline_igd_df, args.output_dir)

    print(f"\n{'=' * 60}")
    print("Analysis 2: IGD+ — baseline best run vs HPO best trial")
    print(f"{'=' * 60}")
    analysis_igd_comparison(hpo_igd_df, baseline_igd_df, args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"Done. Outputs in {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()