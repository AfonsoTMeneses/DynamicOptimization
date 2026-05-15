# Optuna HPO pipeline

Scripts for running Optuna-driven hyperparameter search on four `Metaheuristics.jl` algorithms (NSGA2, SPEA2, SMS_EMOA, MOEAD_DE) and building the thesis tables and figures from the results. Run from the project root, with `Optuna_code/` and `Results/` as siblings.

## Requirements

- Julia 1.x with `--project=.`
- Python 3 with `pandas`, `numpy`, `matplotlib`
- Truss workflow only: `KhepriFrame3DD`

## Run

```bash
julia --project=. Optuna_code/baseline_benchmark.jl
julia --project=. Optuna_code/hpo_benchmark.jl
julia --project=. Optuna_code/compute_cPF.jl --experiment benchmark
julia --project=. Optuna_code/extract_db_metrics.jl        --output-dir analysis_inputs
julia --project=. Optuna_code/extract_best_trial_fronts.jl --output-dir analysis_inputs
julia --project=. Optuna_code/compute_igd.jl --best-trials-csv analysis_inputs/best_trial_fronts.csv --output-dir analysis_inputs
julia --project=. Optuna_code/compute_baseline_igd.jl --output-dir analysis_inputs
python3 Optuna_code/analysis.py \
    --results-dir Results/ \
    --db-metrics-dir analysis_inputs/ \
    --baseline-igd-dir analysis_inputs/ \
    --hpo-igd-csv analysis_inputs/igd_results.csv \
    --output-dir analysis_outputs/
```

For the truss workflow, add `--experiment truss` to every step from `compute_cPF.jl` onward, redirect each `--output-dir` to a separate directory (e.g. `analysis_inputs_truss`), and finally:

```bash
python3 Optuna_code/plot_convergence.py \
    --db-dir Results/hpo_truss_Results/optuna_studies \
    --output-dir convergence_plots/
```

## Notes

- Pipeline outputs go under `Results/`; analysis scripts refuse `--output-dir` paths inside it.
- Optuna trials index from 0; `all_fronts_*.csv` from 1 (`csv.trial = db.number + 1`).
- Samplers are set in `hyperoptimization_intervals.jl`; worker count is `addprocs(2)` inside `hpo_*.jl`.
- `audit_completion.jl` reports unfinished tasks; `inventory_optuna_studies.jl` checks whether each task's best trial is extractable from its `all_fronts` CSV.
