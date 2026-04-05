# DynamicOptimization

Hyperparameter optimization for multi-objective metaheuristic algorithms, benchmarked on the RW-MOP-2021 problem set (50 constrained real-world problems) and a parametric truss structural design case study.

MSc Thesis — Instituto Superior Técnico, Universidade de Lisboa.

## Project Structure

```
Optuna/
├── Project.toml                        # Julia project (dependencies)
├── setup.jl                            # Run once: installs all packages
├── Optuna_code/
│   ├── optuna_utils.jl                 # Core: normalization, HV, IGD+, HPO pipeline, CSV I/O
│   ├── utils_minimum_runs.jl           # Min runs computation (CI formula)
│   ├── Hyperoptimization_intervals.jl  # Search spaces, sampler list (user config)
│   ├── truss_problem.jl               # Parametric truss: geometry, objectives, variables
│   │
│   ├── baseline_benchmark.jl           # Phase 1: 50 problems × 4 algs × 100 runs (defaults)
│   ├── hpo_benchmark.jl               # Phase 2: HPO on 50 problems (Optuna)
│   ├── baseline_truss.jl              # Phase 1: truss × 4 algs × 100 runs (defaults)
│   ├── hpo_truss.jl                   # Phase 4: HPO on truss (3 samplers, SQLite)
│   └── analysis.py                    # Phase 5: figures, LaTeX tables, stats
│
├── Results/
│   ├── minimum_runs_baseline_benchmark_*.csv
│   ├── empirical_bounds_baseline_benchmark_*.csv
│   ├── reference_fronts_baseline_benchmark_*.csv
│   ├── hpo_benchmark_Results/
│   ├── baseline_truss_Results/
│   └── hpo_truss_Results/
│
├── rename_csvs.sh                      # One-time: rename old CSV files
├── CHANGES.md                          # Detailed changelog
├── TODO.md                             # Step-by-step next actions
└── README.md                           # This file
```

## Prerequisites

- **Julia** >= 1.9
- **Python** >= 3.8 with `pip install optuna`
- **KhepriFrame3DD.jl** (truss only — benchmarks work without it)

## Setup

```bash
cd Optuna/
julia --project=. setup.jl
```

If migrating from old script names:
```bash
cd Results/
bash ../rename_csvs.sh
```

## Workflow

### Phase 1 — Baseline

Each algorithm runs with default parameters, 100 times per problem. Produces empirical bounds (ideal/nadir), reference fronts (for IGD+), and minimum runs (95% CI).

```bash
julia --project=.. Optuna_code/baseline_benchmark.jl   # 50 benchmarks
julia --project=.. Optuna_code/baseline_truss.jl        # truss
```

### Phase 2 — HPO (benchmarks)

Optuna trials with multiple samplers. Each trial proposes hyperparameters, runs the optimizer num_runs times, computes normalized HV and IGD+. Tracks convergence, timing, param importance (fANOVA), and Pareto fronts.

```bash
julia --project=.. Optuna_code/hpo_benchmark.jl
```

### Phase 3 — Sampler comparison

Analyze Phase 2 results. Pick top 3 samplers for the truss.

```bash
python3 Optuna_code/analysis.py --results-dir ./Results --output-dir ./thesis_figures
```

### Phase 4 — HPO (truss)

Apply selected samplers to the truss with 50 iterations. Includes SQLite persistence and Pareto front collection.

```bash
julia --project=.. Optuna_code/hpo_truss.jl
```

### Phase 5 — Analysis

Re-run analysis with full data. Generates all thesis deliverables.

```bash
python3 Optuna_code/analysis.py --results-dir ./Results --output-dir ./thesis_figures
```

## Algorithms

| Algorithm | Key Hyperparameters |
|-----------|-------------------|
| NSGA-II | N, eta_cr, p_cr, eta_m, p_m |
| SPEA2 | N, eta_cr, p_cr, eta_m, p_m |
| SMS-EMOA | N, eta_cr, p_cr, eta_m, p_m, n_samples |
| MOEA/D-DE | npartitions, F, CR, eta, p_m, delta, s1, s2 |

## Performance Indicators

**Hypervolume (HV):** Measures the volume of objective space dominated by the obtained front. Higher is better. Computed in normalized space with R = 1.01 × ones.

**IGD+ (Inverted Generational Distance Plus):** Measures how close the obtained front is to the reference front (union of all baseline ND solutions). Lower is better. Weakly Pareto-compliant variant.

Both indicators use the same normalization frame: empirical ideal/nadir from Phase 1.

## Normalization Method

Following Mejía de Dios (personal communication):

1. Union all feasible ND solutions across 100 baseline runs
2. Empirical ideal = componentwise min, nadir = componentwise max
3. Normalize: `f' = (f - ideal) / (nadir - ideal)`
4. Reference point: R = 1.01 × ones(n_objectives)

For DEAD problems (no feasible solutions with defaults): HPO uses library nadir + empirical ideal from the trial's own solutions.

## Analysis Outputs

| Output | Description |
|--------|-------------|
| `baseline_summary.csv` | Mean HV, std, min_runs, feasibility per problem × algorithm |
| `baseline_hv_boxplot.pdf` | Box plot of HV distributions |
| `feasibility_heatmap.pdf` | Feasibility rate heatmap |
| `comparison_table.tex` | LaTeX: ΔHV, IGD+, timing, significance counts |
| `win_tie_loss_table.tex` | LaTeX: Mann-Whitney U win/tie/loss |
| `param_importance_table.tex` | LaTeX: fANOVA importance per param |
| `computational_cost_table.tex` | LaTeX: wall time per study |
| `hpo_improvement_barchart.pdf` | ΔHV grouped by sampler |
| `convergence_*.pdf` | Sampler convergence curves |
| `pareto_front_*.pdf` | Solutions in objective space |
| `param_importance_*.pdf` | Importance bar charts per algorithm |

## Optuna Dashboard (truss only)

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///Results/hpo_truss_Results/optuna_studies/study_NSGA2_TPESampler_Problem_1.db
```

## Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `FALLBACK_RUNS` | 30 | When min_runs > 100 or Inf |
| `NORMALIZED_REFERENCE_MARGIN` | 0.01 | R = (1+margin) × ones |
| `MAX_MOEAD_POPULATION` | 500 | Cap for MOEA/D-DE weights |
