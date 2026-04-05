# DynamicOptimization

Hyperparameter optimization for multi-objective metaheuristic algorithms, benchmarked on the RW-MOP-2021 problem set (50 constrained real-world problems) and a parametric truss structural design case study.

MSc Thesis — Instituto Superior Técnico, Universidade de Lisboa.

## Prerequisites

- **Julia** >= 1.9
- **Python** >= 3.8 with `pip install optuna`
- **KhepriFrame3DD.jl** (truss only — benchmarks work without it)

## Setup

```bash
cd Optuna/
julia --project=. setup.jl
```

## Workflow

baseline_benchmark.jl  →  hpo_benchmark.jl  →  compute_cPF.jl  →  analysis.py

baseline_truss.jl      →  hpo_truss.jl      →  compute_cPF.jl  →  analysis.py


### Phase 1 — Baseline

Each algorithm runs with default parameters, 100 times per problem. Produces empirical bounds (ideal/nadir), reference fronts (for IGD+), and minimum runs (95% CI).

```bash
julia --project=.. Optuna_code/baseline_benchmark.jl   # 50 benchmarks
julia --project=.. Optuna_code/baseline_truss.jl        # truss
```

### Phase 2 — HPO (benchmarks)

Optuna trials with multiple samplers. Each trial proposes hyperparameters, runs the optimizer num_runs times, computes normalized HV and IGD+.

```bash
julia --project=.. Optuna_code/hpo_benchmark.jl
```

### Phase 3 — Sampler comparison

Analyze Phase 2 results. Pick top 3 samplers for the truss.

```bash
python3 Optuna_code/analysis.py --results-dir ./Results --output-dir ./thesis_figures
```

### Phase 4 — HPO (truss)

Apply selected samplers to the truss with 50 iterations.

```bash
julia --project=.. Optuna_code/hpo_truss.jl
```

### Phase 5 — Analysis

Re-run analysis with full data.

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

1. Union all feasible ND solutions across 100 baseline runs
2. Empirical ideal = componentwise min, nadir = componentwise max
3. Normalize: `f' = (f - ideal) / (nadir - ideal)`
4. Reference point: R = 1.01 × ones(n_objectives)

For DEAD problems (no feasible solutions with defaults): HPO uses library nadir + empirical ideal from the trial's own solutions.

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
