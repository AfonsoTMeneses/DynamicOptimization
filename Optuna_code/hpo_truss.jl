using Distributed

let
    baseline_dir = normpath(dirname(@__DIR__), "Results/baseline_truss_Results")
    algs = ["NSGA2", "SPEA2", "SMS_EMOA", "MOEAD_DE"]
    needed = vcat(
        ["empirical_bounds_baseline_truss_$alg.csv" for alg in algs],
        ["minimum_runs_baseline_truss_$alg.csv"    for alg in algs],
    )
    missing_files = filter(f -> !isfile(joinpath(baseline_dir, f)), needed)
    if !isempty(missing_files)
        @info "Baseline files missing, running baseline_truss.jl first" missing_files
        run(`julia --project=$(dirname(@__DIR__)) $(joinpath(@__DIR__, "baseline_truss.jl"))`)
        @info "Baseline done, continuing to HPO"
    else
        @info "Baseline files present, skipping baseline_truss.jl"
    end
end

addprocs(2)

@everywhere begin
    using PyCall
    optuna = pyimport("optuna")
    using KhepriFrame3DD
    using Distributed
    using Metaheuristics
    using Metaheuristics: optimize, get_non_dominated_solutions, pareto_front, Options
    import Metaheuristics.PerformanceIndicators: hypervolume
    using DataStructures
    using CSV
    using DataFrames
    using Statistics
    using JSON
    using Logging
    using HardTestProblems
    include(joinpath(@__DIR__, "optuna_utils.jl"))
    include(joinpath(@__DIR__, "truss_problem.jl"))
end

@everywhere begin
    main_script_name = split(basename(abspath(@__FILE__)), ".jl")[1]
    algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]
    results_path = normpath(dirname(@__DIR__),"Results/$(main_script_name)_Results")
    mkpath(results_path)
    cd(results_path)
end

for alg in algorithms
    suffix = split(alg, "_searchspace")[1]
    path = String(joinpath(results_path, suffix))
    remove_existing_csv(path)
end

@everywhere begin
    n_trials = 100
    All_Algorithm_structure = initialize_algorithm_structures(algorithms)
end

log_file = joinpath(abspath(joinpath(@__DIR__, "../..")), "log_$(main_script_name).txt")
isfile(log_file) && rm(log_file)

@everywhere begin
    options_dataframe = DataFrame(
        x_tol               = 1e-8,
        f_tol               = 1e-12,
        f_tol_rel           = eps(),
        f_tol_abs           = 0.0,
        g_tol               = 0.0,
        h_tol               = 0.0,
        f_calls_limit       = 1000000000,
        time_limit          = Inf,
        iterations          = 50,
        store_convergence   = false,
        debug               = false,
        parallel_evaluation = false,
        verbose             = false,
    )
    options_dict = push_options(options_dataframe)
end

baseline_name = "baseline_truss"
baseline_results_path = normpath(dirname(@__DIR__), "Results/$(baseline_name)_Results")

empirical_bounds_dict = load_merged_empirical_bounds(baseline_results_path, baseline_name)
println("Empirical bounds available: $(length(empirical_bounds_dict)) problem(s)")

if !haskey(empirical_bounds_dict, 1)
    error("Empirical bounds for problem 1 not found under $baseline_results_path. " *
          "Cannot proceed without a valid truss baseline.")
end
ideal_pt, nadir_pt = empirical_bounds_dict[1]
println("  Truss reference point (empirical nadir from baseline): $nadir_pt")

problem_dataframe = DataFrame(
    problems_names    = "parametric_truss",
    problem_function  = problem,
    problem_bounds    = integer_space,
    problem_ref_point = [nadir_pt],
)

for Algorithm_structure in All_Algorithm_structure
    fname = joinpath(baseline_results_path, "minimum_runs_$(baseline_name)_$(Algorithm_structure.Name).csv")

    if !isfile(fname)
        @warn "$fname not found — using FALLBACK_RUNS for $(Algorithm_structure.Name)"
        problem_dataframe[!, Algorithm_structure.Name] = [FALLBACK_RUNS]
        continue
    end

    df = with_logger(SimpleLogger(stderr, Logging.Error)) do
        CSV.read(fname, DataFrame; pool=false)
    end

    runs_dicts = Dict()
    runs_dicts = get_df_column_values(df, RUNS_CI_COLUMN, RUNS_CI_POSITION, Algorithm_structure.Name, runs_dicts)
    loaded_runs = runs_dicts[Algorithm_structure.Name][1:1]
    problem_dataframe[!, Algorithm_structure.Name] = loaded_runs
    println("  $(Algorithm_structure.Name): num_runs = $(loaded_runs[1])  (from $(basename(fname)))")
end

optuna_storage_dir = joinpath(results_path, "optuna_studies")

elapsed_time = @elapsed results = run_HPO(
    sampler_vector,
    options_dict,
    results_path,
    All_Algorithm_structure,
    problem_dataframe,
    main_script_name, n_trials,
    empirical_bounds_dict;
    collect_fronts=true,
    storage_dir=optuna_storage_dir,
)

open("time_run_HPO_$(main_script_name).txt", "a") do io
    println(io, "run_HPO elapsed time: $(round(elapsed_time, digits=2))s ($(round(elapsed_time/60, digits=2)) min)")
end

write_HPO_data_into_csv(results, options_dict, results_path)
