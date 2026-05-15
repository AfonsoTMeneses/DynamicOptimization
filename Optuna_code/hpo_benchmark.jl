using Distributed
addprocs(2)

@everywhere begin
    using Metaheuristics
    using Metaheuristics: optimize, get_non_dominated_solutions, pareto_front, Options
    import Metaheuristics.PerformanceIndicators: hypervolume
    using HardTestProblems
    using DataStructures
    using CSV
    using DataFrames
    using Statistics
    using JSON
    using PyCall
    using Logging
    optuna = pyimport("optuna")
    include(joinpath(@__DIR__, "optuna_utils.jl"))
end

@everywhere begin
    algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]
    main_script_name = split(basename(abspath(@__FILE__)), ".jl")[1]
    results_path = normpath(dirname(@__DIR__),"Results/$(main_script_name)_Results")
    cd(results_path)
end

for alg in algorithms
    suffix = split(alg, "_searchspace")[1]
    remove_existing_csv(joinpath(results_path, suffix))
end

@everywhere begin
    n_trials = 100
    All_Algorithm_structure = initialize_algorithm_structures(algorithms)
end

@everywhere begin
    lb_instaces = 1
    hb_instaces = 50
    baseline_name = "baseline_benchmark"
    baseline_dir = joinpath(dirname(@__DIR__), "Results")
    problem_dataframe = benchmark_handler(All_Algorithm_structure, lb_instaces, hb_instaces, main_script_name;
                                          baseline_name=baseline_name, baseline_dir=baseline_dir)
end

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
        iterations          = 100,
        store_convergence   = false,
        debug               = false,
        parallel_evaluation = false,
        verbose             = false,
    )
    options_dict = push_options(options_dataframe)
end

bounds_base_path = joinpath(dirname(@__DIR__), "Results")
empirical_bounds_dict = load_merged_empirical_bounds(bounds_base_path, "baseline_benchmark")
println("Total problems with empirical bounds: $(length(empirical_bounds_dict))/50")

elapsed_time = @elapsed results = run_HPO(sampler_vector, options_dict, results_path, All_Algorithm_structure, problem_dataframe, main_script_name, n_trials, empirical_bounds_dict;
                                          collect_fronts=true,
                                          storage_dir=joinpath(results_path, "optuna_studies"))
open("time_run_HPO_$(main_script_name).txt", "a") do io
    println(io, "run_HPO elapsed time: $(round(elapsed_time, digits=2))s ($(round(elapsed_time/60, digits=2)) min)")
end
