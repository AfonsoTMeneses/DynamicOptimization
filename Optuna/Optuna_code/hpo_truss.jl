using Distributed
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
if isfile(log_file)
    rm(log_file)
end


@everywhere begin
    
    reference_point = [10, 4000]

    problem_dataframe = DataFrame(
                problems_names   = "parametric_truss",
                problem_function = problem,
                problem_bounds   = integer_space,
                problem_ref_point = [reference_point],
    )

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
                        iterations          = 50,
                        store_convergence   = false,
                        debug               = false,
                        parallel_evaluation = false,
                        verbose             = false
                    )

    options_dict = push_options(options_dataframe)
end

# load baseline data
baseline_name = "baseline_truss"
baseline_results_path = normpath(dirname(@__DIR__), "Results/$(baseline_name)_Results")


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
    problem_dataframe[!, Algorithm_structure.Name] = runs_dicts[Algorithm_structure.Name][1:1]
end


empirical_bounds_dict = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
for Algorithm_structure in All_Algorithm_structure
    bounds_csv = joinpath(baseline_results_path, "empirical_bounds_$(baseline_name)_$(Algorithm_structure.Name).csv")
    if isfile(bounds_csv)
        alg_bounds = load_empirical_bounds(bounds_csv)
        for (inst, (ideal, nadir)) in alg_bounds
            if haskey(empirical_bounds_dict, inst)
                existing_ideal, existing_nadir = empirical_bounds_dict[inst]
                merged_ideal = [min(existing_ideal[i], ideal[i]) for i in eachindex(ideal)]
                merged_nadir = [max(existing_nadir[i], nadir[i]) for i in eachindex(nadir)]
                empirical_bounds_dict[inst] = (merged_ideal, merged_nadir)
            else
                empirical_bounds_dict[inst] = (ideal, nadir)
            end
        end
        println("Loaded empirical bounds from $bounds_csv")
    else
        @warn "Bounds file not found: $bounds_csv"
    end
end
println("Empirical bounds available: $(length(empirical_bounds_dict)) problem(s)")


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
                                    storage_dir=optuna_storage_dir)

open("time_run_HPO_$(main_script_name).txt", "a") do io
        println(io, "run_HPO elapsed time: $(round(elapsed_time, digits=2))s ($(round(elapsed_time/60, digits=2)) min)")
end

write_HPO_data_into_csv(results, options_dict, results_path)
