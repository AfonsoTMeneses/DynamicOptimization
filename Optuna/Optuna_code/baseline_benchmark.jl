using Metaheuristics
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
using HardTestProblems
using DataStructures
using CSV
using DataFrames
using Statistics
using JSON
using Distributed
using Logging
include(joinpath(@__DIR__, "utils_minimum_runs.jl"))
include(joinpath(@__DIR__, "optuna_utils.jl"))
#Metaheuristics v3.4.0 `https://github.com/ines-pereira/Metaheuristics.jl#master`


main_script_name = String(split(basename(abspath(@__FILE__)), ".jl")[1])

algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]

results_path = joinpath(dirname(@__DIR__), "Results")

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
    verbose             = false
)
options = push_options(options_dataframe)


for searchspace in algorithms
    
    CSV_RUNS_FILE_NAME = check_CSV(String(searchspace), main_script_name, results_path)

    Algorithm_structure = detect_searchspaces(searchspace)

    num_runs = 100

    n_iterations = options[:iterations]
    algorithm_instance = Algorithm_structure.Name
    println("Using algorithm: $algorithm_instance")

    bounds_csv = joinpath(results_path, "empirical_bounds_$(main_script_name)_$(Algorithm_structure.Name).csv")
    if isfile(bounds_csv)
        rm(bounds_csv)
    end

    ref_front_csv = joinpath(results_path, "reference_fronts_$(main_script_name)_$(Algorithm_structure.Name).csv")
    if isfile(ref_front_csv)
        rm(ref_front_csv)
    end

    cd(results_path)

    for current_instance in 1:50

        problem_name, f, problem_bounds, reference_point = getproblem(current_instance)

        println("Optimizing problem: ", problem_name)

        nobj = length(reference_point)
        if algorithm_instance == :MOEAD_DE
            moead_weights = set_up_weights_MOEAD_DE(nobj)
            pop_size = length(moead_weights)
        else
            moead_weights = nothing
            pop_size = Algorithm_structure.Parameters[:N]
        end
        max_evals = pop_size * n_iterations
        options[:f_calls_limit] = 3 * max_evals
        println("  nobjectives=$nobj, pop_size=$pop_size, f_calls_limit=$(options[:f_calls_limit])")

        # pass 1 — collect fronts
                all_run_fronts = Vector{Vector{Vector{Float64}}}()

        for i in 1:num_runs
            println("Starting task... run $i / $num_runs")

            options[:seed] = abs(hash((algorithm_instance, problem_name, i))) % 1_000_000
            metaheuristic = set_up_algorithm(algorithm_instance, options; nobjectives=nobj, MOEAD_WEIGHTS=moead_weights)

            status = with_logger(SimpleLogger(stderr, Logging.Error)) do
                optimize(f, problem_bounds, metaheuristic)
            end
            println("Task Finished...")

            front = get_feasible_non_dominated(status.population)
            status = nothing
            push!(all_run_fronts, front)
        end

        # empirical bounds
                ideal, nadir = compute_empirical_bounds(all_run_fronts)

        if isnothing(ideal) || isnothing(nadir)
            @warn "Problem $problem_name: no feasible solutions found in ANY of $num_runs runs. HV will be 0."
            All_HV = Dict(:Hypervolumes => fill(0.0, num_runs))
        else
            println("  Empirical ideal: $ideal")
            println("  Empirical nadir: $nadir")

            save_empirical_bounds(bounds_csv, problem_name, current_instance, ideal, nadir)


            union_front = reduce(vcat, filter(!isempty, all_run_fronts))
            save_reference_front(ref_front_csv, current_instance, union_front)

            # pass 2 — normalized HV
                        All_HV = Dict(:Hypervolumes => Float64[])

            for front in all_run_fronts
                hv = normalized_hypervolume(front, ideal, nadir)
                push!(All_HV[:Hypervolumes], hv)
            end
        end

        hv_mean = mean(All_HV[:Hypervolumes])

        println("Results::$All_HV")
        println("Mean Hypervolume ($(options[:iterations]) iterations): $hv_mean")

        get_minimum_runs(All_HV, problem_name, CSV_RUNS_FILE_NAME, current_instance)

        all_run_fronts = nothing
        GC.gc()

    end 
end  
