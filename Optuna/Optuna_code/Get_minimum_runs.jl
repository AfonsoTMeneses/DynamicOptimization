using PyCall
optuna = pyimport("optuna")
using Metaheuristics
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
using HardTestProblems
using DataStructures
using CSV
using DataFrames
using Statistics
using Distances
using JSON
using Distributed
include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
include(joinpath(@__DIR__, "utils_minimum_runs.jl"))
include(joinpath(@__DIR__, "optuna_utils.jl"))

main_script_name = String(split(basename(abspath(@__FILE__)), ".jl")[1])


result_dir = joinpath(@__DIR__, "Results")

algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]

results_path = joinpath(@__DIR__, "Results")

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
options = push_options(options_dataframe)


for searchspace in algorithms
    
    CSV_RUNS_FILE_NAME = check_CSV(String(searchspace), main_script_name, results_path)

    Algorithm_structure = detect_searchspaces(searchspace)

    optuna_configuration = Optimization_configuration(1, 50, 100)

    num_runs = 100

    for current_instance in optuna_configuration.lb_instances:optuna_configuration.hb_instance

        problem_name, f, problem_bounds, reference_point = getproblem(current_instance)

        println("Optimizing problem: ", problem_name)

        if haskey(ref_points_offset, current_instance)
            reference_point = ref_points_offset[current_instance]
        end

        algorithm_instance = Algorithm_structure.Name
        println("Using algorithm: $algorithm_instance")


        metaheuristic = set_up_algorithm(algorithm_instance, options)

        cd(result_dir)

        All_HV = Dict(:Hypervolumes => Float64[])

        for i in 1:num_runs
            println("Starting task... run $i / $num_runs")

            metaheuristic = set_up_algorithm(algorithm_instance, options)

            status = optimize(f, problem_bounds, metaheuristic)
            println("Task Finished...")
            display(status)
            pf = pareto_front(status) 
            println(pf)

            approx_front = get_non_dominated_solutions(status.population)
            hv = hypervolume(approx_front, reference_point)
            push!(All_HV[:Hypervolumes], hv)
        end


        df = DataFrame(
            problem_name     = problem_name,
            current_instance = current_instance,
            length_HV        = length(All_HV[:Hypervolumes])
        )

        hv_mean = mean(All_HV[:Hypervolumes])

        println("Results::$All_HV")
        println("Mean Hypervolume ($(options[:iterations]) iterations): $hv_mean")

        get_minimum_runs(All_HV, problem_name, CSV_RUNS_FILE_NAME)

    end 
end  
