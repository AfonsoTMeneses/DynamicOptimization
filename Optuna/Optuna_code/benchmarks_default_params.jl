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
using Distributed
optuna = pyimport("optuna")
include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
include(joinpath(@__DIR__, "optuna_utils.jl"))


algorithms = ["MOEAD_DE_searchspace", "NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]
main_script_name = split(basename(abspath(@__FILE__)), ".jl")[1]
results_path = normpath(dirname(@__DIR__),"Results/$(main_script_name)_Results")
cd(results_path)

All_Algorithm_structure = initialize_algorithm_structures(algorithms)

lb_instaces = 1
hb_instaces = 50
problem_dataframe = benchmark_handler(All_Algorithm_structure, lb_instaces, hb_instaces, main_script_name)


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
problem_dataframe = initialize_runs_dicts(All_Algorithm_structure, main_script_name, problem_dataframe, 1, nrow(problem_dataframe))

delete_file("$(main_script_name)_Results.csv", results_path)

for alg in algorithms
    Algorithm_structure = detect_searchspaces(alg)
    folder_name = "$(main_script_name)_Results.csv"

    pop_size     = Algorithm_structure.Parameters[:N]
    n_iterations = options_dict[:iterations]
    max_evals    = pop_size * n_iterations

    algorithm_instance = Algorithm_structure.Name
    println("Using algorithm: $algorithm_instance")
    options_dict[:f_calls_limit] = 3 * max_evals

    for prob_idx in range(1, nrow(problem_dataframe))

        HV = []

       
        results_dict = OrderedDict()  

        result_df = DataFrame(
            algorithm_name = Symbol[],
            problem_name   = String[],
            hv_value       = Vector[],
        )

        metaheuristic = set_up_algorithm(algorithm_instance, options_dict)

        problem_name, problem_function, problem_bounds, problem_ref_point, num_runs =
            unpack_df(problem_dataframe, String(algorithm_instance), prob_idx)

        println("problem_name::      $(problem_name)")
        println("problem_bounds::    $(problem_bounds)")
        println("problem_ref_point:: $(problem_ref_point)")
        println("num_runs::          $(num_runs)")

        if num_runs == "Inf" || num_runs > 100
            println("Skipping problem $problem_name due to invalid run length.")
            push!(result_df, (algorithm_instance, problem_name, [-Inf]))
            write_header = !isfile(folder_name)
            CSV.write(folder_name, result_df, append = true, writeheader = write_header)
            continue
        end

        println(num_runs)

        @time for run_idx in 1:num_runs

            metaheuristic = set_up_algorithm(algorithm_instance, options_dict)

            println("Starting task... run: $run_idx")
            status = optimize(problem_function, problem_bounds, metaheuristic)
            display(status)
            println("Task Finished... run: $run_idx")

            approx_front = get_non_dominated_solutions(status.population)
            hv = hypervolume(approx_front, problem_ref_point)
            push!(HV, hv)
        end

        results_dict[Symbol(algorithm_instance)] = HV

        for val in keys(results_dict)
            push!(result_df, (val, problem_name, results_dict[val]))
        end

        write_header = !isfile(folder_name)
        CSV.write(folder_name, result_df, append = true, writeheader = write_header)

    end
   
end
