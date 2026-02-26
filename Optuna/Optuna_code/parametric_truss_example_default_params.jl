using PyCall
optuna = pyimport("optuna")
using KhepriFrame3DD
using Metaheuristics
using Metaheuristics: optimize, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
using HardTestProblems
using DataStructures
using CSV
using DataFrames
using Statistics
using JSON
using Distributed
include(joinpath(@__DIR__, "optuna_utils.jl"))
include(joinpath(@__DIR__, "parametric_truss_example.jl"))
include(joinpath(@__DIR__, "utils_minimum_runs.jl"))

base_dir = "/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Optuna/Results/"
algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]
main_script_name = split(basename(abspath(@__FILE__)), ".jl")[1]
results_path = joinpath(base_dir, "$(main_script_name)_Results")   
cd(results_path)
All_Algorithm_structure = initialize_algorithm_structures(algorithms)


delete_file("log.txt", "/home/afonso-meneses/Desktop/GitHub/DynamicOptimization")

delete_file("parametric_truss_example_default_params.csv", results_path)

    
reference_point = [10,4000]

problem_data = [problem, integer_space, reference_point]

problem_dataframe = DataFrame(
            problems_names = "parametric_truss_example",
            problem_function = problem_data[1],
            problem_bounds = problem_data[2],
            problem_ref_point = [reference_point],
)


options_dataframe = DataFrame(
                    x_tol = 1e-8,
                    f_tol = 1e-12,
                    f_tol_rel = eps(),
                    f_tol_abs = 0.0,
                    g_tol = 0.0,
                    h_tol = 0.0,
                    f_calls_limit = 1000000000,
                    time_limit = Inf,
                    iterations = 50,
                    store_convergence = false,
                    debug = false,
                    parallel_evaluation = false,
                    verbose = false
                )

run(`clear`)


problem_name, problem_function,
problem_bounds, problem_ref_point = unpack_df_vectors(problem_dataframe, 1)
options_dict = push_options(options_dataframe)
problem_dataframe = initialize_runs_dicts(All_Algorithm_structure, main_script_name, problem_dataframe, 1, nrow(problem_dataframe))

results = []
results_dict = Dict()

for alg in algorithms
    Algorithm_structure = detect_searchspaces(alg)

    pop_size = Algorithm_structure.Parameters[:N]
    n_iterations = options_dict[:iterations]
    max_evals = pop_size * n_iterations
    hv_values = Dict()

    algorithm_instance = Algorithm_structure.Name
    println("Using algorithm: $algorithm_instance")
    options_dict[:f_calls_limit] = 3 * max_evals

    metaheuristic = set_up_algorithm(algorithm_instance, options_dict)

    HV = []

    num_runs = problem_dataframe[1, Symbol(algorithm_instance)]


    @time for i in 1:num_runs
        println("Starting task...")
        status = optimize(problem_function, problem_bounds, metaheuristic)
        display(status)
        println("Task Finished... iteration : $i")
        approx_front = get_non_dominated_solutions(status.population)
        hv = hypervolume(approx_front, problem_ref_point) 
        push!(HV, hv)
    end

    push!(results_dict, Symbol(algorithm_instance) => HV)

    println("Results::$results_dict")

end

results_dict

result_df = DataFrame(
            algorithm_name = Symbol[],
            problem_name = String[],
            hv_value = Vector[],
        )

folder_name = "$(main_script_name).csv"

for val in keys(results_dict)

    push!(result_df, (val, main_script_name, results_dict[val]) )

end

write_header = !isfile(folder_name)
CSV.write(folder_name, result_df, append = true, writeheader = write_header)


