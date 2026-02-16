include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
include(joinpath(@__DIR__, "example_utils.jl"))
include(joinpath(@__DIR__, "utils_minimum_runs.jl"))
using PyCall
using Metaheuristics
using Metaheuristics: TestProblems, optimize, SPEA2, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
using HardTestProblems
using DataStructures
using CSV
using DataFrames
using Statistics
using Distances
using BayesOpt
using JSON

main_script_name = basename(abspath(@__FILE__))


run(`clear`)

base_dir = pwd()

### For CCMO add prefix and check flag in run_optimization

alg = ["SMS_EMOA_searchspace"]



for searchspace in alg

    CSV_RUNS_FILE_NAME, CSV_LENGTH_RESULTS_NAME = check_CSV(searchspace, main_script_name; test = false)

    Algorithm_structure = detect_searchspaces(searchspace)

    optuna_configuration = Optimization_configuration(1,50,100)

    iteration_counts = [100] 

    results_path = joinpath(splitdir(@__DIR__)[1], "Results")

    db_path = joinpath(results_path, "db_$(string(Algorithm_structure.Name))_$(string(iteration_counts[1])).sqlite3")

    results = []

    run(`clear`)

    #########

    ######### GET RUNS WITH DEFAULT PARAMS

    ############

    count = 0
    for current_instance in optuna_configuration.lb_instaces:optuna_configuration.max_instace
        
            hv_values = Dict()
            num_ite = 100
            problem_name = HardTestProblems.NAME_OF_PROBLEMS_RW_MOP_2021[current_instance]
            println("Optimizing problem: ", problem_name)
            f, searchspace, reference_point = getproblem(current_instance) 
            
            if haskey(ref_points_offset, current_instance)
                reference_point = ref_points_offset[current_instance]
            end

            algorithm_instance = Algorithm_structure.Name

            println("Using algorithm: $algorithm_instance")
            options = Metaheuristics.Options(; iterations=num_ite, time_limit=4.0)

            metaheuristic = set_up_algorithm(algorithm_instance, options)

            result_dir = "/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Optuna/Results"

            if pwd() !== result_dir
                cd(result_dir)
            end

            global All_HV = Dict(:Hypervolumes => Float64[])
            global All_dist = Dict(:Distances => Float64[])

            num_runs = 100

            for i in 1:num_runs
                println("Starting task...")
                status = optimize(f, searchspace, metaheuristic)
                println("Task Finished...")
                approx_front = get_non_dominated_solutions(status.population)
                All_HV = return_results(approx_front, reference_point)
            end


            df = DataFrame(
                problem_name = problem_name,
                current_instance = current_instance,
                #Pareto_front_length = length(approx_front),
                length_HV = length(All_HV[:Hypervolumes])
            )

            if count == 0
                CSV.write(CSV_LENGTH_RESULTS_NAME, df; append=true, writeheader = true )
                count = 1
            else
                CSV.write(CSV_LENGTH_RESULTS_NAME, df; append=true)
            end
   
            results = All_HV
            type_of_result = first(keys(results))                


            println("Results::$results")

            hv_values[num_ite] = mean(results[type_of_result])


            println("Hypervolume: $(hv_values[num_ite])")
            
            get_minimum_runs(results, problem_name, current_instance, CSV_RUNS_FILE_NAME)
    end
end



function get_minimum_runs_parametric_truss(alg, main_script_name, n_iterations, num_runs; problem = problem, integer_space = integer_space)
        
    CSV_RUNS_FILE_NAME, _ = check_CSV(alg, main_script_name; test = false)
    Algorithm_structure = detect_searchspaces(alg)
    pop_size = Algorithm_structure.Parameters[:N]
    max_evals = pop_size * n_iterations
    results = []
    hv_values = Dict()

    algorithm_instance = Algorithm_structure.Name
    println("Using algorithm: $algorithm_instance")
    options = Metaheuristics.Options(iterations = n_iterations, f_calls_limit = 3 * max_evals)
    metaheuristic = set_up_algorithm(algorithm_instance, options)

    All_HV = Dict(:Hypervolumes => Float64[])
    
    reference_point = [10, 4000]

    @time for i in 1:num_runs
        println("Starting task...")
        status = optimize(problem, integer_space, metaheuristic)
        display(status)
        println("Task Finished... iteration : $i")
        approx_front = get_non_dominated_solutions(status.population)
        HV = hypervolume(approx_front, reference_point) 
        push!(All_HV[:Hypervolumes], HV) 
    end

    if pwd() !== result_dir
        cd(result_dir)
    end

    results = All_HV


    type_of_result = first(keys(results))                

    println("Results::$results")

    hv_values[n_iterations] = mean(results[type_of_result])

    println("Hypervolume: $(hv_values[n_iterations])")

    problem_name = "parametric_truss_example_n_iterations_$(n_iterations)_num_runs$(num_runs)"
    current_instance = 0
    get_minimum_runs(results, problem_name, current_instance, CSV_RUNS_FILE_NAME)
end

#for alg in algorithms
#     get_minimum_runs_parametric_truss(alg, main_script_name, n_iterations, num_runs)
#end