using KhepriFrame3DD
using Metaheuristics
using Metaheuristics: optimize, get_non_dominated_solutions, pareto_front, Options
import Metaheuristics.PerformanceIndicators: hypervolume
using DataStructures
using CSV
using DataFrames
using Statistics
using JSON
using Distributed
using Logging
include(joinpath(@__DIR__, "optuna_utils.jl"))
include(joinpath(@__DIR__, "truss_problem.jl"))
include(joinpath(@__DIR__, "utils_minimum_runs.jl"))

algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"] #
main_script_name = String(split(basename(abspath(@__FILE__)), ".jl")[1])
results_path = normpath(dirname(@__DIR__),"Results/$(main_script_name)_Results")
mkpath(results_path)
cd(results_path)


All_Algorithm_structure = initialize_algorithm_structures(algorithms)

delete_file("$(main_script_name).csv", results_path)

    
reference_point = [10, 4000] # This is used as a placeholder

problem_dataframe = DataFrame(
            problems_names = "parametric_truss",
            problem_function = problem,
            problem_bounds = integer_space,
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

options_dict = push_options(options_dataframe)

num_runs = 100
results_dict = Dict()

nobj = length(reference_point)

for alg in algorithms
    Algorithm_structure = detect_searchspaces(alg)
    algorithm_instance = Algorithm_structure.Name

    if algorithm_instance == :MOEAD_DE
        moead_weights = set_up_weights_MOEAD_DE(nobj)
        pop_size = length(moead_weights)
    else
        moead_weights = nothing
        pop_size = Algorithm_structure.Parameters[:N]
    end
    n_iterations = options_dict[:iterations]
    max_evals = pop_size * n_iterations
    options_dict[:f_calls_limit] = 3 * max_evals

    println("Using algorithm: $algorithm_instance (nobjectives=$nobj, pop_size=$pop_size)")

    # pass 1 — collect fronts
    all_run_fronts = Vector{Vector{Vector{Float64}}}()

    @time for i in 1:num_runs
        options_dict[:seed] = abs(hash((algorithm_instance, "parametric_truss", i))) % 1_000_000
        metaheuristic = set_up_algorithm(algorithm_instance, options_dict; nobjectives=nobj, MOEAD_WEIGHTS=moead_weights)

        println("Starting task... iteration: $i")
        status = with_logger(SimpleLogger(stderr, Logging.Error)) do
            optimize(problem, integer_space, metaheuristic)
        end
        println("Task Finished... iteration: $i")
        front = get_feasible_non_dominated(status.population)
        push!(all_run_fronts, front)
        status = nothing
        GC.gc(true)
    end


    ideal, nadir = compute_empirical_bounds(all_run_fronts)

    bounds_csv = joinpath(results_path, "empirical_bounds_$(main_script_name)_$(Algorithm_structure.Name).csv")
    if isfile(bounds_csv); rm(bounds_csv); end

    ref_front_csv = joinpath(results_path, "reference_fronts_$(main_script_name)_$(Algorithm_structure.Name).csv")
    if isfile(ref_front_csv); rm(ref_front_csv); end

    CSV_RUNS_FILE_NAME = joinpath(results_path, "minimum_runs_$(main_script_name)_$(Algorithm_structure.Name).csv")
    if isfile(CSV_RUNS_FILE_NAME); rm(CSV_RUNS_FILE_NAME); end

    HV = Float64[]
    if isnothing(ideal) || isnothing(nadir)
        @warn "Parametric truss ($algorithm_instance): no feasible solutions. HV will be 0."
        HV = fill(0.0, num_runs)
    else
        println("  Empirical ideal: $ideal")
        println("  Empirical nadir: $nadir")
        save_empirical_bounds(bounds_csv, "parametric_truss", 1, ideal, nadir)

        # save union front for compute_cPF.jl
        union_front = reduce(vcat, filter(!isempty, all_run_fronts))
        save_reference_front(ref_front_csv, 1, union_front)

        # pass 2 — HV
        for front in all_run_fronts
            push!(HV, normalized_hypervolume(front, ideal, nadir))
        end
    end

    All_HV = Dict(:Hypervolumes => HV)
    hv_mean = mean(HV)
    println("Results::$All_HV")
    println("Mean Hypervolume ($n_iterations iterations): $hv_mean")

    get_minimum_runs(All_HV, "parametric_truss", CSV_RUNS_FILE_NAME, 1)

    results_dict[Symbol(algorithm_instance)] = HV

    all_run_fronts = nothing
    GC.gc(true)
end

result_df = DataFrame(
            algorithm_name = Symbol[],
            problem_name = String[],
            hv_value = Vector[],
        )

folder_name = "$(main_script_name).csv"

for val in keys(results_dict)
    push!(result_df, (val, "parametric_truss", results_dict[val]))
end

write_header = !isfile(folder_name)
CSV.write(folder_name, result_df, append = true, writeheader = write_header)
