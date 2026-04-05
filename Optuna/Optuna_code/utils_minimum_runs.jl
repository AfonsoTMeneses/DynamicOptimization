using DataStructures
using Metaheuristics
using Metaheuristics: pareto_front
using HardTestProblems
using DataFrames
using CSV
using Statistics
using JSON
using Logging


# CSV helpers

function check_CSV(searchspace::String, name_of_script::String, results_path::String)
    println("Currently using $searchspace")
    alg_name = split(searchspace, "_searchspace")[1]

    script_name = endswith(name_of_script, ".jl") ? split(name_of_script, ".jl")[1] : name_of_script

    CSV_RUNS_FILE_NAME = joinpath(results_path, "minimum_runs_$(script_name)_$(alg_name).csv")

    remove_existing_csv(CSV_RUNS_FILE_NAME)

    return CSV_RUNS_FILE_NAME

end

function remove_existing_csv(filepath::String)
    if ispath(filepath)
        println("Removing $(filepath)")
        rm(filepath)
    end
end


# Minimum runs formula

# Given a z-score, standard deviation, and margin of error ϵ,
# returns the minimum number of runs needed.
minimum_runs(z, stdev, ϵ) = ceil((z * stdev / ϵ)^2)


# Core: compute and write minimum runs

function get_minimum_runs(results::Dict, problem_name::String, CSV_RUNS_FILE_NAME::String, current_instance::Int)

    typed_results = results[:Hypervolumes]
    println(typed_results)
    if isempty(typed_results)
        @warn "No HV results for problem $problem_name — skipping."
        return
    end

    mean_hv = mean(typed_results)
    all_std  = std(typed_results)

    if all_std < 0.001
        @warn "HV values are nearly identical for $problem_name (std=$all_std < 0.001), using fallback std=0.001"
        all_std = 0.001
    end

    const_confidence_levels = [
        (90,  1.645),
        (95,  1.96),
        (98,  2.33),
        (99,  2.575)
    ]

 
    runs_dict = OrderedDict{Int, Vector{Union{Int, String}}}(
        level => [] for (level, _) in const_confidence_levels
    )

    last_error = 0.0

    for (level, z) in const_confidence_levels
        for ϵ in 0.01:0.01:0.1
            margin = ϵ * mean_hv
            runs   = minimum_runs(z, all_std, margin)
        
            if isfinite(runs)
                push!(runs_dict[level], Int(floor(runs)))
            else
                push!(runs_dict[level], "Inf")
                @warn "Infinite/too-large run count for $(level)% confidence, ϵ=$(round(ϵ, digits=2)): $runs"
            end
            last_error = margin
        end
    end

    println("Margin of error range: ϵ ∈ [$(round(0.01 * mean_hv, sigdigits=4)), $(round(last_error, sigdigits=4))]")
    for (level, _) in const_confidence_levels
        println("  $(level)% CI: $(runs_dict[level])")
    end

    println("min:  $(minimum(typed_results))")
    println("max:  $(maximum(typed_results))")
    println("std:  $(all_std)")
    println("mean: $(mean_hv)")

    stats_df = DataFrame(
        problem_name           = problem_name,
        best_HV                = maximum(typed_results),
        error                  = last_error / mean_hv,
        current_instance       = current_instance,
        confidence_interval_90 = [JSON.json(runs_dict[90])],
        confidence_interval_95 = [JSON.json(runs_dict[95])],
        confidence_interval_98 = [JSON.json(runs_dict[98])],
        confidence_interval_99 = [JSON.json(runs_dict[99])]
    )

    write_header = !isfile(CSV_RUNS_FILE_NAME)
    CSV.write(CSV_RUNS_FILE_NAME, stats_df; append=true, writeheader=write_header)

    All_HV_df = DataFrame(All_HV = [JSON.json(typed_results)])
    CSV.write(CSV_RUNS_FILE_NAME, All_HV_df; append=true, writeheader=false)

    separator_df = DataFrame(
        problem_name           = [""],
        best_HV                = [missing],
        error                  = [missing],
        current_instance       = [missing],
        confidence_interval_90 = [""],
        confidence_interval_95 = [""],
        confidence_interval_98 = [""],
        confidence_interval_99 = [""]
    )
    CSV.write(CSV_RUNS_FILE_NAME, separator_df; append=true, writeheader=false)

end

function get_minimum_runs(results::Dict, problem_name::String, CSV_RUNS_FILE_NAME::String)
    get_minimum_runs(results, problem_name, CSV_RUNS_FILE_NAME, 0)
end


function get_minimum_runs_parametric_truss(alg, problem_dataframe, options_dict, num_runs, main_script_name, results_path)

    problem_name, problem_function,
    problem_bounds, problem_ref_point = unpack_df_vectors(problem_dataframe, 1)

    CSV_RUNS_FILE_NAME = check_CSV(alg, main_script_name, results_path)

    Algorithm_structure = detect_searchspaces(alg)

    nobj = length(problem_ref_point)
    if Algorithm_structure.Name == :MOEAD_DE
        moead_weights = set_up_weights_MOEAD_DE(nobj)
        pop_size = length(moead_weights)
    else
        moead_weights = nothing
        pop_size = Algorithm_structure.Parameters[:N]
    end
    n_iterations = options_dict[:iterations]
    max_evals    = pop_size * n_iterations

    algorithm_instance = Algorithm_structure.Name
    println("Using algorithm: $algorithm_instance (nobjectives=$nobj, pop_size=$pop_size)")
    options_dict[:f_calls_limit] = 3 * max_evals

    # PASS 1: Collect feasible non-dominated fronts per run
    all_run_fronts = Vector{Vector{Vector{Float64}}}()

    @time for i in 1:num_runs
        metaheuristic = set_up_algorithm(algorithm_instance, options_dict; nobjectives=nobj, MOEAD_WEIGHTS=moead_weights)

        println("Starting task... iteration: $i")
        status = with_logger(SimpleLogger(stderr, Logging.Error)) do
            optimize(problem_function, problem_bounds, metaheuristic)
        end
        println("Task Finished... iteration: $i")
        front = get_feasible_non_dominated(status.population)
        status = nothing
        push!(all_run_fronts, front)
    end

    # Compute empirical bounds from the union of all fronts
    ideal, nadir = compute_empirical_bounds(all_run_fronts)

    if isnothing(ideal) || isnothing(nadir)
        @warn "Parametric truss: no feasible solutions found. HV will be 0."
        All_HV = Dict(:Hypervolumes => fill(0.0, num_runs))
    else
        println("  Empirical ideal: $ideal")
        println("  Empirical nadir: $nadir")

        # Save bounds for later use
        bounds_csv = joinpath(results_path, "empirical_bounds_$(main_script_name)_$(Algorithm_structure.Name).csv")
        save_empirical_bounds(bounds_csv, problem_name, 1, ideal, nadir)

        # PASS 2: Compute normalized HV for each run
        All_HV = Dict(:Hypervolumes => Float64[])
        for front in all_run_fronts
            hv = normalized_hypervolume(front, ideal, nadir)
            push!(All_HV[:Hypervolumes], hv)
        end
    end

    cd(results_path)

    hv_mean = mean(All_HV[:Hypervolumes])
    println("Results::$All_HV")
    println("Mean Hypervolume ($n_iterations iterations): $hv_mean")

    get_minimum_runs(All_HV, problem_name, CSV_RUNS_FILE_NAME, 1)

end
