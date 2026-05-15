using DataStructures
using Metaheuristics
using HardTestProblems
using DataFrames
using CSV
using Statistics
using JSON
using Logging

const CONFIDENCE_LEVELS = [(90, 1.645), (95, 1.96), (98, 2.33), (99, 2.575)]
const MIN_STD_FALLBACK = 0.001

# CSV helpers

function check_CSV(searchspace::String, name_of_script::String, results_path::String)
    println("Currently using $searchspace")
    alg_name = split(searchspace, "_searchspace")[1]
    script_name = endswith(name_of_script, ".jl") ? split(name_of_script, ".jl")[1] : name_of_script
    CSV_RUNS_FILE_NAME = joinpath(results_path, "minimum_runs_$(script_name)_$(alg_name).csv")
    remove_existing_csv(CSV_RUNS_FILE_NAME)
    return CSV_RUNS_FILE_NAME
end

# Minimum runs

minimum_runs(z, stdev, ϵ) = ceil((z * stdev / ϵ)^2)

function get_minimum_runs(results::Dict, problem_name::String, CSV_RUNS_FILE_NAME::String, current_instance::Int)
    typed_results = results[:Hypervolumes]
    println(typed_results)
    if isempty(typed_results)
        @warn "No HV results for problem $problem_name — skipping."
        return
    end

    mean_hv = mean(typed_results)
    all_std = std(typed_results)

    if all_std < MIN_STD_FALLBACK
        @warn "HV values are nearly identical for $problem_name (std=$all_std < $MIN_STD_FALLBACK), using fallback std=$MIN_STD_FALLBACK"
        all_std = MIN_STD_FALLBACK
    end

    runs_dict = OrderedDict{Int, Vector{Union{Int, String}}}(
        level => [] for (level, _) in CONFIDENCE_LEVELS
    )

    last_error = 0.0
    for (level, z) in CONFIDENCE_LEVELS
        for ϵ in 0.01:0.01:0.1
            margin = ϵ * mean_hv
            runs = minimum_runs(z, all_std, margin)
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
    for (level, _) in CONFIDENCE_LEVELS
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
        confidence_interval_99 = [JSON.json(runs_dict[99])],
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
        confidence_interval_99 = [""],
    )
    CSV.write(CSV_RUNS_FILE_NAME, separator_df; append=true, writeheader=false)
end
