
using DataStructures
using Metaheuristics
using Metaheuristics: pareto_front
using HardTestProblems
using DataFrames

function check_CSV(searchspace, name_of_script ; test)
    
    println("Currently using $searchspace")
    split_str = split(searchspace, "_searchspace")

    name_of_script = split(name_of_script, ".jl")

    println(name_of_script)

    if test == true
        CSV_RUNS_FILE_NAME = "$(name_of_script[1])_$(split_str[1])_test.csv"
    else
        CSV_RUNS_FILE_NAME = "$(name_of_script[1])_$(split_str[1]).csv"
    end

    CSV_LENGTH_RESULTS_NAME = "length_results_$CSV_RUNS_FILE_NAME.csv"
    remove_existing_csv(base_dir, CSV_RUNS_FILE_NAME, CSV_LENGTH_RESULTS_NAME, split_str )
    return CSV_RUNS_FILE_NAME, CSV_LENGTH_RESULTS_NAME

end

function remove_existing_csv(base_dir, CSV_RUNS_FILE_NAME, CSV_LENGTH_RESULTS_NAME, split_str)
    if ispath(joinpath(base_dir,"Optuna/Results/$CSV_RUNS_FILE_NAME"))
        println("Eliminating previous CSV for $(split_str[1])")
        rm(joinpath(base_dir,"Optuna/Results/$CSV_RUNS_FILE_NAME"))
    end

    if ispath(joinpath(base_dir,"Optuna/Results/$CSV_LENGTH_RESULTS_NAME"))
        println("Eliminating previous CSV for $(split_str[1])")
        rm(joinpath(base_dir,"Optuna/Results/$CSV_LENGTH_RESULTS_NAME"))
    end

end


minimum_runs(z, stdev, ϵ) = 
    ceil((z * stdev / ϵ)^2)


function get_minimum_runs(results, problem_name, current_instance, CSV_RUNS_FILE_NAME)

    type_of_result = first(keys(results))
    typed_results = results[type_of_result]

    mean_hv = mean(typed_results)
    all_std = std(typed_results)

    if all_std < 0.001
         @warn "All HV values are nearly identical (std < 0.001), using fallback"
         all_std = 0.001
    end

    CONFIDENCE_LEVELS = [
        (90, 1.645),
        (95, 1.96),
        (98, 2.33),
        (99, 2.575)
    ]

    runs_dict = Dict{Int, Vector{Union{Int, String}}}()
    error = 0.0

    for (level, z) in CONFIDENCE_LEVELS
        temp = []
        for ϵ in 0.01:0.01:0.1
            error = ϵ * mean_hv
            runs = minimum_runs(z, all_std, error)
            if isfinite(runs)
                push!(temp,Int(floor(runs)))
            else
                push!(temp,"Inf")
                @warn "Runs too large or infinite for $level% confidence level: $runs"
            end
        end
        runs_dict[level] = temp
    end

    println("Desired margin of error (ϵ): $error")
    for (level, _) in CONFIDENCE_LEVELS
        println("- $(level)% confidence interval: $(runs_dict[level])")
    end

    df = DataFrame(
        problem_name = problem_name,
        best_HV = maximum(typed_results),
        error = error / mean_hv,
        current_instance = current_instance,
        confidence_interval_90 = [JSON.json(runs_dict[90])],
        confidence_interval_95 = [JSON.json(runs_dict[95])],
        confidence_interval_98 = [JSON.json(runs_dict[98])],
        confidence_interval_99 = [JSON.json(runs_dict[99])]
    )

    write_header = !isfile(CSV_RUNS_FILE_NAME)
    CSV.write(CSV_RUNS_FILE_NAME, df; append=true, writeheader=write_header)

    println("Result type key: $type_of_result")
    println("Raw results: $typed_results")

    All_HV_df = DataFrame(All_HV = [JSON.json(typed_results)])
    CSV.write(CSV_RUNS_FILE_NAME, All_HV_df; append=true)

    # Create a visually separating empty row with same columns
    empty_row = DataFrame(
        problem_name = [""],
        best_HV = [""],
        error = [""],
        current_instance = [""],
        confidence_interval_90 = [""],
        confidence_interval_95 = [""],
        confidence_interval_98 = [""],
        confidence_interval_99 = [""]
    )
    CSV.write(CSV_RUNS_FILE_NAME, empty_row; append=true)

    println("min: $(minimum(typed_results))")
    println("max: $(maximum(typed_results))")
    println("std: $(all_std)")
    println("mean: $(mean_hv)")
end
