using CSV
using DataFrames
using JSON
using Statistics
using Printf

include(joinpath(@__DIR__, "optuna_utils.jl"))
include(joinpath(@__DIR__, "pareto_utils.jl"))

# Baseline run loading

function load_baseline_runs_per_problem(reference_fronts_csv::AbstractString)
    df = CSV.read(reference_fronts_csv, DataFrame; pool=false, silencewarnings=true)
    obj_col_pattern = r"^(obj_\d+|Column\d+)$"
    obj_cols = [c for c in names(df) if occursin(obj_col_pattern, String(c))]

    points_per_problem = Dict{Int, Vector{Vector{Float64}}}()
    for row in eachrow(df)
        ismissing(row.current_instance) && continue
        instance = Int(row.current_instance)
        sol = Float64[row[c] for c in obj_cols if !ismissing(row[c])]
        isempty(sol) && continue
        push!(get!(points_per_problem, instance, Vector{Float64}[]), sol)
    end
    return points_per_problem
end

function load_minimum_runs_all_hv(minimum_runs_csv::AbstractString)
    isfile(minimum_runs_csv) || return Dict{Int, Vector{Float64}}()

    all_hv_per_problem = Dict{Int, Vector{Float64}}()
    open(minimum_runs_csv, "r") do io
        readline(io)
        instance = 0
        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue

            if startswith(stripped, "[") || startswith(stripped, "\"[")
                jsstr = startswith(stripped, "\"") ? strip(stripped, '"') : stripped
                jsstr = replace(jsstr, "\"\"" => "\"")
                try
                    arr = JSON.parse(jsstr)
                    all_hv_per_problem[instance] = collect(Float64, arr)
                catch
                end
            else
                fields = split(stripped, ',')
                if length(fields) >= 4
                    try
                        instance = parse(Int, strip(fields[4]))
                    catch
                    end
                end
            end
        end
    end
    return all_hv_per_problem
end

# Driver

function process_baselines(baseline_dir::AbstractString,
                           baseline_name::AbstractString,
                           combined_fronts_dir::AbstractString,
                           bounds_per_problem)
    output_rows = NamedTuple[]
    cpf_cache = Dict{Int, Tuple{Any, Int, Int}}()

    for algorithm in ["NSGA2", "SPEA2", "SMS_EMOA", "MOEAD_DE"]
        ref_csv      = joinpath(baseline_dir, "reference_fronts_$(baseline_name)_$(algorithm).csv")
        min_runs_csv = joinpath(baseline_dir, "minimum_runs_$(baseline_name)_$(algorithm).csv")

        if !isfile(ref_csv)
            @warn "$ref_csv not found, skipping $algorithm"
            continue
        end

        per_problem_points = load_baseline_runs_per_problem(ref_csv)
        all_hv_per_problem = load_minimum_runs_all_hv(min_runs_csv)

        println("\n[$algorithm] $(length(per_problem_points)) problems, $(length(all_hv_per_problem)) HV records")

        for problem_number in sort(collect(keys(per_problem_points)))
            all_hv_vec = get(all_hv_per_problem, problem_number, Float64[])
            ref_data = get(cpf_cache, problem_number, nothing)
            ideal_nadir = get(bounds_per_problem, problem_number, nothing)

            best_run_idx          = isempty(all_hv_vec) ? -1 : argmax(all_hv_vec)
            best_run_hv           = isempty(all_hv_vec) ? NaN : Float64(all_hv_vec[best_run_idx])
            mean_hv               = isempty(all_hv_vec) ? NaN : mean(all_hv_vec)
            n_runs                = length(all_hv_vec)
            cpf_size_for_problem  = 0
            ref_used_size_for_problem = 0
            n_obtained_nd         = 0
            igd_value             = NaN
            note                  = ""

            if isnothing(ideal_nadir)
                note = "no empirical bounds"
            else
                ideal, nadir = ideal_nadir

                if isnothing(ref_data)
                    cpf_points, n_obj = load_combined_front_cpf(combined_fronts_dir, problem_number)
                    if isnothing(cpf_points)
                        note = "no cPF for problem"
                        cpf_cache[problem_number] = (zeros(0, 0), 0, 0)
                    elseif length(ideal) != n_obj
                        note = "bounds dim ($(length(ideal))) != cPF dim ($n_obj)"
                        cpf_cache[problem_number] = (zeros(0, 0), 0, 0)
                    else
                        norm_ref_matrix = build_normalized_reference(cpf_points, ideal, nadir;
                                                                      max_size=MAX_REFERENCE_FRONT_SIZE)
                        ref_size = size(norm_ref_matrix, 1)
                        cpf_cache[problem_number] = (norm_ref_matrix, length(cpf_points), ref_size)
                    end
                    ref_data = cpf_cache[problem_number]
                end
                norm_ref_matrix, cpf_size_for_problem, ref_used_size_for_problem = ref_data

                if ref_used_size_for_problem > 0
                    union_front_points = get(per_problem_points, problem_number, Vector{Float64}[])
                    if !isempty(union_front_points) && length(first(union_front_points)) == length(ideal)
                        n_obtained_nd, igd_value = compute_igd_plus(
                            union_front_points, norm_ref_matrix, ideal, nadir,
                        )
                    else
                        note = "union front empty or wrong dim"
                    end
                end
            end

            push!(output_rows, (
                algorithm     = algorithm,
                problem       = problem_number,
                n_runs        = n_runs,
                best_run_idx  = best_run_idx,
                best_run_hv   = best_run_hv,
                mean_hv       = mean_hv,
                std_hv        = isempty(all_hv_vec) ? NaN : std(all_hv_vec),
                all_hv_json   = isempty(all_hv_vec) ? "" : JSON.json(all_hv_vec),
                n_obtained_nd = n_obtained_nd,
                igd_plus      = igd_value,
                cpf_size      = cpf_size_for_problem,
                ref_used_size = ref_used_size_for_problem,
                notes         = note,
            ))
        end
    end

    return output_rows
end

# CLI

function parse_command_line_args(args::Vector{String})
    output_dir  = nothing
    results_dir = nothing
    experiment  = "benchmark"
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--output-dir"
            i + 1 <= length(args) || error("--output-dir needs a path")
            output_dir = args[i+1]; i += 2
        elseif a == "--results-dir"
            i + 1 <= length(args) || error("--results-dir needs a path")
            results_dir = args[i+1]; i += 2
        elseif a == "--experiment"
            i + 1 <= length(args) || error("--experiment needs benchmark or truss")
            experiment = args[i+1]; i += 2
        elseif a in ("-h", "--help")
            println("""
Usage: julia compute_baseline_igd.jl --output-dir DIR [--results-dir DIR] [--experiment benchmark|truss]

For each (algorithm, problem) baseline run, computes IGD+ of the union of all
baseline-run fronts against the cPF. Resolves paths from --experiment.

Writes: baseline_igd_results.csv
""")
            exit(0)
        else
            error("Unknown arg: $a")
        end
    end
    isnothing(output_dir) && error("--output-dir is required")
    return (output_dir=output_dir, results_dir=results_dir, experiment=experiment)
end

function main()
    parsed = parse_command_line_args(copy(ARGS))

    results_dir = isnothing(parsed.results_dir) ?
        normpath(dirname(@__DIR__), "Results") : parsed.results_dir
    isdir(results_dir) || error("Results dir not found: $results_dir")

    exp = experiment_paths(parsed.experiment, results_dir)

    abs_output, output_csv = refuse_destructive_output(parsed.output_dir, results_dir;
                                                        output_name="baseline_igd_results.csv")
    mkpath(abs_output)

    combined_fronts_dir = exp.cpf_dir
    isdir(combined_fronts_dir) || error("$(basename(combined_fronts_dir)) not found: $combined_fronts_dir")

    println("Experiment:         $(exp.name)")
    println("Results dir:        $results_dir")
    println("Baseline dir:       $(exp.baseline_dir)")
    println("Combined fronts:    $combined_fronts_dir")
    println("Output CSV:         $output_csv")
    println("="^70)

    println("\nLoading merged empirical bounds...")
    bounds_per_problem = load_merged_empirical_bounds(exp.baseline_dir, exp.baseline_name)
    println("  bounds for $(length(bounds_per_problem)) problems")

    rows = process_baselines(exp.baseline_dir, exp.baseline_name, combined_fronts_dir, bounds_per_problem)
    df = DataFrame(rows)
    CSV.write(output_csv, df)

    println("\nWrote $output_csv ($(nrow(df)) rows)")

    println("\nIGD+ summary by algorithm (lower is better):")
    valid = filter(row -> !isnan(row.igd_plus), eachrow(df))
    if !isempty(valid)
        for alg in ["NSGA2", "SPEA2", "SMS_EMOA", "MOEAD_DE"]
            subset = filter(row -> row.algorithm == alg, valid)
            isempty(subset) && continue
            vals = [row.igd_plus for row in subset]
            @printf("  %-10s n=%2d  mean=%.4f  median=%.4f\n",
                    alg, length(vals), mean(vals), median(vals))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
