using CSV
using DataFrames
using JSON
using Metaheuristics
using Statistics
using Printf

include(joinpath(@__DIR__, "optuna_utils.jl"))
include(joinpath(@__DIR__, "pareto_utils.jl"))

const HV_MATCH_TOLERANCE = 1e-6

# Per-task summary CSV loading

function load_summary_hv_per_task(hpo_root::AbstractString, iter_string::AbstractString)
    summary_hv = Dict{Tuple{String, String, Int}, Float64}()
    isdir(hpo_root) || return summary_hv

    for algorithm in ["NSGA2", "SPEA2", "SMS_EMOA", "MOEAD_DE"]
        algorithm_dir = joinpath(hpo_root, algorithm, iter_string)
        isdir(algorithm_dir) || continue
        for filename in readdir(algorithm_dir)
            m = match(Regex("^$(algorithm)_([A-Za-z]+Sampler)\\.csv\$"), filename)
            isnothing(m) && continue
            sampler = String(m.captures[1])
            csv_path = joinpath(algorithm_dir, filename)
            try
                df = CSV.read(csv_path, DataFrame; pool=false, silencewarnings=true)
                for row in eachrow(df)
                    ismissing(row.problem_instance) && continue
                    ismissing(row.hv_value) && continue
                    problem_number = Int(row.problem_instance)
                    summary_hv[(algorithm, sampler, problem_number)] = Float64(row.hv_value)
                end
            catch err
                @warn "Could not parse $csv_path: $err"
            end
        end
    end
    return summary_hv
end

# Best-trial loading

function load_best_trial_fronts(best_trials_csv::AbstractString)
    df = CSV.read(best_trials_csv, DataFrame; pool=false, silencewarnings=true)
    objective_columns = sort([string(c) for c in names(df) if startswith(string(c), "obj_")];
                             by = c -> parse(Int, replace(c, "obj_" => "")))

    grouped = Dict{Tuple{String, String, Int}, NamedTuple}()
    for row in eachrow(df)
        key = (String(row.algorithm), String(row.sampler), Int(row.problem))

        existing = get(grouped, key, nothing)
        if isnothing(existing)
            existing = (
                status               = String(row.status),
                best_trial_db_number = ismissing(row.best_trial_db_number) ? nothing : Int(row.best_trial_db_number),
                best_hv_value        = ismissing(row.best_hv_value)        ? nothing : Float64(row.best_hv_value),
                fallback_used        = ismissing(row.fallback_used)        ? false   : Bool(row.fallback_used),
                top1_trial_db_number = ismissing(row.top1_trial_db_number) ? nothing : Int(row.top1_trial_db_number),
                top1_hv_value        = ismissing(row.top1_hv_value)        ? nothing : Float64(row.top1_hv_value),
                points               = Vector{Vector{Float64}}(),
            )
            grouped[key] = existing
        end

        ismissing(row[Symbol(objective_columns[1])]) && continue

        point = Float64[]
        for c in objective_columns
            v = row[Symbol(c)]
            ismissing(v) && break
            push!(point, Float64(v))
        end
        isempty(point) || push!(existing.points, point)
    end
    return grouped
end

# HV cross-check

function classify_hv_match(best_hv_db, best_hv_summary)
    isnothing(best_hv_summary) && return "summary_missing"
    isnothing(best_hv_db)      && return "db_missing"
    abs(best_hv_db - best_hv_summary) <= HV_MATCH_TOLERANCE && return "true"
    return "false"
end

# CLI

function parse_command_line_args(args::Vector{String})
    best_trials_csv = nothing
    output_dir      = nothing
    output_name     = "igd_results.csv"
    results_dir     = nothing
    experiment      = "benchmark"

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--best-trials-csv"
            i + 1 <= length(args) || error("--best-trials-csv needs a path")
            best_trials_csv = args[i+1]; i += 2
        elseif a == "--output-dir"
            i + 1 <= length(args) || error("--output-dir needs a path")
            output_dir = args[i+1]; i += 2
        elseif a == "--output-name"
            i + 1 <= length(args) || error("--output-name needs a filename")
            output_name = args[i+1]; i += 2
        elseif a == "--results-dir"
            i + 1 <= length(args) || error("--results-dir needs a path")
            results_dir = args[i+1]; i += 2
        elseif a == "--experiment"
            i + 1 <= length(args) || error("--experiment needs benchmark or truss")
            experiment = args[i+1]; i += 2
        elseif a in ("-h", "--help")
            println("""
Usage: julia compute_igd.jl --best-trials-csv FILE --output-dir DIR [--output-name NAME] [--results-dir DIR] [--experiment benchmark|truss]

Reads best_trial_fronts.csv (output of extract_best_trial_fronts.jl) and the
combined_front_Problem_*.csv files (output of compute_cPF.jl) and computes IGD+
of each task's best-trial front against the cPF for that problem.

Required: --best-trials-csv FILE, --output-dir DIR (must be OUTSIDE Results/)
""")
            exit(0)
        else
            error("Unknown argument: $a (use --help)")
        end
    end

    isnothing(best_trials_csv) && error("--best-trials-csv is required")
    isnothing(output_dir)      && error("--output-dir is required")
    return (best_trials_csv=best_trials_csv, output_dir=output_dir,
            output_name=output_name, results_dir=results_dir, experiment=experiment)
end

# Driver

function main()
    parsed = parse_command_line_args(copy(ARGS))

    results_dir = isnothing(parsed.results_dir) ?
        normpath(dirname(@__DIR__), "Results") : parsed.results_dir
    isdir(results_dir) || error("Results dir not found: $results_dir")

    exp = experiment_paths(parsed.experiment, results_dir)

    isfile(parsed.best_trials_csv) || error("best_trials_csv not found: $(parsed.best_trials_csv)")

    abs_output, output_csv = refuse_destructive_output(parsed.output_dir, results_dir;
                                                        output_name=parsed.output_name)
    mkpath(abs_output)

    combined_fronts_dir = exp.cpf_dir
    isdir(combined_fronts_dir) || error("$(basename(combined_fronts_dir)) not found: $combined_fronts_dir")

    if exp.name == "benchmark"
        baseline_name = find_benchmark_baseline_name(exp.baseline_dir)
        isnothing(baseline_name) && error("Could not find benchmark baseline marker file under $(exp.baseline_dir)")
    else
        baseline_name = exp.baseline_name
    end

    println("Experiment:         $(exp.name)")
    println("Results dir:        $results_dir")
    println("best_trials_csv:    $(parsed.best_trials_csv)")
    println("Combined fronts:    $combined_fronts_dir")
    println("Baseline:           $baseline_name")
    println("Baseline dir:       $(exp.baseline_dir)")
    println("HPO dir:            $(exp.hpo_dir)")
    println("Output CSV:         $output_csv")
    println("HV match tolerance: $HV_MATCH_TOLERANCE")
    println("="^70)

    println("\nLoading best-trial fronts ...")
    tasks = load_best_trial_fronts(parsed.best_trials_csv)
    println("  $(length(tasks)) (algorithm, sampler, problem) entries loaded")

    println("\nLoading summary HV for cross-check ...")
    summary_hv = load_summary_hv_per_task(exp.hpo_dir, exp.iter_string)
    println("  $(length(summary_hv)) summary entries loaded")

    println("\nLoading merged empirical bounds ...")
    bounds_per_problem = load_merged_empirical_bounds(exp.baseline_dir, baseline_name)
    println("  bounds for $(length(bounds_per_problem)) problems")

    cpf_cache = Dict{Int, Tuple{Any, Int, Int}}()

    output_rows = NamedTuple[]
    sorted_keys = sort(collect(keys(tasks)))
    total = length(sorted_keys)
    started = time()

    for (idx, key) in enumerate(sorted_keys)
        algorithm, sampler, problem = key
        info = tasks[key]

        if idx == 1 || idx % 100 == 0 || idx == total
            elapsed = time() - started
            @printf("  [%4d/%d] %s_%s_Problem_%d  (elapsed %.1fs)\n",
                    idx, total, algorithm, sampler, problem, elapsed)
            flush(stdout)
        end

        best_hv_summary = get(summary_hv, key, nothing)
        hv_match = classify_hv_match(info.best_hv_value, best_hv_summary)

        n_obtained_nd = 0
        igd_value     = NaN
        cpf_size      = 0
        ref_used_size = 0
        notes_parts   = String[]

        if !haskey(bounds_per_problem, problem)
            push!(notes_parts, "no empirical bounds for problem $problem")
        elseif info.status == "ok" || info.status == "ok_fallback"
            ideal, nadir = bounds_per_problem[problem]

            ref_data = get(cpf_cache, problem, nothing)
            if isnothing(ref_data)
                cpf_points, n_obj = load_combined_front_cpf(combined_fronts_dir, problem)
                if isnothing(cpf_points)
                    push!(notes_parts, "combined_front_Problem_$(problem).csv missing or empty")
                    cpf_cache[problem] = (zeros(0, 0), 0, 0)
                elseif length(ideal) != n_obj
                    push!(notes_parts, "bounds dim ($(length(ideal))) != cPF dim ($n_obj)")
                    cpf_cache[problem] = (zeros(0, 0), 0, 0)
                else
                    norm_ref_matrix = build_normalized_reference(cpf_points, ideal, nadir;
                                                                  max_size=MAX_REFERENCE_FRONT_SIZE)
                    ref_size = size(norm_ref_matrix, 1)
                    cpf_cache[problem] = (norm_ref_matrix, length(cpf_points), ref_size)
                end
                ref_data = cpf_cache[problem]
            end

            norm_ref_matrix, cpf_size, ref_used_size = ref_data

            if !isempty(notes_parts) || ref_used_size == 0
            elseif length(info.points) == 0
                push!(notes_parts, "no points in best-trial front")
            else
                obj_dim_in_points = length(info.points[1])
                if obj_dim_in_points != length(ideal)
                    push!(notes_parts, "best-trial point dim ($obj_dim_in_points) != bounds dim ($(length(ideal)))")
                else
                    n_obtained_nd, igd_value = compute_igd_plus(
                        info.points, norm_ref_matrix, ideal, nadir,
                    )
                end
            end
        end

        push!(output_rows, (
            algorithm            = algorithm,
            sampler              = sampler,
            problem              = problem,
            status               = info.status,
            best_trial_db_number = isnothing(info.best_trial_db_number) ? missing : info.best_trial_db_number,
            best_hv_db           = isnothing(info.best_hv_value)        ? missing : info.best_hv_value,
            best_hv_summary      = isnothing(best_hv_summary)           ? missing : best_hv_summary,
            hv_match             = hv_match,
            fallback_used        = info.fallback_used,
            top1_trial_db_number = isnothing(info.top1_trial_db_number) ? missing : info.top1_trial_db_number,
            top1_hv_value        = isnothing(info.top1_hv_value)        ? missing : info.top1_hv_value,
            n_obtained_nd        = n_obtained_nd,
            igd_plus             = igd_value,
            cpf_size             = cpf_size,
            ref_used_size        = ref_used_size,
            notes                = join(notes_parts, "; "),
        ))
    end

    df = DataFrame(output_rows)
    CSV.write(output_csv, df)

    println("\n" * "="^70)
    println("Wrote $output_csv ($(nrow(df)) rows)")
    println("="^70)

    println("\nStatus breakdown:")
    for status in sort(unique(df.status))
        n = count(==(status), df.status)
        @printf("  %-30s %d\n", status, n)
    end

    println("\nHV cross-check (best_hv_db vs best_hv_summary, tolerance=$HV_MATCH_TOLERANCE):")
    for verdict in sort(unique(df.hv_match))
        n = count(==(verdict), df.hv_match)
        @printf("  %-30s %d\n", verdict, n)
    end

    valid_igd = filter(row -> !isnan(row.igd_plus), eachrow(df))
    if !isempty(valid_igd)
        println("\nIGD+ summary (lower is better) over $(length(valid_igd)) tasks with finite IGD+:")
        algo_groups = Dict{String, Vector{Float64}}()
        for row in valid_igd
            push!(get!(algo_groups, "$(row.algorithm)__$(row.sampler)", Float64[]), row.igd_plus)
        end
        for key in sort(collect(keys(algo_groups)))
            vals = algo_groups[key]
            @printf("  %-40s n=%3d  mean=%.4f  median=%.4f\n",
                    key, length(vals), mean(vals), median(vals))
        end
    end

    mismatches = filter(row -> row.hv_match == "false", eachrow(df))
    if !isempty(mismatches)
        @printf("\n%d task(s) where best_hv_db disagrees with best_hv_summary:\n", length(mismatches))
        n_show = min(10, length(mismatches))
        for row in mismatches[1:n_show]
            @printf("  %s | %s | Problem %d : db=%.6f  summary=%.6f  Δ=%.6f\n",
                    row.algorithm, row.sampler, row.problem,
                    row.best_hv_db, row.best_hv_summary,
                    row.best_hv_summary - row.best_hv_db)
        end
        length(mismatches) > n_show && @printf("  ... %d more in the CSV\n", length(mismatches) - n_show)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
