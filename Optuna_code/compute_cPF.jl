
using CSV
using DataFrames
using JSON
using Statistics
using Logging
using Printf

include(joinpath(@__DIR__, "optuna_utils.jl"))
include(joinpath(@__DIR__, "optuna_db_utils.jl"))
include(joinpath(@__DIR__, "pareto_utils.jl"))

const SPACING_EPS = 0.0
const PROGRESS_MIN_POINTS = 100_000
const PROGRESS_INTERVAL_SECONDS = 5.0

# Filename parsing

function parse_algorithm_and_sampler(filename::String)
    stem = replace(basename(filename), r"\.csv$" => "")
    stem = replace(stem, r"^all_fronts_" => "")
    match_result = match(r"^(.+)_([A-Za-z]+Sampler)_Problem_\d+$", stem)
    isnothing(match_result) && return (nothing, nothing)
    return (match_result.captures[1], match_result.captures[2])
end

function stream_fronts_grouped_by_trial_and_run(fronts_file::String,
                                                objective_columns::Vector{Symbol})
    points_per_trial_per_run = Dict{Int, Dict{Int, Vector{Vector{Float64}}}}()
    malformed_row_count = 0

    row_iterator = CSV.Rows(fronts_file;
                             reusebuffer=true,
                             types=Dict(:trial => Int, :run => Int),
                             silencewarnings=true)

    for row in row_iterator
        local trial_number::Int, run_number::Int, point::Vector{Float64}
        try
            trial_number = row.trial
            run_number   = row.run
            point        = Float64[parse(Float64, getproperty(row, c)) for c in objective_columns]
        catch
            malformed_row_count += 1
            continue
        end
        runs_for_trial = get!(points_per_trial_per_run, trial_number, Dict{Int, Vector{Vector{Float64}}}())
        points_for_run = get!(runs_for_trial, run_number, Vector{Float64}[])
        push!(points_for_run, point)
    end

    return points_per_trial_per_run, malformed_row_count
end

# Source loaders

function load_baseline_fronts(baseline_dir::String, baseline_name::String)
    baseline_points_per_problem = Dict{Int, Vector{Tuple{String, Vector{Float64}}}}()

    for algorithm in ALGORITHMS
        fronts_csv = joinpath(baseline_dir, "reference_fronts_$(baseline_name)_$(algorithm).csv")
        if !isfile(fronts_csv)
            println("  skipping $fronts_csv (not found)")
            continue
        end

        fronts_per_problem = load_reference_fronts(fronts_csv)
        source_label = "baseline_$algorithm"

        for (problem_number, front) in fronts_per_problem
            labelled_points = get!(baseline_points_per_problem, problem_number, Tuple{String, Vector{Float64}}[])
            for point in front
                push!(labelled_points, (source_label, point))
            end
        end
        println("  loaded baseline fronts from $fronts_csv")
    end

    return baseline_points_per_problem
end

function discover_hpo_fronts_files(hpo_dir::String)
    hpo_files_per_problem = Dict{Int, Vector{String}}()

    if !isdir(hpo_dir)
        println("  HPO dir not found: $hpo_dir")
        return hpo_files_per_problem
    end

    for (directory, _, files) in walkdir(hpo_dir)
        for filename in files
            startswith(filename, "all_fronts_") && endswith(filename, ".csv") || continue
            problem_match = match(r"Problem_(\d+)\.csv$", filename)
            isnothing(problem_match) && continue
            problem_number = parse(Int, problem_match.captures[1])
            files_for_problem = get!(hpo_files_per_problem, problem_number, String[])
            push!(files_for_problem, joinpath(directory, filename))
        end
    end

    total_file_count = sum(length(v) for v in values(hpo_files_per_problem); init=0)
    println("  discovered $total_file_count HPO files across $(length(hpo_files_per_problem)) problems from $(basename(hpo_dir))")

    return hpo_files_per_problem
end

function load_hpo_all_nondominated_points_for_problem(hpo_files::Vector{String})
    labelled_points = Tuple{String, Vector{Float64}}[]

    for fronts_file in hpo_files
        algorithm, sampler = parse_algorithm_and_sampler(fronts_file)
        if isnothing(algorithm) || isnothing(sampler)
            @warn "Couldn't parse algorithm/sampler from: $fronts_file"
            continue
        end

        objective_columns = read_objective_columns(fronts_file)
        if isnothing(objective_columns)
            @warn "Couldn't read objective columns from: $fronts_file"
            continue
        end

        points_per_trial_per_run, malformed_row_count =
            stream_fronts_grouped_by_trial_and_run(fronts_file, objective_columns)

        if isempty(points_per_trial_per_run)
            @warn "No trial rows in $fronts_file"
            continue
        end

        all_config_points = Vector{Float64}[]
        for (_, runs_for_trial) in points_per_trial_per_run
            for (_, run_points) in runs_for_trial
                append!(all_config_points, run_points)
            end
        end

        raw_count        = length(all_config_points)
        deduped_points   = dedupe_points(all_config_points)
        nondominated_set = compute_nondominated(deduped_points)

        if isempty(nondominated_set)
            @warn "No non-dominated points after dedup in $fronts_file"
            continue
        end

        source_label = "hpo_$(algorithm)_$(sampler)"
        for point in nondominated_set
            push!(labelled_points, (source_label, point))
        end

        skip_note = malformed_row_count > 0 ? " ($malformed_row_count malformed rows skipped)" : ""
        println("    $(basename(fronts_file)): " *
                "$raw_count raw -> $(length(deduped_points)) unique -> $(length(nondominated_set)) ND points$skip_note")

        points_per_trial_per_run = nothing
        all_config_points = nothing
        GC.gc()
    end

    return labelled_points
end

# Combined-front construction

function split_source_label(source_label::String)
    if startswith(source_label, "baseline_")
        return (replace(source_label, "baseline_" => "", count=1), "baseline")
    elseif startswith(source_label, "hpo_")
        rest = replace(source_label, "hpo_" => "", count=1)
        m = match(r"^(.+)_([A-Za-z]+Sampler)$", rest)
        return isnothing(m) ? (rest, "unknown") : (String(m.captures[1]), String(m.captures[2]))
    end
    return ("unknown", "unknown")
end

function build_combined_front_dataframe(all_labelled_points::Vector{Tuple{String, Vector{Float64}}},
                                        nondominated_set)
    isempty(all_labelled_points) && return DataFrame()
    objective_count = length(all_labelled_points[1][2])

    rows = []
    for (source_label, point) in all_labelled_points
        algorithm, sampler = split_source_label(source_label)
        row = Dict{String, Any}(
            "source"    => source_label,
            "algorithm" => algorithm,
            "sampler"   => sampler,
            "is_nd"     => Tuple(point) in nondominated_set,
        )
        for j in 1:objective_count
            row["obj_$j"] = point[j]
        end
        push!(rows, row)
    end

    df = DataFrame(rows)
    column_order = vcat(["source", "algorithm", "sampler", "is_nd"],
                        ["obj_$j" for j in 1:objective_count])
    return df[:, column_order]
end

function filter_points_to_expected_dimension(labelled_points::Vector{Tuple{String, Vector{Float64}}},
                                              expected_dimension::Int)
    kept_points = Tuple{String, Vector{Float64}}[]
    dropped_count_per_source = Dict{String, Int}()

    for (source_label, point) in labelled_points
        if length(point) == expected_dimension
            push!(kept_points, (source_label, point))
        else
            dropped_count_per_source[source_label] =
                get(dropped_count_per_source, source_label, 0) + 1
        end
    end

    return kept_points, dropped_count_per_source
end

function determine_expected_dimension(labelled_points::Vector{Tuple{String, Vector{Float64}}})
    hpo_point_dims = [length(point) for (label, point) in labelled_points if startswith(label, "hpo_")]
    isempty(hpo_point_dims) || return mode_of(hpo_point_dims)

    baseline_point_dims = [length(point) for (_, point) in labelled_points]
    isempty(baseline_point_dims) && return 0
    return mode_of(baseline_point_dims)
end

function mode_of(values::Vector{Int})
    counts = Dict{Int, Int}()
    for v in values
        counts[v] = get(counts, v, 0) + 1
    end
    return argmax(counts)
end

function prefilter_per_source_then_combine(all_labelled_points::Vector{Tuple{String, Vector{Float64}}},
                                           problem_number::Int)
    points_by_source = Dict{String, Vector{Vector{Float64}}}()
    for (source_label, point) in all_labelled_points
        push!(get!(points_by_source, source_label, Vector{Float64}[]), point)
    end

    prefiltered_points = Vector{Float64}[]
    for (source_label, source_points) in points_by_source
        source_unique_points = dedupe_points(source_points)
        source_nd_points = compute_nondominated(
            source_unique_points;
            progress_label = "P$problem_number prefilter $source_label",
        )
        append!(prefiltered_points, source_nd_points)
        if length(source_points) != length(source_nd_points)
            println("    per-source ND prefilter [$source_label]: $(length(source_points)) -> $(length(source_nd_points))")
        end
    end
    prefiltered_points = dedupe_points(prefiltered_points)
    println("    combined ND input after per-source prefilter: $(length(prefiltered_points)) points")

    return compute_nondominated(prefiltered_points; progress_label = "P$problem_number combined")
end

function build_cpf_for_problem(problem_number::Int,
                               baseline_labelled_points::Vector{Tuple{String, Vector{Float64}}},
                               hpo_files::Vector{String},
                               ideal::Vector{Float64},
                               nadir::Vector{Float64},
                               output_dir::String;
                               overwrite::Bool=false)
    output_csv = joinpath(output_dir, "combined_front_Problem_$(problem_number).csv")
    if isfile(output_csv) && !overwrite
        println("  Problem $problem_number: $output_csv already exists, skipping (use --overwrite to redo)")
        return :skipped
    end

    println("  Problem $problem_number...")
    problem_start_time = time()

    hpo_labelled_points = load_hpo_all_nondominated_points_for_problem(hpo_files)
    all_labelled_points = vcat(baseline_labelled_points, hpo_labelled_points)

    if isempty(all_labelled_points)
        println("    no points, skipping")
        return :empty
    end

    expected_dimension = determine_expected_dimension(all_labelled_points)
    if expected_dimension == 0
        println("    could not determine expected dimension, skipping")
        return :empty
    end

    all_labelled_points, dropped_count_per_source =
        filter_points_to_expected_dimension(all_labelled_points, expected_dimension)

    if !isempty(dropped_count_per_source)
        total_dropped = sum(values(dropped_count_per_source))
        println("    WARNING: dropped $total_dropped points with wrong dimension (expected $expected_dimension):")
        for (source_label, count) in sort(collect(dropped_count_per_source))
            println("      $source_label: $count points")
        end
    end

    if isempty(all_labelled_points)
        println("    no points left after dimension filter, skipping")
        return :empty
    end

    pre_dedupe_count    = length(all_labelled_points)
    all_labelled_points = dedupe_labelled_points(all_labelled_points)
    deduped_count       = length(all_labelled_points)
    if pre_dedupe_count > deduped_count
        println("    cross-source dedup: $pre_dedupe_count -> $deduped_count labelled points")
    end

    combined_pareto_front = prefilter_per_source_then_combine(all_labelled_points, problem_number)

    if isempty(combined_pareto_front)
        println("    empty combined Pareto front, skipping")
        return :empty
    end

    if SPACING_EPS > 0.0
        pre_spacing_count     = length(combined_pareto_front)
        combined_pareto_front = spacing_filter(combined_pareto_front, ideal, nadir; eps_distance=SPACING_EPS)
        post_spacing_count    = length(combined_pareto_front)
        if pre_spacing_count != post_spacing_count
            println("    spacing filter (eps=$SPACING_EPS): $pre_spacing_count -> $post_spacing_count cPF points")
        end
    end

    nondominated_set  = Set(Tuple(p) for p in combined_pareto_front)
    combined_front_df = build_combined_front_dataframe(all_labelled_points, nondominated_set)
    CSV.write(output_csv, combined_front_df)

    total_point_count  = nrow(combined_front_df)
    nondominated_count = count(combined_front_df.is_nd)
    baseline_on_front  = count(combined_front_df.is_nd .& startswith.(combined_front_df.source, "baseline"))
    hpo_on_front       = count(combined_front_df.is_nd .& startswith.(combined_front_df.source, "hpo"))
    println("    $total_point_count points ($nondominated_count on cPF: baseline=$baseline_on_front, hpo=$hpo_on_front)")

    elapsed_minutes = (time() - problem_start_time) / 60
    @printf("    Problem %d done in %.2f min, wrote %s\n", problem_number, elapsed_minutes, output_csv)
    flush(stdout)
    return :written
end

# CLI

function parse_problem_selection(arg::AbstractString)
    selected = Set{Int}()
    for token in split(arg, ',')
        token = strip(token)
        isempty(token) && continue
        if occursin('-', token)
            range_parts = split(token, '-')
            length(range_parts) == 2 || error("Bad range: $token")
            lo = parse(Int, strip(range_parts[1]))
            hi = parse(Int, strip(range_parts[2]))
            for i in lo:hi
                push!(selected, i)
            end
        else
            push!(selected, parse(Int, token))
        end
    end
    return selected
end

function parse_command_line_args(args::Vector{String})
    selection = nothing
    overwrite = false
    experiments = String[]

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--problems"
            i + 1 <= length(args) || error("--problems needs an argument like 1-50 or 11,12,15")
            selection = parse_problem_selection(args[i+1])
            i += 2
        elseif a == "--overwrite"
            overwrite = true
            i += 1
        elseif a == "--experiment"
            i + 1 <= length(args) || error("--experiment needs benchmark or truss")
            push!(experiments, args[i+1])
            i += 2
        elseif a in ("-h", "--help")
            println("""
Usage: julia compute_cPF.jl [options]

Options:
  --problems LIST     Comma/range list of problem numbers, e.g. 1-10 or 11,15-20.
                      Default: all discovered problems.
  --overwrite         Recompute even if combined_front_Problem_N.csv already exists.
  --experiment NAME   Restrict to 'benchmark' or 'truss'. May be repeated.
                      Default: both, if data is present.
  -h, --help          Show this help and exit.
""")
            exit(0)
        else
            error("Unknown argument: $a (use --help)")
        end
    end

    isempty(experiments) && (experiments = ["benchmark", "truss"])
    return (selection=selection, overwrite=overwrite, experiments=experiments)
end

# Driver

function build_for_experiment(baseline_dir::String,
                              hpo_dir::String,
                              baseline_name::String,
                              output_dir::String,
                              selection::Union{Nothing, Set{Int}},
                              overwrite::Bool)
    mkpath(output_dir)

    println("\n  Loading baseline fronts ($baseline_name) from $baseline_dir...")
    baseline_points_per_problem = load_baseline_fronts(baseline_dir, baseline_name)

    println("  Discovering HPO files in $hpo_dir...")
    hpo_files_per_problem = discover_hpo_fronts_files(hpo_dir)

    all_problem_numbers = union(keys(baseline_points_per_problem), keys(hpo_files_per_problem))
    if isempty(all_problem_numbers)
        println("  No fronts found for this experiment.")
        return
    end

    println("  Loading empirical bounds...")
    bounds_per_problem = load_merged_empirical_bounds(baseline_dir, baseline_name)

    target_problem_numbers = sort(collect(all_problem_numbers))
    if !isnothing(selection)
        target_problem_numbers = filter(p -> p in selection, target_problem_numbers)
        if isempty(target_problem_numbers)
            println("  --problems filter excluded everything for this experiment.")
            return
        end
        println("  --problems filter -> $(length(target_problem_numbers)) problem(s) to process: $target_problem_numbers")
    end

    experiment_start_time = time()
    written_count = 0
    skipped_count = 0
    empty_count   = 0

    for (problem_index, problem_number) in enumerate(target_problem_numbers)
        elapsed_minutes = (time() - experiment_start_time) / 60
        @printf("\n[Build progress] %d/%d problems handled | elapsed: %.1f min\n",
                problem_index - 1, length(target_problem_numbers), elapsed_minutes)
        flush(stdout)

        if !haskey(bounds_per_problem, problem_number)
            @warn "Problem $problem_number: no empirical bounds, skipping"
            continue
        end
        ideal, nadir = bounds_per_problem[problem_number]

        baseline_labelled_points = get(baseline_points_per_problem, problem_number, Tuple{String, Vector{Float64}}[])
        hpo_files = get(hpo_files_per_problem, problem_number, String[])

        result = build_cpf_for_problem(
            problem_number, baseline_labelled_points, hpo_files,
            ideal, nadir, output_dir;
            overwrite=overwrite,
        )

        if result == :written
            written_count += 1
        elseif result == :skipped
            skipped_count += 1
        else
            empty_count += 1
        end

        GC.gc()
    end

    total_elapsed_minutes = (time() - experiment_start_time) / 60
    @printf("\n[Build done] written: %d | skipped (already exists): %d | empty: %d | total: %.1f min\n",
            written_count, skipped_count, empty_count, total_elapsed_minutes)
end

function main()
    parsed = parse_command_line_args(copy(ARGS))

    script_dir = @__DIR__
    results_dir = normpath(dirname(script_dir), "Results")
    isdir(results_dir) || error("Results directory not found: $results_dir")
    println("Results directory: $results_dir")

    if "benchmark" in parsed.experiments
        benchmark_baseline_name = find_benchmark_baseline_name(results_dir)
        if !isnothing(benchmark_baseline_name)
            println("\n" * "="^60)
            println("Building cPF for benchmarks (baseline: $benchmark_baseline_name)")
            println("="^60)
            exp = experiment_paths("benchmark", results_dir)
            build_for_experiment(
                exp.baseline_dir,
                exp.hpo_dir,
                benchmark_baseline_name,
                exp.cpf_dir,
                parsed.selection,
                parsed.overwrite,
            )
        else
            println("No benchmark baseline found, skipping.")
        end
    end

    if "truss" in parsed.experiments
        truss_baseline_name = find_truss_baseline_name(results_dir)
        if !isnothing(truss_baseline_name)
            println("\n" * "="^60)
            println("Building cPF for truss (baseline: $truss_baseline_name)")
            println("="^60)
            exp = experiment_paths("truss", results_dir)
            build_for_experiment(
                exp.baseline_dir,
                exp.hpo_dir,
                truss_baseline_name,
                exp.cpf_dir,
                parsed.selection,
                parsed.overwrite,
            )
        else
            println("No truss baseline found, skipping.")
        end
    end

    println("\n" * "="^60)
    println("cPF build done.")
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
