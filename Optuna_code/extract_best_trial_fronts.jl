using DataFrames
using CSV
using Printf

include(joinpath(@__DIR__, "optuna_db_utils.jl"))
include(joinpath(@__DIR__, "pareto_utils.jl"))

const STATUS_OK                          = "ok"
const STATUS_OK_FALLBACK                 = "ok_fallback"
const STATUS_NO_FRONT_EXTRACTABLE        = "no_front_extractable"
const STATUS_NO_CSV                      = "no_csv"
const STATUS_NO_COMPLETE_TRIALS_IN_WIN   = "no_complete_trials_in_window"
const STATUS_ALL_HV_INVALID              = "all_hv_invalid"
const STATUS_DB_UNREADABLE               = "db_unreadable"

# CSV scanning

function scan_csv_for_trials_and_points(fronts_file::AbstractString,
                                        objective_columns::Vector{Symbol})
    seen_trial_set = Set{Int}()
    points_per_trial = Dict{Int, Vector{Tuple{Int, Vector{Float64}}}}()

    rows = CSV.Rows(fronts_file;
                    types=Dict(:trial => Int, :run => Int),
                    reusebuffer=true,
                    silencewarnings=true)

    for row in rows
        local trial_number::Int, run_number::Int, point::Vector{Float64}
        try
            trial_number = row.trial
            run_number   = row.run
            point        = Float64[parse(Float64, getproperty(row, c)) for c in objective_columns]
        catch
            continue
        end
        push!(seen_trial_set, trial_number)
        bucket = get!(points_per_trial, trial_number, Tuple{Int, Vector{Float64}}[])
        push!(bucket, (run_number, point))
    end

    return seen_trial_set, points_per_trial
end

# Best-trial selection

function pick_best_in_window(ranked_trials::DataFrame, csv_trial_set::Set{Int})
    isempty(ranked_trials) && return (
        chosen_db_number = nothing, chosen_hv = nothing, chosen_rank = nothing,
        top1_db_number = nothing, top1_hv = nothing,
    )

    valid_mask = .!ismissing.(ranked_trials.value) .&
                 .!isnan.(coalesce.(ranked_trials.value, NaN)) .&
                 isfinite.(coalesce.(ranked_trials.value, -Inf))
    valid_rows = ranked_trials[valid_mask, :]

    isempty(valid_rows) && return (
        chosen_db_number = nothing, chosen_hv = nothing, chosen_rank = nothing,
        top1_db_number = nothing, top1_hv = nothing,
    )

    top1_db_number = Int(valid_rows[1, :number])
    top1_hv        = Float64(valid_rows[1, :value])

    if isempty(csv_trial_set)
        return (
            chosen_db_number = nothing, chosen_hv = nothing, chosen_rank = nothing,
            top1_db_number = top1_db_number, top1_hv = top1_hv,
        )
    end

    for (rank_index, row) in enumerate(eachrow(valid_rows))
        db_number  = Int(row.number)
        csv_number = db_number + TRIAL_INDEXING_OFFSET
        if csv_number in csv_trial_set
            return (
                chosen_db_number = db_number,
                chosen_hv        = Float64(row.value),
                chosen_rank      = rank_index - 1,
                top1_db_number   = top1_db_number,
                top1_hv          = top1_hv,
            )
        end
    end

    return (
        chosen_db_number = nothing, chosen_hv = nothing, chosen_rank = nothing,
        top1_db_number = top1_db_number, top1_hv = top1_hv,
    )
end

# Per-task pipeline

function build_task_rows(task, results_dir::AbstractString, hpo_dir_name::AbstractString)
    placeholder_metadata = (
        best_trial_db_number = missing, best_hv_value = missing, fallback_used = false,
        top1_trial_db_number = missing, top1_hv_value = missing,
    )

    ranked_trials = try
        read_ranked_complete_trials_in_window(task.db_path, TRIAL_WINDOW_LAST_DB_NUMBER)
    catch err
        @warn "Could not read $(task.db_path): $err"
        return [emit_placeholder_row(task, STATUS_DB_UNREADABLE, placeholder_metadata)]
    end

    if isempty(ranked_trials)
        return [emit_placeholder_row(task, STATUS_NO_COMPLETE_TRIALS_IN_WIN, placeholder_metadata)]
    end

    fronts_path = find_all_fronts_csv(results_dir, hpo_dir_name,
                                       task.algorithm, task.sampler, task.problem)
    if isnothing(fronts_path)
        valid_mask = .!ismissing.(ranked_trials.value) .&
                     .!isnan.(coalesce.(ranked_trials.value, NaN)) .&
                     isfinite.(coalesce.(ranked_trials.value, -Inf))
        valid_rows = ranked_trials[valid_mask, :]

        isempty(valid_rows) &&
            return [emit_placeholder_row(task, STATUS_ALL_HV_INVALID, placeholder_metadata)]

        top1_db_number = Int(valid_rows[1, :number])
        top1_hv        = Float64(valid_rows[1, :value])
        return [emit_placeholder_row(task, STATUS_NO_CSV, (
            best_trial_db_number = missing, best_hv_value = top1_hv, fallback_used = false,
            top1_trial_db_number = top1_db_number, top1_hv_value = top1_hv,
        ))]
    end

    objective_columns = try
        read_objective_columns(fronts_path)
    catch err
        @warn "Could not read objective columns from $fronts_path: $err"
        return [emit_placeholder_row(task, STATUS_DB_UNREADABLE, placeholder_metadata)]
    end
    isnothing(objective_columns) &&
        return [emit_placeholder_row(task, STATUS_DB_UNREADABLE, placeholder_metadata)]

    csv_trial_set, points_per_trial = try
        scan_csv_for_trials_and_points(fronts_path, objective_columns)
    catch err
        @warn "Could not read $fronts_path: $err"
        return [emit_placeholder_row(task, STATUS_DB_UNREADABLE, placeholder_metadata)]
    end

    pick = pick_best_in_window(ranked_trials, csv_trial_set)

    isnothing(pick.top1_db_number) &&
        return [emit_placeholder_row(task, STATUS_ALL_HV_INVALID, placeholder_metadata)]

    if isnothing(pick.chosen_db_number)
        return [emit_placeholder_row(task, STATUS_NO_FRONT_EXTRACTABLE, (
            best_trial_db_number = missing, best_hv_value = missing, fallback_used = true,
            top1_trial_db_number = pick.top1_db_number, top1_hv_value = pick.top1_hv,
        ))]
    end

    chosen_csv_trial = pick.chosen_db_number + TRIAL_INDEXING_OFFSET
    point_records = get(points_per_trial, chosen_csv_trial, Tuple{Int, Vector{Float64}}[])

    if isempty(point_records)
        return [emit_placeholder_row(task, STATUS_NO_FRONT_EXTRACTABLE, (
            best_trial_db_number = pick.chosen_db_number, best_hv_value = pick.chosen_hv,
            fallback_used = pick.chosen_rank > 0,
            top1_trial_db_number = pick.top1_db_number, top1_hv_value = pick.top1_hv,
        ))]
    end

    fallback_used = pick.chosen_rank > 0
    status = fallback_used ? STATUS_OK_FALLBACK : STATUS_OK

    rows = NamedTuple[]
    for (run_number, point) in point_records
        push!(rows, build_point_row(task, status;
            best_trial_db_number = pick.chosen_db_number, best_hv_value = pick.chosen_hv,
            fallback_used = fallback_used,
            top1_trial_db_number = pick.top1_db_number, top1_hv_value = pick.top1_hv,
            run = run_number, point = point))
    end
    return rows
end

function emit_placeholder_row(task, status, metadata)
    return build_point_row(task, status;
        best_trial_db_number = metadata.best_trial_db_number,
        best_hv_value        = metadata.best_hv_value,
        fallback_used        = metadata.fallback_used,
        top1_trial_db_number = metadata.top1_trial_db_number,
        top1_hv_value        = metadata.top1_hv_value,
        run                  = missing,
        point                = nothing,
    )
end

function build_point_row(task, status;
                         best_trial_db_number, best_hv_value, fallback_used,
                         top1_trial_db_number, top1_hv_value, run, point)
    base = Dict{String, Any}(
        "algorithm"             => task.algorithm,
        "sampler"               => task.sampler,
        "problem"               => task.problem,
        "best_trial_db_number"  => best_trial_db_number,
        "best_hv_value"         => best_hv_value,
        "fallback_used"         => fallback_used,
        "top1_trial_db_number"  => top1_trial_db_number,
        "top1_hv_value"         => top1_hv_value,
        "status"                => status,
        "run"                   => run,
    )
    if !isnothing(point)
        for j in 1:length(point)
            base["obj_$j"] = point[j]
        end
    end
    return (; (Symbol(k) => v for (k, v) in base)...)
end

# Output formatting

function harmonize_dataframes(rows::Vector{<:NamedTuple})
    isempty(rows) && return DataFrame()
    all_keys = Set{Symbol}()
    for row in rows
        for k in keys(row)
            push!(all_keys, k)
        end
    end

    obj_keys = sort([k for k in all_keys if startswith(String(k), "obj_")];
                    by = k -> parse(Int, replace(String(k), "obj_" => "")))
    metadata_order = [
        :algorithm, :sampler, :problem,
        :best_trial_db_number, :best_hv_value,
        :fallback_used, :top1_trial_db_number, :top1_hv_value,
        :status, :run,
    ]
    column_order = vcat(metadata_order, obj_keys)

    aligned = NamedTuple[]
    for row in rows
        d = Dict{Symbol, Any}()
        for k in column_order
            d[k] = haskey(row, k) ? row[k] : missing
        end
        push!(aligned, (; (k => d[k] for k in column_order)...))
    end
    return DataFrame(aligned)
end

# CLI

function parse_command_line_args(args::Vector{String})
    output_dir   = nothing
    output_name  = "best_trial_fronts.csv"
    results_dir  = nothing
    experiment   = "benchmark"

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--output-dir"
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
Usage: julia extract_best_trial_fronts.jl --output-dir DIR [--output-name NAME] [--results-dir DIR] [--experiment benchmark|truss]

For every Optuna study, picks the trial with the highest HV among COMPLETE trials with
db.number <= $TRIAL_WINDOW_LAST_DB_NUMBER. Maps to row(s) in the all_fronts CSV using
csv.trial = db.number + $TRIAL_INDEXING_OFFSET. Writes one CSV with every (run, point)
of every chosen trial.

Required: --output-dir DIR (must be OUTSIDE Results/)
Optional: --output-name NAME, --results-dir DIR, --experiment benchmark|truss
""")
            exit(0)
        else
            error("Unknown argument: $a (use --help)")
        end
    end

    isnothing(output_dir) && error("--output-dir is required (use --help for usage)")
    return (output_dir=output_dir, output_name=output_name, results_dir=results_dir, experiment=experiment)
end

function summarize_status(rows::Vector{<:NamedTuple})
    counts = Dict{String, Int}()
    seen_tasks = Dict{Tuple{String, String, Int}, String}()
    for row in rows
        key = (row.algorithm, row.sampler, row.problem)
        haskey(seen_tasks, key) && continue
        seen_tasks[key] = row.status
    end
    for (_, status) in seen_tasks
        counts[status] = get(counts, status, 0) + 1
    end
    return counts, length(seen_tasks)
end

# Driver

function main()
    parsed = parse_command_line_args(copy(ARGS))

    results_dir = isnothing(parsed.results_dir) ?
        normpath(dirname(@__DIR__), "Results") : parsed.results_dir
    isdir(results_dir) || error("Results dir not found: $results_dir")

    exp = experiment_paths(parsed.experiment, results_dir)

    abs_output, output_csv = refuse_destructive_output(parsed.output_dir, results_dir;
                                                        output_name=parsed.output_name)
    mkpath(abs_output)

    optuna_studies_dir = exp.optuna_studies
    isdir(optuna_studies_dir) || error("optuna_studies dir not found: $optuna_studies_dir")

    println("Experiment:        $(exp.name)")
    println("Results dir:       $results_dir")
    println("HPO dir:           $(exp.hpo_dir)")
    println("Output CSV:        $output_csv")
    println("Optuna studies:    $optuna_studies_dir")
    println("Trial window:      db.number in [0, $TRIAL_WINDOW_LAST_DB_NUMBER]")
    println("Indexing offset:   db.number + $TRIAL_INDEXING_OFFSET == csv.trial")
    println("="^60)

    studies = find_optuna_studies(optuna_studies_dir)
    sort!(studies, by = s -> (s.algorithm, s.sampler, s.problem))
    println("Discovered $(length(studies)) study database file(s).")

    all_rows = NamedTuple[]
    start_time = time()
    for (idx, task) in enumerate(studies)
        if idx == 1 || idx % 50 == 0 || idx == length(studies)
            elapsed = time() - start_time
            @printf("  [%4d/%d] %s_%s_Problem_%d  (elapsed %.1fs)\n",
                    idx, length(studies), task.algorithm, task.sampler, task.problem, elapsed)
            flush(stdout)
        end
        append!(all_rows, build_task_rows(task, results_dir, exp.hpo_dir_name))
    end

    if isempty(all_rows)
        println("\nNo rows collected; nothing to write.")
        return
    end

    df = harmonize_dataframes(all_rows)
    CSV.write(output_csv, df)

    println("\n" * "="^60)
    println("Wrote $output_csv ($(nrow(df)) rows)")
    println("="^60)

    counts, n_tasks = summarize_status(all_rows)
    println("\nPer-task status breakdown ($n_tasks unique tasks):")
    for status in sort(collect(keys(counts)))
        @printf("  %-30s %d\n", status, counts[status])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
