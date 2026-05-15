using DataFrames
using CSV
using Printf

include(joinpath(@__DIR__, "optuna_db_utils.jl"))
include(joinpath(@__DIR__, "pareto_utils.jl"))

# Status classification

function classify_task_status(n_complete::Int, n_running::Int, n_with_valid_hv::Int)
    n_complete < EXPECTED_TRIALS_PER_TASK && return "needs_rerun"
    n_with_valid_hv == 0                  && return "all_trials_invalid_hv"
    n_running > 0                          && return "complete_with_orphan_running"
    return "complete"
end

# Per-task inventory

function build_inventory_for_task(task, results_dir::AbstractString, hpo_dir_name::AbstractString)
    algorithm = task.algorithm
    sampler   = task.sampler
    problem   = task.problem
    db_path   = task.db_path

    trials_df, values_df = try
        read_trials_with_values(db_path)
    catch err
        @warn "Could not read $db_path: $err"
        return (
            algorithm=algorithm, sampler=sampler, problem=problem,
            db_path=db_path, db_readable=false,
            n_trials_in_db=0, n_complete=0, n_running=0, n_failed=0, n_pruned=0, n_waiting=0,
            n_with_valid_hv=0,
            best_trial_db_number=missing, best_trial_csv_number=missing, best_hv_value=missing,
            all_fronts_csv_path="", all_fronts_csv_exists=false, n_trials_in_all_fronts=0,
            best_trial_extractable=false, fallback_trial_used=missing, fallback_hv_value=missing,
            top1_minus_fallback=missing,
            status="db_unreadable", notes="db read error: $err",
        )
    end

    state_counts = state_counts_from_trials_df(trials_df)
    n_complete = state_counts["COMPLETE"]
    n_running  = state_counts["RUNNING"]
    n_failed   = state_counts["FAIL"]
    n_pruned   = state_counts["PRUNED"]
    n_waiting  = state_counts["WAITING"]

    complete_trials = filter(row -> row.state == "COMPLETE", trials_df)
    joined = innerjoin(complete_trials, values_df, on=:trial_id)
    valid_mask = .!ismissing.(joined.value) .& .!isnan.(coalesce.(joined.value, NaN)) .&
                 isfinite.(coalesce.(joined.value, -Inf))
    valid_rows = joined[valid_mask, :]
    n_with_valid_hv = nrow(valid_rows)

    if n_with_valid_hv > 0
        sorted_valid = sort(valid_rows, :value, rev=true)
        best_db_number = Int(sorted_valid[1, :number])
        best_hv_value  = Float64(sorted_valid[1, :value])
    else
        sorted_valid    = DataFrame()
        best_db_number  = missing
        best_hv_value   = missing
    end

    all_fronts_path = find_all_fronts_csv(results_dir, hpo_dir_name, algorithm, sampler, problem)
    all_fronts_exists = !isnothing(all_fronts_path)
    csv_trial_set = Set{Int}()
    if all_fronts_exists
        try
            csv_trial_set = read_trial_set_from_all_fronts(all_fronts_path)
        catch err
            @warn "Could not read $all_fronts_path: $err"
            all_fronts_exists = false
        end
    end
    n_in_all_fronts = length(csv_trial_set)

    best_trial_csv_number  = ismissing(best_db_number) ? missing : best_db_number + TRIAL_INDEXING_OFFSET
    best_trial_extractable = !ismissing(best_trial_csv_number) && (best_trial_csv_number in csv_trial_set)

    fallback_trial_used  = missing
    fallback_hv_value    = missing
    top1_minus_fallback  = missing
    if !best_trial_extractable && n_with_valid_hv > 0 && all_fronts_exists
        for row in eachrow(sorted_valid)
            csv_num = Int(row.number) + TRIAL_INDEXING_OFFSET
            if csv_num in csv_trial_set
                fallback_trial_used = csv_num
                fallback_hv_value   = Float64(row.value)
                top1_minus_fallback = best_hv_value - fallback_hv_value
                break
            end
        end
    end

    status = classify_task_status(n_complete, n_running, n_with_valid_hv)

    note_parts = String[]
    n_complete < EXPECTED_TRIALS_PER_TASK && push!(note_parts,
        "only $n_complete/$EXPECTED_TRIALS_PER_TASK COMPLETE trials")
    n_running > 0     && push!(note_parts, "$n_running trial(s) still RUNNING")
    n_failed > 0      && push!(note_parts, "$n_failed FAIL trial(s)")
    !all_fronts_exists && push!(note_parts, "all_fronts CSV not found")
    if !ismissing(best_trial_csv_number) && all_fronts_exists && !best_trial_extractable
        push!(note_parts, "best db trial #$best_db_number (csv #$best_trial_csv_number) NOT in all_fronts")
    end
    !ismissing(fallback_trial_used) && push!(note_parts,
        "fallback to csv trial #$fallback_trial_used (Δhv=$(round(top1_minus_fallback, digits=6)))")
    notes = join(note_parts, "; ")

    return (
        algorithm=algorithm, sampler=sampler, problem=problem,
        db_path=db_path, db_readable=true,
        n_trials_in_db=nrow(trials_df), n_complete=n_complete, n_running=n_running,
        n_failed=n_failed, n_pruned=n_pruned, n_waiting=n_waiting,
        n_with_valid_hv=n_with_valid_hv,
        best_trial_db_number=best_db_number, best_trial_csv_number=best_trial_csv_number,
        best_hv_value=best_hv_value,
        all_fronts_csv_path=isnothing(all_fronts_path) ? "" : all_fronts_path,
        all_fronts_csv_exists=all_fronts_exists, n_trials_in_all_fronts=n_in_all_fronts,
        best_trial_extractable=best_trial_extractable,
        fallback_trial_used=fallback_trial_used, fallback_hv_value=fallback_hv_value,
        top1_minus_fallback=top1_minus_fallback,
        status=status, notes=notes,
    )
end

# CLI

function parse_command_line_args(args::Vector{String})
    results_dir = nothing
    output_dir  = nothing
    experiment  = "benchmark"

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--results-dir"
            i + 1 <= length(args) || error("--results-dir needs a path")
            results_dir = args[i+1]; i += 2
        elseif a == "--output-dir"
            i + 1 <= length(args) || error("--output-dir needs a path")
            output_dir = args[i+1]; i += 2
        elseif a == "--experiment"
            i + 1 <= length(args) || error("--experiment needs benchmark or truss")
            experiment = args[i+1]; i += 2
        elseif a in ("-h", "--help")
            println("""
Usage: julia inventory_optuna_studies.jl --output-dir DIR [--results-dir DIR] [--experiment benchmark|truss]

Reads every Optuna study .db file under the chosen experiment's optuna_studies/
dir and produces a per-task inventory CSV. Cross-references each task's best
DB trial against the corresponding all_fronts CSV. Read-only on Results/.

Required: --output-dir DIR (must be OUTSIDE Results/)
""")
            exit(0)
        else
            error("Unknown argument: $a (use --help)")
        end
    end

    isnothing(output_dir) && error("--output-dir is required (use --help for usage)")
    return (results_dir=results_dir, output_dir=output_dir, experiment=experiment)
end

# Driver

function main()
    parsed = parse_command_line_args(copy(ARGS))

    results_dir = isnothing(parsed.results_dir) ?
        normpath(dirname(@__DIR__), "Results") : parsed.results_dir
    isdir(results_dir) || error("Results dir not found: $results_dir")

    exp = experiment_paths(parsed.experiment, results_dir)

    output_dir = refuse_destructive_output(parsed.output_dir, results_dir;
                                            output_name="optuna_inventory.csv")[1]
    mkpath(output_dir)

    optuna_studies_dir = exp.optuna_studies
    isdir(optuna_studies_dir) || error("optuna_studies dir not found: $optuna_studies_dir")

    println("Experiment:        $(exp.name)")
    println("Results dir:       $results_dir")
    println("Output dir:        $output_dir")
    println("Optuna studies:    $optuna_studies_dir")
    println("Trial offset:      db.number + $TRIAL_INDEXING_OFFSET == csv.trial")
    println("="^60)

    studies = find_optuna_studies(optuna_studies_dir)
    sort!(studies, by = s -> (s.algorithm, s.sampler, s.problem))
    println("Discovered $(length(studies)) study database file(s).")

    inventory_rows = NamedTuple[]
    start_time = time()

    for (idx, task) in enumerate(studies)
        if idx == 1 || idx % 50 == 0 || idx == length(studies)
            elapsed_seconds = time() - start_time
            @printf("  [%4d/%d] %s_%s_Problem_%d  (elapsed %.1fs)\n",
                    idx, length(studies), task.algorithm, task.sampler, task.problem, elapsed_seconds)
            flush(stdout)
        end
        push!(inventory_rows, build_inventory_for_task(task, results_dir, exp.hpo_dir_name))
    end

    inventory_df = DataFrame(inventory_rows)
    inventory_csv = joinpath(output_dir, "optuna_inventory.csv")
    CSV.write(inventory_csv, inventory_df)

    println("\n" * "="^60)
    println("Wrote $inventory_csv ($(nrow(inventory_df)) rows)")
    println("="^60)

    println("\nStatus breakdown:")
    for status in sort(unique(inventory_df.status))
        n = count(==(status), inventory_df.status)
        @printf("  %-30s %d\n", status, n)
    end

    needs_rerun = filter(:status => ==("needs_rerun"), inventory_df)
    if nrow(needs_rerun) > 0
        rerun_csv = joinpath(output_dir, "tasks_needing_rerun.csv")
        CSV.write(rerun_csv, needs_rerun)
        println("\nWrote $rerun_csv ($(nrow(needs_rerun)) tasks with <$EXPECTED_TRIALS_PER_TASK COMPLETE trials)")
    end

    not_extractable = filter(row -> row.db_readable && row.all_fronts_csv_exists &&
                                    !row.best_trial_extractable && !ismissing(row.fallback_trial_used),
                              inventory_df)
    if nrow(not_extractable) > 0
        ne_csv = joinpath(output_dir, "tasks_with_fallback.csv")
        CSV.write(ne_csv, not_extractable)
        println("Wrote $ne_csv ($(nrow(not_extractable)) tasks where best db-trial isn't in CSV; fallback used)")
    end

    no_csv = filter(row -> row.db_readable && !row.all_fronts_csv_exists, inventory_df)
    if nrow(no_csv) > 0
        nc_csv = joinpath(output_dir, "tasks_missing_all_fronts.csv")
        CSV.write(nc_csv, no_csv)
        println("Wrote $nc_csv ($(nrow(no_csv)) tasks with no all_fronts CSV)")
    end

    if nrow(inventory_df) > 0
        clean = count(row -> row.status == "complete" && row.best_trial_extractable, eachrow(inventory_df))
        @printf("\nReady-to-use tasks (status=complete AND best trial extractable): %d / %d (%.1f%%)\n",
                clean, nrow(inventory_df), 100 * clean / nrow(inventory_df))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
