using DataFrames
using CSV
using JSON
using Printf

include(joinpath(@__DIR__, "optuna_db_utils.jl"))
include(joinpath(@__DIR__, "pareto_utils.jl"))

# Per-task processing

function process_task(task)
    algorithm = task.algorithm
    sampler   = task.sampler
    problem   = task.problem

    capped_trials_df = try
        read_capped_complete_trials(task.db_path, TRIAL_WINDOW_LAST_DB_NUMBER)
    catch err
        @warn "Could not read $(task.db_path): $err"
        return ([], "db_unreadable")
    end

    valid_mask = .!ismissing.(capped_trials_df.value) .&
                 .!isnan.(coalesce.(capped_trials_df.value, NaN)) .&
                 isfinite.(coalesce.(capped_trials_df.value, -Inf))
    valid_trials = capped_trials_df[valid_mask, :]

    n_complete_in_window = nrow(capped_trials_df)
    n_valid_in_window    = nrow(valid_trials)

    if n_valid_in_window == 0
        hv_row = (
            algorithm = algorithm, sampler = sampler, problem = problem,
            n_complete_in_window = n_complete_in_window,
            n_valid_in_window = 0,
            best_trial_db_number = missing,
            best_trial_csv_number = missing,
            best_hv_value = missing,
            all_hv_json = "",
            all_hv_mean = missing,
            all_hv_n_runs = 0,
            params_json = "",
            status = "no_valid_trials",
        )
        return ([hv_row], "no_valid_trials")
    end

    sorted_valid = sort(valid_trials, :value, rev=true)
    best_db_number  = Int(sorted_valid[1, :number])
    best_trial_id   = Int(sorted_valid[1, :trial_id])
    best_hv         = Float64(sorted_valid[1, :value])
    best_csv_number = best_db_number + TRIAL_INDEXING_OFFSET

    all_hv_parsed = read_user_attr_for_trial(task.db_path, best_trial_id, "All_HV")
    all_hv_vec = isnothing(all_hv_parsed) ? [best_hv] : collect(Float64, all_hv_parsed)

    params_parsed = read_user_attr_for_trial(task.db_path, best_trial_id, "params")
    params_json   = isnothing(params_parsed) ? "" : JSON.json(params_parsed)

    hv_row = (
        algorithm = algorithm, sampler = sampler, problem = problem,
        n_complete_in_window = n_complete_in_window,
        n_valid_in_window = n_valid_in_window,
        best_trial_db_number = best_db_number,
        best_trial_csv_number = best_csv_number,
        best_hv_value = best_hv,
        all_hv_json = JSON.json(all_hv_vec),
        all_hv_mean = isempty(all_hv_vec) ? NaN : sum(all_hv_vec) / length(all_hv_vec),
        all_hv_n_runs = length(all_hv_vec),
        params_json = params_json,
        status = "ok",
    )

    return ([hv_row], "ok")
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
            output_dir = args[i+1]
            i += 2
        elseif a == "--results-dir"
            i + 1 <= length(args) || error("--results-dir needs a path")
            results_dir = args[i+1]
            i += 2
        elseif a == "--experiment"
            i + 1 <= length(args) || error("--experiment needs benchmark or truss")
            experiment = args[i+1]
            i += 2
        elseif a in ("-h", "--help")
            println("""
Usage: julia extract_db_metrics.jl --output-dir DIR [--results-dir DIR] [--experiment benchmark|truss]

For each Optuna .db file under <hpo_dir>/optuna_studies/ (resolved from --experiment),
picks the best COMPLETE trial with db.number <= $TRIAL_WINDOW_LAST_DB_NUMBER and
extracts All_HV and params for that trial.

Outputs:
  hv_per_task.csv          best trial selection + HV info

Read-only. Refuses to write inside Results/.
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
        normpath(dirname(@__DIR__), "Results") :
        parsed.results_dir
    isdir(results_dir) || error("Results dir not found: $results_dir")

    exp = experiment_paths(parsed.experiment, results_dir)

    abs_output, hv_csv = refuse_destructive_output(parsed.output_dir, results_dir;
                                                     output_name="hv_per_task.csv")
    mkpath(abs_output)

    optuna_studies_dir = exp.optuna_studies
    isdir(optuna_studies_dir) || error("optuna_studies dir not found: $optuna_studies_dir")

    println("Experiment:     $(exp.name)")
    println("Results dir:    $results_dir")
    println("HPO dir:        $(exp.hpo_dir)")
    println("Output dir:     $abs_output")
    println("Trial cap:      db.number <= $TRIAL_WINDOW_LAST_DB_NUMBER")
    println("Indexing rule:  csv.trial = db.number + $TRIAL_INDEXING_OFFSET")
    println("="^70)

    studies = find_optuna_studies(optuna_studies_dir)
    sort!(studies, by = s -> (s.algorithm, s.sampler, s.problem))
    println("Discovered $(length(studies)) study .db file(s).")

    all_hv_rows = NamedTuple[]
    started = time()

    for (idx, task) in enumerate(studies)
        if idx == 1 || idx % 50 == 0 || idx == length(studies)
            @printf("  [%4d/%d] elapsed %.1fs\n", idx, length(studies), time() - started)
            flush(stdout)
        end
        hv_rows, _status = process_task(task)
        append!(all_hv_rows, hv_rows)
    end

    hv_df = DataFrame(all_hv_rows)
    CSV.write(hv_csv, hv_df)
    println("\nWrote $hv_csv ($(nrow(hv_df)) rows)")

    println("\nStatus breakdown:")
    for status in sort(unique(hv_df.status))
        n = count(==(status), hv_df.status)
        @printf("  %-25s %d\n", status, n)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
