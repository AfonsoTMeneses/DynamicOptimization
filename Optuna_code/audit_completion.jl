using DataFrames
using CSV
using DBInterface
using Printf

include(joinpath(@__DIR__, "optuna_db_utils.jl"))
include(joinpath(@__DIR__, "pareto_utils.jl"))

const TARGET_TRIALS_PER_TASK = EXPECTED_TRIALS_PER_TASK
const FAST_TRIAL_SECONDS = 5
const SLOW_TRIAL_SECONDS = 60

# Classification

function classify_task(n_complete::Int)
    n_complete >= TARGET_TRIALS_PER_TASK && return "complete"
    n_complete == 0                       && return "empty"
    return "partial"
end

function categorize_cost(elapsed_per_trial::Union{Float64, Nothing})
    isnothing(elapsed_per_trial) && return :medium
    elapsed_per_trial < FAST_TRIAL_SECONDS    && return :fast
    elapsed_per_trial > SLOW_TRIAL_SECONDS    && return :slow
    return :medium
end

# Time parsing

function parse_iso8601(s::AbstractString)
    s_clean = replace(strip(s), 'T' => ' ')
    if length(s_clean) >= 19
        date_part = s_clean[1:10]
        time_part = s_clean[12:19]
        y = parse(Int, date_part[1:4])
        mo = parse(Int, date_part[6:7])
        d  = parse(Int, date_part[9:10])
        h  = parse(Int, time_part[1:2])
        mi = parse(Int, time_part[4:5])
        sec = parse(Int, time_part[7:8])
        frac = 0.0
        if length(s_clean) > 19 && s_clean[20] == '.'
            frac_str = ""
            for c in s_clean[21:end]
                isdigit(c) || break
                frac_str *= c
            end
            isempty(frac_str) || (frac = parse(Float64, "0." * frac_str))
        end
        epoch = (y - 1970) * 31_557_600.0 + (mo - 1) * 2_629_800.0 + (d - 1) * 86_400.0
        return epoch + h * 3600.0 + mi * 60.0 + sec + frac
    end
    return 0.0
end

function median_seconds_per_trial(db_path::AbstractString)
    with_optuna_db(db_path) do db
        rows = DBInterface.execute(db, """
            SELECT t.datetime_start, t.datetime_complete
            FROM trials t
            WHERE t.state = 'COMPLETE'
              AND t.datetime_start IS NOT NULL
              AND t.datetime_complete IS NOT NULL
        """) |> DataFrame
        isempty(rows) && return nothing
        durations = Float64[]
        for row in eachrow(rows)
            try
                start_t = parse_iso8601(String(row.datetime_start))
                stop_t  = parse_iso8601(String(row.datetime_complete))
                d = stop_t - start_t
                d > 0.0 && push!(durations, d)
            catch
                continue
            end
        end
        isempty(durations) && return nothing
        sort!(durations)
        return durations[ceil(Int, length(durations) / 2)]
    end
end

# Per-task audit

function audit_task(task)
    state_counts = try
        state_counts_dict(task.db_path)
    catch err
        @warn "Could not read $(task.db_path): $err"
        return (
            algorithm=task.algorithm, sampler=task.sampler, problem=task.problem,
            n_total=0, n_complete=0, n_running=0, n_failed=0,
            n_pruned=0, n_waiting=0,
            classification="db_unreadable",
            trials_remaining=TARGET_TRIALS_PER_TASK,
            median_sec_per_trial=missing,
            cost_category="medium",
        )
    end

    n_complete = state_counts["COMPLETE"]
    n_running  = state_counts["RUNNING"]
    n_failed   = state_counts["FAIL"]
    n_pruned   = state_counts["PRUNED"]
    n_waiting  = state_counts["WAITING"]
    n_total    = sum(values(state_counts))

    classification    = classify_task(n_complete)
    trials_remaining  = max(0, TARGET_TRIALS_PER_TASK - n_complete)
    median_sec        = median_seconds_per_trial(task.db_path)
    cost_category     = categorize_cost(median_sec)

    return (
        algorithm = task.algorithm,
        sampler   = task.sampler,
        problem   = task.problem,
        n_total = n_total, n_complete = n_complete, n_running = n_running,
        n_failed = n_failed, n_pruned = n_pruned, n_waiting = n_waiting,
        classification    = classification,
        trials_remaining  = trials_remaining,
        median_sec_per_trial = isnothing(median_sec) ? missing : round(median_sec, digits=1),
        cost_category     = String(cost_category),
    )
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
Usage: julia audit_completion.jl --output-dir DIR [--results-dir DIR] [--experiment benchmark|truss]

Walks the optuna_studies/ for the chosen experiment and reports which tasks
have fewer than $TARGET_TRIALS_PER_TASK COMPLETE trials, plus per-trial cost.

Outputs:
  audit_completion.csv  full per-task audit
  tasks_to_resume.csv   subset that needs more trials

Read-only on Results/. Refuses to write inside Results/.
""")
            exit(0)
        else
            error("Unknown arg: $a")
        end
    end
    isnothing(output_dir) && error("--output-dir is required")
    return (output_dir=output_dir, results_dir=results_dir, experiment=experiment)
end

# Driver

function main()
    parsed = parse_command_line_args(copy(ARGS))

    results_dir = isnothing(parsed.results_dir) ?
        normpath(dirname(@__DIR__), "Results") : parsed.results_dir
    isdir(results_dir) || error("Results dir not found: $results_dir")

    exp = experiment_paths(parsed.experiment, results_dir)

    abs_output = refuse_destructive_output(parsed.output_dir, results_dir)
    mkpath(abs_output)

    optuna_studies_dir = exp.optuna_studies
    isdir(optuna_studies_dir) || error("optuna_studies dir not found: $optuna_studies_dir")

    println("Experiment:    $(exp.name)")
    println("Results dir:   $results_dir")
    println("Optuna studies: $optuna_studies_dir")
    println("Output dir:    $abs_output")
    println("="^70)

    studies = find_optuna_studies(optuna_studies_dir)
    sort!(studies, by = s -> (s.algorithm, s.sampler, s.problem))
    println("Discovered $(length(studies)) study .db file(s).")

    audit_rows = NamedTuple[]
    started = time()
    for (idx, task) in enumerate(studies)
        if idx == 1 || idx % 100 == 0 || idx == length(studies)
            @printf("  [%4d/%d]  elapsed %.1fs\n", idx, length(studies), time() - started)
            flush(stdout)
        end
        push!(audit_rows, audit_task(task))
    end

    audit_df = DataFrame(audit_rows)
    audit_csv = joinpath(abs_output, "audit_completion.csv")
    isfile(audit_csv) && error("Refusing to overwrite $audit_csv")
    CSV.write(audit_csv, audit_df)

    println("\nWrote $audit_csv ($(nrow(audit_df)) rows)")

    println("\nClassification breakdown:")
    for cls in sort(unique(audit_df.classification))
        n = count(==(cls), audit_df.classification)
        @printf("  %-20s %d\n", cls, n)
    end

    needs_resume = filter(row -> row.classification == "partial" || row.classification == "empty", eachrow(audit_df))
    needs_resume_df = DataFrame(needs_resume)
    if nrow(needs_resume_df) == 0
        println("\nAll tasks complete.")
        return
    end

    resume_csv = joinpath(abs_output, "tasks_to_resume.csv")
    CSV.write(resume_csv, needs_resume_df)
    println("\nWrote $resume_csv ($(nrow(needs_resume_df)) tasks needing more trials)")

    println("\nIncomplete tasks by algorithm:")
    for alg in sort(unique(needs_resume_df.algorithm))
        n = count(==(alg), needs_resume_df.algorithm)
        @printf("  %-15s %d tasks\n", alg, n)
    end

    println("\nIncomplete tasks by cost category:")
    for cost in ["fast", "medium", "slow"]
        n = count(==(cost), needs_resume_df.cost_category)
        @printf("  %-10s %d tasks\n", cost, n)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
