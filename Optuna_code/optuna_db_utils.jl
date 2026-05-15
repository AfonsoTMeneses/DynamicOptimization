using SQLite
using DBInterface
using DataFrames
using CSV
using JSON

const STUDY_FILENAME_REGEX = r"^study_(.+)_([A-Za-z]+Sampler)_Problem_(\d+)\.db$"
const TRIAL_WINDOW_LAST_DB_NUMBER = 99
const TRIAL_INDEXING_OFFSET = 1
const EXPECTED_TRIALS_PER_TASK = 100
const ALGORITHMS = ["NSGA2", "SPEA2", "SMS_EMOA", "MOEAD_DE"]
const COUNTED_STATES = ["COMPLETE", "RUNNING", "FAIL", "PRUNED", "WAITING"]

# Connection wrapper

function with_optuna_db(f, db_path::AbstractString)
    db = SQLite.DB(db_path)
    try
        return f(db)
    finally
        SQLite.close(db)
    end
end

# Study discovery

function find_optuna_studies(optuna_studies_dir::AbstractString)
    isdir(optuna_studies_dir) || error("optuna_studies dir not found: $optuna_studies_dir")
    studies = NamedTuple{(:algorithm, :sampler, :problem, :db_path), Tuple{String, String, Int, String}}[]
    for filename in readdir(optuna_studies_dir)
        endswith(filename, ".db") || continue
        m = match(STUDY_FILENAME_REGEX, filename)
        isnothing(m) && continue
        push!(studies, (
            algorithm = String(m.captures[1]),
            sampler   = String(m.captures[2]),
            problem   = parse(Int, m.captures[3]),
            db_path   = joinpath(optuna_studies_dir, filename),
        ))
    end
    return studies
end

# Trial reads

function count_trials_by_state(db_path::AbstractString)
    with_optuna_db(db_path) do db
        return DBInterface.execute(db, "SELECT state, COUNT(*) AS n FROM trials GROUP BY state") |> DataFrame
    end
end

function state_counts_dict(db_path::AbstractString)
    df = count_trials_by_state(db_path)
    counts = Dict{String, Int}(s => 0 for s in COUNTED_STATES)
    for row in eachrow(df)
        counts[String(row.state)] = Int(row.n)
    end
    return counts
end

function state_counts_from_trials_df(trials_df::DataFrame)
    counts = Dict{String, Int}(s => 0 for s in COUNTED_STATES)
    for row in eachrow(trials_df)
        counts[String(row.state)] = get(counts, String(row.state), 0) + 1
    end
    return counts
end

function read_trials_with_values(db_path::AbstractString)
    with_optuna_db(db_path) do db
        trials_df = DBInterface.execute(db, "SELECT trial_id, number, state FROM trials ORDER BY number") |> DataFrame
        values_df = DBInterface.execute(db, "SELECT trial_id, value FROM trial_values") |> DataFrame
        return trials_df, values_df
    end
end

function read_capped_complete_trials(db_path::AbstractString, last_db_number::Int)
    with_optuna_db(db_path) do db
        sql = """
            SELECT t.trial_id, t.number, tv.value, t.state
            FROM trials t
            LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE' AND t.number <= ?
            ORDER BY t.number
        """
        return DBInterface.execute(db, sql, [last_db_number]) |> DataFrame
    end
end

function read_ranked_complete_trials_in_window(db_path::AbstractString, last_db_number::Int)
    with_optuna_db(db_path) do db
        sql = """
            SELECT t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON tv.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE' AND t.number <= ?
            ORDER BY tv.value DESC, t.number ASC
        """
        return DBInterface.execute(db, sql, [last_db_number]) |> DataFrame
    end
end

function read_user_attr_for_trial(db_path::AbstractString, trial_id::Int, key::String)
    return with_optuna_db(db_path) do db
        try
            rows = DBInterface.execute(db, """
                SELECT value_json FROM trial_user_attributes
                WHERE trial_id = ? AND key = ?
            """, [trial_id, key]) |> DataFrame
            nrow(rows) == 0 && return nothing
            return JSON.parse(String(rows[1, :value_json]))
        catch
            return nothing
        end
    end
end

# Front-CSV discovery and reading

function find_all_fronts_csv(results_dir::AbstractString,
                              hpo_dir_name::AbstractString,
                              algorithm::AbstractString,
                              sampler::AbstractString,
                              problem::Int)
    canonical_name = "all_fronts_$(algorithm)_$(sampler)_Problem_$(problem).csv"
    iter_root = joinpath(results_dir, hpo_dir_name, algorithm)
    if isdir(iter_root)
        for iter_subdir in readdir(iter_root)
            iter_path = joinpath(iter_root, iter_subdir)
            isdir(iter_path) || continue
            for problem_subdir in readdir(iter_path)
                startswith(problem_subdir, "Problem_$(problem)_") || continue
                csv_path = joinpath(iter_path, problem_subdir, canonical_name)
                isfile(csv_path) && return csv_path
            end
        end
    end
    fallback_root = joinpath(results_dir, hpo_dir_name)
    if isdir(fallback_root)
        for (dir, _, files) in walkdir(fallback_root)
            canonical_name in files && return joinpath(dir, canonical_name)
        end
    end
    return nothing
end

function read_objective_columns(fronts_file::AbstractString)
    header_sample = CSV.read(fronts_file, DataFrame; limit=1, silencewarnings=true)
    sorted_obj_column_names = sort([string(c) for c in names(header_sample) if startswith(string(c), "obj_")])
    length(sorted_obj_column_names) < 2 && return nothing
    return Symbol.(sorted_obj_column_names)
end

function read_trial_set_from_all_fronts(csv_path::AbstractString)
    trials = Set{Int}()
    rows = CSV.Rows(csv_path; types=Dict(:trial => Int), reusebuffer=true, silencewarnings=true)
    for row in rows
        try
            push!(trials, row.trial)
        catch
            continue
        end
    end
    return trials
end
