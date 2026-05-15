include(joinpath(@__DIR__, "hyperoptimization_intervals.jl"))
using DataStructures
using Metaheuristics
using Metaheuristics: pareto_front, is_feasible
import Metaheuristics.PerformanceIndicators: igd_plus, igd, hypervolume
using HardTestProblems
using DataFrames
using Logging

# Module constants

const NORMALIZED_REFERENCE_MARGIN = 0.01
const MOEAD_DE_INTERNAL_FIELDS = Set([:nobjectives, :N, :λ, :T, :n_r, :z, :B, :τ])
const DEFAULT_MOEAD_NPARTITIONS = 50
const MAX_MOEAD_POPULATION = 500
const RUNS_CI_COLUMN = "confidence_interval_95"
const RUNS_CI_POSITION = 10
const FALLBACK_RUNS = 30

# Feasibility and HV

function filter_feasible(population)
    return filter(is_feasible, population)
end

function get_feasible_non_dominated(population)
    feasible = filter_feasible(population)
    isempty(feasible) && return Vector{Float64}[]
    nd = get_non_dominated_solutions(feasible)
    return [Vector{Float64}(sol.f) for sol in nd]
end

function compute_empirical_bounds(all_fronts::Vector{Vector{Vector{Float64}}})
    union_objectives = reduce(vcat, all_fronts; init=Vector{Float64}[])
    isempty(union_objectives) && return nothing, nothing
    nobj = length(first(union_objectives))
    ideal = [minimum(sol[j] for sol in union_objectives) for j in 1:nobj]
    nadir = [maximum(sol[j] for sol in union_objectives) for j in 1:nobj]
    return ideal, nadir
end

function normalize_objectives(f::Vector{Float64}, ideal::Vector{Float64}, nadir::Vector{Float64})
    return [(f[j] - ideal[j]) / max(nadir[j] - ideal[j], 1e-12) for j in eachindex(f)]
end

function normalized_hypervolume(front::Vector{Vector{Float64}},
                                ideal::Vector{Float64},
                                nadir::Vector{Float64};
                                margin::Float64=NORMALIZED_REFERENCE_MARGIN)
    isempty(front) && return 0.0
    nobj = length(ideal)
    norm_front = [normalize_objectives(sol, ideal, nadir) for sol in front]
    pop = [Metaheuristics.create_child(zeros(1), (nf, Float64[], Float64[])) for nf in norm_front]
    R = fill(1.0 + margin, nobj)
    return hypervolume(pop, R)
end

# Empirical bounds IO

function save_empirical_bounds(filepath::String, problem_name::String,
                               instance::Int, ideal::Vector{Float64}, nadir::Vector{Float64})
    df = DataFrame(
        problem_name = [problem_name],
        current_instance = [instance],
        ideal_point = [JSON.json(ideal)],
        nadir_point = [JSON.json(nadir)],
    )
    write_header = !isfile(filepath)
    CSV.write(filepath, df, append=true, writeheader=write_header)
end

function load_empirical_bounds(filepath::String)
    if !isfile(filepath)
        error("Empirical bounds file not found: $filepath. Run baseline_benchmark.jl first.")
    end
    df = CSV.read(filepath, DataFrame; pool=false)
    bounds = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    for row in eachrow(df)
        ismissing(row.current_instance) && continue
        inst = Int(row.current_instance)
        ideal = Vector{Float64}(JSON.parse(row.ideal_point))
        nadir = Vector{Float64}(JSON.parse(row.nadir_point))
        bounds[inst] = (ideal, nadir)
    end
    return bounds
end

function load_merged_empirical_bounds(bounds_dir::String, baseline_name::String;
                                       algorithms=["NSGA2", "SPEA2", "SMS_EMOA", "MOEAD_DE"])
    merged = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    for algorithm in algorithms
        bounds_csv = joinpath(bounds_dir, "empirical_bounds_$(baseline_name)_$(algorithm).csv")
        isfile(bounds_csv) || (@warn "Bounds file not found: $bounds_csv"; continue)
        for (inst, (ideal, nadir)) in load_empirical_bounds(bounds_csv)
            if haskey(merged, inst)
                existing_ideal, existing_nadir = merged[inst]
                merged[inst] = (
                    [min(existing_ideal[i], ideal[i]) for i in eachindex(ideal)],
                    [max(existing_nadir[i], nadir[i]) for i in eachindex(nadir)],
                )
            else
                merged[inst] = (ideal, nadir)
            end
        end
        println("Loaded empirical bounds from $bounds_csv")
    end
    return merged
end

# Reference fronts IO

function save_reference_front(filepath::String, instance::Int, front::Vector{Vector{Float64}})
    isempty(front) && return
    nobj = length(first(front))
    rows = []
    for sol in front
        row = Dict{String, Any}("current_instance" => instance)
        for j in 1:nobj
            row["obj_$j"] = sol[j]
        end
        push!(rows, row)
    end
    df = DataFrame(rows)
    col_order = vcat(["current_instance"], ["obj_$j" for j in 1:nobj])
    df = df[:, col_order]
    write_header = !isfile(filepath)
    CSV.write(filepath, df; append=true, writeheader=write_header)
end

function load_reference_fronts(filepath::String)
    if !isfile(filepath)
        error("Reference front file not found: $filepath. Run baseline_benchmark.jl first.")
    end
    df = CSV.read(filepath, DataFrame; pool=false, silencewarnings=true)
    fronts = Dict{Int, Vector{Vector{Float64}}}()
    obj_col_pattern = r"^(obj_\d+|Column\d+)$"
    obj_cols = [c for c in names(df) if occursin(obj_col_pattern, String(c))]
    for row in eachrow(df)
        ismissing(row.current_instance) && continue
        inst = Int(row.current_instance)
        sol = Float64[row[c] for c in obj_cols if !ismissing(row[c])]
        isempty(sol) && continue
        if !haskey(fronts, inst)
            fronts[inst] = Vector{Float64}[]
        end
        push!(fronts[inst], sol)
    end
    return fronts
end

# Per-worker shard storage

function _per_worker_path(filepath::String)
    base, ext = splitext(filepath)
    return string(base, "_w", myid(), ext)
end

function save_trial_fronts(filepath::String, fronts::Vector{Vector{Vector{Float64}}},
                          algorithm_name, sampler_name, problem_instance, trial_number)
    isempty(fronts) && return
    nonempty = filter(!isempty, fronts)
    isempty(nonempty) && return
    nobj = length(first(first(nonempty)))

    rows = []
    for (run_idx, front) in enumerate(fronts)
        for sol in front
            row = Dict{String, Any}(
                "algorithm"        => string(algorithm_name),
                "sampler"          => sampler_name,
                "problem_instance" => problem_instance,
                "trial"            => trial_number,
                "run"              => run_idx,
            )
            for j in 1:nobj
                row["obj_$j"] = sol[j]
            end
            push!(rows, row)
        end
    end

    isempty(rows) && return
    df = DataFrame(rows)
    col_order = vcat(["algorithm", "sampler", "problem_instance", "trial", "run"],
                     ["obj_$j" for j in 1:nobj])
    df = df[:, col_order]

    worker_path = _per_worker_path(filepath)
    write_header = !isfile(worker_path)
    CSV.write(worker_path, df; append=true, writeheader=write_header)
end

function merge_worker_fronts(root_dir::String)
    isdir(root_dir) || return
    groups = Dict{String, Vector{String}}()
    for (root, _, files) in walkdir(root_dir)
        for fn in files
            startswith(fn, "all_fronts_") || continue
            m = match(r"^(all_fronts_.+)_w\d+\.csv$", fn)
            isnothing(m) && continue
            canonical = joinpath(root, m.captures[1] * ".csv")
            push!(get!(groups, canonical, String[]), joinpath(root, fn))
        end
    end

    for (canonical, shards) in groups
        sort!(shards)
        try
            open(canonical, "w") do out
                first = true
                for shard in shards
                    open(shard, "r") do inp
                        for (i, line) in enumerate(eachline(inp))
                            if i == 1
                                first || continue
                            end
                            println(out, line)
                        end
                    end
                    first = false
                end
            end
            for shard in shards
                rm(shard)
            end
            println("  merged $(length(shards)) shards -> $canonical")
        catch e
            @warn "Failed to merge shards into $canonical: $e"
        end
    end
end

function merge_result_shards(root_dir::String)
    isdir(root_dir) || return
    groups = Dict{String, Vector{String}}()
    for (dir_path, _, files) in walkdir(root_dir)
        for fn in files
            startswith(fn, "all_fronts_") && continue
            m = match(r"^(.+?)_(\w+Sampler)_w\d+\.csv$", fn)
            isnothing(m) && continue
            canonical = joinpath(dir_path, "$(m.captures[1])_$(m.captures[2]).csv")
            push!(get!(groups, canonical, String[]), joinpath(dir_path, fn))
        end
    end

    for (canonical, shards) in groups
        sort!(shards)
        try
            existing_rows = String[]
            if isfile(canonical)
                open(canonical, "r") do inp
                    for (i, line) in enumerate(eachline(inp))
                        i == 1 && continue
                        push!(existing_rows, line)
                    end
                end
            end

            open(canonical, "w") do out
                header_written = false
                for shard in shards
                    open(shard, "r") do inp
                        for (i, line) in enumerate(eachline(inp))
                            if i == 1
                                header_written && continue
                                header_written = true
                            end
                            println(out, line)
                        end
                    end
                end
                for row in existing_rows
                    println(out, row)
                end
            end
            for shard in shards
                rm(shard)
            end
            println("  merged $(length(shards)) shards -> $canonical")
        catch e
            @warn "Failed to merge shards into $canonical: $e"
        end
    end
end

# Algorithm types

mutable struct Algorithm
    Name::Symbol
    Parameters::OrderedDict{Symbol, Any}
    Parameters_ranges::OrderedDict{Symbol, Any}
end

mutable struct ProblemData
    name::String
    f::Function
    searchspace::Any
    reference_point::Vector
    num_runs::Any
    empirical_ideal::Union{Vector{Float64}, Nothing}
    empirical_nadir::Union{Vector{Float64}, Nothing}
end

# Problem loaders

function getproblem(id::Int)
    f, conf = HardTestProblems.get_RW_MOP_problem(id)
    problem_name = String(nameof(typeof(f)))
    nadir = conf[:nadir]
    bounds = hcat(conf[:xmin], conf[:xmax])
    if id == 25
        bounds = bounds[[2, 1], :]
    end
    return problem_name, f, bounds, nadir
end

# Algorithm setup

function init_algorithm_structure(Name_algorithm::String)
    Algorithm_structure = Algorithm(:none, OrderedDict(), OrderedDict())
    Algorithm_structure.Name = Symbol(Name_algorithm)
    algorithm_instance = getfield(Metaheuristics, Symbol(Algorithm_structure.Name))
    Algorithm_structure.Parameters = get_default_kwargs(algorithm_instance)
    return Algorithm_structure
end

function get_default_kwargs(algorithm)
    if algorithm == MOEAD_DE
        weights = set_up_weights_MOEAD_DE(2)
        instance = algorithm(weights;)
        return OrderedDict(
            field => getfield(instance.parameters, field)
            for field in fieldnames(typeof(instance.parameters))
            if !(field in MOEAD_DE_INTERNAL_FIELDS)
        )
    else
        try
            instance = algorithm()
            if :parameters in fieldnames(typeof(instance))
                params = getfield(instance, :parameters)
                return OrderedDict(field => getfield(params, field) for field in fieldnames(typeof(params)))
            else
                return OrderedDict()
            end
        catch e
            @warn "Error creating instance for $algorithm: $e"
            return OrderedDict()
        end
    end
end

function initialize_algorithm_structures(alg)
    All_Algorithm_structure = Vector{Algorithm}()
    for searchspace in alg
        Algorithm_structure = detect_searchspaces(searchspace)
        push!(All_Algorithm_structure, Algorithm_structure)
    end
    return All_Algorithm_structure
end

function detect_searchspaces(searchspace::String)
    if !occursin("_searchspace", string(searchspace))
        error("Invalid searchspace string: '$searchspace' — must contain '_searchspace'")
    end
    Algorithm_structure = init_algorithm_structure(string(split(string(searchspace), "_searchspace")[1]))
    current_searchspace = getfield(@__MODULE__, Symbol(searchspace))
    for (key, value) in current_searchspace
        symbol = Symbol(key)
        arr_range = length(value) > 2 ? [value[1], value[2], value[3]] : [value[1], value[2]]
        Algorithm_structure.Parameters_ranges[symbol] = arr_range
    end
    return Algorithm_structure
end

# MOEAD/DE weights

function moead_weight_count(nobjectives::Int, npartitions::Int)
    return binomial(nobjectives + npartitions - 1, nobjectives - 1)
end

function max_npartitions_for_cap(nobjectives::Int, cap::Int)
    npart = 2
    while moead_weight_count(nobjectives, npart + 1) <= cap
        npart += 1
    end
    return npart
end

function set_up_weights_MOEAD_DE(nobjectives::Int, npartitions=nothing)
    if isnothing(npartitions)
        npartitions = DEFAULT_MOEAD_NPARTITIONS
    end
    expected_pop = moead_weight_count(nobjectives, npartitions)
    if expected_pop > MAX_MOEAD_POPULATION
        safe_npart = max_npartitions_for_cap(nobjectives, MAX_MOEAD_POPULATION)
        safe_pop   = moead_weight_count(nobjectives, safe_npart)
        @warn "npartitions=$npartitions gives $expected_pop vectors, capping to $safe_npart"
        npartitions = safe_npart
    end
    return gen_ref_dirs(nobjectives, npartitions)
end

function set_up_algorithm(algorithm_instance, options_dict; params=Dict(), HPO=false, use_ccmo=false, MOEAD_WEIGHTS=nothing, nobjectives::Int=2)
    base_algo = getproperty(Metaheuristics, Symbol(algorithm_instance))

    if use_ccmo
        base_algo = CCMO(base_algo)
    end

    options_kwargs = NamedTuple{Tuple(Symbol.(keys(options_dict)))}(values(options_dict))
    options = Options(; options_kwargs...)

    if algorithm_instance == :MOEAD_DE
        if MOEAD_WEIGHTS === nothing
            MOEAD_WEIGHTS = set_up_weights_MOEAD_DE(nobjectives)
        end
        if HPO
            T   = max(3, round(Int, 0.2  * length(MOEAD_WEIGHTS)))
            n_r = max(2, round(Int, 0.05 * length(MOEAD_WEIGHTS)))
            algo_params = filter(kv -> kv.first != :npartitions, params)
            metaheuristic = base_algo(MOEAD_WEIGHTS; algo_params..., T, n_r, options=options)
        else
            metaheuristic = base_algo(MOEAD_WEIGHTS; options=options)
        end
    else
        kwargs = HPO ? (; params..., options=options) : (; options=options)
        metaheuristic = base_algo(; kwargs...)
    end

    return metaheuristic
end

function set_configuration_optuna(trial, Algorithm_structure, sampler_func, reference_point)
    params   = Dict()
    MOEAD_HP = Dict()
    weights  = nothing
    nobjectives = length(reference_point)

    for (hyperparam, range_vals) in Algorithm_structure.Parameters_ranges
        lb, hb = range_vals[1:2]

        if hyperparam == :npartitions
            hb_safe = max_npartitions_for_cap(nobjectives, MAX_MOEAD_POPULATION)
            lb_clamped = min(max(2, lb), hb_safe)
            hb_clamped = min(hb, hb_safe)
            if hb_clamped < lb_clamped
                hb_clamped = lb_clamped
            end
            MOEAD_HP[hyperparam] = trial.suggest_int(hyperparam, lb_clamped, hb_clamped)
        else
            param_type = typeof(Algorithm_structure.Parameters[hyperparam])
            params[hyperparam] = if param_type == Float64
                trial.suggest_float(hyperparam, lb, hb)
            elseif param_type == Int64
                trial.suggest_int(hyperparam, lb, hb)
            elseif param_type == Bool
                parse(Bool, trial.suggest_categorical(hyperparam, ["false", "true"]))
            else
                error("Unsupported parameter type: $param_type")
            end
        end
    end

    if !isempty(MOEAD_HP)
        weights = set_up_weights_MOEAD_DE(length(reference_point), MOEAD_HP[:npartitions])
        merge!(params, MOEAD_HP)
    end

    return params, weights
end

# DataFrame access

function get_df_column_values(df::DataFrame, column_name::String, array_position::Int, Alg_Name, run_dict)
    println("Reading $(Alg_Name) CSV — column: $(column_name)")

    if !(column_name in names(df))
        error("Column '$(column_name)' not found in CSV. Available columns: $(names(df))")
    end

    column_array = []
    for i in df[!, column_name]
        ismissing(i) && continue
        str = strip(string(i))
        isempty(str) && continue

        try
            array = JSON.parse(str)
            if array_position > length(array)
                @warn "Array position $array_position > array length $(length(array)) for $(Alg_Name)"
                push!(column_array, "Inf")
                continue
            end
            val = array[array_position]
            if val isa String && occursin("Inf", val)
                push!(column_array, "Inf")
            else
                push!(column_array, val)
            end
        catch e
            @warn "Failed to parse JSON for $(Alg_Name): $e (raw: $(first(str, 50))...)"
            push!(column_array, "Inf")
        end
    end

    run_dict[Alg_Name] = column_array
    return run_dict
end

function push_options(options_dataframe::DataFrame)
    return Dict(Symbol(col) => options_dataframe[1, col] for col in names(options_dataframe))
end

function unpack_df_vectors(problem_dataframe::DataFrame, problem_instance::Int)
    return (
        problem_dataframe[!, 1][problem_instance],
        problem_dataframe[!, 2][problem_instance],
        problem_dataframe[!, 3][problem_instance],
        problem_dataframe[!, 4][problem_instance],
    )
end

function unpack_df(problem_dataframe::DataFrame, algorithm_name::String, problem_instance::Int)
    problem_name, problem_function, problem_bounds, problem_ref_point =
        unpack_df_vectors(problem_dataframe, problem_instance)
    problem_algo_run = problem_dataframe[problem_instance, algorithm_name]
    return problem_name, problem_function, problem_bounds, problem_ref_point, problem_algo_run
end

# Filesystem helpers

function create_directories(metaheuristic_str::String, num_ite::Int64, problem_folder_name::String, path::String)
    iter_dir = joinpath(path, metaheuristic_str, string(num_ite))
    problem_dir = joinpath(iter_dir, problem_folder_name)
    mkpath(problem_dir)
    return problem_dir, iter_dir
end

function delete_file(file::String, path::String)
    if file in readdir(path)
        rm(joinpath(path, file))
    end
end

function remove_existing_csv(filepath::String)
    if ispath(filepath)
        println("Removing $(filepath)")
        rm(filepath, recursive=true)
    end
end

# Benchmark loading

function benchmark_handler(All_Algorithm_structure, lower_bound::Int64, upper_bound::Int64, main_script_name;
                           baseline_name::String=main_script_name,
                           baseline_dir::String=".")
    if !(lower_bound < upper_bound && upper_bound <= 50 && lower_bound <= 50)
        error("Invalid bounds")
    end

    problem_dataframe = DataFrame(
        problems_names       = String[],
        problem_function     = Function[],
        problem_bounds       = Any[],
        problem_reference_point = Any[],
    )

    for i in lower_bound:upper_bound
        probl_name, f, bounds, reference_point = getproblem(i)
        push!(problem_dataframe, (
            problems_names          = probl_name,
            problem_function        = f,
            problem_bounds          = bounds,
            problem_reference_point = reference_point,
        ))
    end

    return initialize_runs_dicts(All_Algorithm_structure, problem_dataframe, lower_bound, upper_bound;
                                 baseline_name=baseline_name, baseline_dir=baseline_dir)
end

function initialize_runs_dicts(All_Algorithm_structure, problem_dataframe::DataFrame, lower_bound::Int64, upper_bound::Int64;
                               baseline_name::String="", baseline_dir::String=".")
    for Algorithm_structure in All_Algorithm_structure
        fname = joinpath(baseline_dir, "minimum_runs_$(baseline_name)_$(Algorithm_structure.Name).csv")

        if !isfile(fname)
            error("$fname not found")
        end

        df = with_logger(SimpleLogger(stderr, Logging.Error)) do
            CSV.read(fname, DataFrame; pool=false)
        end

        runs_dicts = Dict()
        runs_dicts = get_df_column_values(df, RUNS_CI_COLUMN, RUNS_CI_POSITION, Algorithm_structure.Name, runs_dicts)
        problem_dataframe[!, Algorithm_structure.Name] = runs_dicts[Algorithm_structure.Name][lower_bound:upper_bound]
    end
    return problem_dataframe
end

# Optimization runner

function run_optimization(problem_data::ProblemData, params,
                          Algorithm_structure, options;
                          MOEAD_WEIGHTS=nothing,
                          empirical_ideal::Union{Vector{Float64}, Nothing}=nothing,
                          empirical_nadir::Union{Vector{Float64}, Nothing}=nothing,
                          collect_fronts::Bool=false)

    All_HV     = Float64[]
    all_fronts = collect_fronts ? Vector{Vector{Vector{Float64}}}() : nothing

    algorithm_instance = Algorithm_structure.Name
    options = copy(options)
    num_ite = options[:iterations]

    nobj = length(problem_data.reference_point)

    if algorithm_instance == :MOEAD_DE && MOEAD_WEIGHTS === nothing
        MOEAD_WEIGHTS = set_up_weights_MOEAD_DE(nobj)
    end

    pop_size = algorithm_instance == :MOEAD_DE ?
        length(MOEAD_WEIGHTS) :
        get(params, :N, Algorithm_structure.Parameters[:N])
    max_evals = pop_size * num_ite
    options[:f_calls_limit] = 3 * max_evals

    num_runs = problem_data.num_runs
    if num_runs isa String
        num_runs = tryparse(Int, num_runs)
        if isnothing(num_runs) || num_runs > 100
            println("Problem $(problem_data.name): min_runs unparseable or >100, using fallback=$FALLBACK_RUNS")
            num_runs = FALLBACK_RUNS
        end
    elseif num_runs isa Number
        num_runs = Int(floor(num_runs))
        if num_runs > 100
            println("Problem $(problem_data.name): min_runs=$num_runs is too high, using fallback=$FALLBACK_RUNS")
            num_runs = FALLBACK_RUNS
        end
    else
        num_runs = FALLBACK_RUNS
    end

    has_bounds = !isnothing(empirical_ideal) && !isnothing(empirical_nadir)

    if has_bounds
        for i in 1:num_runs
            println("Starting task... run $i / $num_runs for $(problem_data.name) with $algorithm_instance")
            options[:seed] = abs(hash((algorithm_instance, problem_data.name, i))) % 1_000_000
            metaheuristic = set_up_algorithm(algorithm_instance, options; params, HPO=true, MOEAD_WEIGHTS, nobjectives=nobj)

            status = with_logger(SimpleLogger(stderr, Logging.Error)) do
                optimize(problem_data.f, problem_data.searchspace, metaheuristic)
            end
            println("  Run $i/$num_runs done — $(length(status.population)) solutions, $(count(is_feasible, status.population)) feasible")

            front = get_feasible_non_dominated(status.population)
            status = nothing
            metaheuristic = nothing
            GC.gc()
            @static if Sys.islinux()
                ccall(:malloc_trim, Cint, (Cint,), 0)
            end

            hv = isempty(front) ? 0.0 : normalized_hypervolume(front, empirical_ideal, empirical_nadir)
            push!(All_HV, hv)
            collect_fronts && push!(all_fronts, front)
        end
    else
        all_run_fronts = Vector{Vector{Vector{Float64}}}()
        for i in 1:num_runs
            println("Starting task... run $i / $num_runs for $(problem_data.name) [DEAD] with $algorithm_instance")
            options[:seed] = abs(hash((algorithm_instance, problem_data.name, i))) % 1_000_000
            metaheuristic = set_up_algorithm(algorithm_instance, options; params, HPO=true, MOEAD_WEIGHTS, nobjectives=nobj)

            status = with_logger(SimpleLogger(stderr, Logging.Error)) do
                optimize(problem_data.f, problem_data.searchspace, metaheuristic)
            end
            println("  Run $i/$num_runs done — $(length(status.population)) solutions, $(count(is_feasible, status.population)) feasible")

            push!(all_run_fronts, get_feasible_non_dominated(status.population))
            status = nothing
            metaheuristic = nothing
            GC.gc()
            @static if Sys.islinux()
                ccall(:malloc_trim, Cint, (Cint,), 0)
            end
        end

        trial_ideal, _ = compute_empirical_bounds(all_run_fronts)
        library_nadir = Vector{Float64}(problem_data.reference_point)

        if isnothing(trial_ideal)
            All_HV = fill(0.0, num_runs)
        else
            println("  DEAD problem found feasible solutions! ideal=$trial_ideal, nadir(library)=$library_nadir")
            for front in all_run_fronts
                hv = isempty(front) ? 0.0 : normalized_hypervolume(front, trial_ideal, library_nadir)
                push!(All_HV, hv)
            end
        end

        if collect_fronts
            all_fronts = all_run_fronts
        end
        all_run_fronts = nothing
    end

    mean_hv = isempty(All_HV) ? 0.0 : mean(All_HV)
    return mean_hv, All_HV, all_fronts
end

# Optuna driver

function objective(trial, sampler_func, Algorithm_structure, problem_data::ProblemData, main_script_name, options;
                  fronts_csv_path::Union{String,Nothing}=nothing, sampler_name::String="", problem_instance::Int=0)

    params, weights = set_configuration_optuna(trial, Algorithm_structure, sampler_func, problem_data.reference_point)
    println("  Trial $(trial.number): $(problem_data.name) | $(Algorithm_structure.Name) | params=$(params)")

    do_collect = !isnothing(fronts_csv_path)

    trial_elapsed = @elapsed begin
        mean_hv, All_HV, trial_fronts = run_optimization(problem_data, params,
                                            Algorithm_structure,
                                            options; MOEAD_WEIGHTS=weights,
                                            empirical_ideal=problem_data.empirical_ideal,
                                            empirical_nadir=problem_data.empirical_nadir,
                                            collect_fronts=do_collect)
    end

    if do_collect && !isnothing(trial_fronts) && !isempty(trial_fronts)
        try
            save_trial_fronts(fronts_csv_path, trial_fronts,
                              Algorithm_structure.Name, sampler_name,
                              problem_instance, trial.number)
        catch e
            @warn "Failed to save trial fronts: $e"
        end
        trial_fronts = nothing
    end

    params_json = JSON.json(Dict(string(k) => v for (k, v) in params))
    all_hv_json = JSON.json(All_HV)
    trial.set_user_attr("problem_name", problem_data.name)
    trial.set_user_attr("params", params_json)
    trial.set_user_attr("elapsed_seconds", trial_elapsed)

    if mean_hv <= 0.0 && isempty(All_HV)
        GC.gc()
        @static if Sys.islinux()
            ccall(:malloc_trim, Cint, (Cint,), 0)
        end
        return -Inf
    end

    trial.set_user_attr("All_HV", all_hv_json)

    GC.gc()
    @static if Sys.islinux()
        ccall(:malloc_trim, Cint, (Cint,), 0)
    end
    try
        pyimport("gc").collect()
    catch
    end

    return mean_hv
end

function run_trial(sampler_instance::Int, Algorithm_structure, sampler_vector, results_path::String, options, problem_instance, problem_dataframe, main_script_name, n_trials, empirical_bounds_dict;
                   collect_fronts::Bool=false, storage_dir::Union{String, Nothing}=nothing)

    problem_name, f, searchspace, reference_point, num_runs =
        unpack_df(problem_dataframe, String(Algorithm_structure.Name), problem_instance)

    emp_ideal, emp_nadir = if haskey(empirical_bounds_dict, problem_instance)
        empirical_bounds_dict[problem_instance]
    else
        @warn "no bounds for problem $problem_instance, HV will be 0"
        (nothing, nothing)
    end

    problem_data = ProblemData(problem_name, f, searchspace, reference_point, num_runs, emp_ideal, emp_nadir)

    sampler_name = sampler_vector[sampler_instance]
    println("sampler_name :: $sampler_name")
    sampler_module = optuna.samplers
    sampler_func   = getproperty(sampler_module, Symbol(sampler_name))

    sampler_constructor = sampler_name == "GridSampler" ?
        sampler_func(Algorithm_structure.Parameters_ranges) :
        sampler_func()

    study_kwargs = Dict(
        :study_name => "$(Algorithm_structure.Name)_$(sampler_name)_Problem_$(problem_instance)",
        :direction  => "maximize",
        :sampler    => sampler_constructor,
    )
    if !isnothing(storage_dir)
        mkpath(storage_dir)
        db_name = "study_$(Algorithm_structure.Name)_$(sampler_name)_Problem_$(problem_instance).db"
        db_path = joinpath(storage_dir, db_name)
        study_kwargs[:storage] = "sqlite:///$db_path"
        study_kwargs[:load_if_exists] = true
        println("  Optuna storage: $db_path")
    end

    println("sampler_func: ", sampler_func)
    study = optuna.create_study(; study_kwargs...)

    num_ite = options[:iterations]

    fronts_csv_path = nothing
    if collect_fronts
        problem_folder_name = "Problem_$(problem_instance)_$(problem_name)"
        problem_dir, _ = create_directories(String(Algorithm_structure.Name), num_ite, problem_folder_name, results_path)
        fronts_csv_path = joinpath(problem_dir, "all_fronts_$(Algorithm_structure.Name)_$(sampler_name)_Problem_$(problem_instance).csv")
    end

    trial_state_module = pyimport("optuna.trial")
    countable_states = [trial_state_module.TrialState.COMPLETE,
                        trial_state_module.TrialState.PRUNED]
    completed_trial_count = length(study.get_trials(deepcopy=false, states=countable_states))
    remaining_trials = max(0, n_trials - completed_trial_count)

    if remaining_trials == 0
        println("  Study already has $completed_trial_count completed/pruned trials (target=$n_trials), skipping optimize")
        study_elapsed = 0.0
    else
        if completed_trial_count > 0
            println("  Resuming study: $completed_trial_count already done, running $remaining_trials more (target=$n_trials)")
        end
        study_elapsed = @elapsed study.optimize(
            trial -> objective(trial, sampler_func, Algorithm_structure, problem_data, main_script_name, options;
                               fronts_csv_path=fronts_csv_path, sampler_name=sampler_name, problem_instance=problem_instance),
            n_trials = remaining_trials,
        )
        println("  Study completed in $(round(study_elapsed, digits=1))s for $problem_name ($(Algorithm_structure.Name) / $sampler_name)")
    end

    if isnan(study.best_value) || study.best_value == -Inf || !haskey(study.best_trial.user_attrs, "All_HV")
        println("No valid result for $problem_name")
        study = nothing
        GC.gc()
        pyimport("gc").collect()
        return nothing
    end

    all_hv_raw    = study.best_trial.user_attrs["All_HV"]
    all_hv_parsed = all_hv_raw isa AbstractString ? JSON.parse(all_hv_raw) : all_hv_raw
    All_HV        = collect(Float64, all_hv_parsed)
    sampler_class = study[:sampler][:__class__][:__name__]
    best_value    = Float64(study.best_value)
    best_params   = Dict(study.best_params)

    param_importances = Dict{String, Float64}()
    try
        importance_module = pyimport("optuna.importance")
        importances = importance_module.get_param_importances(study)
        for (k, v) in importances
            param_importances[string(k)] = Float64(v)
        end
        println("  Param importances: $param_importances")
    catch e
        @warn "Could not extract param importances for $problem_name: $e"
    end

    trial_values = Float64[]
    for t in study.trials
        try
            val = Float64(t.value)
            push!(trial_values, isnan(val) ? -Inf : val)
        catch
            push!(trial_values, -Inf)
        end
    end
    best_so_far = accumulate(max, trial_values)

    study = nothing
    GC.gc()
    pyimport("gc").collect()

    result = (
        algorithm_name    = Algorithm_structure.Name,
        sampler           = sampler_class,
        problem_name      = problem_name,
        problem_instance  = problem_instance,
        hv_value          = best_value,
        params            = best_params,
        All_HV            = All_HV,
        num_runs          = length(All_HV),
        empirical_ideal   = emp_ideal,
        empirical_nadir   = emp_nadir,
        trial_values      = trial_values,
        best_so_far       = best_so_far,
        elapsed_seconds   = study_elapsed,
        param_importances = param_importances,
    )

    try
        write_single_result(result, options, results_path)
    catch e
        @warn "Failed to write per-task summary for $problem_name ($sampler_name): $e"
    end

    return result
end

function init_parallel_arrays(sampler_vector, problem_instances::UnitRange{Int64})
    algo_instances = 1:length(sampler_vector)
    problem_instances_array = [prob_i for prob_i in problem_instances for _ in algo_instances]
    sampler_instances_array = vcat([algo_instances for _ in problem_instances]...)
    return problem_instances_array, sampler_instances_array
end

function run_HPO(sampler_vector, options, results_path, All_Algorithm_structure, problem_dataframe, main_script_name, n_trials, empirical_bounds_dict;
                 collect_fronts::Bool=false, storage_dir::Union{String, Nothing}=nothing)
    problem_instances = 1:nrow(problem_dataframe)
    problem_instances_array, sampler_instances_array = init_parallel_arrays(sampler_vector, problem_instances)

    all_tasks = [
        (alg, s, p)
        for alg in All_Algorithm_structure
        for (s, p) in zip(sampler_instances_array, problem_instances_array)
    ]

    log_path = joinpath(results_path, "log_$main_script_name.txt")
    mkpath(dirname(log_path))
    if isfile(log_path)
        rm(log_path)
    end

    open(log_path, "a") do io
        println(io, "Total tasks: $(length(all_tasks))")
        for (alg, s, p) in all_tasks
            println(io, "  $(alg.Name) | sampler=$s | problem=$p")
        end
    end

    results_flat = pmap(all_tasks) do (Algorithm_structure, sampler_instance, prob)
        try
            println("Currently Testing : $(Algorithm_structure.Name) | problem=$prob")
            result = run_trial(sampler_instance, Algorithm_structure, sampler_vector,
                    results_path, options, prob, problem_dataframe, main_script_name, n_trials, empirical_bounds_dict;
                    collect_fronts=collect_fronts, storage_dir=storage_dir)
            GC.gc()
            return result
        catch e
            @warn "Task failed for $(Algorithm_structure.Name), prob=$prob: $e"
            nothing
        end
    end

    if collect_fronts
        println("Merging per-worker front shards...")
        merge_worker_fronts(results_path)
    end

    println("Merging per-worker summary-CSV shards...")
    merge_result_shards(results_path)

    return results_flat
end

# Result writing

function write_single_result(r, options_dict, results_path)
    isnothing(r) && return
    iteration_counts = options_dict[:iterations]

    println("$(r.algorithm_name) :: $(r.problem_name): value = $(r.hv_value), params = $(r.params)")

    params_str = JSON.json(Dict(string(k) => v for (k, v) in r[:params]))

    result_df = DataFrame(
        algorithm_name    = [r[:algorithm_name]],
        sampler           = [r[:sampler]],
        problem_instance  = [r[:problem_instance]],
        problem_name      = [r[:problem_name]],
        hv_value          = [r[:hv_value]],
        params            = [params_str],
        num_runs          = [r[:num_runs]],
        elapsed_seconds   = [round(r[:elapsed_seconds], digits=2)],
        param_importances = [JSON.json(r[:param_importances])],
        All_HV            = [JSON.json(r[:All_HV])],
        empirical_ideal   = [isnothing(r[:empirical_ideal]) ? "nothing" : JSON.json(r[:empirical_ideal])],
        empirical_nadir   = [isnothing(r[:empirical_nadir]) ? "nothing" : JSON.json(r[:empirical_nadir])],
    )

    problem_folder_name = "Problem_$(r[:problem_instance])_$(r.problem_name)"
    _, iter_dir = create_directories(String(r[:algorithm_name]), iteration_counts, problem_folder_name, results_path)

    canonical_name = "$(r[:algorithm_name])_$(r.sampler).csv"
    canonical_path = joinpath(iter_dir, canonical_name)
    shard_path     = _per_worker_path(canonical_path)

    write_header = !isfile(shard_path)
    CSV.write(shard_path, result_df; append=true, writeheader=write_header)

    if hasproperty(r, :trial_values) && !isempty(r.trial_values)
        conv_df = DataFrame(
            problem_instance = fill(r[:problem_instance], length(r.trial_values)),
            trial            = 1:length(r.trial_values),
            hv_value         = r.trial_values,
            best_so_far      = r.best_so_far,
        )
        conv_canonical_name = "convergence_$(r[:algorithm_name])_$(r.sampler).csv"
        conv_canonical_path = joinpath(iter_dir, conv_canonical_name)
        conv_shard_path     = _per_worker_path(conv_canonical_path)

        conv_write_header = !isfile(conv_shard_path)
        CSV.write(conv_shard_path, conv_df; append=true, writeheader=conv_write_header)
    end
end

function write_HPO_data_into_csv(results, options_dict, results_path)
    for item in results
        if item isa AbstractArray
            for r in item
                write_single_result(r, options_dict, results_path)
            end
        else
            write_single_result(item, options_dict, results_path)
        end
    end
end
