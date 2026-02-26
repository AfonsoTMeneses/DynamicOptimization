include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
using DataStructures
using Metaheuristics
using Metaheuristics: pareto_front
using HardTestProblems
using DataFrames


# ─────────────────────────────────────────────
# Reference point offsets for benchmark problems
# ─────────────────────────────────────────────

const ref_points_offset = Dict(
    #1 => [370000, -7330],
    2  => [1, 1000],
    3  => [2, 10000],
    4  => [40, 0.2],
    7  => [10, 100],
    9  => [3200, 0.05],
    11 => [93450.511, 2000, 4.853469e6, 9.620032e6, 25000.0],
    12 => [1200, 0.1],
    13 => [6500, 1300.0, 2000],
    14 => [2, 0.02],
    15 => [35, 190000],
    16 => [20, 0.0020408763],
    17 => [1000000, 1000000, 10000],
    20 => [10000, 5e-5],
    #22 => [-9000.90871, -20000.0],
    24 => [8000, 400000, 100000000],
    27 => [1, 0],
    28 => [300, 50],
    29 => [10, 25],
    30 => [10, 30],
    31 => [10, 30],
    32 => [10, 30],
    33 => [10, 30],
    34 => [10, 30],
    35 => [5, 30],
    36 => [1000, 3000],
    37 => [1000, 3000],
    38 => [3000, 3000],
    39 => [1000, 3000, 3000],
    40 => [40, 40],
    41 => [40, 40, 40],
    42 => [40, 40],
    43 => [40, 40],
    44 => [40, 40, 60],   #[5.720257, 1.0231384, 4.0471196]
    45 => [40, 40, 40],
    46 => [80, 80, 80, 80], #[5.7381081, 2.3578333, 8.1909894, 3.9032935]
    47 => [100, 100],
    48 => [100, 100],
    49 => [100, 100, 100],
    50 => [100000, 2000]
)


# ─────────────────────────────────────────────
# Structs
# ─────────────────────────────────────────────

mutable struct Algorithm
    Name::Symbol
    Parameters::OrderedDict{Symbol, Any}
    Parameters_ranges::OrderedDict{Symbol, Any}
end

mutable struct Optimization_configuration
    lb_instances::Int
    hb_instance::Int
    max_trials::Int
end

mutable struct ProblemData
    name::String
    f::Function
    searchspace::Any
    reference_point::Vector
    num_runs::Any
end


# ─────────────────────────────────────────────
# Problem loading
# ─────────────────────────────────────────────

function getproblem(id::Int)
    f, conf = HardTestProblems.get_RW_MOP_problem(id)
    problem_name = String(nameof(typeof(f)))
    reference_point = conf[:nadir]
    bounds = hcat(conf[:xmin], conf[:xmax])
    if id == 25
        bounds = bounds[[2, 1], :]
    end
    return problem_name, f, bounds, reference_point
end


# ─────────────────────────────────────────────
# Algorithm initialisation
# ─────────────────────────────────────────────

function init_algorithm_structure(Name_algorithm::String)
    Algorithm_structure = Algorithm(:none, OrderedDict(), OrderedDict())
    Algorithm_structure.Name = Symbol(Name_algorithm)
    algorithm_instance = getfield(Metaheuristics, Symbol(Algorithm_structure.Name))
    Algorithm_structure.Parameters = get_default_kwargs(algorithm_instance)
    return Algorithm_structure
end

function get_default_kwargs(algorithm)
    if algorithm == MOEAD_DE
        weights = set_up_weights_MOEAD_DE()
        instance = algorithm(weights;)
        return OrderedDict(fields => getfield(instance.parameters, fields) for fields in fieldnames(algorithm))
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
    # FIX (minor): typed Vector{Algorithm} instead of Any[]
    All_Algorithm_structure = Vector{Algorithm}()
    for searchspace in alg
        Algorithm_structure = detect_searchspaces(searchspace)
        push!(All_Algorithm_structure, Algorithm_structure)
    end
    return All_Algorithm_structure
end

function detect_searchspaces(searchspace::String)
    if occursin("_searchspace", string(searchspace))
        Algorithm_structure = init_algorithm_structure(string(split(string(searchspace), "_searchspace")[1]))
        current_searchspace = getfield(@__MODULE__, Symbol(searchspace))
        for (key, value) in current_searchspace
            symbol = Symbol(key)
            arr_range = length(value) > 2 ? [value[1], value[2], value[3]] : [value[1], value[2]]
            Algorithm_structure.Parameters_ranges[symbol] = arr_range
        end
    end
    return Algorithm_structure
end


# ─────────────────────────────────────────────
# Weight / algorithm setup
# ─────────────────────────────────────────────

function set_up_weights_MOEAD_DE(nobjectives=nothing, npartitions=nothing)
    if isnothing(nobjectives) && isnothing(npartitions)
        nobjectives = 2
        npartitions = 50
    end
    return gen_ref_dirs(nobjectives, npartitions)
end

function set_up_algorithm(algorithm_instance, options_dict; params=Dict(), HPO=false, CCMO=false, MOEAD_WEIGHTS=nothing)
    base_algo = getproperty(Metaheuristics, Symbol(algorithm_instance))

    if CCMO
        base_algo = CCMO(base_algo)
    end

    options_kwargs = NamedTuple{Tuple(Symbol.(keys(options_dict)))}(values(options_dict))
    options = Options(; options_kwargs...)

    if algorithm_instance == :MOEAD_DE
        if MOEAD_WEIGHTS === nothing
            MOEAD_WEIGHTS = set_up_weights_MOEAD_DE()
        end
        if HPO
            T   = max(3, round(Int, 0.2  * length(MOEAD_WEIGHTS)))
            n_r = max(2, round(Int, 0.05 * length(MOEAD_WEIGHTS)))
            metaheuristic = base_algo(MOEAD_WEIGHTS; params..., T, n_r, options=options)
        else
            metaheuristic = base_algo(MOEAD_WEIGHTS; options=options)
        end
    else
        kwargs = HPO ? (; params..., options=options) : (; options=options)
        metaheuristic = base_algo(; kwargs...)
    end

    return metaheuristic
end


# ─────────────────────────────────────────────
# Optuna configuration
# ─────────────────────────────────────────────

function set_configuration_optuna(trial, Algorithm_structure, sampler_func, reference_point)
    params   = Dict()
    MOEAD_HP = Dict()
    weights  = nothing

    for (hyperparam, range_vals) in Algorithm_structure.Parameters_ranges
        lb, hb = range_vals[1:2]

        if hyperparam == :npartitions
            MOEAD_HP[hyperparam] = trial.suggest_int(hyperparam, lb, hb)
        else
            param_type = typeof(Algorithm_structure.Parameters[hyperparam])
            params[hyperparam] = if param_type == Float64
                if !(sampler_func == optuna.samplers.BruteForceSampler)
                    trial.suggest_float(hyperparam, lb, hb)
                else
                    step = range_vals[3]
                    trial.suggest_float(hyperparam, lb, hb, step=step)
                end
            elseif param_type == Int64
                trial.suggest_int(hyperparam, lb, hb)
            elseif param_type == Bool
                trial.suggest_categorical(hyperparam, ["false", "true"])
            else
                error("Unsupported parameter type: $param_type")
            end
        end
    end

    if !isempty(MOEAD_HP)
        weights = set_up_weights_MOEAD_DE(length(reference_point), MOEAD_HP[:npartitions])
    end

    return params, weights
end


# ─────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────

function get_df_column_values(df::DataFrame, column::Int, array_position::Int, Alg_Name, run_dict)
    if 1 ≤ column ≤ length(names(df))
        column_array = []
        for i in df[2:end, column]
            if !ismissing(i)
                str = String(i)
                if !occursin("Inf", str)
                    array = eval(Meta.parse(str))
                    push!(column_array, array[array_position])
                else
                    push!(column_array, i)
                end
            end
        end
    else
        error("Invalid column index: $column")
    end
    run_dict[Alg_Name] = column_array
    return run_dict
end

function push_options(options_dataframe::DataFrame)
    return Dict(Symbol(col) => options_dataframe[1, col] for col in names(options_dataframe))
end

function unpack_df_vectors(problem_dataframe::DataFrame, problem_instance::Int)
    problem_name_vector     = problem_dataframe[!, 1]
    problem_function_vector = problem_dataframe[!, 2]
    problem_bounds          = problem_dataframe[!, 3]
    problem_ref_point       = problem_dataframe[!, 4]

    return (
        problem_name_vector[problem_instance],
        problem_function_vector[problem_instance],
        problem_bounds[problem_instance],
        problem_ref_point[problem_instance]
    )
end

function unpack_df(problem_dataframe::DataFrame, algorithm_name::String, problem_instance::Int)
    problem_name, problem_function, problem_bounds, problem_ref_point =
        unpack_df_vectors(problem_dataframe, problem_instance)
    problem_algo_run = problem_dataframe[problem_instance, algorithm_name]
    return problem_name, problem_function, problem_bounds, problem_ref_point, problem_algo_run
end


# ─────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────

function create_directories(metaheuristic_str::String, num_ite::Int64, problem_folder_name::String, path::String)
    algorithm_dir = joinpath(string(path), metaheuristic_str)
    mkpath(algorithm_dir)
    iter_dir = joinpath(algorithm_dir, string(num_ite))
    mkpath(iter_dir)
    problem_dir = joinpath(iter_dir, problem_folder_name)
    mkpath(problem_dir)
    return problem_dir, iter_dir
end

function delete_file(file::String, path::String)
    if file in readdir(path)
        rm(joinpath(path, file))
    end
end


# ─────────────────────────────────────────────
# Benchmark / run initialisation
# ─────────────────────────────────────────────

function benchmark_handler(All_Algorithm_structure, lower_bound::Int64, upper_bound::Int64, main_script_name)
    if !(lower_bound < upper_bound && upper_bound <= 50 && lower_bound <= 50)
        error("Invalid bounds")
    end

    problem_dataframe = DataFrame(
        problems_names       = String[],
        problem_function     = Function[],
        problem_bounds       = Any[],
        problem_reference_point = Any[]
    )

    for i in lower_bound:upper_bound
        probl_name, f, bounds, reference_point = getproblem(i)
        if haskey(ref_points_offset, i)
            reference_point = ref_points_offset[i]
        end
        push!(problem_dataframe, (
            problems_names          = probl_name,
            problem_function        = f,
            problem_bounds          = bounds,
            problem_reference_point = reference_point
        ))
    end

    problem_dataframe = initialize_runs_dicts(All_Algorithm_structure, main_script_name, problem_dataframe, lower_bound, upper_bound)
    return problem_dataframe
end

function initialize_runs_dicts(All_Algorithm_structure, main_script_name, problem_dataframe::DataFrame, lower_bound::Int64, upper_bound::Int64)
    for Algorithm_structure in All_Algorithm_structure
        fname = "minimum_runs_$(main_script_name)_$(Algorithm_structure.Name).csv"
        if isfile(fname)
            df = CSV.read(fname, DataFrame; header=false)
        else
            error("$fname not found in the current folder. Please switch to the correct folder or run get_minimum_runs() to generate it.")
        end
        runs_dicts = Dict()
        runs_dicts = get_df_column_values(df, 6, 10, Algorithm_structure.Name, runs_dicts)
        problem_dataframe[!, Algorithm_structure.Name] = runs_dicts[Algorithm_structure.Name][lower_bound:upper_bound]
    end
    return problem_dataframe
end


# ─────────────────────────────────────────────
# Core optimisation
# ─────────────────────────────────────────────

function run_optimization(problem_data::ProblemData, params,
                          Algorithm_structure, result_dir::String, options; MOEAD_WEIGHTS=nothing)

    All_HV            = Float64[]
    hv_values         = Dict()
    all_pareto_fronts = Dict()

    algorithm_instance = Algorithm_structure.Name
    num_ite   = options[:iterations]
    pop_size  = Algorithm_structure.Parameters[:N]
    max_evals = pop_size * num_ite
    options[:f_calls_limit] = 3 * max_evals

    log_path = joinpath(dirname(result_dir), "log.txt")
    open(log_path, "a") do io
        println(io, "Optimizing problem:: $(problem_data.name)",
                    " reference_point:: $(problem_data.reference_point)",
                    " num_runs:: $(problem_data.num_runs)",
                    " algorithm_instance:: $algorithm_instance")
    end

    metaheuristic = set_up_algorithm(algorithm_instance, options; params, HPO=true, MOEAD_WEIGHTS)

    if problem_data.num_runs == "Inf" || problem_data.num_runs > 100
        println("Skipping problem $(problem_data.name) due to invalid run length.")
        return -Inf, -Inf, -Inf
    end

    for i in 1:problem_data.num_runs
        println("Starting task... run $i / $(problem_data.num_runs) for $(problem_data.name)")
        status = optimize(problem_data.f, problem_data.searchspace, metaheuristic)
        display(status)
        println("Task Finished...")
        approx_front = get_non_dominated_solutions(status.population)
        push!(All_HV, hypervolume(approx_front, problem_data.reference_point))
        front_objectives = get_non_dominated_solutions([sol.f for sol in approx_front])
        all_pareto_fronts[Symbol("Run_$(i)_$(problem_data.name)")] = front_objectives
    end

    hv_values[num_ite] = mean(All_HV)
    return hv_values, All_HV, all_pareto_fronts
end

function objective(trial, sampler_func, Algorithm_structure, result_dir, problem_data::ProblemData, options)
    params, weights = set_configuration_optuna(trial, Algorithm_structure, sampler_func, problem_data.reference_point)

    hv_value, All_HV, all_pareto_fronts = run_optimization(problem_data, params,
                                                            Algorithm_structure,
                                                            result_dir, options; MOEAD_WEIGHTS=weights)

    if hv_value == -Inf && All_HV == -Inf
        return -Inf
    end

    isempty(hv_value) && return -Inf

    hv_max = maximum(values(hv_value))
    trial.set_user_attr("problem_name", problem_data.name)
    trial.set_user_attr("PF",           all_pareto_fronts)
    trial.set_user_attr("All_HV",       All_HV)

    return hv_max
end


# ─────────────────────────────────────────────
# Trial runner
# ─────────────────────────────────────────────

function run_trial(sampler_instance::Int, Algorithm_structure, sampler_vector, result_dir::String, options, problem_instance, problem_dataframe, n_trials)
    
    problem_name, f, searchspace, reference_point, num_runs =
        unpack_df(problem_dataframe, String(Algorithm_structure.Name), problem_instance)

    problem_data = ProblemData(problem_name, f, searchspace, reference_point, num_runs)

    sampler_name = sampler_vector[sampler_instance]
    println("sampler_name :: $sampler_name")
    sampler_module = optuna.samplers
    sampler_func   = getproperty(sampler_module, Symbol(sampler_name))

    sampler_constructor = sampler_name == "GridSampler" ?
        sampler_func(Algorithm_structure.Parameters_ranges) :
        sampler_func()

    println("sampler_func: ", sampler_func)
    study = optuna.create_study(
        study_name = problem_name,
        direction  = "maximize",
        sampler    = sampler_constructor
    )

    num_ite = options[:iterations]

    study.optimize(
        trial -> objective(trial, sampler_func, Algorithm_structure, result_dir, problem_data, options),
        n_trials = n_trials
    )

    if isnan(study.best_value) || study.best_value == -Inf || !haskey(study.best_trial.user_attrs, "All_HV")
        println("No valid result for $problem_name")
        return nothing
    end

    All_HV  = study.best_trial.user_attrs["All_HV"]
    PF_best = study.best_trial.user_attrs["PF"]

    opt_results_df = DataFrame(
        algorithm_name = Symbol[],
        sampler        = String[],
        solutions      = Vector{Any}[],
    )
    sampler_class = study[:sampler][:__class__][:__name__]
    for (key, PF) in PF_best
        push!(opt_results_df, (algorithm_name=Symbol(key), sampler=sampler_class, solutions=PF))
    end

    problem_folder_name = "Problem_$(problem_instance)_$(problem_name)"
    problem_dir, _ = create_directories(String(Algorithm_structure.Name), num_ite, problem_folder_name, result_dir)

    CSV_NAME = "$(problem_name)_$(problem_instance)_$(sampler_class)_$(Algorithm_structure.Name)_obtained_solutions.csv"
    CSV.write(joinpath(problem_dir, CSV_NAME), opt_results_df)

    return (
        algorithm_name   = Algorithm_structure.Name,
        sampler          = sampler_class,
        problem_name     = problem_name,
        problem_instance = problem_instance,
        hv_value         = study.best_value,
        params           = study.best_params,
    )
end


# ─────────────────────────────────────────────
# HPO runner
# ─────────────────────────────────────────────

function init_parallel_arrays(sampler_vector, problem_instances::UnitRange{Int64})
    algo_instances = 1:length(sampler_vector)
    problem_instances_array = [prob_i for prob_i in problem_instances for _ in algo_instances]
    sampler_instances_array = vcat([algo_instances for _ in problem_instances]...)
    return problem_instances_array, sampler_instances_array
end

function run_HPO(sampler_vector, options, result_dir, All_Algorithm_structure, problem_dataframe, main_script_name, n_trials)
    problem_instances = 1:nrow(problem_dataframe)
    problem_instances_array, sampler_instances_array = init_parallel_arrays(sampler_vector, problem_instances)

    problem_dataframe = initialize_runs_dicts(All_Algorithm_structure, main_script_name, problem_dataframe, 1, nrow(problem_dataframe))
    println(problem_dataframe)

    results = @distributed (vcat) for Algorithm_structure in collect(All_Algorithm_structure)
        println("Currently Testing : $(Algorithm_structure.Name)")
        task = (sampler_instance, prob) -> run_trial(sampler_instance, Algorithm_structure, sampler_vector, result_dir, options, prob, problem_dataframe, n_trials)
        pmap_results = pmap(task, sampler_instances_array, problem_instances_array)
        [pmap_results]
    end

    return results
end


# ─────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────


function remove_existing_csv(filepath::String)
    if ispath(filepath)
        println("Removing $(filepath)")
        rm(filepath, recursive=true)
    end
end

function write_HPO_data_into_csv(results, options_dict, results_path)
    for alg_results in results
        for r in alg_results
            if !isnothing(r)
                result_df = DataFrame(
                    algorithm_name   = Symbol[],
                    sampler          = String[],
                    problem_instance = Int[],
                    problem_name     = String[],
                    hv_value         = Float64[],
                    params           = String[],
                )

                iteration_counts = options_dict[:iterations]

                println("$(r.algorithm_name) :: $(r.problem_name): value = $(r.hv_value), params = $(r.params)")

                params_str = string(r[:params])
                if occursin("Dict{Any, Any}", params_str)
                    params_str = replace(params_str, r"Dict\{Any, Any\}\(" => "", ")" => "", "\"" => "", "=>" => "=")
                end
                println("params_str:: $params_str")

                push!(result_df, (r[:algorithm_name], r[:sampler], r[:problem_instance],
                                  r[:problem_name], r[:hv_value], params_str))

                problem_folder_name = "Problem_$(r[:problem_instance])_$(r.problem_name)"
                problem_dir, iter_dir = create_directories(String(r[:algorithm_name]), iteration_counts, problem_folder_name, results_path)

                CSV_NAME = "$(r[:algorithm_name])_$(r.sampler).csv"
                csv_path = joinpath(iter_dir, CSV_NAME)
                
                write_header = !isfile(csv_path)
                
                if occursin(String(r.sampler), CSV_NAME)
                    CSV.write(csv_path, result_df, append=true, writeheader=write_header)
                end
            end
        end
    end
end



