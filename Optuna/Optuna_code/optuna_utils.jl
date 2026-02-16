

include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
using DataStructures
using Metaheuristics
using Metaheuristics: pareto_front
using HardTestProblems
using DataFrames



    ref_points_offset = Dict(
        #1 => [370000, -7330],
        2 => [1, 1000], 
        4 => [40, 0.2],
        7 => [4, 50],
        9 =>[3200, 0.05],
        11 =>  [93450.511, 2000, 4.853469e6, 9.620032e6, 25000.0],
        12 => [1200, 0.1],
        13 => [6500, 1300.0, 2000],
        14 => [2, 0.02],
        15 => [35, 190000],
        16 => [20, 0.0020408763],
        17 => [1000000, 1000000, 10000],
        20 =>[10000, 5e-5],
        #22 => [-9000.90871, -20000.0],
        24 => [8000, 400000, 100000000], #--
        27 => [1, 0],
        28 => [300, 50],
        29 => [10, 25],
        30 => [10, 30],
        31 => [10, 30],
        32 => [10, 30],
        33 => [10, 30], # -- 
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
        44 => [40, 40, 60], #[5.720257, 1.0231384, 4.0471196]
        45 => [40, 40, 40],
        46 => [80, 80, 80, 80],  #[5.7381081, 2.3578333, 8.1909894, 3.9032935] 
        47 => [100, 100],
        48 => [100, 100],
        49 => [100, 100, 100],
        50 => [100000, 2000]
    )



    
mutable struct Algorithm
    Name::Symbol
    Parameters::OrderedDict{Symbol, Any}
    Parameters_ranges::OrderedDict{Symbol, Any}    
end

mutable struct Optimization_configuration
    lb_instaces::Int
    hb_instace::Int
    max_trials::Int
end

mutable struct Problem


end




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
        instance = algorithm(weights;) # TODO:: Melhorar estes unecessary checks 
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
        catch err
            #println("Error creating instance: ", err)
            return OrderedDict()
        end
    end
end

function create_directories(metaheuristic_str::String, iteration_counts::Int64, problem_folder_name::String, path::String)
    algorithm_dir = joinpath(string(path), metaheuristic_str)
    mkpath(algorithm_dir)

    iter_dir = joinpath(algorithm_dir, string(iteration_counts))
    mkpath(iter_dir)

    problem_dir = joinpath(iter_dir, problem_folder_name)
    mkpath(problem_dir)

    return problem_dir, iter_dir
end

function detect_searchspaces(searchspace::String)
    if occursin("_searchspace", string(searchspace))
        Algorithm_structure = init_algorithm_structure(string(split(string(searchspace), "_searchspace")[1]))
        current_searchspace = getfield(@__MODULE__, Symbol(searchspace))
        #println( "Algorithm to be used for optimization::: $(Algorithm_structure.Name)")
        #println("Searchspace in consideration::")
        for (key, value) in current_searchspace
            symbol = Symbol(key)
            #println("Key: $(key), Value: $(value)")
            length(value) > 2 ? arr_range = [value[1], value[2], value[3]] : arr_range = [value[1], value[2]]
            Algorithm_structure.Parameters_ranges[symbol] = arr_range
        end
        
        #println(Algorithm_structure.Parameters_ranges)
    end
    Algorithm_structure.Parameters_ranges
    
    return Algorithm_structure
end


function set_configuration_optuna(trial, Algorithm_structure, sampler_func,reference_point)
    params = Dict()
    MOEAD_HP = Dict() 
    weights = nothing

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
                    trial.suggest_float(hyperparam, lb, hb, step = step)
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

function get_df_column_values(df::DataFrame, column::Int, array_position::Int, Alg_Name, run_dict)
    
    if 1 ≤ column ≤ length(names(df))
        collumn_array = []

        for i in df[2:end, column] 
            if !ismissing(i)
                str = String(i)
                if !occursin("Inf", str)
                    array = eval(Meta.parse(str)) 
                    push!(collumn_array, array[array_position])
                    
                else
                    push!(collumn_array, i)
                end
            end
        end
    else
        error("Invalid column index: $column")
    end
    run_dict[Alg_Name] = collumn_array
    return run_dict

end

function set_up_weights_MOEAD_DE(nobjectives = nothing , npartitions = nothing)
    if  isnothing(nobjectives) && isnothing(npartitions)
        nobjectives = 2  
        npartitions = 50
    end
    weights = gen_ref_dirs(nobjectives, npartitions)
    return weights
   
end

function set_up_algorithm(algorithm_instance, options:: Options; params=Dict(), HPO=false, CCMO=false, MOEAD_WEIGHTS=nothing)
    base_algo = getproperty(Metaheuristics, Symbol(algorithm_instance))
    
    if CCMO
        base_algo = CCMO(base_algo)
    end

    if algorithm_instance == :MOEAD_DE
        if MOEAD_WEIGHTS === nothing
            MOEAD_WEIGHTS = set_up_weights_MOEAD_DE()
        end
        
        if HPO
            
            T = max(3, round(Int, 0.2*length(MOEAD_WEIGHTS)))
            n_r = max(2, round(Int, 0.05*length(MOEAD_WEIGHTS)))

            metaheuristic = base_algo(MOEAD_WEIGHTS; params..., T, n_r, options)

        else
            metaheuristic = base_algo(MOEAD_WEIGHTS; options)
        end
    else
        kwargs = HPO ? (; params..., options) : (; options)
        metaheuristic = base_algo(; kwargs...)
    end

    return metaheuristic
end


function run_optimization(problem_data, params,
                    Algorithm_structure, result_dir; MOEAD_WEIGHTS = nothing)

    hv_values = Dict()
    All_HV = Float64[]
    all_pareto_fronts = Dict()
    num_ite = 20

    
    problem_name = problem_data[1]
    f = problem_data[2]
    searchspace = problem_data[3]
    reference_point = problem_data[4]
    num_runs = problem_data[5]

    algorithm_instance = Algorithm_structure.Name

   
    cd("/home/afonso-meneses/Desktop/GitHub/DynamicOptimization")
    open("log.txt", "a") do io

        println(io, "Optimizing problem:: $problem_name", " reference_point:: $reference_point " 
        , "num_runs:: $num_runs ", "algorithm_instance:: $algorithm_instance")
    end

    options = Options(iterations = num_ite, f_calls_limit = 1000000000)
    
    metaheuristic = set_up_algorithm(algorithm_instance, options; params, HPO = true, MOEAD_WEIGHTS)#, CCMO = true) ### CCMO parameter
    
    cd(result_dir)

    if num_runs == "Inf" || num_runs > 100
        println("Skipping problem $problem_name due to invalid run length.")
        return -Inf, -Inf, -Inf
    end

    for i in 1:num_runs
        println("Starting task...")
        status = optimize(f, searchspace, metaheuristic)
        display(status)
        println("Task Finished...")
        approx_front = get_non_dominated_solutions(status.population)
        println(typeof(reference_point))
        push!(All_HV, hypervolume(approx_front, reference_point))
        front_objectives = get_non_dominated_solutions([sol.f for sol in approx_front])   
        field = "Run_$(i)_$(problem_name)"
        all_pareto_fronts[Symbol(field)] = front_objectives
    end


    #println(all_pareto_fronts)

    hv_values[num_ite] = mean(All_HV)


    #println("Hypervolume: $(hv_values[num_ite])")


    return hv_values, All_HV, all_pareto_fronts

end

function objective(trial, sampler_func, Algorithm_structure, result_dir, problem_data)

    problem_name = problem_data[1] 
    reference_point = problem_data[4]

    params, weights = set_configuration_optuna(trial, Algorithm_structure, sampler_func, reference_point)
  
    hv_value, All_HV, all_pareto_fronts = run_optimization(problem_data, params, 
                                                            Algorithm_structure, 
                                                            result_dir; MOEAD_WEIGHTS = weights)

                                                            
    if hv_value == -Inf && All_HV == -Inf
        return -Inf
    end

    isempty(hv_value) && return -Inf

    hv_max = maximum(values(hv_value))

    trial.set_user_attr("problem_name", problem_name)
    trial.set_user_attr("PF", all_pareto_fronts)
    trial.set_user_attr("All_HV", All_HV)

    return hv_max
end

function init_parallel_arrays(sampler_vector, problem_instances::UnitRange{Int64})
    algo_instances = 1:length(sampler_vector)
    algo_instances_array = vcat([algo_instances for _  in problem_instances]...)    
    problem_instances_array = [prob_i for prob_i in problem_instances for _ in algo_instances]
    return problem_instances_array, algo_instances_array
end


function initialize_algorithm_structures(alg)
        
        All_Algorithm_structure = Any[]

        for searchspace in alg
            
            Algorithm_structure = detect_searchspaces(searchspace)
            push!(All_Algorithm_structure, Algorithm_structure)
           
        end

    return All_Algorithm_structure
end

function initialize_runs_dicts(All_Algorithm_structure, problem_dataframe :: DataFrame, lower_bound::Int64, upper_bound::Int64 )

    
    for Algorithm_structure in All_Algorithm_structure
        runs_dicts = Dict()
        if isfile("minimum_runs_$(Algorithm_structure.Name).csv")
            df = CSV.read("minimum_runs_$(Algorithm_structure.Name).csv", DataFrame; header=false)
        else
            error("minimum_runs_$(Algorithm_structure.Name).csv is not in the current folder, please switch to the folder where its located or run get_minimum_runs() to generate it")
        end
    
        runs_dicts = get_df_column_values(df,6, 10, Algorithm_structure.Name, runs_dicts)
        problem_dataframe[!, Algorithm_structure.Name] = runs_dicts[Algorithm_structure.Name][lower_bound:upper_bound]
    end

    return problem_dataframe
end

function benchmark_handler(All_Algorithm_structure, lower_bound::Int64, upper_bound::Int64)

    if !(lower_bound < upper_bound && upper_bound <= 50 && lower_bound <= 50)
        error("Invalid bounds")
    end

    problem_dataframe = DataFrame(
        problems_names = String[],
        problem_function = Function[],
        problem_bounds = Any[],
        problem_reference_point = Any[]
    )

    for i in lower_bound:upper_bound

        probl_name, f, bounds, reference_point = getproblem(i)

        if haskey(ref_points_offset, i)
            reference_point = ref_points_offset[i]
        end

        push!(problem_dataframe, (
            problems_names = probl_name,
            problem_function = f,
            problem_bounds = bounds,
            problem_reference_point = reference_point
        ))
    end

    problem_dataframe = initialize_runs_dicts(All_Algorithm_structure, problem_dataframe, lower_bound, upper_bound)

    return problem_dataframe
end


function unravel_df(problem_dataframe::DataFrame, algorithm_name::String, problem_instance::Int)

    DataFrame_options = OrderedDict(
                                "problems_names" => 1, 
                                "problem_function" => 2,
                                "problem_bounds" => 3, 
                                "problem_ref_point" => 4,
                                "algorithm_name"=> 5) 

  
    problem_name_vector = problem_dataframe[!, 1]
    problem_function_vector = problem_dataframe[!, 2]
    problem_bounds =  problem_dataframe[!, 3]
    problem_ref_point = problem_dataframe[!, 4]

    problem_name = problem_name_vector[problem_instance]
    problem_function = problem_function_vector[problem_instance]
    problem_bounds = problem_bounds[problem_instance]
    problem_ref_point = problem_ref_point[problem_instance] 
    #problem_algo_run = problem_dataframe[problem_dataframe.problems_names .== problem_name , algorithm_name]
    problem_algo_run = problem_dataframe[problem_instance, algorithm_name]

    return problem_name, problem_function, problem_bounds, problem_ref_point, problem_algo_run

end

function run_trial(sampler_instance::Int, Algorithm_structure, sampler_vector, result_dir::String, iteration_counts, problem_instance, problem_dataframe)
    
    problem_name, f, searchspace, reference_point, num_runs = unravel_df(problem_dataframe, String(Algorithm_structure.Name), problem_instance)

    problem_data = [problem_name, f, searchspace, reference_point, num_runs]

    sampler_name = sampler_vector[sampler_instance]
    println("sampler_name :: $sampler_name")
    sampler_module = optuna.samplers
    sampler_func = getproperty(sampler_module, Symbol(sampler_name))

    if sampler_name == "GridSampler"
        sampler_constructor = sampler_func(Algorithm_structure.Parameters_ranges)
    else
        sampler_constructor = sampler_func()
    end
    
    println("sampler_func  " ,sampler_func)
    study = optuna.create_study(
        study_name = problem_name,
        direction = "maximize",
        sampler = sampler_constructor
    )
    

 
    study.optimize(trial -> objective(trial, sampler_func, Algorithm_structure, result_dir, problem_data), n_trials = n_trials)

    if isnan(study.best_value) || study.best_value == -Inf || !haskey(study.best_trial.user_attrs, "All_HV")
        println("No valid result for $problem_name")
        return nothing
    end

    All_HV = study.best_trial.user_attrs["All_HV"]
    PF_best = study.best_trial.user_attrs["PF"]

    opt_results_df = DataFrame(
        algorithm_name = Symbol[],
        sampler = String[],
        solutions = Vector{Any}[],
    )
    for (key, PF) in PF_best
    
        push!(opt_results_df, (algorithm_name = Symbol(key), sampler = study[:sampler][:__class__][:__name__], solutions = PF,))
    end

    problem_folder_name = "Problem_$(problem_instance)_$(problem_name)"
    problem_dir, iter_dir = create_directories(String(Algorithm_structure.Name), iteration_counts, problem_folder_name, result_dir)
    cd(problem_dir)

    CSV_NAME = "$(Symbol(problem_name))_$(problem_instance)_$(study[:sampler][:__class__][:__name__])_$(Algorithm_structure.Name)_obtained_solutions.csv"
    CSV.write(CSV_NAME, opt_results_df)

    return (
        algorithm_name = Algorithm_structure.Name,
        sampler = study[:sampler][:__class__][:__name__],
        problem_name = problem_name,
        problem_instance = problem_instance,
        hv_value = study.best_value,
        params = study.best_params,
        #all_hv = [JSON.json(All_HV)]
    )

end

function run_HPO(sampler_vector, iteration_counts, result_dir, All_Algorithm_structure, problem_dataframe; )
        
        results = []
       
        problem_instances = 1:nrow(problem_dataframe)
        problem_instances_array, sampler_instances_array = init_parallel_arrays(sampler_vector, problem_instances) 
        results = @distributed (vcat) for Algorithm_structure in collect(All_Algorithm_structure)
            
            
            println("Currently Testing : $(Algorithm_structure.Name)")
            task = (sampler_instance, prob) -> run_trial(sampler_instance, Algorithm_structure, sampler_vector, result_dir, iteration_counts, prob, problem_dataframe)
            pmap_results = pmap(task, sampler_instances_array, problem_instances_array)
            [pmap_results]
        end
   

    return results
end


        

#for alg in algorithms

#    CSV_RUNS_FILE_NAME, CSV_LENGTH_RESULTS_NAME = check_CSV(alg, main_script_name; test = false)
#    Algorithm_structure = detect_searchspaces(alg)
#    pop_size = Algorithm_structure.Parameters[:N]
#    max_evals = pop_size * n_iterations
#    results = []
#    hv_values = Dict()

#    algorithm_instance = Algorithm_structure.Name
#    println("Using algorithm: $algorithm_instance")
#    options = Metaheuristics.Options(iterations = n_iterations, f_calls_limit = 3 * max_evals)
#    metaheuristic = set_up_algorithm(algorithm_instance, options)
#    #metaheuristic = MixedInteger(metaheuristic)
#    All_HV = Dict(:Hypervolumes => Float64[])
#    num_runs = 10
#    reference_point = [10, 4000]

#    @time for i in 1:num_runs
#        println("Starting task...")
#        status = optimize(problem, mixed_space, metaheuristic)
#        display(status)
#        println("Task Finished... iteration : $i")
#        approx_front = get_non_dominated_solutions(status.population)
#        HV = hypervolume(approx_front, reference_point) 
#        push!(All_HV[:Hypervolumes], HV) 
#        #push!(All_SOL[:Solutions], [JSON.json(approx_front)]) 
#    end

#    if pwd() !== result_dir
#        cd(result_dir)
#    end

#    results = All_HV

#    all_sol = All_SOL

#    type_of_result = first(keys(results))                

#    println("Results::$results")

#    hv_values[n_iterations] = mean(results[type_of_result])

#   println("Hypervolume: $(hv_values[n_iterations])")

#    problem_name = "parametric_truss_example"
#    current_instance = 0
#    get_minimum_runs(results, problem_name, current_instance, CSV_RUNS_FILE_NAME)

#end