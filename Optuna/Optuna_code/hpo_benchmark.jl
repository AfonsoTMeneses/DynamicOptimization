using Distributed
addprocs(2)

@everywhere begin
    #using Pkg
    #ENV["PYTHON"] = "/home/afonso-meneses/Desktop/GitHub/python_env/bin/python" 
    #Pkg.build("PyCall")
    using Metaheuristics
    using Metaheuristics: optimize, get_non_dominated_solutions, pareto_front, Options
    import Metaheuristics.PerformanceIndicators: hypervolume
    using HardTestProblems
    using DataStructures
    using CSV
    using DataFrames
    using Statistics
    using JSON
    using PyCall
    using Logging
    optuna = pyimport("optuna")
    include(joinpath(@__DIR__, "optuna_utils.jl"))
end  


@everywhere begin
    algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"] 
    main_script_name = split(basename(abspath(@__FILE__)), ".jl")[1]
    results_path = normpath(dirname(@__DIR__),"Results/$(main_script_name)_Results")
    cd(results_path)
end

for alg in algorithms
    
    suffix = split(alg, "_searchspace")[1]   
    remove_existing_csv(joinpath(results_path, suffix))
    
end


@everywhere begin
    n_trials = 5
    All_Algorithm_structure = initialize_algorithm_structures(algorithms)
end
 
@everywhere begin
    lb_instaces = 1
    hb_instaces = 5
    baseline_name = "baseline_benchmark"
    baseline_dir = joinpath(dirname(@__DIR__), "Results")
    problem_dataframe = benchmark_handler(All_Algorithm_structure, lb_instaces, hb_instaces, main_script_name;
                                          baseline_name=baseline_name, baseline_dir=baseline_dir)
end


@everywhere begin
   
    options_dataframe = DataFrame(
                    x_tol = 1e-8,
                    f_tol = 1e-12,
                    f_tol_rel = eps(),
                    f_tol_abs = 0.0,
                    g_tol = 0.0,
                    h_tol = 0.0,
                    f_calls_limit = 1000000000,
                    time_limit = Inf,
                    iterations = 100,
                    store_convergence = false,
                    debug = false,
                    parallel_evaluation = false,
                    verbose = false
                )

    options_dict = push_options(options_dataframe)

end

get_min_runs_name = "baseline_benchmark"
bounds_base_path = joinpath(dirname(@__DIR__), "Results")

empirical_bounds_dict = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
for alg in algorithms
    alg_name = split(alg, "_searchspace")[1]
    bounds_csv = joinpath(bounds_base_path, "empirical_bounds_$(get_min_runs_name)_$(alg_name).csv")
    if isfile(bounds_csv)
        alg_bounds = load_empirical_bounds(bounds_csv)
        for (inst, (ideal, nadir)) in alg_bounds
            if haskey(empirical_bounds_dict, inst)
                existing_ideal, existing_nadir = empirical_bounds_dict[inst]
                merged_ideal = [min(existing_ideal[i], ideal[i]) for i in eachindex(ideal)]
                merged_nadir = [max(existing_nadir[i], nadir[i]) for i in eachindex(nadir)]
                empirical_bounds_dict[inst] = (merged_ideal, merged_nadir)
            else
                empirical_bounds_dict[inst] = (ideal, nadir)
            end
        end
        println("Loaded empirical bounds from $bounds_csv ($(length(alg_bounds)) problems)")
    else
        @warn "Bounds file not found: $bounds_csv"
    end
end
println("Total problems with empirical bounds: $(length(empirical_bounds_dict))/50")

elapsed_time = @elapsed results = run_HPO(sampler_vector, options_dict, results_path, All_Algorithm_structure, problem_dataframe, main_script_name, n_trials, empirical_bounds_dict;
                                          collect_fronts=true)
open("time_run_HPO_$(main_script_name).txt", "a") do io
    println(io, "run_HPO elapsed time: $(round(elapsed_time, digits=2))s ($(round(elapsed_time/60, digits=2)) min)")
end

write_HPO_data_into_csv(results, options_dict, results_path)
