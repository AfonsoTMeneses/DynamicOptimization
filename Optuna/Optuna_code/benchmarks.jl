using Distributed
addprocs(4)

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
    optuna = pyimport("optuna")
    include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
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
    n_trials = 100
    All_Algorithm_structure = initialize_algorithm_structures(algorithms)
end
 
@everywhere begin
    lb_instaces = 1
    hb_instaces = 50
    problem_dataframe = DataFrame()
    problem_dataframe = benchmark_handler(All_Algorithm_structure, lb_instaces, hb_instaces, main_script_name)
    results = []
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
                    iterations = 50,
                    store_convergence = false,
                    debug = false,
                    parallel_evaluation = false,
                    verbose = false
                )

    options_dict = push_options(options_dataframe)

end


problem_instances = 1:nrow(problem_dataframe)
problem_instances_array, sampler_instances_array = init_parallel_arrays(sampler_vector, problem_instances)

    all_tasks = [
        (alg, s, p)
        for alg in All_Algorithm_structure
        for (s, p) in zip(sampler_instances_array, problem_instances_array)
    ]
    



elapsed_time = @elapsed results = run_HPO(sampler_vector, options_dict, results_path, All_Algorithm_structure, problem_dataframe, main_script_name, n_trials)
open("time_run_HPO_$(main_script_name).txt", "a") do io
    println(io, "run_HPO elapsed time: $(round(elapsed_time, digits=2))s ($(round(elapsed_time/60, digits=2)) min)")
end

write_HPO_data_into_csv(results, options_dict, results_path)



#=
a = 1:4

b = vcat([a for _ in 1:3]...)

c = [prob_i for prob_i in 1:3 for _ in 1:4]

length(b)

=#