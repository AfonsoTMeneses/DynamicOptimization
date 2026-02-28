
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
    using Distributed
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



@everywhere begin
    n_trials = 2
    All_Algorithm_structure = initialize_algorithm_structures(algorithms)
end

    ###########
    ########### HPO
    ###########
@everywhere begin
    lb_instaces = 1
    hb_instaces = 2
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

    elapsed_time = @elapsed results = run_HPO(sampler_vector, options_dict, results_path, All_Algorithm_structure, problem_dataframe, main_script_name, n_trials)
    # pmap -- 25.687724 seconds (2.98 M allocations: 203.145 MiB, 0.33% gc time, 8.33% compilation time: 4% of which was recompilation)
    # map -- 38.523086 seconds (38.27 M allocations: 14.290 GiB, 5.12% gc time, 25.31% compilation time: 18% of which was recompilation)
    open("time_run_HPO_$(main_script_name).txt", "a") do io
        println(io, "run_HPO elapsed time: $(round(elapsed_time, digits=2))s ($(round(elapsed_time/60, digits=2)) min)")
    end


   
    #println("Summary of best trials:")


write_HPO_data_into_csv(results, options_dict, results_path)

#=
a = 1:4

b = vcat([a for _ in 1:3]...)

c = [prob_i for prob_i in 1:3 for _ in 1:4]

length(b)

=#