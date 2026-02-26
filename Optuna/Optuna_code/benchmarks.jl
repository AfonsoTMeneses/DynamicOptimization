
#using Distributed
#addprocs(3)

#@everywhere begin
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
#end  

run(`clear`)

base_dir = "/home/afonso-meneses/Desktop/GitHub/DynamicOptimization/Optuna/Results/"
algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]
main_script_name = split(basename(abspath(@__FILE__)), ".jl")[1]
results_path = joinpath(base_dir, "$(main_script_name)_Results") 
cd(results_path)

#@everywhere begin
    n_trials = 100
    All_Algorithm_structure = initialize_algorithm_structures(algorithms)
#end

    ###########
    ########### HPO
    ###########

    lb_instaces = 1
    hb_instaces = 50
    problem_dataframe = DataFrame()
    problem_dataframe = benchmark_handler(All_Algorithm_structure, lb_instaces, hb_instaces, main_script_name)
    results = []
    run(`clear`)


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

    @time results = run_HPO(sampler_vector, options_dict, results_path, All_Algorithm_structure, problem_dataframe, main_script_name, n_trials)
    # pmap -- 25.687724 seconds (2.98 M allocations: 203.145 MiB, 0.33% gc time, 8.33% compilation time: 4% of which was recompilation)
    # map -- 38.523086 seconds (38.27 M allocations: 14.290 GiB, 5.12% gc time, 25.31% compilation time: 18% of which was recompilation)

   
    #println("Summary of best trials:")



write_HPO_data_into_csv(results, options_dict, results_path)

#=
a = 1:4

b = vcat([a for _ in 1:3]...)

c = [prob_i for prob_i in 1:3 for _ in 1:4]

length(b)

=#