
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
    include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
    include(joinpath(@__DIR__, "optuna_utils.jl"))
    optuna = pyimport("optuna")
#end  

PyCall.python 

run(`clear`)
result_dir = normpath(@__DIR__,"..","Results")
cd(result_dir)

#@everywhere begin
    alg = ["MOEAD_DE_searchspace","NSGA2_searchspace"] #[ "SPEA2_searchspace", "SMS_EMOA_searchspace"]
    n_trials = 2
    All_Algorithm_structure = initialize_algorithm_structures(alg)
    iteration_counts = 100
#end

    ###########
    ########### HPO
    ###########

    lb_instaces = 1 
    hb_instaces = 5
    problem_dataframe = DataFrame()
    problem_dataframe = benchmark_handler(All_Algorithm_structure, lb_instaces, hb_instaces)
    results = []
    run(`clear`)


    include(joinpath(@__DIR__, "optuna_utils.jl"))

    run(`clear`)
    @time results = run_HPO(sampler_vector, iteration_counts,result_dir, All_Algorithm_structure, problem_dataframe)
    # pmap -- 25.687724 seconds (2.98 M allocations: 203.145 MiB, 0.33% gc time, 8.33% compilation time: 4% of which was recompilation)
    # map -- 38.523086 seconds (38.27 M allocations: 14.290 GiB, 5.12% gc time, 25.31% compilation time: 18% of which was recompilation)

   
    #println("Summary of best trials:")
   
   function write_data_into_csv(results)
        temp = results
        
        for i in range(1, length(results))
            temp = results[i]
            for r in range(1, length(temp))
            
                if !isnothing(temp[r]) 
                    CSV_NAME = ""

                    result_df = DataFrame(
                        algorithm_name = Symbol[],
                        sampler = String[],
                        problem_instance = Int[],
                        problem_name = String[],
                        hv_value = Float64[],
                        params = String[], 
                    )


            
                    println("$(temp[r].algorithm_name) :: $(temp[r].problem_name): value = $(temp[r].hv_value), params = $(temp[r].params)")
                    if occursin("Dict{Any, Any}", string(temp[r][:params]))    
                        params_str = replace(string(temp[r][:params]), r"Dict\{Any, Any\}\(" => "", ")" => "", "\"" => "", "=>" => "=")
                    end

                    println("params_str:: $params_str")

                    push!(result_df, (temp[r][:algorithm_name], temp[r][:sampler], temp[r][:problem_instance],
                    temp[r][:problem_name], temp[r][:hv_value], params_str ))

                    problem_folder_name = "Problem_$(temp[r][:problem_instance])_$(temp[r].problem_name)"
                    problem_dir, iter_dir = create_directories(String(temp[r][:algorithm_name]), iteration_counts, problem_folder_name, result_dir)
                    cd(iter_dir)
                    
                    CSV_NAME = "$(temp[r][:algorithm_name])_$(temp[r].sampler).csv"
                    

                    write_header = !isfile(CSV_NAME)

                    if occursin(string(temp[r].sampler),CSV_NAME)
                        CSV.write(CSV_NAME, result_df, append = true, writeheader = write_header)
                    end
                end

            end 
        end
    end


#write_data_into_csv(results)

#=
a = 1:4

b = vcat([a for _ in 1:3]...)

c = [prob_i for prob_i in 1:3 for _ in 1:4]

length(b)

=#