#using KhepriAutoCAD
#Pkg.add(url="https://github.com/aptmcl/KhepriFrame3DD.jl")
#Pkg.add(url="https://github.com/ines-pereira/Metaheuristics.jl")
using Distributed
addprocs(3)


#ENV["PYTHON"] = "/home/afonso-meneses/Desktop/GitHub/python3.11_env/bin/python3.11" 
#Pkg.build("PyCall")


@everywhere begin
    #using Pkg
    #ENV["PYTHON"] = "/home/afonso-meneses/Desktop/GitHub/new_env/bin/python3"
    #ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
    #Pkg.build("PyCall")
    using PyCall
    optuna = pyimport("optuna")
    using KhepriFrame3DD
    using Distributed
    include(joinpath(@__DIR__, "Hyperoptimization_intervals.jl"))
    include(joinpath(@__DIR__, "utils_minimum_runs.jl"))
    include(joinpath(@__DIR__, "optuna_utils.jl"))
    using Metaheuristics
    using Metaheuristics: optimize, get_non_dominated_solutions, pareto_front, Options
    import Metaheuristics.PerformanceIndicators: hypervolume
    using DataStructures
    using CSV
    using DataFrames
    using Statistics
    using JSON
    using HardTestProblems
end
    
# Truss Geometry ---------------------------------------------------------------
    
@everywhere begin
    my_free_truss_node_family = truss_node_family_element(default_truss_node_family(), support=false)
    free_node(pt) = truss_node(pt, family=my_free_truss_node_family)
    fixed_node(pt) = truss_node(pt, family=fixed_truss_node_family)
end

@everywhere begin
    space_frame(ptss) =
        let ais = ptss[1],
            bis = ptss[2],
            cis = ptss[3]

            fixed_node(ais[1])
            free_node.(ais[2:end-1])
            fixed_node(ais[end])
            free_node.(bis)
            truss_bar.(ais, cis)
            truss_bar.(bis, ais[1:end-1])
            truss_bar.(bis, cis[1:end-1])
            truss_bar.(bis, ais[2:end])
            truss_bar.(bis, cis[2:end])
            truss_bar.(ais[2:end], ais[1:end-1])
            truss_bar.(bis[2:end], bis[1:end-1])
            if ptss[4:end] == []
                fixed_node(cis[1])
                free_node.(cis[2:end-1])
                fixed_node(cis[end])
                truss_bar.(cis[2:end], cis[1:end-1])
            else
                truss_bar.(bis, ptss[4])
                space_frame(ptss[3:end])
            end
        end

end


@everywhere begin
    parametric_truss(x11, y11, z11, x12, y12, z12, x13, y13, z13, x21, y21, z21, x22, y22, z22, x31, y31, z31, x32, y32, z32, x33, y33, z33, x41, y41, z41, x42, y42, z42, x51, y51, z51, x52, y52, z52, x53, y53, z53) =
        let p11 = xyz(x11, y11, z11),
            p12 = xyz(x12, y12, z12),
            p13 = xyz(x13, y13, z13),
            p21 = xyz(x21, y21, z21),
            p22 = xyz(x22, y22, z22),
            p31 = xyz(x31, y31, z31),
            p32 = xyz(x32, y32, z32),
            p33 = xyz(x33, y33, z33),
            p41 = xyz(x41, y41, z41),
            p42 = xyz(x42, y42, z42),
            p51 = xyz(x51, y51, z51),
            p52 = xyz(x52, y52, z52),
            p53 = xyz(x53, y53, z53)

            space_frame([[p11, p12, p13],
                [p21, p22],
                [p31, p32, p33],
                [p41, p42],
                [p51, p52, p53]])
        end
end
#delete_all_shapes()
#parametric_truss(0, 0, 0, 1, 0, 0, 2, 0, 0, 0.5, 0.5, 1, 1.5, 0.5, 1, 0, 1, 0, 1, 1, 0, 2, 1, 0, 0.5, 1.5, 1, 1.5, 1.5, 1, 0, 2, 0, 1, 2, 0, 2, 2, 0)

@everywhere begin
    fixed_parametric_truss(
        x12, y12, z12,
        x21, y21, z21,
        x22, y22, z22,
        x32, y32, z32,
        x41, y41, z41,
        x42, y42, z42,
        x52, y52, z52) =
        begin
            delete_all_shapes()
            parametric_truss(
                0, 0, 0, x12, y12, z12, 20, 0, 0,
                x21, y21, z21, x22, y22, z22,
                0, 10, 0, x32, y32, z32, 20, 10, 0,
                x41, y41, z41, x42, y42, z42,
                0, 20, 0, x52, y52, z52, 20, 20, 0)
        end
end
#delete_all_shapes()
#fixed_parametric_truss(
#    1.3, 0, 0,
#    0.5, 0.5, 1, 1.5, 0.5, 1,
#    1, 1, 0,
#    0.5, 1.5, 1, 1.5, 1.5, 1,
#    1, 2, 0)

#=
random_fixed_parametric_truss(r) =
    let rnd(v) = random_range(v - r, v + r)*10
        fixed_parametric_truss(
            rnd(1), rnd(0), rnd(0),
            rnd(0.5), rnd(0.5), rnd(1),
            rnd(1.5), rnd(0.5), rnd(1),
            rnd(1), rnd(1), rnd(0),
            rnd(0.5), rnd(1.5), rnd(1),
            rnd(1.5), rnd(1.5), rnd(1),
            rnd(1), rnd(2), rnd(0))
    end
=#

#delete_all_shapes()
#random_fixed_parametric_truss(0.4)

# Truss Analysis and Optimization ----------------------------------------------

## Helper Functions
@everywhere begin
    
    step_size = 0.01    
    int2float(x, min, step = step_size) =
        min + step * x

    bounds_coordinates(v, r=0.3) = (v - r, v + r) .* 10
end
## Materials Young's Modulus and Cost

@everywhere begin
    materials_e = [
        1.6409e11, # Cast Iron ASTM A536
        1.86e11,   # Steel, stainless AISI 302
        2.e11,     # Carbon Steel, Structural ASTM A516
        2.0684e11, # Alloy Steel, ASTM A242
        2.047e11,  # Alloy Steel, AISI 4140
        1.93e11,   # Stainless Steel AISI 201
    ]

    materials_cost = [
        460.0,     # Cast Iron ASTM A536
        1480.0,    # Steel, stainless AISI 302
        860.0,     # Carbon Steel, Structural ASTM A516
        950.0,     # Alloy Steel, ASTM A242
        2750.0,    # Alloy Steel, AISI 4140
        1825.0,    # Stainless Steel AISI 201
    ]
end
## Objectives
#=
The objectives for the optimization are:
  (1) minimizing the maximum displacement,
  (2) minimizing the material cost of the truss structure.
=#
@everywhere begin

    r = 0.4
    x12_interval = bounds_coordinates(1, r)
    y12_interval = bounds_coordinates(0, r)
    z12_interval = bounds_coordinates(0, r)
    x21_interval = bounds_coordinates(0.5, r)
    y21_interval = bounds_coordinates(0.5, r)
    z21_interval = bounds_coordinates(1, r)
    x22_interval = bounds_coordinates(1.5, r)
    y22_interval = bounds_coordinates(0.5, r)
    z22_interval = bounds_coordinates(1, r)
    x32_interval = bounds_coordinates(1, r)
    y32_interval = bounds_coordinates(1, r)
    z32_interval = bounds_coordinates(0, r)
    x41_interval = bounds_coordinates(0.5, r)
    y41_interval = bounds_coordinates(1.5, r)
    z41_interval = bounds_coordinates(1, r)
    x42_interval = bounds_coordinates(1.5, r)
    y42_interval = bounds_coordinates(1.5, r)
    z42_interval = bounds_coordinates(1, r)
    x52_interval = bounds_coordinates(1, r)
    y52_interval = bounds_coordinates(2, r)
    z52_interval = bounds_coordinates(0, r)


    n_objs = 2

    cost(truss_volume, material) =
        truss_volume * materials_cost[Int(material)]

    objectives(
        material, bar_radius,
        x12, y12, z12,
        x21, y21, z21,
        x22, y22, z22,
        x32, y32, z32,
        x41, y41, z41,
        x42, y42, z42,
        x52, y52, z52) =
        let b_radius = int2float(bar_radius, 0.035, 0.005),
            load = vz(-3500.0) * 20 * 20 # total load applied to the truss
            x12 = int2float(x12, x12_interval[1])
            y12 = int2float(y12, y12_interval[1])
            z12 = int2float(z12, z12_interval[1])
            x21 = int2float(x21, x21_interval[1])
            y21 = int2float(y21, y21_interval[1])
            z21 = int2float(z21, z21_interval[1])
            x22 = int2float(x22, x22_interval[1])
            y22 = int2float(y22, y22_interval[1])
            z22 = int2float(z22, z22_interval[1])
            x32 = int2float(x32, x32_interval[1])
            y32 = int2float(y32, y32_interval[1])
            z32 = int2float(z32, z32_interval[1])
            x41 = int2float(x41, x41_interval[1])
            y41 = int2float(y41, y41_interval[1])
            z41 = int2float(z41, z41_interval[1])
            x42 = int2float(x42, x42_interval[1])
            y42 = int2float(y42, y42_interval[1])
            z42 = int2float(z42, z42_interval[1])
            x52 = int2float(x52, x52_interval[1])
            y52 = int2float(y52, y52_interval[1])
            z52 = int2float(z52, z52_interval[1])
            set_backend_family(
                default_truss_bar_family(),
                frame3dd,
                frame3dd_truss_bar_family(
                    E=materials_e[Int(material)], # (Young's modulus)
                    #G=8.1e10,                    # (Kirchoff's or Shear modulus)
                    G=7.95e10,                    # (Kirchoff's or Shear modulus)
                    p=0.0,                        # Roll angle
                    d=7850.0))                    # Density
                    #d=77010.0))                   # Density
            with_truss_node_family(radius=b_radius * 2.4) do
                with_truss_bar_family(radius=b_radius, inner_radius=b_radius - 0.02) do
                    fixed_parametric_truss(x12, y12, z12, x21, y21, z21, x22, y22, z22, x32, y32, z32, x41, y41, z41, x42, y42, z42, x52, y52, z52)
                    free_ns = length(filter(!KhepriBase.truss_node_is_supported, frame3dd.truss_nodes))
                    truss_volume = truss_bars_volume()
                    analysis = truss_analysis(load / free_ns)
                    max_disp = max_displacement(analysis)
                    # show_truss_deformation(analysis, autocad, factor=100) # to visualize in AutoCAD
                    [max_disp, cost(truss_volume, material)]
            end
        end
    end
end

@everywhere begin
    
    problem(x) = 
        (objectives(
            x[1], x[2], x[3], x[4], x[5],
            x[6], x[7], x[8], x[9], x[10],
            x[11], x[12], x[13], x[14], x[15],
            x[16], x[17], x[18], x[19], x[20],
            x[21], x[22], x[23]),
        [0.0], [0.0])
end
#@everywhere begin
#    problem(x) =
#    (objectives(
#            x[:integer][1], x[:integer][2],
#            x[:continuous][1], x[:continuous][2], x[:continuous][3],
#            x[:continuous][4], x[:continuous][5], x[:continuous][6],
#            x[:continuous][7], x[:continuous][8], x[:continuous][9],
#            x[:continuous][10], x[:continuous][11], x[:continuous][12], 
#            x[:continuous][13], x[:continuous][14], x[:continuous][15],
#            x[:continuous][16], x[:continuous][17], x[:continuous][18],
#            x[:continuous][19], x[:continuous][20], x[:continuous][21]),
#        [0.0], [0.0])


#end
## Variables
@everywhere begin
    n_vars = 23

    material_idx = 1:6

    bar_radius = 0:8 # min=0.035, max=0.075, step=0.005

    upper_bound(interval, step_size = step_size) = Int((interval[end] - interval[1])/step_size)

    x12_upper_bound = upper_bound(x12_interval)
    y12_upper_bound = upper_bound(y12_interval) 
    z12_upper_bound = upper_bound(z12_interval)
    x21_upper_bound = upper_bound(x21_interval)
    y21_upper_bound = upper_bound(y21_interval)
    z21_upper_bound = upper_bound(z21_interval)
    x22_upper_bound = upper_bound(x22_interval) 
    y22_upper_bound = upper_bound(y22_interval) 
    z22_upper_bound = upper_bound(z22_interval) 
    x32_upper_bound = upper_bound(x32_interval) 
    y32_upper_bound = upper_bound(y32_interval) 
    z32_upper_bound = upper_bound(z32_interval) 
    x41_upper_bound = upper_bound(x41_interval) 
    y41_upper_bound = upper_bound(y41_interval) 
    z41_upper_bound = upper_bound(z41_interval) 
    x42_upper_bound = upper_bound(x42_interval) 
    y42_upper_bound = upper_bound(y42_interval) 
    z42_upper_bound = upper_bound(z42_interval) 
    x52_upper_bound = upper_bound(x52_interval) 
    y52_upper_bound = upper_bound(y52_interval) 
    z52_upper_bound = upper_bound(z52_interval) 


    points = [x12_interval, y12_interval, z12_interval,
            x21_interval, y21_interval, z21_interval,
            x22_interval, y22_interval, z22_interval,
            x32_interval, y32_interval, z32_interval,
            x41_interval, y41_interval, z41_interval,
            x42_interval, y42_interval, z42_interval,
            x52_interval, y52_interval, z52_interval]

    #points_lb = [p[1] for p in points]
    #points_ub = [p[end] for p in points]

    lower_bound = 0


    lb_points = [material_idx[1], bar_radius[1],
            fill(lower_bound, 21)...]

    ub_points = [material_idx[end], bar_radius[end],
            x12_upper_bound, y12_upper_bound, z12_upper_bound,
            x21_upper_bound, y21_upper_bound, z21_upper_bound,
            x22_upper_bound, y22_upper_bound, z22_upper_bound,
            x32_upper_bound, y32_upper_bound, z32_upper_bound,
            x41_upper_bound, y41_upper_bound, z41_upper_bound,
            x42_upper_bound, y42_upper_bound, z42_upper_bound,
            x52_upper_bound, y52_upper_bound, z52_upper_bound]

    integer_space = BoxConstrainedSpace(lb_points, ub_points)

    #continuous_space = BoxConstrainedSpace(points_lb, points_ub)

    #mixed_space = MixedSpace(:integer => integer_space, :continuous => continuous_space)
end
#end 
#=
vars_bounds =
    [material_idx[1] bar_radius[1] x12[1] y12[1] z12[1] [
     x21[1]] y21[1] z21[1] x22[1] y22[1] z22[1] [
     x32[1]] y32[1] z32[1] x41[1] y41[1] z41[1] [
     x42[1]] y42[1] z42[1] x52[1] y52[1] z52[1];
     material_idx[end] bar_radius[end] x12[end] y12[end] z12[end] [
     x21[end]] y21[end] z21[end] x22[end] y22[end] z22[end] [
     x32[end]] y32[end] z32[end] x41[end] y41[end] z41[end] [
     x42[end]] y42[end] z42[end] x52[end] y52[end] z52[end]]
=#

#print("integer_space: ", typeof(integer_space))
#options = Options(iterations = 1000, f_calls_limit = 1000000000);
#status = optimize(problem, integer_space, NSGA2(options=options))
#display(status)


## Test Optimization
@everywhere begin
    main_script_name = split(basename(abspath(@__FILE__)), ".jl")[1]
    algorithms = ["MOEAD_DE_searchspace","NSGA2_searchspace", "SPEA2_searchspace", "SMS_EMOA_searchspace"]
    results_path = normpath(dirname(@__DIR__),"Results/$(main_script_name)_Results")
    cd(results_path)
end

for alg in algorithms
    suffix = split(alg, "_searchspace")[1]
    path = String(joinpath(results_path, suffix))
    remove_existing_csv(path)
end



@everywhere begin
    n_trials = 100
    All_Algorithm_structure = initialize_algorithm_structures(algorithms)
end

    log_file = joinpath(base_dir, "log.txt")
    if isfile(log_file)
        rm(log_file)
    end


@everywhere begin
    
    reference_point = [10, 4000]

    problem_dataframe = DataFrame(
                problems_names   = "parametric_truss_example",
                problem_function = problem,
                problem_bounds   = integer_space,
                problem_ref_point = [reference_point],
    )

end

options_dataframe = DataFrame(
                    x_tol               = 1e-8,
                    f_tol               = 1e-12,
                    f_tol_rel           = eps(),
                    f_tol_abs           = 0.0,
                    g_tol               = 0.0,
                    h_tol               = 0.0,
                    f_calls_limit       = 1000000000,
                    time_limit          = Inf,
                    iterations          = 50,
                    store_convergence   = false,
                    debug               = false,
                    parallel_evaluation = false,
                    verbose             = false
                )

options_dict = push_options(options_dataframe)

@time results = run_HPO(
                sampler_vector,
                options_dict,
                results_path,
                All_Algorithm_structure,
                problem_dataframe,
                main_script_name, n_trials)

write_HPO_data_into_csv(results, options_dict, results_path)


