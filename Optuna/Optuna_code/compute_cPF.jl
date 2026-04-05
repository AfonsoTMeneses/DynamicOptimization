#!/usr/bin/env julia
# compute_cPF.jl
# Builds the combined Pareto front per problem from all baseline + HPO solutions,
# then computes IGD+ for each config against it.
# Run after all experiments are done.
#
using CSV
using DataFrames
using JSON
using Metaheuristics
using Metaheuristics: pareto_front, is_feasible, get_non_dominated_solutions
import Metaheuristics.PerformanceIndicators: igd_plus
using Statistics

include(joinpath(@__DIR__, "optuna_utils.jl"))

ALGORITHMS = ["NSGA2", "SPEA2", "SMS_EMOA", "MOEAD_DE"]


function compute_nondominated(pool::Vector{Vector{Float64}})
    isempty(pool) && return Vector{Float64}[]

    nobj = length(first(pool))

    pop = [
        Metaheuristics.create_child(zeros(1), (sol, Float64[], Float64[]))
        for sol in pool
    ]

    nd = get_non_dominated_solutions(pop)

    return [Float64[nd[i].f[j] for j in 1:nobj] for i in eachindex(nd)]
end


function compute_igd_plus(obtained::Vector{Vector{Float64}}, reference::Vector{Vector{Float64}})
    (isempty(obtained) || isempty(reference)) && return Inf

    A = reduce(hcat, obtained)'
    B = reduce(hcat, reference)'

    return igd_plus(A, B)
end


function load_baseline_fronts(results_dir, baseline_name)
    all_fronts = Dict{Int, Vector{Tuple{String, Vector{Float64}}}}()

    for alg in ALGORITHMS
        csv_path = joinpath(results_dir, "reference_fronts_$(baseline_name)_$(alg).csv")
        if !isfile(csv_path)
            println("  skipping $csv_path (not found)")
            continue
        end

        fronts = load_reference_fronts(csv_path)
        source = "baseline_$alg"

        for (inst, front) in fronts
            if !haskey(all_fronts, inst)
                all_fronts[inst] = Tuple{String, Vector{Float64}}[]
            end
            for sol in front
                push!(all_fronts[inst], (source, sol))
            end
        end
        println("  loaded baseline fronts from $csv_path")
    end

    return all_fronts
end


function load_hpo_fronts(results_dir, hpo_dir_name)
    all_fronts = Dict{Int, Vector{Tuple{String, Vector{Float64}}}}()
    hpo_dir = joinpath(results_dir, hpo_dir_name)

    if !isdir(hpo_dir)
        println("  HPO dir not found: $hpo_dir")
        return all_fronts
    end

    for (root, dirs, files) in walkdir(hpo_dir)
        for fn in files
            startswith(fn, "all_fronts_") && endswith(fn, ".csv") || continue

            fpath = joinpath(root, fn)
            try
                df = CSV.read(fpath, DataFrame)
                obj_cols = sort([c for c in names(df) if startswith(c, "obj_")])
                length(obj_cols) < 2 && continue

                for row in eachrow(df)
                    alg = string(row.algorithm)
                    sampler = string(row.sampler)
                    prob = Int(row.problem_instance)
                    source = "hpo_$(alg)_$(sampler)"

                    if !haskey(all_fronts, prob)
                        all_fronts[prob] = Tuple{String, Vector{Float64}}[]
                    end

                    sol = Float64[row[c] for c in obj_cols]
                    push!(all_fronts[prob], (source, sol))
                end
            catch e
                @warn "Failed to parse $fpath: $e"
            end
        end
    end

    n_total = sum(length(v) for v in values(all_fronts); init=0)
    println("  loaded $n_total HPO solutions across $(length(all_fronts)) problems from $hpo_dir_name")

    return all_fronts
end


function process_experiment(results_dir, baseline_name, hpo_dir_name, output_dir)
    mkpath(output_dir)

    println("\n  Loading baseline fronts ($baseline_name)...")
    baseline_fronts = load_baseline_fronts(results_dir, baseline_name)

    println("  Loading HPO fronts ($hpo_dir_name)...")
    hpo_fronts = load_hpo_fronts(results_dir, hpo_dir_name)

    all_problems = union(keys(baseline_fronts), keys(hpo_fronts))

    if isempty(all_problems)
        println("  No fronts found for this experiment.")
        return
    end

    println("  Loading empirical bounds...")
    bounds_dict = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    for alg in ALGORITHMS
        bounds_csv = joinpath(results_dir, "empirical_bounds_$(baseline_name)_$(alg).csv")
        if isfile(bounds_csv)
            alg_bounds = load_empirical_bounds(bounds_csv)
            for (inst, (ideal, nadir)) in alg_bounds
                if haskey(bounds_dict, inst)
                    ei, en = bounds_dict[inst]
                    merged_ideal = [min(ei[i], ideal[i]) for i in eachindex(ideal)]
                    merged_nadir = [max(en[i], nadir[i]) for i in eachindex(nadir)]
                    bounds_dict[inst] = (merged_ideal, merged_nadir)
                else
                    bounds_dict[inst] = (ideal, nadir)
                end
            end
        end
    end

    igd_rows = DataFrame(
        problem = Int[],
        config = String[],
        source_type = String[],
        algorithm = String[],
        sampler = String[],
        igd_plus = Float64[],
    )

    for prob in sort(collect(all_problems))
        println("  Problem $prob...")

        merged = Tuple{String, Vector{Float64}}[]
        if haskey(baseline_fronts, prob)
            append!(merged, baseline_fronts[prob])
        end
        if haskey(hpo_fronts, prob)
            append!(merged, hpo_fronts[prob])
        end

        if isempty(merged)
            continue
        end

        all_sols = [sol for (_, sol) in merged]

        nd_sols = compute_nondominated(all_sols)

        if isempty(nd_sols)
            println("    empty ND set, skipping")
            continue
        end

        nd_set = Set(Tuple(s) for s in nd_sols)
        nobj = length(first(nd_sols))

        # save ALL solutions with is_nd flag
        all_rows = []
        for (src, sol) in merged
            parts = split(src, "_")
            if startswith(src, "baseline_")
                alg = join(parts[2:end], "_")
                sampler = "baseline"
            elseif startswith(src, "hpo_")
                sampler_idx = findfirst(p -> occursin("Sampler", p), parts)
                if !isnothing(sampler_idx)
                    alg = join(parts[2:sampler_idx-1], "_")
                    sampler = parts[sampler_idx]
                else
                    alg = join(parts[2:end], "_")
                    sampler = "unknown"
                end
            else
                alg = src
                sampler = "unknown"
            end

            row = Dict{String, Any}(
                "source"    => src,
                "algorithm" => alg,
                "sampler"   => sampler,
                "is_nd"     => Tuple(sol) in nd_set,
            )
            for j in 1:nobj
                row["obj_$j"] = sol[j]
            end
            push!(all_rows, row)
        end

        cpf_df = DataFrame(all_rows)
        col_order = vcat(["source", "algorithm", "sampler", "is_nd"], ["obj_$j" for j in 1:nobj])
        cpf_df = cpf_df[:, col_order]

        cpf_path = joinpath(output_dir, "combined_front_Problem_$(prob).csv")
        CSV.write(cpf_path, cpf_df)

        n_total = nrow(cpf_df)
        n_nd = count(cpf_df.is_nd)
        n_bl_nd = count(cpf_df.is_nd .& startswith.(cpf_df.source, "baseline"))
        n_hpo_nd = count(cpf_df.is_nd .& startswith.(cpf_df.source, "hpo"))
        println("    $n_total solutions ($n_nd on front: baseline=$n_bl_nd, hpo=$n_hpo_nd)")

        ideal, nadir = if haskey(bounds_dict, prob)
            bounds_dict[prob]
        else
            compute_empirical_bounds([nd_sols])
        end

        if isnothing(ideal) || isnothing(nadir)
            println("    no bounds available, skipping IGD+")
            continue
        end

        norm_reference = [normalize_objectives(s, ideal, nadir) for s in nd_sols]

        config_groups = Dict{String, Vector{Vector{Float64}}}()
        for (src, sol) in merged
            if !haskey(config_groups, src)
                config_groups[src] = Vector{Float64}[]
            end
            push!(config_groups[src], sol)
        end

        for (config, sols) in sort(collect(config_groups))
            norm_obtained = [normalize_objectives(s, ideal, nadir) for s in sols]

            A = reduce(hcat, norm_obtained)'
            B = reduce(hcat, norm_reference)'
            igd_val = igd_plus(A, B)

            parts = split(config, "_")
            if startswith(config, "baseline_")
                alg = join(parts[2:end], "_")
                push!(igd_rows, (prob, config, "baseline", alg, "", igd_val))
            elseif startswith(config, "hpo_")
                sampler_idx = findfirst(p -> occursin("Sampler", p), parts)
                if !isnothing(sampler_idx)
                    alg = join(parts[2:sampler_idx-1], "_")
                    sampler = parts[sampler_idx]
                    push!(igd_rows, (prob, config, "hpo", alg, sampler, igd_val))
                end
            end
        end
    end

    if nrow(igd_rows) > 0
        igd_path = joinpath(output_dir, "igd_results.csv")
        CSV.write(igd_path, igd_rows)
        println("\n  Saved IGD+ results: $(nrow(igd_rows)) entries to $igd_path")

        println("\n  IGD+ summary (lower is better):")
        summary = combine(groupby(igd_rows, [:source_type, :algorithm]),
            :igd_plus => mean => :mean_igd,
            :igd_plus => std => :std_igd,
            :problem => length => :n,
        )
        for row in eachrow(summary)
            println("    $(row.source_type) | $(row.algorithm): mean=$(round(row.mean_igd, digits=4)) ± $(round(row.std_igd, digits=4)) (n=$(row.n))")
        end
    end
end


function main()
    script_dir = @__DIR__
    results_dir = normpath(dirname(script_dir), "Results")

    if !isdir(results_dir)
        error("Results directory not found: $results_dir")
    end

    println("Results directory: $results_dir")

    baseline_name = nothing
    for prefix in ["baseline_benchmark", "Get_minimum_runs"]
        if isfile(joinpath(results_dir, "reference_fronts_$(prefix)_NSGA2.csv"))
            baseline_name = prefix
            break
        end
    end

    # benchmarks
    if !isnothing(baseline_name)
        println("\n" * "="^60)
        println("Processing benchmarks (baseline: $baseline_name)")
        println("="^60)

        output_dir = joinpath(results_dir, "combined_fronts_benchmark")
        process_experiment(results_dir, baseline_name, "hpo_benchmark_Results", output_dir)
    else
        println("No benchmark baseline found, skipping.")
    end

    # truss
    truss_baseline = nothing
    for prefix in ["baseline_truss", "parametric_truss_example_default_params"]
        test_dir = joinpath(results_dir, "$(prefix)_Results")
        if isdir(test_dir)
            truss_baseline = prefix
            break
        end
    end

    if !isnothing(truss_baseline)
        println("\n" * "="^60)
        println("Processing truss (baseline: $truss_baseline)")
        println("="^60)

        truss_results = joinpath(results_dir, "$(truss_baseline)_Results")
        output_dir = joinpath(results_dir, "combined_fronts_truss")
        process_experiment(truss_results, truss_baseline, "hpo_truss_Results", output_dir)
    else
        println("No truss baseline found, skipping.")
    end

    println("\n" * "="^60)
    println("Done.")
    println("="^60)
end

main()
