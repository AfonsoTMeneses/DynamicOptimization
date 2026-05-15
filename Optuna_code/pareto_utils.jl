using CSV
using DataFrames
using Printf
import Metaheuristics.PerformanceIndicators: igd_plus

const MAX_REFERENCE_FRONT_SIZE = 2000

# Dedup

function dedupe_points(points::Vector{Vector{Float64}})
    isempty(points) && return points
    seen = Set{Tuple}()
    unique_points = Vector{Vector{Float64}}()
    for point in points
        key = Tuple(point)
        if !(key in seen)
            push!(seen, key)
            push!(unique_points, point)
        end
    end
    return unique_points
end

function dedupe_labelled_points(labelled_points::Vector{Tuple{String, Vector{Float64}}})
    isempty(labelled_points) && return labelled_points
    seen = Set{Tuple}()
    unique_labelled_points = Vector{Tuple{String, Vector{Float64}}}()
    for (source_label, point) in labelled_points
        key = Tuple(point)
        if !(key in seen)
            push!(seen, key)
            push!(unique_labelled_points, (source_label, point))
        end
    end
    return unique_labelled_points
end

# Non-dominance (lex-sorted Kung variant)

function nondominated_2d(points::Vector{Vector{Float64}})
    isempty(points) && return Vector{Float64}[]
    sorted_points = sort(points, by = p -> (p[1], p[2]))
    nondominated = Vector{Vector{Float64}}()
    smallest_obj2_seen = Inf
    for point in sorted_points
        if point[2] < smallest_obj2_seen
            push!(nondominated, point)
            smallest_obj2_seen = point[2]
        end
    end
    return nondominated
end

function nondominated_kd(points::Vector{Vector{Float64}}; progress_label::String="")
    isempty(points) && return Vector{Float64}[]
    point_count = length(points)
    point_count == 1 && return copy(points)
    objective_count = length(first(points))

    lex_sorted_indices = sortperm(points)
    kept_points = Vector{Vector{Float64}}()
    sizehint!(kept_points, min(point_count, 1 << 16))

    last_progress = time()
    @inbounds for (i, source_index) in enumerate(lex_sorted_indices)
        candidate = points[source_index]
        is_dominated = false
        for kept_index in length(kept_points):-1:1
            existing = kept_points[kept_index]
            all_le = true
            any_lt = false
            for j in 1:objective_count
                ej = existing[j]; cj = candidate[j]
                if ej > cj
                    all_le = false
                    break
                elseif ej < cj
                    any_lt = true
                end
            end
            if all_le && any_lt
                is_dominated = true
                break
            end
        end
        is_dominated || push!(kept_points, candidate)

        if !isempty(progress_label) && time() - last_progress > 5.0
            @printf("    %s: processed %d/%d, kept=%d\n",
                    progress_label, i, point_count, length(kept_points))
            flush(stdout)
            last_progress = time()
        end
    end
    return kept_points
end

function compute_nondominated(points::Vector{Vector{Float64}}; progress_label::String="")
    isempty(points) && return Vector{Float64}[]
    return length(first(points)) == 2 ?
        nondominated_2d(points) :
        nondominated_kd(points; progress_label=progress_label)
end

# Spacing filter

function spacing_filter(points::Vector{Vector{Float64}},
                        ideal::Vector{Float64},
                        nadir::Vector{Float64};
                        eps_distance::Float64=0.0)
    (eps_distance <= 0.0 || length(points) <= 1) && return points

    objective_count = length(first(points))
    ranges = [max(nadir[j] - ideal[j], 1e-12) for j in 1:objective_count]
    normalized_points = [[(p[j] - ideal[j]) / ranges[j] for j in 1:objective_count] for p in points]

    extreme_indices = Set{Int}()
    for j in 1:objective_count
        push!(extreme_indices, argmin([np[j] for np in normalized_points]))
    end

    sort_order = sortperm(normalized_points; by = p -> p[1])

    kept_indices    = Int[]
    kept_normalized = Vector{Vector{Float64}}()

    function admit!(i)
        push!(kept_indices, i)
        push!(kept_normalized, normalized_points[i])
    end

    for i in sort_order
        i in extreme_indices && admit!(i)
    end

    eps_squared = eps_distance^2
    for i in sort_order
        i in extreme_indices && continue
        candidate = normalized_points[i]
        too_close = false
        for kept in kept_normalized
            dist_squared = 0.0
            @inbounds for j in 1:objective_count
                d = candidate[j] - kept[j]
                dist_squared += d * d
                dist_squared >= eps_squared && break
            end
            if dist_squared < eps_squared
                too_close = true
                break
            end
        end
        too_close || admit!(i)
    end

    return [points[i] for i in kept_indices]
end

# Downsampling

function downsample_front_2d(front::Vector{Vector{Float64}}, target_size::Int)
    sorted_front = sort(front, by = p -> (p[1], p[2]))
    stride = max(1, ceil(Int, length(sorted_front) / target_size))
    downsampled = sorted_front[1:stride:end]
    first_extreme, last_extreme = sorted_front[1], sorted_front[end]
    downsampled[1]   == first_extreme || pushfirst!(downsampled, first_extreme)
    downsampled[end] == last_extreme  || push!(downsampled, last_extreme)
    return downsampled
end

function downsample_front_kd(front::Vector{Vector{Float64}}, target_size::Int)
    objective_count = length(first(front))
    axis_minimums = [minimum(p[j] for p in front) for j in 1:objective_count]
    axis_maximums = [maximum(p[j] for p in front) for j in 1:objective_count]
    axis_spans    = [max(axis_maximums[j] - axis_minimums[j], 1.0) for j in 1:objective_count]

    cells_per_axis = max(2, ceil(Int, (2 * target_size)^(1 / objective_count)))
    representative_per_cell = Dict{NTuple{objective_count, Int}, Vector{Float64}}()

    for point in front
        cell_key = ntuple(objective_count) do j
            clamp(floor(Int, (point[j] - axis_minimums[j]) / axis_spans[j] * cells_per_axis),
                  0, cells_per_axis - 1)
        end
        haskey(representative_per_cell, cell_key) || (representative_per_cell[cell_key] = point)
    end
    return collect(values(representative_per_cell))
end

function downsample_reference_front(front::Vector{Vector{Float64}}, target_size::Int)
    length(front) <= target_size && return front
    return length(first(front)) == 2 ?
        downsample_front_2d(front, target_size) :
        downsample_front_kd(front, target_size)
end

# Combined-front IO

function load_combined_front_cpf(combined_fronts_dir::AbstractString, problem_number::Int)
    csv_path = joinpath(combined_fronts_dir, "combined_front_Problem_$(problem_number).csv")
    isfile(csv_path) || return (nothing, nothing)

    df = CSV.read(csv_path, DataFrame; pool=false, silencewarnings=true)
    objective_columns = sort([string(c) for c in names(df) if startswith(string(c), "obj_")])
    isempty(objective_columns) && return (nothing, nothing)

    cpf_points = Vector{Float64}[]
    for row in eachrow(df)
        row.is_nd || continue
        push!(cpf_points, Float64[row[Symbol(c)] for c in objective_columns])
    end
    return (cpf_points, length(objective_columns))
end

# Reference normalization

function build_normalized_reference(cpf_points::Vector{Vector{Float64}},
                                     ideal::Vector{Float64},
                                     nadir::Vector{Float64};
                                     max_size::Int=MAX_REFERENCE_FRONT_SIZE)
    downsampled = length(cpf_points) > max_size ?
        downsample_reference_front(cpf_points, max_size) : cpf_points
    ranges = [max(nadir[j] - ideal[j], 1e-12) for j in eachindex(ideal)]
    normalized_rows = [[(p[j] - ideal[j]) / ranges[j] for j in eachindex(p)] for p in downsampled]
    return reduce(hcat, normalized_rows)'
end

# IGD+ on dedup+ND-filtered, normalized obtained points

function compute_igd_plus(obtained_points::Vector{Vector{Float64}},
                          normalized_reference_matrix,
                          ideal::Vector{Float64},
                          nadir::Vector{Float64})
    isempty(obtained_points) && return (0, NaN)
    unique_points = dedupe_points(obtained_points)
    nd_points = compute_nondominated(unique_points)
    isempty(nd_points) && return (0, NaN)

    ranges = [max(nadir[j] - ideal[j], 1e-12) for j in eachindex(ideal)]
    normalized = [[(p[j] - ideal[j]) / ranges[j] for j in eachindex(p)] for p in nd_points]
    obtained_matrix = reduce(hcat, normalized)'
    return (length(nd_points), igd_plus(obtained_matrix, normalized_reference_matrix))
end

# Experiment paths

function experiment_paths(experiment::AbstractString, results_dir::AbstractString)
    if experiment == "benchmark"
        hpo_dir = joinpath(results_dir, "hpo_benchmark_Results")
        return (
            name              = "benchmark",
            hpo_dir           = hpo_dir,
            hpo_dir_name      = "hpo_benchmark_Results",
            optuna_studies    = joinpath(hpo_dir, "optuna_studies"),
            cpf_dir           = joinpath(results_dir, "combined_fronts_benchmark"),
            baseline_dir      = results_dir,
            baseline_name     = "baseline_benchmark",
            iter_string       = "100",
        )
    elseif experiment == "truss"
        baseline_dir = joinpath(results_dir, "baseline_truss_Results")
        hpo_dir      = joinpath(results_dir, "hpo_truss_Results")
        return (
            name              = "truss",
            hpo_dir           = hpo_dir,
            hpo_dir_name      = "hpo_truss_Results",
            optuna_studies    = joinpath(hpo_dir, "optuna_studies"),
            cpf_dir           = joinpath(results_dir, "combined_fronts_truss"),
            baseline_dir      = baseline_dir,
            baseline_name     = "baseline_truss",
            iter_string       = "50",
        )
    else
        error("Unknown --experiment: '$experiment'. Use 'benchmark' or 'truss'.")
    end
end

# Baseline-name discovery (looks for legacy names too)

function find_benchmark_baseline_name(results_dir::AbstractString)
    for candidate in ["baseline_benchmark", "Get_minimum_runs"]
        marker = joinpath(results_dir, "reference_fronts_$(candidate)_NSGA2.csv")
        isfile(marker) && return candidate
    end
    return nothing
end

function find_truss_baseline_name(results_dir::AbstractString)
    for candidate in ["baseline_truss", "parametric_truss_example_default_params"]
        isdir(joinpath(results_dir, "$(candidate)_Results")) && return candidate
    end
    return nothing
end

# Output safeguards

function refuse_destructive_output(output_dir::AbstractString,
                                    results_dir::AbstractString;
                                    output_name::Union{AbstractString, Nothing}=nothing)
    abs_output  = abspath(output_dir)
    abs_results = abspath(results_dir)
    if startswith(abs_output, abs_results * "/") || abs_output == abs_results
        error("Refusing to write inside Results/ ($abs_results). Choose --output-dir OUTSIDE the Results tree.")
    end
    if !isnothing(output_name)
        output_csv = joinpath(output_dir, output_name)
        if isfile(output_csv)
            error("Refusing to overwrite existing $output_csv. Move or rename it first.")
        end
        return (abs_output, output_csv)
    end
    return abs_output
end
