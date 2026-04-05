# setup.jl — Install all dependencies for DynamicOptimization
# 
# Run this once from the project root (Optuna/ directory):
#   cd Optuna/
#   julia --project=. setup.jl


using Pkg

Pkg.add([
    "CSV",
    "DataFrames",
    "DataStructures",
    "JSON",
    "PyCall",
    "Statistics",
])


Pkg.add("Metaheuristics")
Pkg.add("HardTestProblems")


try
    Pkg.add(url="https://github.com/aptmcl/KhepriFrame3DD.jl")
    println("KhepriFrame3DD installed successfully.")
catch e
    @warn "KhepriFrame3DD installation failed: $e\n  Truss scripts will not work, but benchmark scripts are unaffected."
end

println("Setup complete!")
println("\nNext steps:")
println("  1. Ensure Optuna is installed in your Python environment:")
println("     pip install optuna")
println("  2. Run the baseline: julia --project=.. Optuna_code/baseline_benchmark.jl")
println("  3. Run HPO:          julia --project=.. Optuna_code/hpo_benchmark.jl")
