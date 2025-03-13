include("internal_tide_setup.jl")
# Parameters modified at run time
const tp_end = 460
analysis_round = 2
# If the simulation is not yet run to the analysis period, set analysis_round to nothing
analysis_round = tp_end >= 450 ? analysis_round : nothing
# Derived parameters
const ω₀ = 1.4e-4     # tidal freq.
const tᶠ  = tp_end*2π/ω₀
const θ = 0      # slope angle
run_internal_tide(; tᶠ,θ, analysis_round)