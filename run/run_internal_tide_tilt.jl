include("internal_tide_setup.jl")
# Parameters modified at run time
const tp_end = 10
# Derived parameters
const ω₀ = 1.4e-4     # tidal freq.
const tᶠ  = tp_end*2π/ω₀
θ = 3.6e-3      # slope angle
run_internal_tide(tᶠ,θ)
