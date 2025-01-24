include("internal_tide_setup.jl")

tᶠ = 10*2π/ω₀    # endtime of the simulation 
θ = 3.6e-3      # slope angle
run_internal_tide(tᶠ,θ)
