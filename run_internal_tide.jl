## This script uses initialize_internal_tide.jl file to initialize and run a simulation.
# The script should be run on a GPU node by submitting a bash run script as a slurm batch job.

include("initialize_internal_tide.jl")

simname = "smag_tilt"
const Nx = 500
const Ny = 1000
const Nz = 250        
const Δtᵒ = 30minutes # interval for saving output
const tᶠ = 1days      # endtime of the simulation
const θ = 3.6e-3      # slope angle
const U₀ = 0.025      # tidal amplitude
const N = 1.e-3       # Buoyancy frequency
closure = SmagorinskyLilly()
# 3 modes to choose: "spinup", "test", "analysis"
output_mode = "test"
output_writer = true
simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                    Δtᵒ = Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, 
                                    output_mode=output_mode, output_writer=output_writer,
                                    closure=closure)
## Running the simulation!
run!(simulation; pickup=false)
@info """
    Simulation complete.
"""