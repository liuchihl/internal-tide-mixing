## This script uses initialize_internal_tide.jl file to initialize and run a simulation.
# The script should be run on a GPU node by submitting a bash run script as a slurm batch job.

include("initialize_internal_tide.jl")

simname = "smag_tilt"
const Nx = 500
const Ny = 1000
const Nz = 250        
const Δtᵒ = 30minutes # interval for saving output
const tᶠ = 3days      # endtime of the simulation
const θ = 3.6e-3      # slope angle
const U₀ = 0.025      # tidal amplitude
const N = 1.e-3       # Buoyancy frequency
const f₀ = -0.53e-4   # Coriolis frequency
threeD_snapshot_interval = 2Δtᵒ
closure = SmagorinskyLilly()
# 3 modes to choose: "spinup", "test", "analysis"
output_mode = "test"
output_writer = true
clean = false         # cleanup and overwrite existing output (if pickup=true, clean=false, and vice versa)
simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                    Δtᵒ=Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, f₀=f₀,
                                    output_mode=output_mode, output_writer=output_writer,
                                    threeD_snapshot_interval=threeD_snapshot_interval, 
                                    clean=clean, closure=closure)
## Running the simulation!
run!(simulation; pickup=true)
@info """
    Simulation complete.
"""