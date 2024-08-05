## This script uses initialize_internal_tide.jl file to initialize and run a simulation.
# The script should be run on a GPU node by submitting a bash run script as a slurm batch job.

include("initialize_internal_tide.jl")

simname = "tests"
const Nx = 10
const Ny = 4
const Nz = 20        
const Δtᵒ = 30minutes # interval for saving output
const tᶠ = 4days      # endtime of the simulation
const θ = 3.6e-3      # slope angle
const U₀ = 0.025      # tidal amplitude
const N = 1.e-3       # Buoyancy frequency
const f₀ = -0.53e-4   # Coriolis frequency
threeD_snapshot_interval = 12Δtᵒ  # effective only when output_mode="analysis"
closure = SmagorinskyLilly()
# 3 modes to choose: "spinup", "test", "analysis"
output_mode = "test"
output_writer = true
clean_checkpoint = false         # cleanup checkpoint
overwrite_output = true          # overwrite existing output (if pickup=true, clean=false, and vice versa)
timerange = "3.5-4"
simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                Δtᵒ=Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, f₀=f₀,
                                output_mode=output_mode, output_writer=output_writer,
                                threeD_snapshot_interval=threeD_snapshot_interval, 
                                clean_checkpoint=clean_checkpoint, overwrite_output=overwrite_output, 
                                closure=closure, timerange=timerange)
## Running the simulation!
pickup = string("output/", simname, "/internal_tide_theta=0.0036_realtopo3D_Nx=10_Nz=20_3-3.5_checkpoint_iteration11156.jld2")
run!(simulation; pickup=pickup)
@info """
    Simulation complete.
"""