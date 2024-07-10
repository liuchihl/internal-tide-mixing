## This script uses initialize_internal_tide.jl file to initialize and run a simulation.
# The script should be run on a GPU node by submitting a bash run script as a slurm batch job.

include("initialize_internal_tide.jl")

const simname = "reference"
const Nx = 500
const Ny = 1000
const Nz = 250
const θ = 3.6e-3
const U₀ = 0.025
closure = SmagorinskyLilly()
simulation = initialize_internal_tide(simname, Nx, Ny, Nz; θ=θ, U₀=U₀, closure=closure)

## Running the simulation!
run!(simulation; pickup=true)
@info """
    Simulation complete.
"""