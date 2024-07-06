## this is a script that calls the initialize_internal_tide.jl file to initialize the simulation. Then run the simulation.
# The script should be submitted in the bash file
# change the tilting angle here rather than the source code.


include("initialize_internal_tide.jl")

const θ = 3.6e-3 #1.2e-3
Nx = 500
Ny = 1000
Nz = 250
U₀ = 0.025
# closure = ScalarDiffusivity(; ν=1e-4, κ=1e-4),
closure = SmagorinskyLilly()
simulation = initialize_internal_tide(θ; Nx, Ny, Nz, U₀, closure)

## Running the simulation!
run!(simulation; pickup=true)
@info """
    Simulation complete.
"""