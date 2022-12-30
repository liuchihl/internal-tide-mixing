using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra

suffix = "sinusoidal_20-days"

## Simulation parameters
Nx = 400
Ny = 1
Nz = 128

tᶠ = 20days # simulation run time
Δtᵒ = 30minutes # interval for saving output

H = 3kilometers # 6.e3 # vertical extent
L = 2*2H # 60.e3 # horizontal extent

n = 2 # topographic wave number
h = 200meters # topographic height

## Create grid
# Creates a vertical grid with near-constant spacing `refinement * Lz / Nz` near the bottom:
# "Warped" coordinate
kwarp(k, N) = (N + 1 - k) / N
# Linear near-surface generator
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
# Bottom-intensified stretching function
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))

# Generating function
z_faces(k) = - H * (ζ(k, Nz, 1.8) * Σ(k, Nz, 10) - 1)

grid = RectilinearGrid(size=(Nx, Ny, Nz), 
        x = (0, L),
        y = (0, L), 
        z = z_faces,
        halo = (4,4,4),
        topology = (Periodic, Periodic, Bounded)
)

# Sinusoidal topography of height h and mode number n
topog(x, y, z) = h * (1 + cos(n*2π*x/L + π))/2 + 2minimum(grid.Δzᵃᵃᶜ)
topog_mask(x, y, z) = z < topog(x, y, z)

# Create immersed boundary grid
grid_with_bumps = ImmersedBoundaryGrid(grid, GridFittedBoundary(topog_mask))

# Environmental parameters
N = 1.e-3
f₀ = 0.53e-4

# Tidal forcing
U₀ = 0.025
ω₀ = 1.4e-4
tidal_forcing(x, y, z, t) = U₀*ω₀*cos(ω₀*t)

s = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))
γ = h*π/(s*6kilometers)
print("Wave steepness of γ=",round(γ, digits=3))

model = NonhydrostaticModel(
    grid = grid_with_bumps,
    advection = WENO(),
    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-4, κ=1e-4),
    tracers = :b,
    buoyancy = BuoyancyTracer(),
    coriolis = FPlane(f=f₀),
    forcing = (u = tidal_forcing,))

# Linear stratification
bᵢ(x, y, z) = N^2*z + 1e-9*rand()
uᵢ(x, y, z) = 0.
vᵢ(x, y, z) = U₀*(ω₀*f₀)/(ω₀^2 - f₀^2)
set!(model, b=bᵢ, u=uᵢ, v=vᵢ)

## Configure simulation
Δt = (1/N)*0.03
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

## Diagnostics
u, v, w = model.velocities
ν = model.closure.ν
ε = Field(ν*(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2 + ∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2 + ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2))
custom_diags = (ε=ε,)
all_diags = merge(model.velocities, model.tracers, custom_diags)

fname = string("internal_tide_", suffix,".jld2")
simulation.output_writers[:fields] = JLD2OutputWriter(model, all_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        filename = fname,
                                        overwrite_existing = true)
## Progress messages
progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))

## Running the simulation!
run!(simulation)

@info """
    Simulation complete.
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""