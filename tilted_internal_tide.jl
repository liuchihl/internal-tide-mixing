using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra

suffix = "tilted-test"

## Simulation parameters
Nx = 200
Ny = 1
Nz = 128

tᶠ = 60minutes # simulation run time
Δtᵒ = 30minutes # interval for saving output

H = 3kilometers # 6.e3 # vertical extent
L = 2H # 60.e3 # horizontal extent

n = 1 # topographic wave number
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
N = 1.e-3 # Brunt-Väisälä buoyancy frequency
f₀ = 0.53e-4 # Coriolis frequency
θ = 0.1 # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
ĝ = (sin(θ), 0, cos(θ))

# Tidal forcing
U₀ = 0.025
ω₀ = 1.4e-4
u_tidal_forcing(x, y, z, t) = U₀*ω₀*cos(ω₀*t)*(+ĝ[3])
w_tidal_forcing(x, y, z, t) = U₀*ω₀*cos(ω₀*t)*(-ĝ[1])

s = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))
γ = h*π/(s*6kilometers)
print("Wave steepness of γ=",round(γ, digits=3))

# Rotate gravity vector
buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = ĝ)
coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)

# Linear background stratification (in ẑ)
@inline ẑ(x, z, ĝ) = x*ĝ[1] .+ z*ĝ[3]
@inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

model = NonhydrostaticModel(
    grid = grid_with_bumps,
    advection = WENO(),
    buoyancy = buoyancy,
    coriolis = coriolis,
    forcing = (u = u_tidal_forcing, w = w_tidal_forcing),
    closure = ScalarDiffusivity(; ν=1e-4, κ=1e-4),
    tracers = :b,
    timestepper = :RungeKutta3,
    background_fields = (; b=B̄_field),
)

bᵢ(x, y, z) = 1e-9*rand()
uᵢ(x, y, z) = 0.
vᵢ(x, y, z) = U₀*(ω₀*f₀)/(ω₀^2 - f₀^2)
wᵢ(x, y, z) = 0.
set!(model, b=bᵢ, u=uᵢ, v=vᵢ, w=wᵢ)

## Configure simulation
Δt = (1/N)*0.03
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

## Diagnostics
b = model.tracers.b
B̄ = model.background_fields.tracers.b
B = B̄ + b # total buoyancy field

u, v, w = model.velocities
û = @at (Face, Center, Center) u*ĝ[3] + w*ĝ[1] # true zonal velocity
ŵ = @at (Center, Center, Face) w*ĝ[3] - u*ĝ[1] # true vertical velocity

ν = model.closure.ν
ε = Field(ν*(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2 + ∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2 + ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2))

custom_diags = (B=B, û=û, ŵ=ŵ, ε=ε,)
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