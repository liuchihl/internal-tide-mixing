using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra

suffix = "0.1days"

## Simulation parameters
Nx = 400
Ny = 1
Nz = 128

tᶠ = 0.1days # simulation run time
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
N = 1.e-3 # Brunt-Väisälä buoyancy frequency
f₀ = 0.53e-4 # Coriolis frequency
θ = 2.e-3 # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates

# Tidal forcing
U₀ = 0.025
ω₀ = 1.4e-4
u_tidal_forcing(x, y, z, t) = U₀*ω₀*sin(ω₀*t)

# IC such that flow is in phase with predicted linear response, but otherwise quiescent
Uᵣ = U₀ * ω₀^2/(ω₀^2 - f₀^2 - (N*sin(θ))^2) # quasi-resonant linear barotropic response
uᵢ(x, y, z) = -Uᵣ
vᵢ(x, y, z) = 0.
bᵢ(x, y, z) = 1e-9*rand() # seed infinitesimal perturbations in buoyancy field

s = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))
γ = h*π/(s*6kilometers)
print("Steepness parameter of γ=",round(γ, digits=3))

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
    forcing = (u = u_tidal_forcing,),
    closure = ScalarDiffusivity(; ν=1e-4, κ=1e-4),
    tracers = :b,
    timestepper = :RungeKutta3,
    background_fields = (; b=B̄_field),
)

set!(model, b=bᵢ, u=uᵢ, v=vᵢ)

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
κ = model.closure.κ
ε = Field(ν*(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2 + ∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2 + ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2))
χ = @at (Center, Center, Center) κ[1] * (∂x(b)^2 + ∂z(b)^2)

custom_diags = (B=B, uhat=û, what=ŵ, ε=ε, χ=χ)
all_diags = merge(model.velocities, model.tracers, custom_diags)

fname = string("internal_tide_", suffix,"-theta=",string(θ),".jld2")
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