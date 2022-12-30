using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra
using Plots

suffix = "10-day-test"

## Simulation parameters
Nx = 256
Ny = 1
Nz = 128

tᶠ = 10days # simulation run time
Δtᵒ = 30minutes # interval for saving output

H = 3kilometers # 6.e3 # vertical extent
L = H # 60.e3 # horizontal extent

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

plot(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz], xlim=(0, NaN))

# Sinusoidal topography of height h and mode number n
h = 200meters
n = 3
topog_mask(x, y, z) = z < h * (1 + cos(n*(2π*x/L + π)))/2 + 2minimum(grid.Δzᵃᵃᶜ)

# Create immersed boundary grid
grid_with_bumps = ImmersedBoundaryGrid(grid, GridFittedBoundary(topog_mask))

# Environmental parameters
N = 1.e-3
f₀ = 0.53e-4

# Tidal forcing
U₀ = 0.025
ω₀ = 1.4e-4
tidal_forcing(x, y, z, t) = U₀*ω₀*cos(ω₀*t)

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

fname = string("internal_tide_NL2010_", suffix,".jld2")
simulation.output_writers[:fields] = JLD2OutputWriter(model, all_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        filename = fname,
                                        overwrite_existing = true)
## Progress messages

progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))

run!(simulation)

@info """
    Simulation complete.
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""

using JLD2
using Plots

function nice_divergent_levels(c, clim; nlevels=20)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return (-clim, clim), levels
end

function nan_solid(x, z, u, topog)
    Nx, Nz = size(u)
    x2 = reshape(x, (Nx, 1))
    z2 = reshape(z, (1, Nz))
    u[topog.(x2, 0., z2)] .= NaN
    return nothing
end

function visualize_internal_tide_simulation(prefix)

    filename = prefix * ".jld2"
    file = jldopen(filename)

    grid = file["serialized/grid"]

    xu, yu, zu = nodes((Face, Center, Center), grid)
    xw, yw, zw = nodes((Center, Center, Face), grid)
    xb, yb, zb = nodes((Center, Center, Center), grid)

    b₀ = file["timeseries/b/0"][:, 1, :]

    iterations = parse.(Int, keys(file["timeseries/t"]))    

    anim = @animate for (i, iter) in enumerate(iterations)

        @info "Plotting iteration $iter of $(iterations[end])..."

        u = file["timeseries/u/$iter"][:, 1, :]
        b = file["timeseries/b/$iter"][:, 1, :]
        ε = file["timeseries/ε/$iter"][:, 1, :]
        t = file["timeseries/t/$iter"]

        b′ = b .- b₀

        ulims, ulevels = nice_divergent_levels(u, 2*U₀)
        blims, blevels = nice_divergent_levels(b′, N^2*grid.Lz/1000.)
        
        nan_solid(xu, zu, u, topog_mask)
        nan_solid(xb, zb, ε, topog_mask)
        nan_solid(xb, zb, b, topog_mask)

        u_title = @sprintf("x velocity, t = %.2f", t)

        u_plot = contourf(xu, zu, u'; title = u_title,                  color = :balance, linewidth = 0, levels = ulevels, clims = ulims)
        ε_plot = contourf(xb, zb, log10.(ε)'; title = "dissipation",    color = :matter, linewidth = 0, clims = (-11., -7.))
        b_plot = contourf(xb, zb, b′'; title = "buoyancy perturbation", color = :delta, linewidth = 0, levels = blevels, clims = blims)

        plot(u_plot, b_plot, ε_plot, layout = (3, 1), xlims=(0, L), ylims=(0, H/2), size = (1200, 1600), dpi=200)
    end

    mp4(anim, prefix * ".mp4", fps = 20)

    close(file)
end

visualize_internal_tide_simulation(string("internal_tide_NL2010_", suffix))
