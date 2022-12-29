using Printf
using Oceananigans
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra

## parameters
Nt = 10
H = 6.e3
L = H # 60.e3

grid = RectilinearGrid(size=(256, 1, 256), 
        x = (0, L),
        y = (0, L), 
        z = (0, H),
        halo = (4,4,4),
        topology = (Periodic, Periodic, Bounded)
)

# Sinusoidal topography of height h and mode number 6
h = 200
n = 6
sinusoidal_bumps(x, y, z) = z < h * (1 + cos(n*(2π*x/L + π)))/2 + 2grid.Δzᵃᵃᶜ # ensure boundary is immersed

# Create immersed boundary grid
grid_with_bumps = ImmersedBoundaryGrid(grid, GridFittedBoundary(sinusoidal_bumps))

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
vᵢ(x, y, z) = 0.#U₀*(ω₀*f₀)/(ω₀^2 - f₀^2)
set!(model, b=bᵢ, u=uᵢ, v=vᵢ)

progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w))

Δt = 0.02/N

simulation = Simulation(model, Δt = Δt, stop_iteration = Nt)

fname = string("internal_tide_NL2010_testing.jld2")
simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                        schedule = IterationInterval(1),
                                                        filename = fname,
                                                        overwrite_existing = true)

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(1))

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
    x2 = reshape(x, Nx, 1)
    z2 = reshape(z, 1, Nz)
    u[topog.(x2, 0, z2)] .= NaN
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
        w = file["timeseries/w/$iter"][:, 1, :]
        b = file["timeseries/b/$iter"][:, 1, :]
        t = file["timeseries/t/$iter"]

        b′ = b .- b₀

        ulims, ulevels = nice_divergent_levels(u, 2*U0)
        wlims, wlevels = nice_divergent_levels(w, U0/5)
        blims, blevels = nice_divergent_levels(b′, N^2*grid.Lz/1000.)
        
        nan_solid(xu, zu, u, sinusoidal_bumps)
        nan_solid(xw, zw, w, sinusoidal_bumps)
        nan_solid(xb, zb, b, sinusoidal_bumps)

        u_title = @sprintf("x velocity, t = %.2f", t)

        u_plot = contourf(xu, zu, u'; title = u_title,                  color = :balance, linewidth = 0, levels = ulevels, clims = ulims)
        w_plot = contourf(xw, zw, w'; title = "z velocity",             color = :balance, linewidth = 0, levels = wlevels, clims = wlims)
        b_plot = contourf(xb, zb, b′'; title = "buoyancy perturbation", color = :balance, linewidth = 0, levels = blevels, clims = blims)

        plot(u_plot, w_plot, b_plot, layout = (3, 1), xlims=(0, L), ylims=(0, H), size = (1200, 1600), dpi=200)
    end

    mp4(anim, prefix * ".mp4", fps = 20)

    close(file)
end

visualize_internal_tide_simulation(string("internal_tide_NL2010_testing"))
