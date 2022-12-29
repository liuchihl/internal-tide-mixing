using Printf
using Oceananigans
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra

## parameters
model_modes = [HydrostaticFreeSurfaceModel, NonhydrostaticModel]
Nt = 43200
h = 200
n = 6 # horizontal mode number
H = 4.e3
L = 2π*H

function boundary_clustered(N, L, ini)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .* L .+ ini
    return z_faces
end

function center_clustered(N, L, ini)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + 3 - Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .* L .+ ini
    return z_faces
end

grid = RectilinearGrid(size=(256, 128), 
        x = (0, L), 
        z = (0, H),
        halo = (4,4),
        topology = (Periodic, Flat, Bounded)
)

# Sinusoidal topography of height h
sinusoidal_bumps(x, y, z) = z < h * (1 + cos(n*(2π*x/L + π)))/2 + 2grid.Δzᵃᵃᶜ # ensure boundary is immersed
grid_with_bumps = ImmersedBoundaryGrid(grid, GridFittedBoundary(sinusoidal_bumps))

# Environmental parameters
N = 1.e-3
f0 = -5.e-5

# Tidal forcing
U0 = 0.01
omega0 = 2π/(12.42*60*60)
tidal_forcing(x, y, z, t) = U0*omega0*cos(omega0*t)

for mode in model_modes
    
    # select model mode
    if mode == HydrostaticFreeSurfaceModel
        model = HydrostaticFreeSurfaceModel(
                    grid = grid_with_bumps,
                    momentum_advection = WENO(),
                    tracer_advection = WENO(),
                    free_surface = ImplicitFreeSurface(gravitational_acceleration=9.81),
                    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-4, κ=1e-4),
                    tracers = :b,
                    buoyancy = BuoyancyTracer(),
                    coriolis = FPlane(f=f0),
                    forcing = (u = tidal_forcing,))
    elseif mode == NonhydrostaticModel
        model = NonhydrostaticModel(
                    grid = grid_with_bumps,
                    advection = WENO(),
                    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-4, κ=1e-4),
                    tracers = :b,
                    buoyancy = BuoyancyTracer(),
                    coriolis = FPlane(f=f0),
                    forcing = (u = tidal_forcing,))
    end

    # Linear stratification
    set!(model, b = (x, y, z) -> N^2 * z)

    progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                                100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                                s.model.clock.time, maximum(abs, model.velocities.w))
    
    Δt = 0.02/N
    
    simulation = Simulation(model, Δt = Δt, stop_iteration = Nt)

    fname = string("internal_tide_", string(mode),".jld2")
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = IterationInterval(50),
                                                          filename = fname,
                                                          overwrite_existing = true)

    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(200))

    run!(simulation)

    @info """
        Simulation complete.
        Output: $(abspath(simulation.output_writers[:fields].filepath))
    """
end

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

for mode in model_modes
    visualize_internal_tide_simulation(string("internal_tide_", string(mode)))
end
