using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using JLD2
using Statistics

function nice_divergent_levels(c, clim; nlevels=20)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return (-clim, clim), levels
end

function nan_solid(x, z, u, mask)
    Nx, Nz = size(u)
    x2 = reshape(x, (Nx, 1))
    z2 = reshape(z, (1, Nz))
    u[mask] .= NaN
    return u
end

saved_output_filename = "internal_tide_40days-theta=0.0.jld2"
file = jldopen(saved_output_filename)

ig = file["serialized/grid"]
ug = ig.underlying_grid
ĝ = file["serialized/buoyancy"].gravity_unit_vector

ε = FieldTimeSeries(saved_output_filename, "ε")
ε̄ = log10.(mean(interior(ε, :, 1, :, :)[:,:,end-200:end], dims=(3))[:,:])

t = ε.times

xb, yb, zb = nodes(ε[1])

begin
    fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10))
    axis_kwargs = (xlabel = "zonal distance (x)",
                      ylabel = "elevation (z)",
                      limits = ((0, ug.Lx), (0, ug.Lz)),
                      )
    ax_u = Axis(fig[2, 1]; title = "zonal velocity (u) and equally-spaced buoyancy contours (B)", axis_kwargs...)
    ax_ε = Axis(fig[3, 1]; title = "TKE dissipation rate (ε) and equally-spaced buoyancy contours (B)", axis_kwargs...)
end
fig[1, :] = Label(fig, "t = 1", fontsize=20, tellwidth=false)

hm_ε = heatmap!(ax_ε, xb, zb, ε̄,
    colorrange = (-11, -9), colormap = :matter,
    lowclip=cgrad(:matter)[1], highclip=cgrad(:matter)[end])
Colorbar(fig[2,2], hm_ε)
Colorbar(fig[3,2], hm_ε)
current_figure()

frames = (1:length(t))
filename = join(split(saved_output_filename, ".")[1:end-1], ".")
record(fig, string(filename,".mp4"), frames, framerate=16) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end