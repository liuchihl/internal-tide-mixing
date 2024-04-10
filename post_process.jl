using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations


function deriv(x,y)
spl = interpolate(x, y, BSplineOrder(6))
D1f = diff(spl, Derivative(1))   # change
    return D1f.(x)
end

# function nice_divergent_levels(c, clim; nlevels=20)
#     levels = range(-clim, stop=clim, length=nlevels)
#     cmax = maximum(abs, c)
#     clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
#     return (-clim, clim), levels
# end

## load data
filename_field = "output/internal_tide_3days-theta=0.0036_realtopo3D_Nx10_field.nc"
ds_field = Dataset(filename_field,"r")
bathy_data = "output/bathymetry.nc"
Bathy = Dataset(bathy_data,"r")
# filename_slice = "internal_tide_0.1days-theta=0.002.jld2.nc"
# ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC)
xC = ds_field["xC"]; Nx=length(xC)
yC = ds_field["yC"]; Ny=length(yC)
t = ds_field["time"];

# load all data
B = ds_field["B"].var;       B = B[:,:,:,:];
uhat = ds_field["uhat"].var; uhat = uhat[:,:,:,:];
what = ds_field["what"].var; what = what[:,:,:,:];
wb = ds_field["wb"].var;     wb = wb[:,:,:,:];
ε = ds_field["ε"].var; ε = ε[:,:,:,:];
χ = ds_field["χ"].var; χ = χ[:,:,:,:];

# set topography to NaN
uhat[uhat.==0] .= NaN
what[what.==0] .= NaN
wb[isnan.(uhat)] .= NaN
ε[isnan.(uhat)] .= NaN
B[isnan.(uhat)] .= NaN
# dBdz[isnan.(dBdz)] .= NaN
# dûdz[isnan.(dûdz)] .= NaN

# close(ds_field)
# close(ds_slice)


# derivatives of B and u. Only slices are used because it can be memory intensive
# dBdz = zeros(size(B[:,1,:,:])); dûdz = zeros(size(uhat[:,1,:,:]));

# for i in 1:length(t)
#     for j in 1:length(xC)
#       k = Int(round(length(yC)*0.5))
#         dBdz[j,:,i] = deriv(zC,B[j,k,:,i])
#         dûdz[j,:,i] = deriv(zC,uhat[j,k,:,i])
#     end
# end

n = Observable(1)
ûₙ = @lift(uhat[:,1,:,$n])
εₙ = @lift(ε[:,1,:,$n])
Bₙ = @lift(B[:,1,:,$n])
# dBdzₙ = @lift interior(dBdz[$n], :, 1, :)
# dûdzₙ = @lift interior(dûdz[$n], :, 1, :)
ω₀ = 1.4e-4
M₂_period = 2π/ω₀
title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)


fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,600))
axis_kwargs = (xlabel = "zonal distance (x)",
                  ylabel = "elevation (z)",
                  limits = ((0, ds_field["xF"][end]), (0, ds_field["zF"][end])),
                  )

ax_u = Axis(fig[1, 1]; title = "u and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_uz = Axis(fig[1, 2]; title = "du/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_bz = Axis(fig[2, 1]; title = "db/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_ε = Axis(fig[2, 1]; title = "TKE dissipation rate (ε) and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_B = Axis(fig[3, 1]; title = "B", axis_kwargs...)


U₀ = 0.025
hm_u = heatmap!(ax_u, xC[:], zC[:], ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :balance,
    lowclip=cgrad(:balance)[1], highclip=cgrad(:balance)[end])
ct_u = contour!(ax_u, xC, zC, Bₙ,
    levels=0.:0.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u)

hm_ε = heatmap!(ax_ε, xC[:], zC[:], εₙ,
    colorrange = (-10.5, -8.5), colormap = :matter,
    lowclip=cgrad(:matter)[1], highclip=cgrad(:matter)[end])
ct_ε = contour!(ax_ε, xC, zC, Bₙ,
    levels=0.:0.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε)

frames = (1:length(t))
filename = join(split(filename_field, ".")[1:end-1], ".")
record(fig, string(filename,".mp4"), frames, framerate=16) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end
