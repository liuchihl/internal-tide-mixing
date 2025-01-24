using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
# using BSplineKit
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
# filename_field = "output/internal_tide_3days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_xz.nc"
# ds_field = Dataset(filename_field,"r")
# bathy_data = "output/bathymetry.nc"
# Bathy = Dataset(bathy_data,"r")
filename_slice = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_yz.nc"
ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_slice["zC"]; Nz=length(zC)
xC = ds_slice["xC"]; Nx=length(xC)
yC = ds_slice["yC"]; Ny=length(yC)
t = ds_slice["time"];

# load all data
B = ds_slice["B"].var;        B = B[:,:,:,:];
uhat = ds_slice["uhat"].var; uhat = uhat[:,:,:,:];
# what = ds_slice["what"].var; #what = what[:,:,:,:];
# wb = ds_slice["wb"].var;     #wb = wb[:,:,:,:];
ε = ds_slice["ε"].var;       ε = ε[:,:,:,:];
χ = ds_slice["χ"].var;       χ = χ[:,:,:,:];


# set topography to NaN
uhat[uhat.==0] .= NaN
# what[what.==0] .= NaN
# wb[isnan.(uhat)] .= NaN
ε[isnan.(uhat)] .= NaN
B[isnan.(uhat)] .= NaN
χ[isnan.(uhat)] .= NaN

# dBdz[isnan.(dBdz)] .= NaN
# dûdz[isnan.(dûdz)] .= NaN

# close(ds_field)


# derivatives of B and u. Only slices are used because it can be memory intensive
# dBdz = zeros(size(B[:,1,:,:])); dûdz = zeros(size(uhat[:,1,:,:]));

# for i in 1:length(t)
#     for j in 1:length(xC)
#       k = Int(round(length(yC)*0.5))
#         dBdz[j,:,i] = deriv(zC,B[j,k,:,i])
#         dûdz[j,:,i] = deriv(zC,uhat[j,k,:,i])
#     end
# end






# plot
n = Observable(1)
ûₙ = @lift(uhat[1,:,:,$n])
εₙ = @lift(log10.(ε[1,:,:,$n]))
Bₙ = @lift(B[1,:,:,$n])
χₙ = @lift(log10.(χ[1,:,:,$n]))
# dBdzₙ = @lift interior(dBdz[$n], :, 1, :)
# dûdzₙ = @lift interior(dûdz[$n], :, 1, :)
ω₀ = 1.4e-4
M₂_period = 2π/ω₀



fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs = (xlabel = "Cross canyon distance y (km)",
                  ylabel = "Elevation z (m)",
                  limits = ((0, ds_slice["yF"][end]), (0, ds_slice["zF"][end]-1000)),
                  )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "u and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_uz = Axis(fig[1, 2]; title = "du/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_bz = Axis(fig[2, 1]; title = "db/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_ε = Axis(fig[3, 1]; title = "TKE dissipation rate (log ε) and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_χ = Axis(fig[4, 1]; title = "Scalar variance dissipation rate (log χ) and equally-spaced buoyancy contours (B)", axis_kwargs...)


using ColorSchemes
U₀ = 0.025
hm_u = heatmap!(ax_u, yC[:], zC[:], ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :rainbow,
    lowclip=cgrad(:rainbow)[1], highclip=cgrad(:rainbow)[end],clim = (-3U₀,3U₀),
    nan_color = :gray)
ct_u = contour!(ax_u, yC, zC, Bₙ,
    levels=0.:.2e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u; label = "m/s")

hm_ε = heatmap!(ax_ε, yC[:], zC[:], εₙ,
    colorrange = (-10.5, -8.5), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    clim = (-10.5,-8.5),nan_color = :gray)
ct_ε = contour!(ax_ε, yC, zC, Bₙ,
    levels=0.:.2e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε)


hm_χ = heatmap!(ax_χ, yC[:], zC[:], χₙ,
    colorrange = (-17, -13), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    clim = (-17,-13),nan_color = :gray)
ct_χ = contour!(ax_χ, yC, zC, Bₙ,
    levels=0.:2e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[4,2], hm_χ)

frames = (1:length(t))

filename = join(split(filename_slice, ".")[1:end-1], ".")

record(fig, string(filename,".mp4"), frames, framerate=16) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_slice)
