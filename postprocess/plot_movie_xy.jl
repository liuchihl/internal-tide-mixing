using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using LinearAlgebra
using Interpolations
using MAT

# function deriv(x,y)
# spl = interpolate(x, y, BSplineOrder(6))
# D1f = diff(spl, Derivative(1))   # change
#     return D1f.(x)
# end

function deriv(x,y)
    dydx =  diff(y[:,:,:,:],dims=3)./reshape(diff(x[:]),1,1,length(zC)-1)
    return dydx
 end

# function nice_divergent_levels(c, clim; nlevels=20)
#     levels = range(-clim, stop=clim, length=nlevels)
#     cmax = maximum(abs, c)
#     clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
#     return (-clim, clim), levels
# end

## load data
simname = "tilt"
tꜝ = "10"

## load data
# filename_field = "output/internal_tide_3days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_xz.nc"
# ds_field = Dataset(filename_field,"r")
# bathy_data = "output/bathymetry.nc"
# Bathy = Dataset(bathy_data,"r")
# filename_slice = "output/supercritical_tilt/backgroundfluxdivergence_smagorinky/backgroundfluxdivergence_smagorinkyinternal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_0_5_xz.nc"
filename_slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_",tꜝ,"_slices_xy.nc")
# filename_slice = "output/no_tilt/internal_tide_5days-theta=0_realtopo3D_Nx500_Nz250_slices_30_50_xz.nc"
ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_slice["zC"]; Nz=length(zC)
xC = ds_slice["xC"]; Nx=length(xC)
yC = ds_slice["yC"]; Ny=length(yC)
t = ds_slice["time"];

# load all data
B = ds_slice["B"].var;        B = B[:,:,:,:];
b = ds_slice["b"].var;        b = b[:,:,:,:];
uhat = ds_slice["uhat"].var; uhat = uhat[:,:,:,:];
# what = ds_slice["what"].var; #what = what[:,:,:,:];
# wb = ds_slice["wb"].var;     #wb = wb[:,:,:,:];
ε = ds_slice["ε"].var;       ε = ε[:,:,:,:];
χ = ds_slice["χ"].var;       χ = χ[:,:,:,:];

Bz = zeros(size(B[:,1:1,:,:])); #dûdz = zeros(size(uhat[:,1,:,:]));
Bz = deriv(zC,B);
θ = 3.6e-3; N = 1e-3;
Bz_bc = -N^2*cos(θ).*ones(Nx,Ny,1,length(t));
Bz = cat(Bz, Bz_bc, dims=3);
χ = χ./Bz
# set topography to NaN
uhat[uhat.==0] .= NaN
# what[what.==0] .= NaN
# wb[isnan.(uhat)] .= NaN
ε[isnan.(uhat)] .= NaN
# b[b.==0] .= NaN
χ[b.==0] .= NaN

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



file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=Ny)

using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)



## plot
n = Observable(1)
ûₙ = @lift(uhat[:,:,1,$n])
εₙ = @lift(log10.(ε[:,:,1,$n]))
Bₙ = @lift(B[:,:,1,$n])
# χₙ = @lift(sign.(χ[:,:,1,$n]).*log10.(abs.(χ[:,:,1,$n])))
χₙ = @lift(log10.(abs.(χ[:,:,1,$n])))

# dBdzₙ = @lift interior(dBdz[$n], :, 1, :)
# dûdzₙ = @lift interior(dûdz[$n], :, 1, :)
ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = Figure(resolution = (1000,1000), figure_padding=(5,5,5,5), size=(800,800))
rowgap!(fig.layout, 0) # Reduce gap between rows
colgap!(fig.layout, 0) # Reduce gap between columns
axis_kwargs1 = (xlabel = "Zonal distance x (m)",
                  ylabel = "Cross canyon distance y (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end])))
                #   aspect = DataAspect())
axis_kwargs2 = (xlabel = "Zonal distance x (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end])),
                  yticklabelsvisible=false)
                #   aspect = DataAspect())
title = @lift @sprintf("z=1300 m, t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
# fig[1, 1] = Label(fig, title, fontsize=20, tellwidth=false)
fig[1, 1:3] = Label(fig, title, fontsize=20, padding=(0, 0, 5, 0))


ax_u = Axis(fig[2,1]; title = "u", axis_kwargs1...); 
# ax_uz = Axis(fig[1, 2]; title = "du/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_bz = Axis(fig[2, 1]; title = "db/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_ε = Axis(fig[2,2]; title = "log ε", axis_kwargs2...)
ax_χ = Axis(fig[2,3]; title = "log χ", axis_kwargs2...)
ax_u.aspect = 0.5
ax_ε.aspect = 0.5
ax_χ.aspect = 0.5

using ColorSchemes
U₀ = 0.025
hm_u = heatmap!(ax_u, xC[:], yC[:], ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_u, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_u = contour!(ax_u, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,1], hm_u; label = "m/s", vertical=false)
# colsize!(fig.layout, 1, Aspect(1, 0.5))

hm_ε = heatmap!(ax_ε, xC[:], yC[:], εₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
contour!(ax_ε, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_ε = contour!(ax_ε, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε, vertical=false)
# colsize!(fig.layout, 1, Aspect(1, 0.5))

hm_χ = heatmap!(ax_χ, xC[:], yC[:], χₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
contour!(ax_χ, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_χ = contour!(ax_χ, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,3], hm_χ, vertical=false)
# colsize!(fig.layout, 1, Aspect(1, 0.33333))
rowsize!(fig.layout, 1, 50)
rowsize!(fig.layout, 3, 50)
frames = 1:2:length(t)

filename = join(split(filename_slice, ".")[1:end-1], ".")

record(fig, string(filename,".mp4"), frames, framerate=14) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

# close(ds_slice)

#####
# plot x-y in terms of vorticity
using NCDatasets
include("functions/mmderiv.jl")
tᶠ = 460
θ = 3.6e-3
if θ==3.6e-3
    simname = "tilt"
else 
    simname = "flat"
end
filename_slice = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_xy.nc")
ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_slice["zC"]; Nz=length(zC)
xC = ds_slice["xC"]; Nx=length(xC)
yC = ds_slice["yC"]; Ny=length(yC)
xF = ds_slice["xF"]
yF = ds_slice["yF"]
t = ds_slice["time"]

# load u and v velocity data
u = ds_slice["uhat"][:,:,:,:]
v = ds_slice["v"][:,:,:,:]

# Set zero values to NaN
u[u.==0] .= NaN
v[v.==0] .= NaN

# Calculate vorticity (∂v/∂x - ∂u/∂y)
ω = zeros(size(u[:,:,:,:]))
for i in 1:length(t)
        
end
for i in 1:length(t)
    ω[:,:,:,i] = mmderiv(xC, v[:,:,1,i]) .- mmderiv(yC, u[:,:,1,i]')'
end

using MAT
using CairoMakie
using Printf
using Statistics

# Create an Observable for the time step
n = Observable(1)
f = -5.3e-5
vortₙ = @lift(ω[:,:,1,$n]./f)
ω₀ = 1.4e-4
M₂_period = 2π/ω₀
# Create the figure with appropriate dimensions - using just resolution
fig = Figure(resolution = (800, 600))

# Create a reactive title
title = @lift @sprintf("z=1300 m, t=%1.2f M₂ tidal periods", t[$n]/M₂_period)

# Set common axis parameters
axis_kwargs = (
    xlabel = "Zonal distance x (m)",
    ylabel = "Cross canyon distance y (m)",
    limits = ((0, xF[end]), (0, yF[end])),
)

# Create main axis for the vorticity plot
ax_vort = Axis(fig[2,1]; title = "ω/f", axis_kwargs...)

# Use DataAspect to maintain proper proportions
ax_vort.aspect = DataAspect()

# Calculate appropriate color range by excluding outliers
vort_values = filter(!isnan, ω[:,:,1,:]./f)
vort_range = maximum(abs.(vort_values)) * 0.05

# Create the heatmap
hm_vort = heatmap!(ax_vort, xC[:], yC[:], vortₙ,
    colorrange = (-vort_range, vort_range), 
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)

# Add contour lines for topography
contour!(ax_vort, xC[:], yC[:], z_interp, linewidth = 1.5, color = :black)

# Add title to the first row, spanning both columns
fig[1, 1:2] = Label(fig, title, fontsize=20)

# Add colorbar to the right of the heatmap
fig[2, 2] = Colorbar(fig, hm_vort; vertical=true, width=20)

# Set appropriate spacing between elements
colgap!(fig.layout, 5)
rowgap!(fig.layout, 5)

# Ensure the layout is tight
trim!(fig.layout)

# Define which frames to render
frames = 1:length(t)  # Use all frames, or subset with 1:2:length(t)
filename = join(split(filename_slice, ".")[1:end-1], ".")

# Record the animation
record(fig, string("output/", simname, "/", simname, "_vorticity_xy.mp4"), frames, framerate=14) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

#################



## plot x-y in terms of tidally-averaged cross-slope velocity and vertical velocity 

## load data
simname = "tilt"
timerange = "0-40"
filename_slice = string("output/",simname,"/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_",timerange,"_threeD_timeavg.nc")
ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_slice["zC"]; Nz=length(zC)
xC = ds_slice["xC"]; Nx=length(xC)
yC = ds_slice["yC"]; Ny=length(yC)
t = ds_slice["time"];

# load all data
# B = ds_slice["B"].var;        B = B[:,:,:,:];
# b = ds_slice["b"].var;        b = b[:,:,:,:];
uhat = ds_slice["uhat"][:,:,:,end];
what = ds_slice["what"][:,:,:,end];

# set topography to NaN
uhat[uhat.==0] .= NaN
what[what.==0] .= NaN

file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=Ny)

using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)

## plot
n = Observable(1)
ûₙ = @lift(uhat[:,:,$n,1])
ŵₙ = @lift(what[:,:,$n,1])
û_intₙ = @lift(nansum(uhat[:,:,1:$n,1],dim=3).*sum(zC[1:$n]))
ŵ_intₙ = @lift(nansum(what[:,:,1:$n,1],dim=3).*sum(zC[1:$n]))
max_ûₙ = @lift(maximum(nansum(uhat[:,:,1:$n,1],dim=3)).*sum(zC[1:$n]))
max_uhat = max_ûₙ[]
max_ŵₙ = @lift(maximum(nansum(what[:,:,1:$n,1],dim=3)).*sum(zC[1:$n]))
max_what = max_ŵₙ[]
ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = Figure(resolution = (2000,1000), figure_padding=(10, 40, 10, 10), size=(1600,800), fontsize = 30)
axis_kwargs1 = (xlabel = "Zonal distance x (m)",
                  ylabel = "Cross canyon distance y (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end])))
                #   aspect = DataAspect())
axis_kwargs2 = (xlabel = "Zonal distance x (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end])),
                  yticklabelsvisible=false)
                #   aspect = DataAspect())
title = @lift @sprintf("t = 40 tidal period, z = %1.2f m", zC[$n])

fig[1, 1] = Label(fig, title, fontsize=30, tellwidth=false)
ax_u = Axis(fig[2,1]; title = L"\text{Cross-slope velocity},~ \hat{u}", axis_kwargs1...); 
ax_w = Axis(fig[2,2]; title = L"\text{Slope-normal velocity},~ \hat{w}", axis_kwargs2...)
ax_u_int = Axis(fig[2,3]; title = L"\int_0^z \hat{u} dz", xlabel = "Zonal distance x (m)", 
                                limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end]))); 
ax_w_int = Axis(fig[2,4]; title = L"\int_0^z \hat{w} dz", axis_kwargs2...)
ax_u.aspect = 0.5
ax_w.aspect = 0.5
ax_u_int.aspect = 0.5
ax_w_int.aspect = 0.5

using ColorSchemes
U₀ = 0.025
hm_u = heatmap!(ax_u, xC[:], yC[:], ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_u, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_u = contour!(ax_u, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,1], hm_u; label = "m/s", vertical=false,
        ticks=[-3U₀, 3U₀],  # Custom ticks
        )

hm_w = heatmap!(ax_w, xC[:], yC[:], ŵₙ,
    colorrange = (-.5U₀, .5U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_w, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_ε = contour!(ax_ε, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_w, vertical=false,
        ticks=[-.5U₀, .5U₀],  # Custom ticks
        )

hm_u_int = heatmap!(ax_u_int, xC[:], yC[:], û_intₙ,
    colorrange = @lift((-maximum($û_intₙ)*0.8 - 1e-10, maximum($û_intₙ)*0.8 + 1e-10)),
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_u_int, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_u = contour!(ax_u, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,3], hm_u_int; label = "m²/s", vertical=false)

hm_w_int = heatmap!(ax_w_int, xC[:], yC[:], ŵ_intₙ,
colorrange = @lift((-maximum($ŵ_intₙ)*0.6 - 1e-10, maximum($ŵ_intₙ)*0.6 + 1e-10)),
colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_w_int, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_ε = contour!(ax_ε, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,4], hm_w_int, vertical=false)
# colsize!(fig.layout, 1, Aspect(1, 0.5))

frames = 1:1:length(zC)

filename = join(split(filename_slice, ".")[1:end-1], ".")

record(fig, string(filename,"_moving_z.mp4"), frames, framerate=16) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

##

## plot x-y in terms of tidally-averaged cross-slope velocity and vertical velocity 

## load data
simname = "tilt"
timerange = "0-40"
filename_slice = string("output/",simname,"/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_",timerange,"_threeD_timeavg.nc")
ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_slice["zC"]; Nz=length(zC)
xC = ds_slice["xC"]; Nx=length(xC)
yC = ds_slice["yC"]; Ny=length(yC)
t = ds_slice["time"];

# load all data
# B = ds_slice["B"].var;        B = B[:,:,:,:];
# b = ds_slice["b"].var;        b = b[:,:,:,:];
uhat = ds_slice["uhat"][:,:,:,end];
what = ds_slice["what"][:,:,:,end];

# set topography to NaN
uhat[uhat.==0] .= NaN
what[what.==0] .= NaN

file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=Ny)

using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)


## plot diffusive flux
n = Observable(1)
ûₙ = @lift(uhat[:,:,$n,1])
ŵₙ = @lift(what[:,:,$n,1])
û_intₙ = @lift(nansum(uhat[:,:,1:$n,1],dim=3).*sum(zC[1:$n]))
ŵ_intₙ = @lift(nansum(what[:,:,1:$n,1],dim=3).*sum(zC[1:$n]))
max_ûₙ = @lift(maximum(nansum(uhat[:,:,1:$n,1],dim=3)).*sum(zC[1:$n]))
max_uhat = max_ûₙ[]
max_ŵₙ = @lift(maximum(nansum(what[:,:,1:$n,1],dim=3)).*sum(zC[1:$n]))
max_what = max_ŵₙ[]
ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = Figure(resolution = (2000,1000), figure_padding=(10, 40, 10, 10), size=(1600,800), fontsize = 30)
axis_kwargs1 = (xlabel = "Zonal distance x (m)",
                  ylabel = "Cross canyon distance y (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end])))
                #   aspect = DataAspect())
axis_kwargs2 = (xlabel = "Zonal distance x (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end])),
                  yticklabelsvisible=false)
                #   aspect = DataAspect())
title = @lift @sprintf("t = 40 tidal period, z = %1.2f m", zC[$n])

fig[1, 1] = Label(fig, title, fontsize=30, tellwidth=false)
ax_u = Axis(fig[2,1]; title = L"\text{Cross-slope velocity},~ \hat{u}", axis_kwargs1...); 
ax_w = Axis(fig[2,2]; title = L"\text{Slope-normal velocity},~ \hat{w}", axis_kwargs2...)
ax_u_int = Axis(fig[2,3]; title = L"\int_0^z \hat{u} dz", xlabel = "Zonal distance x (m)", 
                                limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end]))); 
ax_w_int = Axis(fig[2,4]; title = L"\int_0^z \hat{w} dz", axis_kwargs2...)
ax_u.aspect = 0.5
ax_w.aspect = 0.5
ax_u_int.aspect = 0.5
ax_w_int.aspect = 0.5

using ColorSchemes
U₀ = 0.025
hm_u = heatmap!(ax_u, xC[:], yC[:], ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_u, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_u = contour!(ax_u, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,1], hm_u; label = "m/s", vertical=false,
        ticks=[-3U₀, 3U₀],  # Custom ticks
        )

hm_w = heatmap!(ax_w, xC[:], yC[:], ŵₙ,
    colorrange = (-.5U₀, .5U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_w, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_ε = contour!(ax_ε, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_w, vertical=false,
        ticks=[-.5U₀, .5U₀],  # Custom ticks
        )

hm_u_int = heatmap!(ax_u_int, xC[:], yC[:], û_intₙ,
    colorrange = @lift((-maximum($û_intₙ)*0.8 - 1e-10, maximum($û_intₙ)*0.8 + 1e-10)),
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_u_int, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_u = contour!(ax_u, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,3], hm_u_int; label = "m²/s", vertical=false)

hm_w_int = heatmap!(ax_w_int, xC[:], yC[:], ŵ_intₙ,
colorrange = @lift((-maximum($ŵ_intₙ)*0.6 - 1e-10, maximum($ŵ_intₙ)*0.6 + 1e-10)),
colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
contour!(ax_w_int, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_ε = contour!(ax_ε, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,4], hm_w_int, vertical=false)
# colsize!(fig.layout, 1, Aspect(1, 0.5))

frames = 1:1:length(zC)

filename = join(split(filename_slice, ".")[1:end-1], ".")

record(fig, string(filename,"_moving_z.mp4"), frames, framerate=16) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end












## plot u,v,w, tau, with a moving z
# load 3D snapshots
using Interpolations
using NCDatasets

function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end
include("functions/interpolation_z_dimension.jl")
simname = "tilt"
# timerange = "80-120"

tᶠ = 454.0
θ = 3.6e-3
if θ == 3.6e-3
    simname = "tilt"
else
    simname = "flat"
end

if tᶠ ≤ 10
    output_mode = "verification"
    Nt = 11      # number of time averages
    endtime = tᶠ
elseif tᶠ ≤ 450
    output_mode = "spinup"
    Nt = 4
    if tᶠ == 450
        endtime = ["$i" for i in 50:40:450]
        # endtime = tᶠ   # if you want to run only one case
    end
else
    output_mode = "analysis"
    Nt = 1    # only 1 tidal average
    endtime = tᶠ
end


bin_edge = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
bin_mask = hab


# Load two consecutive half tidal periods for full cycle averaging from snapshots
# because the time average outputs are not calculated correctly, so we decided to just use 3D snapshots
global total_steps
tᶠ_0 = tᶠ - 1
tᶠ_first = tᶠ - 0.5
tᶠ_second = tᶠ

filename_3D_0 = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_0, "_analysis_round=all_threeD.nc")
filename_3D_first = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_first, "_analysis_round=all_threeD.nc")
filename_3D_second = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_second, "_analysis_round=all_threeD.nc")
filename_verification = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", 10, "_threeD_timeavg.nc")
ds_3D_0 = Dataset(filename_3D_0, "r")
ds_3D_first = Dataset(filename_3D_first, "r")
ds_3D_second = Dataset(filename_3D_second, "r")
ds_verification = Dataset(filename_verification, "r")

# Get dimensions
zC = ds_3D_first["z_aac"][:]
zF = ds_3D_first["z_aaf"][:]
xC = ds_3D_first["x_caa"][:]
xF = ds_3D_first["x_faa"][:]
yC = ds_3D_first["y_aca"][:]
yF = ds_3D_first["y_afa"][:]
Nz = length(zC[:])
Nx = length(xC[:])
dx = xF[end] - xF[end-1]
Ny = length(yC[:])
dy = yF[end] - yF[end-1]
z_face = zF

# Initialize variables for running sums
dB̄dx = zeros(Nx, Ny, Nz, 1)
dB̄dy = zeros(Nx, Ny, Nz, 1)
dB̄dz = zeros(Nx, Ny, Nz, 1)
b = ds_verification["b"][:, :, :, 1:1]

# Combined running sums
# first define the variables using the final timestep of tf-0.5 chunk 
# (for example, the first timestep of 452.5 chunk is 452.0833, but we want 452.0 and it is at the final timestep  in tf-0.5 chunk)
B_sum = ds_3D_0["B"][:, :, :, end:end]
uhat_sum = ds_3D_0["uhat"][:, :, :, end:end]
what_sum = ds_3D_0["what"][:, :, :, end:end]
v_sum = ds_3D_0["v"][:, :, :, end:end]
∇κ∇B_sum = ds_3D_0["∇κ∇B"][:, :, :, end:end]
div_uB_sum = ds_3D_0["div_uB"][:, :, :, end:end]
ε_sum = ds_3D_0["ε"][:, :, :, end:end]
χ_sum = ds_3D_0["χ"][:, :, :, end:end]
# Count total timesteps processed
total_steps = 1   # already have one timestep from the first dataset

# Process first dataset
Nt_first = length(ds_3D_first["time"][:])
for n in 1:Nt_first
    # Read data for this timestep
    B_sum .+= ds_3D_first["B"][:, :, :, n:n]
    uhat_sum .+= ds_3D_first["uhat"][:, :, :, n:n]
    what_sum .+= ds_3D_first["what"][:, :, :, n:n]
    v_sum .+= ds_3D_first["v"][:, :, :, n:n]
    ∇κ∇B_sum .+= ds_3D_first["∇κ∇B"][:, :, :, n:n]
    div_uB_sum .+= ds_3D_first["div_uB"][:, :, :, n:n]
    ε_sum .+= ds_3D_first["ε"][:, :, :, n:n]
    χ_sum .+= ds_3D_first["χ"][:, :, :, n:n]
    # Free memory by explicitly clearing variables if needed
    GC.gc()
    global total_steps += 1
    println(total_steps)
end

# Process second dataset
Nt_second = length(ds_3D_second["time"][:])
for n in 1:Nt_second
    # Read data for this timestep
    B_sum .+= ds_3D_second["B"][:, :, :, n:n]
    uhat_sum .+= ds_3D_second["uhat"][:, :, :, n:n]
    what_sum .+= ds_3D_second["what"][:, :, :, n:n]
    v_sum .+= ds_3D_second["v"][:, :, :, n:n]
    ∇κ∇B_sum .+= ds_3D_second["∇κ∇B"][:, :, :, n:n]
    div_uB_sum .+= ds_3D_second["div_uB"][:, :, :, n:n]
    ε_sum .+= ds_3D_second["ε"][:, :, :, n:n]
    χ_sum .+= ds_3D_second["χ"][:, :, :, n:n]

    global total_steps += 1

    # Free memory by explicitly clearing variables if needed
    GC.gc()
end

# Calculate averages
B = B_sum ./ total_steps
uhat = uhat_sum ./ total_steps
what = what_sum ./ total_steps
v = v_sum ./ total_steps
∇κ∇B = ∇κ∇B_sum ./ total_steps
div_uB = div_uB_sum ./ total_steps
ε = ε_sum ./ total_steps
χ = χ_sum ./ total_steps


# Clear the intermediate sums to free memory
B_sum = nothing
uhat_sum = nothing
what_sum = nothing
v_sum = nothing
∇κ∇B_sum = nothing
div_uB_sum = nothing
GC.gc()

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
uhat_cen = (uhat[1:end,:,:] .+ vcat(uhat[2:end,:,:],uhat[1:1,:,:]))./2 # u at center in the x dimension

what_cen = (what[:, :, 1:end-1, 1] .+ what[:, :, 2:end, 1]) ./ 2 # what at center
# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end, :, :], what_cen[1:end-1, :, :]) .+ what_cen[:, :, :]) ./ 2
u = uhat[:, :, :, 1] * cos(θ) .+ wtemp * sin(θ) # cross-slope velocity
w = -uhat[:, :, :, 1] * sin(θ) .+ wtemp * cos(θ)# slope-normal velocity
v_cen = (v[:,1:end,:,1] .+ cat(v[:,2:end,:,1], v[:,1:1,:,1], dims=2))./2 # v at center in the x dimension
# topostrophy
uhat_cen[uhat_cen.==0] .= NaN
what_cen[what_cen.==0] .= NaN
v_cen[v_cen.==0] .= NaN

# calculate the topostrophy
include("functions/mmderiv.jl")
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];
bin_mask = hab
dHdy = zeros(size(hab))
dHdx = zeros(size(hab))
for k in 1:size(hab,3)
dHdy[:,:,k] = mmderiv(yC,hab[:,:,k]')'
dHdx[:,:,k] = mmderiv(xC,hab[:,:,k])
end

τ = @. -(uhat_cen*dHdy - v_cen*dHdx)


N = 1.e-3
Bz = deriv(zC, B)
Bz[b[:, :, 1:end-1, :].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
dx =  30
dy = 30
dB̄dx = diff(vcat(B[end:end,:,:,1], B[:,:,:,1]), dims=1) ./ dx
dB̄dy = diff(hcat(B[:, end:end, :, 1], B[:, :, :, 1]), dims=2) ./ dy # dB̄dy is the derivative of B with respect to y at the center cell

# interpolate Bz from faces to center cell
using Interpolations
# Interpolate each row
Bz_center = zeros(size(Bz, 1), size(Bz, 2), length(zC), 1)
for i in 1:size(Bz, 1)
    for j in 1:size(Bz, 2)
        itp = linear_interpolation(zF[2:end-1], Bz[i, j, :, 1], extrapolation_bc=Line())
        Bz_center[i, j, :, :] = itp(zC)
    end
end

# or try the simplest way (piecewise linear manually)
Bx_center = (vcat(dB̄dx[2:end, :, :, 1], dB̄dx[1:1, :, :]) .+ dB̄dx[:, :, :]) ./ 2
By_center = (hcat(dB̄dy[:, 2:end, :, 1], dB̄dy[:, 1:1, :, 1]) .+ dB̄dy[:, :, :]) ./ 2
Bx_center[abs.(Bx_center).>0.1*nanmaximum(abs.(Bx_center))] .= 0  # set the points inside and right above the immersed boundary to NaN
By_center[abs.(By_center).>0.1*nanmaximum(abs.(By_center))] .= 0 

# u_bar_∇B_bar = u .* Bx_center .+ v .* By_center .+ w .* Bz_center
u_bar_∇B_bar = u .* Bx_center .+ v .* By_center .+ w .* Bz_center
# u_bar_∇B_bar[b.==0] .= NaN  # set the points inside and right above the immersed boundary to NaN
u_bar_∇B_bar[b.==0] .= NaN  # set the points inside and right above the immersed boundary to NaN

div_uB[b.==0] .= NaN
u_prime∇B_prime = div_uB .- u_bar_∇B_bar


#######) plot u,v,w, and topostrophy
using CairoMakie
using Printf
using Statistics

# Create the output directory if it doesn't exist
output_dir = string("output/", simname)
# Full animation filename
anim_filename = string(output_dir, "/raw_u_v_w_tau_xy_animation.mp4")

# Create Observable for the z-index
n = Observable(1)

# Create lift expressions for the raw data slices
what_slice = @lift(what_cen[:,:,$n])
uhat_slice = @lift(uhat_cen[:,:,$n])
v_slice = @lift(v_cen[:,:,$n])
tau_slice = @lift(τ[:,:,$n])

# Create the figure with a resolution that fits a 2x2 layout
fig = CairoMakie.Figure(resolution=(1200, 1000), fontsize=16)

# Create a custom grid layout
layout = GridLayout()

# Assign the layout to the figure
fig[1:3, 1] = layout

# Create title with changing z value
title = @lift("z = $(round(zC[$n], digits=1)) m")
layout[1, 1:8] = Label(fig, title, fontsize=24)

# Create axes for 2x2 layout
# Top left: u
ax1 = Axis(layout[2, 1], 
           title = L"\bar{u}",
           ylabel = "y [km]",
           titlesize = 22)

# Top right: v
ax2 = Axis(layout[2, 5], 
           title = L"\bar{v}",
           titlesize = 22)

# Bottom left: w
ax3 = Axis(layout[3, 1], 
           title = L"\bar{w}",
           xlabel = "x [km]", 
           ylabel = "y [km]",
           titlesize = 22)

# Bottom right: topostrophy
ax4 = Axis(layout[3, 5], 
           title = L"τ~\mathrm{(topostrophy)}",
           xlabel = "x [km]",
           titlesize = 22)

# Set consistent aspect ratios
for ax in [ax1, ax2, ax3, ax4]
    ax.aspect = 1.0
end

# Plot each variable with a colorbar
# u (top left)
hm1 = heatmap!(ax1, xC[:]*1e-3, yC[:]*1e-3, uhat_slice,
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.1, 0.1),
               nan_color = :gray)
cb1 = Colorbar(layout[2, 3], hm1, width=8)
contour!(ax1, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# v (top right)
hm2 = heatmap!(ax2, xC[:]*1e-3, yC[:]*1e-3, v_slice,
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.05, 0.05),
               nan_color = :gray)
cb2 = Colorbar(layout[2, 7], hm2, width=8)
contour!(ax2, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# w (bottom left)
hm3 = heatmap!(ax3, xC[:]*1e-3, yC[:]*1e-3, what_slice,
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.005, 0.005),
               nan_color = :gray)
cb3 = Colorbar(layout[3, 3], hm3, width=8)
contour!(ax3, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# τ (bottom right)
hm4 = heatmap!(ax4, xC[:]*1e-3, yC[:]*1e-3, tau_slice,
               colormap = reverse(cgrad(:RdBu)),
               colorrange = (-0.02, 0.02),
               nan_color = :gray)
cb4 = Colorbar(layout[3, 7], hm4, width=8)
contour!(ax4, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Hide y-axis for right plots
hideydecorations!(ax2, grid=false)
hideydecorations!(ax4, grid=false)

# Hide x-axis for top plots
hidexdecorations!(ax1, grid=false)
hidexdecorations!(ax2, grid=false)

# Set column widths for the layout
colsize!(layout, 1, Relative(0.38))  # Left plots
colsize!(layout, 3, Relative(0.04))  # Left colorbars
colsize!(layout, 5, Relative(0.38))  # Right plots
colsize!(layout, 7, Relative(0.04))  # Right colorbars

# Add spacing between columns
colgap!(layout, 5)

# Add minimal padding to figure
fig.attributes.margin = (20, 20, 20, 20)

# Define which z-levels to include in animation - use a subset for clearer visualization
# Ensure z_levels doesn't exceed the actual data dimensions
max_z_level = min(200, length(zC))
z_levels = 1:2:max_z_level

# Create the animation
record(fig, anim_filename, 1:length(z_levels); framerate=5) do i
    @info "Rendering frame $i of $(length(z_levels))..."
    n[] = z_levels[i]
end
