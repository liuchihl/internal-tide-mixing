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

