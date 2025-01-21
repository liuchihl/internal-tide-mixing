# This script plots the cross sections of a tidal-averaged 3D field
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
# using BSplineKit
# using Statistics
# using LinearAlgebra
# using Interpolations
using MAT
filename_field = "output/higher_top_resolution/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_10-30_threeD_timeavg.nc"
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
yF = ds_field["yF"]; 
t = ds_field["time"][:];

n=58
b = ds_field["b"][:,:,:,n:n];
Bz = ds_field["Bz"][:,:,:,n:n];
epsilon = ds_field["ε"][:,:,:,n:n];
uhat = ds_field["uhat"][:,:,:,n:n];    # true u
what = ds_field["what"][:,:,:,n:n];    # true w
chi = ds_field["χ"][:,:,:,n:n];   
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
θ = 0.0036
u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ)

# mask out
Bz[b.==0] .= NaN
epsilon[epsilon.==0] .= NaN;
chi[b.==0] .= NaN;
u[uhat[:,:,:,1].==0] .= NaN;
uhat[uhat.==0] .= NaN;
b[b.==0] .= NaN;
what[what.==0] .= NaN;

# topography
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


## x-y sections
    
ind500 = argmin(abs.(zC .- 500))   # 500 m height above bottom
ind900 = argmin(abs.(zC .- 900))   # 900 m height above bottom
ind2000 = argmin(abs.(zC .- 2000))   # 1300 m height above bottom


fig = Figure(size=(1000,800))
ax = Axis3(fig[1, 1], aspect = (1, 2, 1.5), limits=((0,15),(0,30),(0,zF[end]/1e3)))

x = LinRange(1, xC[end]/1e3, Nx)
y = LinRange(1, yC[end]/1e3, Ny)
# z = LinRange(0, zF[end]/1e3, Nz)
z = LinRange(zC[ind500]/1e3,zC[ind2000]/1e3,2)
# sgrid = SliderGrid(
#     fig[2, 1],
#     (label = "1300", range = 1:length(x)),
#     (label = "1300", range = 1:length(y)),
#     (label = "1300", range = 1:length(z))
# )

vol = u[:,:,[ind500,ind2000]]

plt = volumeslices!(ax, x, y, z, vol[:,:,:],
    colorrange = (-0.01, 0.01), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray, bbox_color=:black);


# connect sliders to `volumeslices` update methods
# sl_xy3 = sgrid.sliders
# # on(sl_xy1.value) do v; plt[:update_xy][](v) end
# # on(sl_xy2.value) do v; plt[:update_xy][](v) end
# on(sl_xy3[1].value) do v; plt[:update_xy][](v) end
# set_close_to!(sl_xy1, ind500)
# set_close_to!(sl_xy2, ind900)
# set_close_to!(sl_xy3[1], ind1300)
sl_yz, sl_xz, sl_xy = sgrid.sliders
on(sl_yz.value) do v; plt[:update_yz][](v) end
on(sl_xz.value) do v; plt[:update_xz][](v) end
on(sl_xy.value) do v; plt[:update_xy][](v) end

set_close_to!(sl_yz, 0)
set_close_to!(sl_xz, .5length(y))
set_close_to!(sl_xy, .5length(z))

save("output/higher_top_resolution/volumeslices_u.png",fig)

















fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(800,400))
axis_kwargs1 = (xlabel = "Zonal distance x (m)",
                  ylabel = "Cross canyon distance y (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end])))
                #   aspect = DataAspect())
axis_kwargs2 = (xlabel = "Zonal distance x (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["yF"][end])),
                  yticklabelsvisible=false)
                #   aspect = DataAspect())

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
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256, flipaxis = false,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],clim = (-3U₀,3U₀),
    nan_color = :gray)
contour!(ax_u, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_u = contour!(ax_u, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,1], hm_u; label = "m/s", vertical=false)
# colsize!(fig.layout, 1, Aspect(2, 0.5))

hm_ε = heatmap!(ax_ε, xC[:], yC[:], εₙ,
    colorrange = (-10.5, -8.5), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    clim = (-10.5,-8.5),nan_color = :gray)
contour!(ax_ε, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_ε = contour!(ax_ε, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε, vertical=false)
# colsize!(fig.layout, 1, Aspect(1, 0.5))

hm_χ = heatmap!(ax_χ, xC[:], yC[:], χₙ,
    colorrange = (-10.5, -8.5), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    clim = (-10.5,-8.5),nan_color = :gray)
contour!(ax_χ, xC[:], yC[:], z_interp, linewidth = 1.5, color =:black)
# ct_χ = contour!(ax_χ, xC, yC, Bₙ,
#     levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,3], hm_χ, vertical=false)
# colsize!(fig.layout, 1, Aspect(1, 0.5))

frames = 1:length(t)

filename = join(split(filename_slice, ".")[1:end-1], ".")

record(fig, string(filename,".mp4"), frames, framerate=16) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end


