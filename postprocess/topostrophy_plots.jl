using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
# using PyPlot
using NCDatasets
using Statistics
using NaNStatistics
using LinearAlgebra
using Interpolations
using CairoMakie

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/mmderiv.jl")

tᶠ = 330
θ = 3.6e-3
if θ==3.6e-3
    simname = "tilt"
else 
    simname = "flat"
end

if  tᶠ ≤ 10
    output_mode = "verification"
    Nt = 10      # number of time averages
    endtime = tᶠ
elseif tᶠ ≤ 1010
    output_mode = "spinup"
    Nt = 4
    if tᶠ == 1010
        endtime = ["$i" for i in 50:40:1010]
    else
        endtime = tᶠ
    end
else
    output_mode = "analysis"
    Nt = 20
    endtime = tᶠ
end

filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

t = ds_field["time"][:];
m = length(t)
# grids
zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]); 
xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];
yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];


# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];
bin_mask = hab
# topostrophy
dHdy = zeros(size(hab))
dHdx = zeros(size(hab))
for k in 1:size(hab,3)
dHdy[:,:,k] = mmderiv(yC,hab[:,:,k]')'
dHdx[:,:,k] = mmderiv(xC,hab[:,:,k])
end

uhat = ds_field["uhat"][:,:,:,m:m];    # true u
v = ds_field["v"][:,:,:,m:m];    # true u
v_cen = (v[:,1:end,:,1] .+ cat(v[:,2:end,:,1], v[:,1:1,:,1], dims=2))./2 # u at center in the x dimension
what = ds_field["what"][:,:,:,m:m];    # true w
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
u_cen = (u[1:end,:,:] .+ vcat(u[2:end,:,:],u[1:1,:,:]))./2 # u at center in the x dimension
# topostrophy
u_cen[u_cen.==0] .= NaN
v_cen[v_cen.==0] .= NaN

τ = @. -(u_cen*dHdy - v_cen*dHdx)

######################## plot
using PyPlot
close("all")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
ind = argmin(abs.(zC.-1000))
ind_y = argmin(abs.(yC.-12000))

# x-z slice
ax = axs[1]
c = ax.pcolor(xC, zC, τ[:, ind_y, :]', cmap="coolwarm",vmin=-0.01, vmax=0.01)
ax.set_facecolor("gray")
fig.colorbar(c, ax=ax, label="τ")
CS=ax.contour(xC, zC, v_cen[:, div(Ny, 2), :]', cmap="PiYG_r", linewidths=1.5,-0.08:0.01:0.08)
ax.clabel(CS, inline=true, fontsize=8, colors=["k"])
ax.plot([xC[1],xC[end]],[zC[ind],zC[ind]],"k--")
ax.set_title(string("t =",Int(round(t[n]/(2*pi/1.4e-4))-10),"-",Int(round(t[n]/(2*pi/1.4e-4))),
 " tidal average, x-z slice, v contours"))
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")

# x-y slice
ax = axs[2]
c = ax.pcolor(xC, yC, τ[:, :, ind]', cmap="coolwarm",vmin=-0.01, vmax=0.01)
ax.set_facecolor("gray")
fig.colorbar(c, ax=ax, label="τ")
# Add topography contours
ax.contour(xC, yC, hab[:,:,1]', colors="k", linewidths=1.3, linestyles="-")
# Add velocity quivers, subsample for clarity
skip = 15  # Plot every 15th point
X = [x for x in xC[1:skip:end], y in yC[1:skip:end]]
Y = [y for x in xC[1:skip:end], y in yC[1:skip:end]]
U = u_cen[1:skip:end, 1:skip:end, ind]
V = v_cen[1:skip:end, 1:skip:end, ind]
ax.quiver(X, Y, U, V, scale=1.5)
ax.plot([xC[1],xC[end]],[yC[ind_y],yC[ind_y]],"k--")
ax.plot([xC[Nx÷2],xC[Nx÷2]],[yC[1],yC[end]],"k--")


ax.set_title(string("x-y slice, z=",Int(round(zC[ind]))  ," m"))
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

# y-z slice
ax = axs[3]
c = ax.pcolor(yC, zC, τ[div(Nx, 2), :, :]', cmap="coolwarm",vmin=-0.01, vmax=0.01)
ax.set_facecolor("gray")
fig.colorbar(c, ax=ax, label="τ")
CS = ax.contour(yC, zC, u_cen[div(Nx, 2), :, :]', cmap="PiYG_r", linewidths=1.5,-0.08:0.01:0.08)
ax.clabel(CS, inline=true, fontsize=8, colors=["k"])
ax.set_title("y-z slice, u contours")
ax.set_xlabel("y [m]")
ax.set_ylabel("z [m]")
ax.plot([xC[1],xC[end]],[zC[ind],zC[ind]],"k--")
plt.tight_layout()
plt.show()
savefig(string("output/", simname, "/topostrophy_",simname,"_tᶠ=",tᶠ,".png"))


# make an animation
using CairoMakie

# plot
n = Observable(1)
τₙ = @lift(τ[$n,:,:])
uₙ = @lift(u[$n,:,:])

fig = CairoMakie.Figure(resolution = (800, 600), size=(800,600))
axis_kwargs1 = (xlabel = "y (km)",
                  ylabel = "z (m)",
                  limits = ((0, 1e-3*yF[end]), (0, zF[end]) )
                  )

ax_τ = Axis(fig[1,1]; 
title = @lift(string(Int(round(t[m]/(2*pi/1.4e-4))-10),"-",Int(round(t[m]/(2*pi/1.4e-4))),
" tidal average topostrophy, u contours, x = ", round(xC[$n]/1e3, digits=1), " km")), axis_kwargs1...)

using ColorSchemes

hm_τ = CairoMakie.heatmap!(ax_τ, 1e-3*yC[:], zC[:], τₙ,
    colorrange = (-0.01,0.01), colormap=:coolwarm,
    lowclip=cgrad(:coolwarm)[1], highclip=cgrad(:coolwarm)[end],
    nan_color = :gray)

cont = CairoMakie.contour!(ax_τ, 1e-3*yC[:], zC[:], uₙ, 
    levels = -0.1:0.01:0.1,
    linewidth = 1.8, 
    colormap = reverse(ColorSchemes.PiYG))

Colorbar(fig[2,1], hm_τ; label = "topostrophy [m/s]", vertical=false)
arrows = arrows!(ax_τ, 
    1e-3*yC[1:15:end], zC[1:10:end],
    @lift(v_cen[$n,1:15:end,1:10:end]),            # y direction
    @lift(1000*w[$n,1:15:end,1:10:end]),                # z direction
    arrowsize = 7,                                     # smaller arrowhead
    lengthscale = 40,                                # longer tail
    linewidth = 1.0,                                  # thicker lines
    color = :black
)
# Add reference arrows
y_ref = 2200  # position for reference arrows
z_ref = 500    # position for reference arrows
arrows!(ax_τ,
    [1e-3*y_ref], [z_ref],
    [0.01], [0.0],  # horizontal reference: u = 0.1
    arrowsize = 7,
    lengthscale = 40,
    linewidth = 1.2,
    color = :black)
arrows!(ax_τ,
    [1e-3*y_ref], [z_ref],
    [0.0], [0.001*1000],  # vertical reference: w = 0.01 (scaled by 1000)
    arrowsize = 7,
    lengthscale = 40,
    linewidth = 1.2,
    color = :black)
text!(ax_τ, 1e-3*y_ref+0.5, z_ref+40, text="v = 0.01 m/s")
text!(ax_τ, 1e-3*y_ref+0.5, z_ref-50, text="w = 0.001 m/s")
frames =  1:2:length(xC)

CairoMakie.record(fig, string("output/", simname, "/topostrophy_animation_yz_",simname,"_tᶠ=", tᶠ, ".mp4"), frames, framerate = 12) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

# Create y-z slice animation zoomed in from 0-1500 m
using ColorSchemes
# Find the indices where hab is 0 along the z dimension
bottom_indices = zeros(Int, size(hab, 1), size(hab, 2))
for i in 1:size(hab, 1)
    for j in 1:size(hab, 2)
        bottom_indices[i, j] = findfirst(hab[i, j, :] .> 0)
    end
end

ind = argmin(abs.(zC.-1500))
n = Observable(1)
τₙ = @lift(τ[$n, :, 1:ind])
uₙ = @lift(u_cen[$n, :, 1:ind])

fig = CairoMakie.Figure(resolution = (800, 600), size=(800,600))
axis_kwargs1 = (xlabel = "y (km)",
                ylabel = "z (m)",
                limits = ((0, 1e-3*yF[end]), (0, zF[ind])))

ax_τ = Axis(fig[1,1]; 
    title = @lift(string(Int(round(t[m]/(2*pi/1.4e-4))-10),"-",Int(round(t[m]/(2*pi/1.4e-4))),
    " tidal average topostrophy, u contours, x = ", round(xC[$n]/1e3, digits=1), " km")), axis_kwargs1...)

hm_τ = CairoMakie.heatmap!(ax_τ, 1e-3*yC[:], zC[1:ind], τₙ,
    colorrange = (-0.01,0.01), colormap=:coolwarm,
    lowclip=cgrad(:coolwarm)[1], highclip=cgrad(:coolwarm)[end],
    nan_color = :gray)

cont = CairoMakie.contour!(ax_τ, 1e-3*yC[:], zC[1:ind], uₙ, 
    levels = -0.1:0.01:0.1,
    linewidth = 1.8, 
    colormap = reverse(ColorSchemes.PiYG))

Colorbar(fig[2,1], hm_τ; label = "topostrophy [m/s]", vertical=false)

# plot the arrow in the domain
arrows!(ax_τ, 
    1e-3*yC[1:15:end], zC[1:10:ind],
    @lift(v_cen[$n,1:15:end,1:10:ind]),
    @lift(1000*w[$n,1:15:end,1:10:ind]),
    arrowsize = 7,
    lengthscale = 20,
    linewidth = 1.0,
    color = :black)
# plot the arrow at the bottom
arrows!(ax_τ, 
    [1e-3*yC[j] for j in 1:5:Ny], 
    @lift([zC[bottom_indices[$n, j]] for j in 1:5:Ny]),
    @lift([v_cen[$n,j,bottom_indices[$n, j]] for j in 1:5:Ny]),
    @lift([1000*w[$n,j,bottom_indices[$n, j]] for j in 1:5:Ny]),
    arrowsize = 7,
    lengthscale = 50,
    linewidth = 1.0,
    color = :red)

# Reference arrows
y_ref = 2200
z_ref = 400
arrows!(ax_τ,
    [1e-3*y_ref], [z_ref],
    [0.01], [0.0],
    arrowsize = 7,
    lengthscale = 20,
    linewidth = 1.0,
    color = :black)
arrows!(ax_τ,
    [1e-3*y_ref], [z_ref],
    [0.0], [0.001*1000],
    arrowsize = 7,
    lengthscale = 20,
    linewidth = 1.0,
    color = :black)
text!(ax_τ, 1e-3*y_ref+0.5, z_ref+40, text="v = 0.01 m/s")
text!(ax_τ, 1e-3*y_ref+0.5, z_ref-50, text="w = 0.001 m/s")

y_ref = 2200
z_ref = 200
arrows!(ax_τ,
    [1e-3*y_ref], [z_ref],
    [0.01], [0.0],
    arrowsize = 7,
    lengthscale = 50,
    linewidth = 1.0,
    color = :red)
arrows!(ax_τ,
    [1e-3*y_ref], [z_ref],
    [0.0], [0.001*1000],
    arrowsize = 7,
    lengthscale = 50,
    linewidth = 1.0,
    color = :red)
text!(ax_τ, 1e-3*y_ref+0.5, z_ref+40, text="v = 0.01 m/s", color=:red)
text!(ax_τ, 1e-3*y_ref+0.5, z_ref-50, text="w = 0.001 m/s", color=:red)

frames = 1:2:Nx

CairoMakie.record(fig, string("output/", simname, "/topostrophy_animation_yz_bottom_",simname,"_tᶠ=", tᶠ, ".mp4"), frames, framerate = 12) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end






######################## Create x-z slice animation
n = Observable(1)
τₙ = @lift(τ[:, $n, :])
vₙ = @lift(v_cen[:, $n, :])

fig = CairoMakie.Figure(resolution = (800, 600), size=(800,600))
axis_kwargs1 = (xlabel = "x (km)",
                ylabel = "z (m)",
                limits = ((0, 1e-3*xF[end]), (0, zF[end])))

ax_τ = Axis(fig[1,1]; 
    title = @lift(string(Int(round(t[m]/(2*pi/1.4e-4))-10),"-",Int(round(t[m]/(2*pi/1.4e-4))),
    " tidal average topostrophy, v contours, y = ", round(yC[$n]/1e3, digits=1), " km")), axis_kwargs1...)

hm_τ = CairoMakie.heatmap!(ax_τ, 1e-3*xC[:], zC[:], τₙ,
    colorrange = (-0.01,0.01), colormap=:coolwarm,
    lowclip=cgrad(:coolwarm)[1], highclip=cgrad(:coolwarm)[end],
    nan_color = :gray)

cont = CairoMakie.contour!(ax_τ, 1e-3*xC[:], zC[:], vₙ, 
    levels = -0.1:0.01:0.1,
    linewidth = 1.8, 
    colormap = reverse(ColorSchemes.PiYG))

Colorbar(fig[2,1], hm_τ; label = "topostrophy [m/s]", vertical=false)

arrows!(ax_τ, 
    1e-3*xC[1:15:end], zC[1:10:end],
    @lift(u_cen[1:15:end,$n,1:10:end]),
    @lift(1000*w[1:15:end,$n,1:10:end]),
    arrowsize = 7,
    lengthscale = 40,
    linewidth = 1.0,
    color = :black)

# Reference arrows
x_ref = 7500
z_ref = 200
arrows!(ax_τ,
    [1e-3*x_ref], [z_ref],
    [0.01], [0.0],
    arrowsize = 7,
    lengthscale = 40,
    linewidth = 1.2,
    color = :blue)
arrows!(ax_τ,
    [1e-3*x_ref], [z_ref],
    [0.0], [0.001*1000],
    arrowsize = 7,
    lengthscale = 40,
    linewidth = 1.2,
    color = :blue)
text!(ax_τ, 1e-3*x_ref+1.5, z_ref+40, text="u = 0.01 m/s")
text!(ax_τ, 1e-3*x_ref+1.5, z_ref-50, text="w = 0.001 m/s")

frames = 1:2:Ny

CairoMakie.record(fig, string("output/", simname, "/topostrophy_animation_xz_",simname,"_tᶠ=", tᶠ, ".mp4"), frames, framerate = 12) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end


###################### zoom in on bottom few cells
using ColorSchemes
# Find the indices where hab is 0 along the z dimension
bottom_indices = zeros(Int, size(hab, 1), size(hab, 2))
for i in 1:size(hab, 1)
    for j in 1:size(hab, 2)
        bottom_indices[i, j] = findfirst(hab[i, j, :] .> 0)
    end
end

# Create x-z slice animation zoomed in from 0-1500 m
ind = argmin(abs.(zC.-1500))
n = Observable(1)
τₙ = @lift(τ[:, $n, 1:ind])
vₙ = @lift(v_cen[:, $n, 1:ind])
fig = CairoMakie.Figure(resolution = (800, 600), size=(800,600))
axis_kwargs1 = (xlabel = "x (km)",
                ylabel = "z (m)",
                limits = ((0, 1e-3*xF[end]), (0, zF[ind])))

ax_τ = Axis(fig[1,1]; 
    title = @lift(string(Int(round(t[m]/(2*pi/1.4e-4))-10),"-",Int(round(t[m]/(2*pi/1.4e-4))),
    " tidal average topostrophy, v contours, y = ", round(yC[$n]/1e3, digits=1), " km")), axis_kwargs1...)

hm_τ = CairoMakie.heatmap!(ax_τ, 1e-3*xC[:], zC[1:ind], τₙ,
    colorrange = (-0.01,0.01), colormap=:coolwarm,
    lowclip=cgrad(:coolwarm)[1], highclip=cgrad(:coolwarm)[end],
    nan_color = :gray)

cont = CairoMakie.contour!(ax_τ, 1e-3*xC[:], zC[1:ind], vₙ, 
    levels = -0.1:0.01:0.1,
    linewidth = 1.8, 
    colormap = reverse(ColorSchemes.PiYG))

Colorbar(fig[2,1], hm_τ; label = "topostrophy [m/s]", vertical=false)

# all arrow in the domain 
arrows!(ax_τ, 
    1e-3*xC[1:15:end], zC[1:10:ind],
    @lift(u_cen[1:15:end,$n,1:10:ind]),
    @lift(1000*w[1:15:end,$n,1:10:ind]),
    arrowsize = 7,
    lengthscale = 20,
    linewidth = 1.0,
    color = :black)
# only the bottom arrow
arrows!(ax_τ, 
    [1e-3*xC[i] for i in 1:5:Nx], 
    @lift([zC[bottom_indices[i, $n]] for i in 1:5:Nx]),
    @lift([u_cen[i,$n,bottom_indices[i, $n]] for i in 1:5:Nx]),
    @lift([1000*w[i,$n,bottom_indices[i, $n]] for i in 1:5:Nx]),
    arrowsize = 7,
    lengthscale = 50,
    linewidth = 1.0,
    color = :red)
# Reference arrows
x_ref = 7000
z_ref = 400
arrows!(ax_τ,
    [1e-3*x_ref], [z_ref],
    [0.01], [0.0],
    arrowsize = 7,
    lengthscale = 20,
    linewidth = 1.0,
    color = :black)
arrows!(ax_τ,
    [1e-3*x_ref], [z_ref],
    [0.0], [0.001*1000],
    arrowsize = 7,
    lengthscale = 20,
    linewidth = 1.0,
    color = :black)
text!(ax_τ, 1e-3*x_ref+0.5, z_ref+40, text="u = 0.01 m/s")
text!(ax_τ, 1e-3*x_ref+0.5, z_ref-50, text="w = 0.001 m/s")


x_ref = 7000
z_ref = 200
arrows!(ax_τ,
    [1e-3*x_ref], [z_ref],
    [0.01], [0.0],
    arrowsize = 7,
    lengthscale = 50,
    linewidth = 1.0,
    color = :red)
arrows!(ax_τ,
    [1e-3*x_ref], [z_ref],
    [0.0], [0.001*1000],
    arrowsize = 7,
    lengthscale = 50,
    linewidth = 1.0,
    color = :red)
text!(ax_τ, 1e-3*x_ref+0.5, z_ref+40, text="u = 0.01 m/s", color=:red)
text!(ax_τ, 1e-3*x_ref+0.5, z_ref-50, text="w = 0.001 m/s", color=:red)

frames = 1:2:Ny

CairoMakie.record(fig, string("output/", simname, "/topostrophy_animation_xz_bottom",simname,"_tᶠ=", tᶠ, ".mp4"), frames, framerate = 12) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end





################ histogram of topostrophy 
using StatsBase

include("functions/interpolation_z_dimension.jl")
tᶠ = 330
θ = 0#3.6e-3
if θ==3.6e-3
    simname = "tilt"
else 
    simname = "flat"
end

if  tᶠ ≤ 10
    output_mode = "verification"
    Nt = 10      # number of time averages
    endtime = tᶠ
elseif tᶠ ≤ 1010
    output_mode = "spinup"
    Nt = 4
    if tᶠ == 1010
        endtime = ["$i" for i in 50:40:1010]
    else
        endtime = tᶠ
    end
else
    output_mode = "analysis"
    Nt = 20
    endtime = tᶠ
end

filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

t = ds_field["time"][:];
m = length(t)
# grids
zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]); 
xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];
yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];

include("functions/mmderiv.jl")
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];
bin_mask = hab
# topostrophy
dHdy = zeros(size(hab))
dHdx = zeros(size(hab))
for k in 1:size(hab,3)
dHdy[:,:,k] = mmderiv(yC,hab[:,:,k]')'
dHdx[:,:,k] = mmderiv(xC,hab[:,:,k])
end

# for m in 1:length(t)
    m = 4
    uhat = ds_field["uhat"][:,:,:,m:m];    # true u
    v = ds_field["v"][:,:,:,m:m];    # true u
    v_cen = (v[:,1:end,:,:] .+ cat(v[:,2:end,:,:], v[:,1:1,:,:], dims=2))./2 # u at center in the x dimension
    what = ds_field["what"][:,:,:,m:m];    # true w
    what_cen = (what[:,:,1:end-1,:] .+ what[:,:,2:end,:])./2 # what at center
    # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
    wtemp = (vcat(what_cen[end:end,:,:,:],what_cen[1:end-1,:,:,:]) .+ what_cen[:,:,:,:])./2
    u = uhat[:,:,:,:]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
    w = -uhat[:,:,:,:]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
    u_cen = (u[1:end,:,:,:] .+ vcat(u[2:end,:,:,:],u[1:1,:,:,:]))./2 # u at center in the x dimension
    # topostrophy
    u_cen[u_cen.==0] .= NaN
    v_cen[v_cen.==0] .= NaN

    τ = @. -(u_cen*dHdy - v_cen*dHdx)
# end
# interpolate τ to the bins
bin_edges = 0:5:500
bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) / 2
τ_interp,_ =  interpolate_z_dimension(τ, hab, bin_edges)

# Create histogram bins for topostrophy
N_τ = 100
τ_bins = range(-0.04, 0.04, length=N_τ+1)
τ_centers = (τ_bins[1:end-1] + τ_bins[2:end]) / 2

# Initialize array to store histograms
hist_data = zeros(length(τ_centers), length(bin_centers))

# Calculate histogram for each z level
for z_idx in 1:length(bin_centers)
    # Extract all topostrophy values at this z level
    τ_slice = τ_interp[:, :, z_idx, 1]
    # Remove NaN values
    τ_valid = filter(!isnan, vec(τ_slice))
    if !isempty(τ_valid)
        # Compute histogram
        hist = StatsBase.fit(Histogram, τ_valid, τ_bins, closed=:left)
        # Store normalized counts
        hist_data[:, z_idx] = hist.weights / sum(hist.weights)
    end
end

# Plot
fig = Figure(resolution=(800, 600))
ax = Axis(fig[1,1], 
    xlabel="Topostrophy τ [m/s]",
    ylabel="Height above bottom [m]",
    title="PDF of topostrophy")

hm = heatmap!(ax, τ_centers, bin_centers, log10.(hist_data), 
    colormap=:batlowK100, colorrange = (-5,-1)		
    )
Colorbar(fig[1,2], hm, label="log10(PDF)")

save(string("output/", simname, "/topostrophy_histogram_",simname,"_tᶠ=",tᶠ,".png"), fig)









# # Create a snapshot of arrows at a specific x-position
# snapshot_n = 1  # You can change this to any index you want

# fig_snapshot = CairoMakie.Figure(resolution = (800, 600))
# ax_snapshot = Axis(fig_snapshot[1, 1];
#     xlabel = "y (km)",
#     ylabel = "z (m)",
#     title = "Velocity vectors at x = $(round(xC[snapshot_n]/1e3, digits=1)) km",
#     limits = ((0, yF[end]*1e-3), (0, zF[end])))

# arrows!(ax_snapshot,
#     1e-3*yC[1:15:end], zC[1:15:end],
#     v_cen[snapshot_n,1:15:end,1:15:end],
#     1e3*w[snapshot_n,1:15:end,1:15:end],
#     arrowsize = 10,
#     lengthscale = 100,  # Increased lengthscale to make tails more visible
#     linewidth = 1.2,
#     color = :black)

# save(string("output/", simname, "/velocity_vectors_snapshot_",simname,"_tᶠ=",tᶠ,".png"), fig_snapshot)
