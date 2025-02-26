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
using MAT

simname = "2D_tilt"
timerange = "0-500"
θ = 0.0036
## load data
filename_slice = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_",timerange, "_1TP_timeavg.nc")
ds = Dataset(filename_slice,"r")

# grids
zC = ds["zC"][:]; Nz=length(zC); 
zF = ds["zF"][:]; 
xF = ds["xF"][:];
xC = ds["xC"][:]; Nx=length(xC)
yC = ds["yC"][:]; Ny=length(yC)
t = ds["time"][:];

# load all data
B = ds["B"][:,:,:,:]; 
Bz = ds["Bz"][:,:,:,:];
b = ds["b"][:,:,:,:];
uhat = ds["uhat"][:,:,:,:];
what = ds["what"][:,:,:,:];
v = ds["v"][:,:,:,:];


# set topography to NaN
uhat[uhat.==0] .= NaN
what[what.==0] .= NaN
v[b.==0] .= NaN
B[b.==0] .= NaN
Bz[b.==0] .= NaN

# plot
n = Observable(1)
ûₙ = @lift(uhat[:,1,:,$n+$n+10])
vₙ = @lift(v[:,1,:,$n])
ŵₙ = @lift(what[:,1,:,$n])
Bₙ = @lift(B[:,1,:,$n])
Bzₙ = @lift(Bz[:,1,:,$n])
ω₀ = 1.4e-4
M₂_period = 2π/ω₀



fig = Figure(resolution = (1200, 800), size=(1200,800))
axis_kwargs = (xlabel = "Zonal distance x (km)",
                  ylabel = "Elevation z (m)",
                  limits = ((0, ds["xF"][end]*1e-3), (0, ds["zF"][end])),
                  )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "ū and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_v = Axis(fig[2, 3]; title = "v̄ and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_w = Axis(fig[3, 1]; title = "w̄ and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_Bz = Axis(fig[3, 3]; title = L"\overline{dB/dz} \text{~and equally-spaced buoyancy contours (B)}", axis_kwargs...)

using ColorSchemes
U₀ = 0.025
hm_u = heatmap!(ax_u, xF*1e-3, zC, ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_u = contour!(ax_u, xC*1e-3, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u; label = "m/s")

hm_v = heatmap!(ax_v, xC*1e-3, zC, vₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_v = contour!(ax_v, xC*1e-3, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,4], hm_v)

hm_w = heatmap!(ax_w, xC*1e-3, zF, ŵₙ,
    colorrange = (-.5U₀, .5U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_w = contour!(ax_w, xC*1e-3, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_w)


hm_Bz = heatmap!(ax_Bz, xC*1e-3, zC, Bzₙ, colorrange=(-1e-6,1e-6),
    colormap = reverse(cgrad(:RdYlBu_9)),
    lowclip=reverse(cgrad(:RdYlBu_9))[1], highclip=reverse(cgrad(:RdYlBu_9))[end],
    nan_color = :gray)

ct_Bz = contour!(ax_Bz, xC*1e-3, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,4], hm_Bz)


frames =  (1:length(t))

filename = join(split(filename_slice, ".")[1:end-1], ".")

record(fig, string(filename,".mp4"), frames, framerate=8) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

# close(ds)

# plot the timeaveraged velocities 
what_timeavg = dropdims(mean(what[:,:,:,400:500], dims=4), dims=4)
uhat_timeavg = dropdims(mean(uhat[:,:,:,400:500], dims=4), dims=4)
v_timeavg = dropdims(mean(v[:,:,:,400:500], dims=4), dims=4)

# Create figure for time-averaged velocities
fig_timeavg = Figure(resolution=(1000, 800),size=(1000,800))
axis_kwargs = (xlabel = "Zonal distance x (km)",
              ylabel = "Elevation z (m)",
              limits = ((0, xF[end]*1e-3), (0, zF[end])))

ax_u = Axis(fig_timeavg[1, 1]; title = "400-500 TP averaged zonal velocity", axis_kwargs...)
ax_v = Axis(fig_timeavg[2, 1]; title = "400-500 TP meridional velocity", axis_kwargs...)
ax_w = Axis(fig_timeavg[3, 1]; title = "400-500 TP vertical velocity", axis_kwargs...)

using ColorSchemes
U₀ = 0.025
    
# Plot heatmaps
hm_u = heatmap!(ax_u, xF*1e-3, zC, uhat_timeavg[:,1,:],
    colorrange = (-3U₀, 3U₀),
    colormap = :diverging_bwr_20_95_c54_n256,
    nan_color = :gray)
Colorbar(fig_timeavg[1, 2], hm_u, label="m/s")

hm_v = heatmap!(ax_v, xC*1e-3, zC, v_timeavg[:,1,:],
    colorrange = (-3U₀, 3U₀),
    colormap = :diverging_bwr_20_95_c54_n256,
    nan_color = :gray)
Colorbar(fig_timeavg[2, 2], hm_v, label="m/s")

hm_w = heatmap!(ax_w, xC*1e-3, zF, what_timeavg[:,1,:],
    colorrange = (-.25U₀, .25U₀),
    colormap = :diverging_bwr_20_95_c54_n256,
    nan_color = :gray)
Colorbar(fig_timeavg[3, 2], hm_w, label="m/s")

display(fig_timeavg)
save(string("output/",simname,"/velocities_timeavg_400-500.png"), fig_timeavg)

######### plot the terrain following w averages (height above bottom), compare f=0 and f≠0

simname = "2D_tilt"
timerange = "0-500"
θ = 0.0036
## load data
filename_tilt = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_",timerange, "_1TP_timeavg.nc")
ds_tilt = Dataset(filename_tilt,"r")

simname = "2D_tilt_f=0"
filename_f0 = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_",timerange, "_1TP_timeavg.nc")
ds_f0 = Dataset(filename_f0,"r")

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")
bin_edge = 0:9:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2


# grids
zC = ds_tilt["zC"][:]; Nz=length(zC); 
zF = ds_tilt["zF"][:]; 
xF = ds_tilt["xF"][:]; dx = xF[end]-xF[end-1];
xC = ds_tilt["xC"][:]; Nx=length(xC)
yC = ds_tilt["yC"][:]; Ny=length(yC)
t = ds_tilt["time"][:];
z_face = xF * sin(θ) .+ zF' * cos(θ)     # Cartesian coordinate
t = ds_tilt["time"][:];
# mask it with buoyancy 
# Find height above bottom from buoyancy field
hab_mask = zeros(Nx,Ny,Nz)
w_data = ds_tilt["what"][:,:,:,1]  # get buoyancy data for first time step
for i in 1:Nx    # loop over x dimension
    for j in 1:Ny # loop over y dimension
        first_nonzero = findfirst(w_data[i,j,:] .> 0)  # find first index where b >= 0 in z dimension
        if first_nonzero !== nothing
            for k in first_nonzero:Nz
                hab_mask[i,j,k] = zF[k] - zF[first_nonzero]  # distance above topography
            end
        end
    end
end

Nt = length(t)
what_avg = zeros(length(bin_edge)-1,1)
what_avg_f0 = zeros(length(bin_edge)-1,1)
uhat_avg = zeros(length(bin_edge)-1,1)
uhat_avg_f0 = zeros(length(bin_edge)-1,1)
    uhat = mean(ds_tilt["uhat"][:,:,:,400:500],dims=4);     # true u
    what = mean(ds_tilt["what"][:,:,:,400:500],dims=4);     # true w
    what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
    # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
    wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
    u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
    w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
    
    uhat_f0 = mean(ds_f0["uhat"][:,:,:,400:500],dims=4);     # true u
    what_f0 = mean(ds_f0["what"][:,:,:,400:500],dims=4);     # true w
    what_cen_f0 = (what_f0[:,:,1:end-1,1] .+ what_f0[:,:,2:end,1])./2 # what at center
    # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
    wtemp_f0 = (vcat(what_cen_f0[end:end,:,:],what_cen_f0[1:end-1,:,:]) .+ what_cen_f0[:,:,:])./2
    u_f0 = uhat_f0[:,:,:,1]*cos(θ) .+ wtemp_f0*sin(θ) # cross-slope velocity
    w_f0 = -uhat_f0[:,:,:,1]*sin(θ) .+ wtemp_f0*cos(θ)# slope-normal velocity
    @time what_avg, _ = bins(what_cen,bin_edge,hab_mask,dx=dx,dy=1,z_face=z_face,normalize=true)
    @time what_avg_f0, _ = bins(what_cen_f0,bin_edge,hab_mask,dx=dx,dy=1,z_face=z_face,normalize=true)
    @time u_avg, _ = bins(u,bin_edge,hab_mask,dx=dx,dy=1,z_face=z_face,normalize=true)        
    @time u_avg_f0, _ = bins(u_f0,bin_edge,hab_mask,dx=dx,dy=1,z_face=z_face,normalize=true)        

    z = bin_center;
fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(1000,800),fontsize=20)
    axis_kwargs_ln = (ylabel = "hab (m)", yminorticksvisible = true, limits = (nothing,(0, 500)) )   
    
ax_u_ln = Axis(fig[1, 1]; title = "Cross-slope velocity", yminorticks = IntervalsBetween(5),axis_kwargs_ln...)
ax_what_ln = Axis(fig[1, 2]; 
                  title = "True vertical velocity", 
                  yminorticks = IntervalsBetween(5),
                  axis_kwargs_ln...)

lines!(ax_u_ln, u_avg[:,1], z[:], linewidth=3,color=:black)
lines!(ax_u_ln, u_avg_f0[:,1], z[:], linewidth=3,color=:red, linestyle=:dash)

ln1 = lines!(ax_what_ln, what_avg[:,1], z[:], linewidth=3,color=:black)
ln2 = lines!(ax_what_ln, what_avg_f0[:,1], z[:], linewidth=3,color=:red, linestyle=:dash)

# Add the legend to the axis
axislegend(ax_what_ln, [ln1, ln2], ["f = -5.3e-5 1/s","f = 0"],
    position = :lt,
    orientation = :vertical
)
display(fig)
save(string("output/",simname,"/hab_velocities_compare_with&without_f_400-500timeavg.png"),fig)

# 
bottom_indices = zeros(Int, size(hab, 1),1)
for i in 1:size(hab, 1)
        bottom_indices[i] = findfirst(hab[i, 1, :] .> 0)
end

w_abovetopography = [what_cen[i,1,bottom_indices[i]]  for i in 1:size(what_cen,1)]
mean(w_abovetopography)

################### plot the terrain following averages (height above bottom)

simname = "2D_tilt"
timerange = "0-500"
θ = 0.0036

## load data
filename_slice = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_",timerange, "_1TP_timeavg.nc")
ds = Dataset(filename_slice,"r")

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")
bin_edge = 0:9:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2

# grids
zC = ds["zC"][:]; Nz=length(zC); 
zF = ds["zF"][:]; dx = xF[end]-xF[end-1];
xF = ds["xF"][:];
xC = ds["xC"][:]; Nx=length(xC)
yC = ds["yC"][:]; Ny=length(yC)
t = ds["time"][:];
z_face = xF * sin(θ) .+ zF' * cos(θ)     # Cartesian coordinate

Nt = length(t)
b_avg = zeros(length(bin_edge)-1,Nt)
Bz_avg = zeros(length(bin_edge)-1,Nt)
uhat_avg = zeros(length(bin_edge)-1,Nt)
u_avg = zeros(length(bin_edge)-1,Nt)
what_avg = zeros(length(bin_edge)-1,Nt)
τ_avg = zeros(length(bin_edge)-1,Nt)
# Find height above bottom from buoyancy field
hab_mask = zeros(Nx,Ny,Nz)
w_data = ds["what"][:,:,:,1]  # get buoyancy data for first time step
for i in 1:Nx    # loop over x dimension
    for j in 1:Ny # loop over y dimension
        first_nonzero = findfirst(w_data[i,j,:] .> 0)  # find first index where b >= 0 in z dimension
        if first_nonzero !== nothing
            for k in first_nonzero:Nz
                hab_mask[i,j,k] = zF[k] - zF[first_nonzero]  # distance above topography
            end
        end
    end
end
for n in 1:Nt
    b = ds["b"][:,:,:,n:n];           # buoyancy perturbation
    B = ds["B"][:,:,:,n:n];           # total buoyancy
    Bz = ds["Bz"][:,:,:,n:n];           # total buoyancy
    uhat = ds["uhat"][:,:,:,n:n];     # true u
    what = ds["what"][:,:,:,n:n];     # true w
    what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
    # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
    wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
    u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
    w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
    @time Bz_avg[:,n], _ = bins(Bz,bin_edge,hab_mask,dx=dx,dy=30,z_face=z_face,normalize=true)
    @time what_avg[:,n], _ = bins(what_cen,bin_edge,hab_mask,dx=dx,dy=30,z_face=z_face,normalize=true)
    @time u_avg[:,n], _ = bins(u,bin_edge,hab_mask,dx=dx,dy=30,z_face=z_face,normalize=true)        
    @time b_avg[:,n], _ = bins(b,bin_edge,hab_mask,dx=dx,dy=30,z_face=z_face,normalize=true)        
@info n
end

what_avg_100TPaverage = mean(what_avg[:,400:501], dims=2)
fig = Figure(resolution = (1600, 1600), figure_padding=(10, 40, 10, 10), size=(1600,1600), fontsize=20)

# Define common axis arguments
axis_kwargs_hm = (xlabel = "time (tidal cycle)",
                 ylabel = "hab (m)",
                 yminorticksvisible = true,
                 limits = ((t[1], t[end]), (0, 1500)))

axis_kwargs_line = (ylabel = "hab (m)", 
                   yminorticksvisible = true, 
                   limits = (nothing,(0, 500)))

# Create all axes
ax_b = Axis(fig[1, 1]; title = "b (buoyancy perturbation)", yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_b_ln = Axis(fig[1, 3]; title = "b (buoyancy perturbation)", yminorticks = IntervalsBetween(5), limits = ((0,maximum(b_avg)),(0, 500)))

ax_Bz = Axis(fig[2, 1]; title = "dB/dz (Total buoyancy gradient)", yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_Bz_ln = Axis(fig[2, 3]; title = "10⁻⁶ x dB/dz (Total buoyancy gradient)", yminorticks = IntervalsBetween(5), axis_kwargs_line...)

ax_u = Axis(fig[3, 1]; title = "Cross-slope velocity", yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_u_ln = Axis(fig[3, 3]; title = "Cross-slope velocity", yminorticks = IntervalsBetween(5), axis_kwargs_line...)

ax_what = Axis(fig[4, 1]; title = "True vertical velocity", yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_what_ln = Axis(fig[4, 3]; title = "True vertical velocity", 
                  yminorticks = IntervalsBetween(5),
                  xticks = ([-0.0015, -0.001, -0.0005, 0], ["-0.0015", "-0.001", "-0.0005", "0"]),
                  axis_kwargs_line...)

# Plot heatmaps
hm_b = heatmap!(ax_b, t[:], bin_center[:], b_avg',
    colorrange = (0,0.00012), 
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
Colorbar(fig[1,2], hm_b)

hm_Bz = heatmap!(ax_Bz, t[:], bin_center[:], Bz_avg',
    colormap = :diverging_bwr_20_95_c54_n256, 
    colorrange=(4.e-7,10.e-7),
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
Colorbar(fig[2,2], hm_Bz)

hm_u = heatmap!(ax_u, t[:], bin_center[:], u_avg',
    colorrange = (-0.01, 0.01), 
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
Colorbar(fig[3,2], hm_u)

hm_what = heatmap!(ax_what, t[:], bin_center[:], what_avg',
    colorrange = (-0.001, 0.001), 
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
Colorbar(fig[4,2], hm_what)

# Plot line graphs
ind = argmin(abs.(t/(2*pi/1.4e-4) .- 300))

# Buoyancy lines
lines!(ax_b_ln, b_avg[:,ind], bin_center[:], linewidth=3, color=:black)
lines!(ax_b_ln, b_avg[:,end], bin_center[:], linewidth=3, color=:red)

lines!(ax_Bz_ln, 1e6*Bz_avg[:,ind], bin_center[:], linewidth=3, color=:black)
lines!(ax_Bz_ln, 1e6*Bz_avg[:,end], bin_center[:], linewidth=3, color=:red)

# Velocity lines
lines!(ax_u_ln, u_avg[:,ind], bin_center[:], linewidth=3, color=:black)
lines!(ax_u_ln, u_avg[:,end], bin_center[:], linewidth=3, color=:red)
lines!(ax_u_ln, u_avg_20TPaverage[:], bin_center[:], linewidth=3, color=:blue)
lines!(ax_u_ln, [0,0], [0,bin_center[end]], color=:black)

lines!(ax_what_ln, what_avg[:,ind], bin_center[:], linewidth=3, color=:black)
lines!(ax_what_ln, what_avg[:,end], bin_center[:], linewidth=3, color=:red)
lines!(ax_what_ln, what_avg_100TPaverage[:], bin_center[:], linewidth=3, color=:blue)
lines!(ax_what_ln, [0,0], [0,bin_center[end]], color=:black)

# Add legends
label1 = "$(Int(round(t[ind]/(2*pi/1.4e-4)))) tidal cycle"
label2 = "$(Int(round(t[end]/(2*pi/1.4e-4)))) tidal cycle"
label3 = "$(Int(round(t[400]/(2*pi/1.4e-4))))-$(Int(round(t[501]/(2*pi/1.4e-4)))) tidal average"

axislegend(ax_what_ln, 
    [LineElement(color=:black), LineElement(color=:red), LineElement(color=:blue)],
    [label1, label2, label3],
    position = :lt,
    orientation = :vertical)

display(fig)
save(string("output/",simname,"/hab_buoyancy_velocities_",timerange,".png"), fig)
