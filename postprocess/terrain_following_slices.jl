# this script plots the cross sections of terrain following heatmaps
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

# preallocate
B_avg = zeros(length(bin_edge) - 1, Nt)
Bz_avg = zeros(length(bin_edge) - 1, Nt)
uhat_avg = zeros(length(bin_edge) - 1, Nt)
u_avg = zeros(length(bin_edge) - 1, Nt)
what_avg = zeros(length(bin_edge) - 1, Nt)

∇κ∇B_avg = zeros(length(bin_edge) - 1, Nt)
div_uB_avg = zeros(length(bin_edge) - 1, Nt)
u_bar_∇B_bar_avg = zeros(length(bin_edge) - 1, Nt)
u_prime∇B_prime_avg = zeros(length(bin_edge) - 1, Nt)
dBdt_avg = zeros(length(bin_edge) - 1, Nt)

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

new_bin_edges = 0:8:1500
bin_centers = @. (new_bin_edges[1:end-1] + new_bin_edges[2:end]) / 2
interpolated_what = zeros(Nx,Ny,size(bin_centers,1))
interpolated_uhat = zeros(Nx,Ny,size(bin_centers,1))
interpolated_v = zeros(Nx,Ny,size(bin_centers,1))
interpolated_chi = zeros(Nx,Ny,size(bin_centers,1))
interpolated_eps = zeros(Nx,Ny,size(bin_centers,1))
interpolated_Bz = zeros(Nx,Ny,size(bin_centers,1))
interpolated_u_bar_∇B_bar = zeros(Nx,Ny,size(bin_centers,1))
interpolated_u_prime∇B_prime = zeros(Nx,Ny,size(bin_centers,1))
interpolated_∇κ∇B = zeros(Nx,Ny,size(bin_centers,1))
interpolated_wBz_center = zeros(Nx,Ny,size(bin_centers,1))

# interpolation
@time interpolated_what[:,:,:],_ =  interpolate_z_dimension(what_cen, hab, new_bin_edges)
@time interpolated_uhat[:,:,:],_ =  interpolate_z_dimension(uhat_cen, hab, new_bin_edges)
@time interpolated_v[:,:,:],_ =  interpolate_z_dimension(v_cen, hab, new_bin_edges)
@time interpolated_chi[:,:,:],_ =  interpolate_z_dimension(χ, hab, new_bin_edges)
@time interpolated_eps[:,:,:],_ =  interpolate_z_dimension(ε, hab, new_bin_edges)
@time interpolated_Bz[:,:,:],_ =  interpolate_z_dimension(Bz_center, hab, new_bin_edges)
@time interpolated_u_bar_∇B_bar[:,:,:],_ = interpolate_z_dimension(u_bar_∇B_bar, hab, new_bin_edges)
@time interpolated_u_prime∇B_prime[:,:,:],_ = interpolate_z_dimension(u_prime∇B_prime, hab, new_bin_edges)
@time interpolated_∇κ∇B[:,:,:],_ = interpolate_z_dimension(∇κ∇B[:,:,:,1], hab, new_bin_edges)
@time interpolated_wBz_center[:,:,:],_ = interpolate_z_dimension(w .* Bz_center[:,:,:,1], hab, new_bin_edges)

# bathymetry
using MAT
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


##########1) simple figure: take the average from the bottom to some height above the bottom
using PyPlot

bin_size = 8
z_position = 4:8:84
# for i in 1:length(z_position)
# z_upper = argmin(abs.(bin_centers.-z_position[i]))
# end
z_upper = 3
close(gcf())
fig, (ax1, ax2, ax3) = subplots(1, 3, figsize=(15, 5))

# Import the necessary colors module from matplotlib
mpl_colors = PyPlot.pyimport("matplotlib.colors")

# Define parameters for symmetric log scale
# linthresh defines the linear range around zero
linthresh = 0.05  # Adjust this value based on your data
vmax = maximum(abs.(nansum(interpolated_what[:,:,1:z_upper]*bin_size,dim=(3))))
vmin = -vmax

# Create a symmetric log norm
norm = mpl_colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)

c1 = ax1.pcolor(xC[:]*1e-3, yC[:]*1e-3, nansum(interpolated_what[:,:,1:z_upper]*bin_size,dim=(3))',
                cmap="RdYlBu_r", 
                norm=norm)  # Apply the symmetric log norm here
colorbar(c1, ax=ax1, extend="both")
ax1.set_title("\$\\int_{0}^{" * string(round(Int, bin_centers[z_upper])) * "} \\bar{w} \\, dHAB\$")
h1=ax1.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
ax1.clabel(h1, inline=true, fontsize=8)  # Add labels to contour lines
ax1.set_xlabel("x [km]")
ax1.set_ylabel("y [km]")

c2 = ax2.pcolor(xC[:]*1e-3, yC[:]*1e-3, nanmean(interpolated_uhat[:,:,1:z_upper]*bin_size,dim=(3))',
                cmap="RdYlBu_r",
                vmin=-maximum(nanmean(interpolated_uhat[:,:,1:z_upper]*bin_size,dim=(3))), vmax=maximum(nanmean(interpolated_uhat[:,:,1:z_upper]*bin_size,dim=(3))))
colorbar(c2, ax=ax2, extend="both")
ax2.set_title("\$\\int_{0}^{" * string(round(Int, bin_centers[z_upper])) * "} \\bar{u} \\, dHAB\$")
h2=ax2.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
ax2.clabel(h2, inline=true, fontsize=8)  # Add labels to contour lines
ax2.set_xlabel("x [km]")
ax2.set_ylabel("y [km]")

c3 = ax3.pcolor(xC[:]*1e-3, yC[:]*1e-3, (nanmean(interpolated_Bz[:,:,1:z_upper],dim=(3))'),
                cmap="GnBu",
                vmin=0, vmax=2e-6)
colorbar(c3, ax=ax3, extend="both")
ax3.set_title(L"\bar{B}_z" * string("[0-", round(Int,bin_centers[z_upper]), "m]"))
h3=ax3.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
ax3.clabel(h3, inline=true, fontsize=8)  # Add labels to contour lines
ax3.set_xlabel("x [km]")
ax3.set_ylabel("y [km]")
savefig(string("output/",simname,"/hab_u_w_Bz_xy_heatmap_hab=",z_position[z_upper],".png"))






############## animation: 
#1) taking the average from the bottom to some height above the bottom
using CairoMakie
using Printf
using Statistics

# Setting up the animation
bin_size = 7
z_position = 7:7:500

# Create the output directory if it doesn't exist
output_dir = string("output/", simname)
isdir(output_dir) || mkpath(output_dir)

# Full animation filename
anim_filename = string(output_dir, "/hab_u_w_Bz_xy_animation.mp4")

# Create Observable for the z-index
n = Observable(1)

# Create lift expressions for the data
what_slice = @lift(nansum(interpolated_what[:,:,1:argmin(abs.(bin_centers .- z_position[$n]))]*bin_size, dims=3))
uhat_slice = @lift(nanmean(interpolated_uhat[:,:,1:argmin(abs.(bin_centers .- z_position[$n]))]*bin_size, dims=3))
Bz_slice = @lift(nanmean(interpolated_Bz[:,:,1:argmin(abs.(bin_centers .- z_position[$n]))], dims=3))

# Create the figure
# Create the figure with slightly more horizontal space to accommodate colorbars
# Create the figure with more horizontal space to accommodate colorbars
# Create the figure with a resolution that better fits the content
# Create the figure with a resolution that fits the content
fig = CairoMakie.Figure(resolution=(1200, 600), fontsize=16)

# Create a custom grid layout with explicit column widths
layout = GridLayout()

# Assign the layout to the figure
fig[1:2, 1] = layout

# Create title with changing z value
title = @lift("z = $(round(Int, bin_centers[argmin(abs.(bin_centers .- z_position[$n]))])) m")
layout[1, 1:6] = Label(fig, title, fontsize=24)

# Create axes with specific aspect ratios
# Subplots will be in columns 1, 3, and 5; colorbars in columns 2, 4, and 6
ax1 = Axis(layout[2, 1], 
           title = L"\int_0^z \bar{w} \, dHAB",
           xlabel = "x [km]", 
           ylabel = "y [km]",
           titlesize = 22)

ax2 = Axis(layout[2, 3], 
           title = L"\int_0^z \bar{u} \, dHAB",
           xlabel = "x [km]",
           titlesize = 22)

ax3 = Axis(layout[2, 5], 
           title = L"\bar{B}_z (0-z)",
           xlabel = "x [km]",
           titlesize = 22)

# Set consistent aspect ratios
ax1.aspect = 0.5
ax2.aspect = 0.5
ax3.aspect = 0.5

# First plot: Vertical velocity
hm1 = heatmap!(ax1, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($what_slice, dims=3)),
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = @lift((-maximum(abs.(dropdims($what_slice, dims=3)))*0.5, 
                                   maximum(abs.(dropdims($what_slice, dims=3)))*0.5)))
cb1 = Colorbar(layout[2, 2], hm1, width=8)  # Place in column 2, slim width
contour!(ax1, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Second plot: Horizontal velocity
hm2 = heatmap!(ax2, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($uhat_slice, dims=3)),
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = @lift((-maximum(abs.(dropdims($uhat_slice, dims=3)))*0.8, 
                                   maximum(abs.(dropdims($uhat_slice, dims=3)))*0.8)))
cb2 = Colorbar(layout[2, 4], hm2, width=8)  # Place in column 4, slim width
contour!(ax2, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Third plot: Stratification
hm3 = heatmap!(ax3, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($Bz_slice, dims=3)),
               colormap = :GnBu,
               colorrange = (0, 2e-6))
cb3 = Colorbar(layout[2, 6], hm3, width=8)  # Place in column 6, slim width
contour!(ax3, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Set column widths: larger for subplots, smaller for colorbars
colsize!(layout, 1, Relative(0.3))  # Subplot 1
colsize!(layout, 2, Relative(0.03))  # Colorbar 1
colsize!(layout, 3, Relative(0.3))  # Subplot 2
colsize!(layout, 4, Relative(0.03))  # Colorbar 2
colsize!(layout, 5, Relative(0.3))  # Subplot 3
colsize!(layout, 6, Relative(0.03))  # Colorbar 3

# Minimize spacing between columns
colgap!(layout, 0)  # Set to 0 to eliminate gaps

# Add minimal padding to figure
fig.attributes.margin = (10, 10, 10, 10)

# Create the animation
record(fig, anim_filename, 1:length(z_position); framerate=5) do i
    @info "Rendering frame $i of $(length(z_position))..."
    n[] = i
end

##############2) taking the moving average from the bottom to some height above the bottom
using CairoMakie
using Printf
using Statistics

# Setting up the animation
bin_size = 8
z_position = 8:8:500

# Create the output directory if it doesn't exist
output_dir = string("output/", simname)
# Full animation filename
anim_filename = string(output_dir, "/hab_moving_average_u_v_w_Bz_xy_animation.mp4")

# Create Observable for the z-index
n = Observable(1)

# Create lift expressions for the data
what_slice = @lift(nanmean(interpolated_what[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3))
uhat_slice = @lift(nanmean(interpolated_uhat[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3))
Bz_slice = @lift(nanmean(interpolated_Bz[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3))
v_slice = @lift(nanmean(interpolated_v[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3))

# Create the figure with a resolution that fits a 2x2 layout
fig = CairoMakie.Figure(resolution=(1200, 1000), fontsize=16)

# Create a custom grid layout
layout = GridLayout()

# Assign the layout to the figure
fig[1:3, 1] = layout

# Create title with changing z value
title = @lift("z = $(round(Int, bin_centers[argmin(abs.(bin_centers .- z_position[$n]))]))-$(round(Int, bin_centers[argmin(abs.(bin_centers .- z_position[$n]))+3])) m")
layout[1, 1:8] = Label(fig, title, fontsize=24)

# Create axes for 2x2 layout
# Top left: u
ax1 = Axis(layout[2, 1], 
           title = L"\langle\bar{u}\rangle",
           ylabel = "y [km]",
           titlesize = 22)

# Top right: v
ax2 = Axis(layout[2, 5], 
           title = L"\langle\bar{v}\rangle",
           titlesize = 22)

# Bottom left: w
ax3 = Axis(layout[3, 1], 
           title = L"\langle\bar{w}\rangle",
           xlabel = "x [km]", 
           ylabel = "y [km]",
           titlesize = 22)

# Bottom right: Bz
ax4 = Axis(layout[3, 5], 
           title = L"\langle\bar{B}_z\rangle",
           xlabel = "x [km]",
           titlesize = 22)

# Set consistent aspect ratios
for ax in [ax1, ax2, ax3, ax4]
    ax.aspect = 1.0
end

# Plot each variable with a colorbar
# u (top left)
hm1 = heatmap!(ax1, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($uhat_slice, dims=3)),
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.1, 0.1))
cb1 = Colorbar(layout[2, 3], hm1, width=8)
contour!(ax1, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# v (top right)
hm2 = heatmap!(ax2, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($v_slice, dims=3)),
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.05, 0.05))
cb2 = Colorbar(layout[2, 7], hm2, width=8)
contour!(ax2, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# w (bottom left)
hm3 = heatmap!(ax3, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($what_slice, dims=3)),
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.02, 0.02))
cb3 = Colorbar(layout[3, 3], hm3, width=8)
contour!(ax3, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Bz (bottom right)
hm4 = heatmap!(ax4, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($Bz_slice, dims=3)),
               colormap = :GnBu,
               colorrange = (0, 2e-6))
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

# Create the animation
record(fig, anim_filename, 1:length(z_position); framerate=5) do i
    @info "Rendering frame $i of $(length(z_position))..."
    n[] = i
end


#######3) plot u,v,w, and topostrophy
using CairoMakie
using Printf
using Statistics



new_bin_edges = 0:8:1500
bin_centers = @. (new_bin_edges[1:end-1] + new_bin_edges[2:end]) / 2
interpolated_what = zeros(Nx,Ny,size(bin_centers,1))
interpolated_uhat = zeros(Nx,Ny,size(bin_centers,1))
interpolated_v = zeros(Nx,Ny,size(bin_centers,1))
interpolated_tau = zeros(Nx,Ny,size(bin_centers,1))
# interpolation
@time interpolated_what[:,:,:],_ =  interpolate_z_dimension(what_cen, hab, new_bin_edges)
@time interpolated_uhat[:,:,:],_ =  interpolate_z_dimension(uhat_cen, hab, new_bin_edges)
@time interpolated_v[:,:,:],_ =  interpolate_z_dimension(v_cen, hab, new_bin_edges)
@time interpolated_tau[:,:,:],_ = interpolate_z_dimension(τ, hab, new_bin_edges)
# Setting up the animation
bin_size = 8
z_position = 8:8:500

# Create the output directory if it doesn't exist
output_dir = string("output/", simname)
# Full animation filename
anim_filename = string(output_dir, "/hab_moving_average_u_v_w_tau_xy_animation.mp4")

# Create Observable for the z-index
n = Observable(1)

# Create lift expressions for the data
what_slice = @lift(nanmean(interpolated_what[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3))
uhat_slice = @lift(nanmean(interpolated_uhat[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3))
v_slice = @lift(nanmean(interpolated_v[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3))
tau_slice = @lift(nanmean(interpolated_tau[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3))
# Create the figure with a resolution that fits a 2x2 layout
fig = CairoMakie.Figure(resolution=(1200, 1000), fontsize=16)

# Create a custom grid layout
layout = GridLayout()

# Assign the layout to the figure
fig[1:3, 1] = layout

# Create title with changing z value
title = @lift("z = $(round(Int, bin_centers[argmin(abs.(bin_centers .- z_position[$n]))]))-$(round(Int, bin_centers[argmin(abs.(bin_centers .- z_position[$n]))+3])) m")
layout[1, 1:8] = Label(fig, title, fontsize=24)

# Create axes for 2x2 layout
# Top left: u
ax1 = Axis(layout[2, 1], 
           title = L"\langle\bar{u}\rangle",
           ylabel = "y [km]",
           titlesize = 22)

# Top right: v
ax2 = Axis(layout[2, 5], 
           title = L"\langle\bar{v}\rangle",
           titlesize = 22)

# Bottom left: w
ax3 = Axis(layout[3, 1], 
           title = L"\langle\bar{w}\rangle",
           xlabel = "x [km]", 
           ylabel = "y [km]",
           titlesize = 22)

# Bottom right: Bz
ax4 = Axis(layout[3, 5], 
           title = L"\langle τ~\mathrm{(topostrophy)} \rangle",
           xlabel = "x [km]",
           titlesize = 22)

# Set consistent aspect ratios
for ax in [ax1, ax2, ax3, ax4]
    ax.aspect = 1.0
end

# Plot each variable with a colorbar
# u (top left)
hm1 = heatmap!(ax1, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($uhat_slice, dims=3)),
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.1, 0.1))
cb1 = Colorbar(layout[2, 3], hm1, width=8)
contour!(ax1, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# v (top right)
hm2 = heatmap!(ax2, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($v_slice, dims=3)),
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.05, 0.05))
cb2 = Colorbar(layout[2, 7], hm2, width=8)
contour!(ax2, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# w (bottom left)
hm3 = heatmap!(ax3, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($what_slice, dims=3)),
               colormap = reverse(cgrad(:RdYlBu)),
               colorrange = (-0.005, 0.005))
cb3 = Colorbar(layout[3, 3], hm3, width=8)
contour!(ax3, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# τ (bottom right)
hm4 = heatmap!(ax4, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($tau_slice, dims=3)),
               colormap = reverse(cgrad(:RdBu)),
               colorrange = (-0.1, 0.1).*0.2)
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

# Create the animation
record(fig, anim_filename, 1:length(z_position); framerate=5) do i
    @info "Rendering frame $i of $(length(z_position))..."
    n[] = i
end



######4) animation for chi and epsilon: hab average from the bottom to some height above the bottom
using CairoMakie
using Printf
using Statistics

# Setting up the animation
bin_size = 7
z_position = 7:7:500

output_dir = string("output/", simname)
# Full animation filename
anim_filename = string(output_dir, "/hab_chi_eps_xy_animation.mp4")

# Create Observable for the z-index
n = Observable(1)

# Create lift expressions for the data
chi_slice = @lift(log10.(nanmean(interpolated_chi[:,:,1:argmin(abs.(bin_centers .- z_position[$n]))]
            ./abs.(interpolated_Bz[:,:,1:argmin(abs.(bin_centers .- z_position[$n]))]), dims=3)) )
eps_slice = @lift(log10.(nanmean(interpolated_eps[:,:,1:argmin(abs.(bin_centers .- z_position[$n]))], dims=3)) )

# Create the figure
fig = CairoMakie.Figure(resolution=(800, 600), fontsize=16)

# Create a custom grid layout with explicit column widths
layout = GridLayout()

# Assign the layout to the figure
fig[1:2, 1] = layout

# Create title with changing z value
title = @lift("z = $(round(Int, bin_centers[argmin(abs.(bin_centers .- z_position[$n]))])) m")
layout[1, 1:4] = Label(fig, title, fontsize=24)

# Create axes with specific aspect ratios
ax1 = Axis(layout[2, 1], 
           title = L"log_{10}\frac{1}{z} ∫_0^z \bar{\chi}~ dHAB",
           xlabel = "x [km]", 
           ylabel = "y [km]",
           titlesize = 22,
           limits = ((0,15),(0,30)))

ax2 = Axis(layout[2, 3], 
           title = L"log_{10}\frac{1}{z} ∫_0^z \bar{\varepsilon}~ dHAB",
           xlabel = "x [km]",
           titlesize = 22)

# Set consistent aspect ratios
ax1.aspect = 0.5
ax2.aspect = 0.5

# First plot: Chi
hm1 = heatmap!(ax1, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($chi_slice, dims=3)),
               colormap = reverse(cgrad(:Spectral)),
               colorrange =(-9, -6))
cb1 = Colorbar(layout[2, 2], hm1, width=8)
contour!(ax1, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Second plot: Epsilon
hm2 = heatmap!(ax2, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($eps_slice, dims=3)),
               colormap = reverse(cgrad(:Spectral)),
               colorrange = (-9,-6))
cb2 = Colorbar(layout[2, 4], hm2, width=8)
contour!(ax2, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Set column widths: larger for subplots, smaller for colorbars
colsize!(layout, 1, Relative(0.45))  # Subplot 1
colsize!(layout, 2, Relative(0.05))  # Colorbar 1
colsize!(layout, 3, Relative(0.45))  # Subplot 2
colsize!(layout, 4, Relative(0.05))  # Colorbar 2

# Minimize spacing between columns
colgap!(layout, 0)

# Add minimal padding to figure
fig.attributes.margin = (10, 10, 10, 10)

# Create the animation
record(fig, anim_filename, 1:length(z_position); framerate=5) do i
    @info "Rendering frame $i of $(length(z_position))..."
    n[] = i
end



####4) moving average for chi and epsilon: hab average from the bottom to some height above the bottom
######3) animation for chi and epsilon: hab average from the bottom to some height above the bottom
using CairoMakie
using Printf
using Statistics

# Setting up the animation
bin_size = 7
z_position = 7:7:500

output_dir = string("output/", simname)
# Full animation filename
anim_filename = string(output_dir, "/hab_moving_average_chi_eps_xy_animation.mp4")

# Create Observable for the z-index
n = Observable(1)

# Create lift expressions for the data
chi_slice = @lift(log10.(nanmean(interpolated_chi[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3]
            ./abs.(interpolated_Bz[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3]), dims=3)) )
eps_slice = @lift(log10.(nanmean(interpolated_eps[:,:,argmin(abs.(bin_centers .- z_position[$n])):argmin(abs.(bin_centers .- z_position[$n]))+3], dims=3)) )

# Create the figure
fig = CairoMakie.Figure(resolution=(800, 600), fontsize=16)

# Create a custom grid layout with explicit column widths
layout = GridLayout()

# Assign the layout to the figure
fig[1:2, 1] = layout

# Create title with changing z value
title = @lift("z = $(round(Int, bin_centers[argmin(abs.(bin_centers .- z_position[$n]))]))-$(round(Int, bin_centers[argmin(abs.(bin_centers .- z_position[$n]))+3])) m")
layout[1, 1:4] = Label(fig, title, fontsize=24)

# Create axes with specific aspect ratios
ax1 = Axis(layout[2, 1], 
           title = L"log_{10}\langle\bar{\chi}\rangle",
           xlabel = "x [km]", 
           ylabel = "y [km]",
           titlesize = 22,
           limits = ((0,15),(0,30)))

ax2 = Axis(layout[2, 3], 
           title = L"log_{10}\langle\bar{\varepsilon}\rangle",
           xlabel = "x [km]",
           titlesize = 22)

# Set consistent aspect ratios
ax1.aspect = 0.5
ax2.aspect = 0.5

# First plot: Chi
hm1 = heatmap!(ax1, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($chi_slice, dims=3)),
               colormap = reverse(cgrad(:Spectral)),
               colorrange =(-9, -6))
cb1 = Colorbar(layout[2, 2], hm1, width=8)
contour!(ax1, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Second plot: Epsilon
hm2 = heatmap!(ax2, xC[:]*1e-3, yC[:]*1e-3, @lift(dropdims($eps_slice, dims=3)),
               colormap = reverse(cgrad(:Spectral)),
               colorrange = (-9,-6))
cb2 = Colorbar(layout[2, 4], hm2, width=8)
contour!(ax2, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)

# Set column widths: larger for subplots, smaller for colorbars
colsize!(layout, 1, Relative(0.45))  # Subplot 1
colsize!(layout, 2, Relative(0.05))  # Colorbar 1
colsize!(layout, 3, Relative(0.45))  # Subplot 2
colsize!(layout, 4, Relative(0.05))  # Colorbar 2

# Minimize spacing between columns
colgap!(layout, 0)

# Add minimal padding to figure
fig.attributes.margin = (10, 10, 10, 10)

# Create the animation
record(fig, anim_filename, 1:length(z_position); framerate=5) do i
    @info "Rendering frame $i of $(length(z_position))..."
    n[] = i
end







##########plot an animation of u, w, chi, eps, Bz with respect to z (x-y plane)
using CairoMakie
using Printf
using Statistics

# Setting up the animation
z_indices = 1:210#:length(zC)  # Use original z indices instead of interpolated values

# Create output directory if it doesn't exist
output_dir = string("output/", simname)
anim_filename = string(output_dir, "/xy_5var_slices_animation.mp4")

# Create Observable for the z-index
n = Observable(1)

# Create mask based on buoyancy field
uhat[uhat .== 0] .= NaN
what_center[what_center .== 0] .= NaN
chi[b[:,:,:,1] .== 0] .= NaN
Bz_center[b[:,:,:,1] .== 0] .= NaN
eps[eps .== 0] .= NaN
# Create lift expressions for sliced data at original z levels
uhat_slice = @lift(uhat[:,:,$n])
what_center_slice = @lift(what_center[:,:,$n])
chi_slice = @lift(log10.(chi[:,:,$n]./abs.(Bz_center[:,:,$n])))
eps_slice = @lift(log10.(eps[:,:,$n]))
Bz_center_slice = @lift(Bz_center[:,:,$n])

# # Replace NaNs and Infs for log scaled plots
# chi_slice = @lift begin
#     slice = copy($chi_slice)
#     replace!(slice, -Inf => NaN)
#     replace!(slice, Inf => NaN)
#     slice
# end

# eps_slice = @lift begin
#     slice = copy($eps_slice)
#     replace!(slice, -Inf => NaN)
#     replace!(slice, Inf => NaN)
#     slice
# end

# Create the figure
fig = CairoMakie.Figure(resolution=(1400, 900), fontsize=18)  # Increased figure height and fontsize

# Create the grid layout
layout = GridLayout()
fig[1:4, 1] = layout  # Increased rows to accommodate horizontal colorbars

# Create title with current depth
title = @lift("Depth = $(round(zC[$n], digits=1)) m")
layout[1, 1:5] = Label(fig, title, fontsize=30)  # Increased title fontsize

# Create the axes (in row 2)
ax1 = Axis(layout[2, 1], title = L"\bar{u}", ylabel = "y [km]", titlesize=24, ylabelsize=20, aspect=DataAspect())
ax2 = Axis(layout[2, 2], title = L"\bar{w}", titlesize=24, aspect=DataAspect())
ax3 = Axis(layout[2, 3], title = L"log_{10}(\bar{\chi})", titlesize=24, aspect=DataAspect())
ax4 = Axis(layout[2, 4], title = L"log_{10}(\bar{\varepsilon})", titlesize=24, aspect=DataAspect())
ax5 = Axis(layout[2, 5], title = L"\bar{B}_z", titlesize=24, aspect=DataAspect())

# Add x-axis labels to row 3 (below the plots)
Label(layout[3, 1], "x [km]", fontsize=20)
Label(layout[3, 2], "x [km]", fontsize=20)
Label(layout[3, 3], "x [km]", fontsize=20)
Label(layout[3, 4], "x [km]", fontsize=20)
Label(layout[3, 5], "x [km]", fontsize=20)

# Add heatmaps
hm1 = heatmap!(ax1, xC[:]*1e-3, yC[:]*1e-3, uhat_slice, colormap = reverse(cgrad(:RdYlBu)), colorrange=(-0.08, 0.08))
hm2 = heatmap!(ax2, xC[:]*1e-3, yC[:]*1e-3, what_center_slice, colormap = reverse(cgrad(:RdYlBu)), colorrange=(-0.01, 0.01))
hm3 = heatmap!(ax3, xC[:]*1e-3, yC[:]*1e-3, chi_slice, colormap=reverse(cgrad(:Spectral)), colorrange=(-10, -6))
hm4 = heatmap!(ax4, xC[:]*1e-3, yC[:]*1e-3, eps_slice, colormap=reverse(cgrad(:Spectral)), colorrange=(-10, -6))
hm5 = heatmap!(ax5, xC[:]*1e-3, yC[:]*1e-3, Bz_center_slice, colormap=:GnBu, colorrange=(0, 2e-6))

# Add horizontal colorbars below each plot (in row 4)
cb1 = Colorbar(layout[4, 1], hm1, vertical=false, ticklabelsize=16, height=10, width=Relative(0.8))
cb2 = Colorbar(layout[4, 2], hm2, vertical=false, ticklabelsize=16, height=10, width=Relative(0.8))
cb3 = Colorbar(layout[4, 3], hm3, vertical=false, ticklabelsize=16, height=10, width=Relative(0.8))
cb4 = Colorbar(layout[4, 4], hm4, vertical=false, ticklabelsize=16, height=10, width=Relative(0.8))
cb5 = Colorbar(layout[4, 5], hm5, vertical=false, ticklabelsize=16, height=10, width=Relative(0.8),
               ticks=[0.0, 1.0e-6, 2.0e-6])  # Explicitly specify just 3 tick positions

# Add contours of topography to all subplots
for ax in [ax1, ax2, ax3, ax4, ax5]
    contour!(ax, xC[:]*1e-3, yC[:]*1e-3, z_interp, linewidth=1.5, color=:black, levels=10)
    # Hide x-axis labels since we have them as separate labels
    hidexdecorations!(ax, grid=false)
    # Set consistent limits for all subplots
    ax.limits = ((0, maximum(xC)*1e-3), (0, maximum(yC)*1e-3))
end

# Hide y-axis for all but the first plot
for ax in [ax2, ax3, ax4, ax5]
    hideydecorations!(ax, grid=false)
end

# Set column sizes (all equal width)
for i in 1:5
    colsize!(layout, i, Relative(1/5))
end

# Adjust row sizes
rowsize!(layout, 1, Relative(0.1))  # Title
rowsize!(layout, 2, Relative(0.7))  # Plots
rowsize!(layout, 3, Relative(0.05)) # X-labels
rowsize!(layout, 4, Relative(0.15)) # Colorbars

# Add spacing between columns
colgap!(layout, 10)
rowgap!(layout, 5)

# Add padding to figure
fig.attributes.margin = (20, 20, 20, 20)

frames = 1:length(z_indices)
# Create the animation
record(fig, anim_filename, z_indices[frames]; framerate=10) do i
    @info "Rendering frame $i of $(z_indices[length(z_indices)])..."
    n[] = i
end


##### create x-z plot that loops over y
##########plot an animation of u, w, chi, eps, Bz in x-z plane, looping over y
using CairoMakie
using Printf
using Statistics

# Setting up the animation
y_indices = 1:2:length(yC)

# Create output directory if it doesn't exist
output_dir = string("output/", simname)
anim_filename = string(output_dir, "/xz_5var_slices_animation.mp4")

# Create Observable for the y-index
n = Observable(1)

# Create mask based on buoyancy field
uhat[uhat .== 0] .= NaN
what_center[what_center .== 0] .= NaN
chi[b[:,:,:,1] .== 0] .= NaN
Bz_center[b[:,:,:,1] .== 0] .= NaN
eps[eps .== 0] .= NaN

# Create lift expressions for sliced data
uhat_slice = @lift(uhat[:,$(n),:])  # Transpose to have z on vertical axis
what_center_slice = @lift(what_center[:,$(n),:])
chi_slice = @lift(log10.(chi[:,$(n),:]./abs.(Bz_center[:,$(n),:])))
eps_slice = @lift(log10.(eps[:,$(n),:]))
Bz_center_slice = @lift(Bz_center[:,$(n),:])

# Create the figure with 5 rows and 2 columns
fig = CairoMakie.Figure(resolution=(1000, 1400), fontsize=16)

# Create title with current y position
title = @lift("y = $(round(yC[$(n)], digits=1)) m")
Label(fig[1, 1:2], title, fontsize=24)

# Create axes for each variable
ax1 = Axis(fig[2, 1], title=L"\bar{u}", ylabel="z [m]", titlesize=20)
ax2 = Axis(fig[3, 1], title=L"\bar{w}", ylabel="z [m]", titlesize=20)
ax3 = Axis(fig[4, 1], title=L"log_{10}(\bar{\chi})", ylabel="z [m]", titlesize=20)
ax4 = Axis(fig[5, 1], title=L"log_{10}(\bar{\varepsilon})", ylabel="z [m]", titlesize=20)
ax5 = Axis(fig[6, 1], title=L"\bar{B}_z", ylabel="z [m]", xlabel="x [km]", titlesize=20)

# Create heatmaps
hm1 = heatmap!(ax1, xC[:]*1e-3, zC[:], uhat_slice, colormap=reverse(cgrad(:RdYlBu)), colorrange=(-0.08, 0.08),nan_color = :gray)
hm2 = heatmap!(ax2, xC[:]*1e-3, zC[:], what_center_slice, colormap=reverse(cgrad(:RdYlBu)), colorrange=(-0.01, 0.01),nan_color = :gray)
hm3 = heatmap!(ax3, xC[:]*1e-3, zC[:], chi_slice, colormap=reverse(cgrad(:Spectral)), colorrange=(-10, -6),nan_color = :gray)
hm4 = heatmap!(ax4, xC[:]*1e-3, zC[:], eps_slice, colormap=reverse(cgrad(:Spectral)), colorrange=(-10, -6),nan_color = :gray)
hm5 = heatmap!(ax5, xC[:]*1e-3, zC[:], Bz_center_slice, colormap=:GnBu, colorrange=(0, 2e-6),nan_color = :gray)

# Create colorbars in the second column
cb1 = Colorbar(fig[2, 2], hm1, width=15)
cb2 = Colorbar(fig[3, 2], hm2, width=15)
cb3 = Colorbar(fig[4, 2], hm3, width=15)
cb4 = Colorbar(fig[5, 2], hm4, width=15)
cb5 = Colorbar(fig[6, 2], hm5, width=15)

# Extract bathymetry profile for current y slice
bathy_profile = @lift(z_interp[:, $(n)])

# Add seafloor contour to each subplot
for (i, ax) in enumerate([ax1, ax2, ax3, ax4, ax5])
    # seafloor = @lift([(xC[j]*1e-3, z_interp[j, $(n)]) for j in 1:length(xC)])
    # lines!(ax, @lift([p[1] for p in $(seafloor)]), @lift([p[2] for p in $(seafloor)]), 
    #        color=:black, linewidth=2)
    
    # # Set x axis limits the same for all plots
    ax.limits = ((0, maximum(xC)*1e-3), nothing)
    
    # Only show x axis labels for bottom plot
    if i != 5
        hidexdecorations!(ax, grid=false, ticks=false)
    end
end

# Set column widths to give more space to the plots
colsize!(fig.layout, 1, Relative(0.85))
colsize!(fig.layout, 2, Relative(0.15))
frames = 1:length(y_indices)
# Create the animation
record(fig, anim_filename, y_indices[frames]; framerate=14) do i
    @info "Rendering frame $i of $(y_indices[length(y_indices)]))..."
    n[] = i
end


# using PyPlot
# close(gcf())
# PyPlot.plot(nanmean(interpolated_what_tavg[:,:,:],dim=(1,2)),bin_centers)
# gcf()
# savefig("output/tilt/interpolated_what_ln.png")


# # plot only Bz 
# using PyPlot

# close(gcf())
# fig, ax = subplots(1, 1, figsize=(5, 5))
#     # mpl_colors = PyPlot.pyimport("matplotlib.colors")
#     # linthresh = 1e-8

#     # norm_log = mpl_colors["SymLogNorm"](linthresh=linthresh, vmin=-1e-7, vmax=1e-5, base=10)
# c = ax.pcolor(xC[:]*1e-3, yC[:]*1e-3, (nanmean(interpolated_Bz_tavg[:,:,1:19],dim=(3))'),
#                 cmap="RdYlBu_r",
#                 vmin=0, vmax=2e-6)
# colorbar(c, ax=ax, extend="both")
# ax.set_title(L"\bar{B}_z (0-92.5 m)")
# h=ax.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
# PyPlot.clabel(h, inline=true, fontsize=8)  # Add labels to contour lines
# ax.set_xlabel("x [km]")
# ax.set_ylabel("y [km]")
# savefig("output/tilt/interpolated_Bz_heatmap_0-92.5.png")
