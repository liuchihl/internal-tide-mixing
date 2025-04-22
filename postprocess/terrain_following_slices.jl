# this script plots the cross sections of terrain following heatmaps
function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end
include("functions/interpolation_z_dimension.jl")
# test
using Interpolations
using NCDatasets
simname = "flat"
# timerange = "80-120"
tᶠ = 460 
θ=3.6e-3
# load data

filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];

filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_const_dt_Bbudget-wb-eps-chi.nc")
ds_field = Dataset(filename_field,"r")
filename_velocity = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_const_dt_u-v-w-Rig.nc")
ds_velocity = Dataset(filename_velocity,"r")
filename_B = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_B-c.nc")
ds_B = Dataset(filename_B,"r")
filename_verification = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",10, "_threeD_timeavg.nc")   
ds_verification = Dataset(filename_verification,"r")


zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]); 

xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
t = ds_field["time"][:];

new_bin_edges = 0:7:1500
bin_centers = @. (new_bin_edges[1:end-1] + new_bin_edges[2:end]) / 2
interpolated_what = zeros(Nx,Ny,size(bin_centers,1))
interpolated_uhat = zeros(Nx,Ny,size(bin_centers,1))
interpolated_Bz = zeros(Nx,Ny,size(bin_centers,1))
uhat = nanmean(ds_velocity["uhat"][:,:,:,:],dim=4);    # true u
what = nanmean(ds_velocity["what"][:,:,:,:],dim=4);    # true w
B = nanmean(ds_B["B"][:,:,:,n:n],dim=4);    # true w
b = ds_verification["b"][:,:,:,1:1];          # buoyancy perturbation

N = 1.e-3
Bz =  deriv(zC,B);
Bz[b[:,:,1:end-1,1:1].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
# interpolate Bz from faces to center cell
# Interpolate each row
Bz_center = zeros(size(Bz,1),size(Bz,2),length(zC))
for i in 1:size(Bz,1)
    for j in 1:size(Bz,2)
            itp = linear_interpolation(zF[2:end-1], Bz[i,j,:], extrapolation_bc=Line())
        Bz_center[i,j,:] = itp(zC)
    end
end
# Interpolate each row
what_center = zeros(size(what,1),size(what,2),length(zC))
for i in 1:size(what,1)
    for j in 1:size(what,2)
        itp = linear_interpolation(zF[:], what[i,j,:], extrapolation_bc=Line())
        what_center[i,j,:] = itp(zC)
    end
end

@time interpolated_what[:,:,:],_ =  interpolate_z_dimension(what_center, hab, new_bin_edges)
@time interpolated_uhat[:,:,:],_ =  interpolate_z_dimension(uhat, hab, new_bin_edges)
@time interpolated_Bz[:,:,:],_ =  interpolate_z_dimension(Bz_center, hab, new_bin_edges)

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



using PyPlot

bin_size = 7
z_position = 7:7:49
for i in 1:length(z_position)
z_upper = argmin(abs.(bin_centers.-z_position[i]))
end

close(gcf())
fig, (ax1, ax2, ax3) = subplots(1, 3, figsize=(15, 5))

c1 = ax1.pcolor(xC[:]*1e-3, yC[:]*1e-3, nansum(interpolated_what[:,:,1:z_upper]*bin_size,dim=(3))',
                cmap="RdYlBu_r",
                vmin=-maximum(nansum(interpolated_what[:,:,1:z_upper]*bin_size,dim=(3))), vmax=maximum(nansum(interpolated_what[:,:,1:z_upper]*bin_size,dim=(3))))
colorbar(c1, ax=ax1, extend="both")
ax1.set_title("\$\\int_{0}^{" * string(round(Int, bin_centers[z_upper])) * "} \\bar{w} \\, dHAB\$")
h1=ax1.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
PyPlot.clabel(h1, inline=true, fontsize=8)  # Add labels to contour lines
ax1.set_xlabel("x [km]")
ax1.set_ylabel("y [km]")

c2 = ax2.pcolor(xC[:]*1e-3, yC[:]*1e-3, nanmean(interpolated_uhat[:,:,1:z_upper]*bin_size,dim=(3))',
                cmap="RdYlBu_r",
                vmin=-maximum(nanmean(interpolated_uhat[:,:,1:z_upper]*bin_size,dim=(3))), vmax=maximum(nanmean(interpolated_uhat[:,:,1:z_upper]*bin_size,dim=(3))))
colorbar(c2, ax=ax2, extend="both")
ax2.set_title("\$\\int_{0}^{" * string(round(Int, bin_centers[z_upper])) * "} \\bar{u} \\, dHAB\$")
h2=ax2.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
PyPlot.clabel(h2, inline=true, fontsize=8)  # Add labels to contour lines
ax2.set_xlabel("x [km]")
ax2.set_ylabel("y [km]")

c3 = ax3.pcolor(xC[:]*1e-3, yC[:]*1e-3, (nanmean(interpolated_Bz[:,:,1:z_upper],dim=(3))'),
                cmap="GnBu",
                vmin=0, vmax=2e-6)
colorbar(c3, ax=ax3, extend="both")
ax3.set_title(L"\bar{B}_z" * string("[0-", round(Int,bin_centers[z_upper]), "m]"))
h3=ax3.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
PyPlot.clabel(h3, inline=true, fontsize=8)  # Add labels to contour lines
ax3.set_xlabel("x [km]")
ax3.set_ylabel("y [km]")


savefig(string("output/",simname,"/hab_u_w_Bz_xy_heatmap.png"))






###### animation: 
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
