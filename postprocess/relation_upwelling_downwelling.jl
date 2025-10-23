# this script checks the relation between upwelling and downwelling
# the argument is: Eulerian downwelling drives more bottom mixing, therefore causes stronger dipycnal upwelling. 

using NCDatasets
using Statistics
function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end

include("functions/mmderiv.jl") # for mmderiv function
include("functions/interpolation_z_dimension.jl")
include("functions/weighted_bin_z_dimension.jl") # for bin_z_dimension function


# bathymetry
Nx = 500
Ny = 1000
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


## load the mask
fmask = "TF_avg_tᶠ=462.0_analysis.nc"
ds_mask = Dataset(string("output/tilt/",fmask), "r")
mask_sill = ds_mask["mask_sill"][:,:]
mask_flanks = ds_mask["mask_flanks"][:,:]
mask_rest = ds_mask["mask_rest"][:,:]

# compute mean w

tᶠ = 462.0
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

# global total_steps
# tᶠ_0 = tᶠ - 1
tᶠ_first = tᶠ - 0.5
# tᶠ_second = tᶠ

# filename_3D_0 = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_0, "_analysis_round=all_threeD.nc")
filename_3D_first = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_first, "_analysis_round=all_threeD.nc")
# filename_3D_second = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_second, "_analysis_round=all_threeD.nc")
filename_verification = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", 10, "_threeD_timeavg.nc")
# ds_3D_0 = Dataset(filename_3D_0, "r")
ds_3D_first = Dataset(filename_3D_first, "r")
# ds_3D_second = Dataset(filename_3D_second, "r")
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

b = ds_verification["b"][:, :, :, 1:1]
#### load data of 4 TP average
ds = Dataset("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=462.0_10tidal_periods_avg.nc")
B = ds["B"][:,:,:,1] 
what = ds["what"][:,:,:,1] 
uhat = ds["uhat"][:,:,:,1] 
v = ds["v"][:,:,:,1] 
ε = ds["ε"][:,:,:,1] # ε at cell centers
∇κ∇B = ds["∇κ∇B"][:,:,:,1] # ∇κ∇B at cell centers
zC = ds["z_aac"][:]
vz = zeros(Nx, Ny, Nz) # vertical velocity at cell centers

uz = deriv(zC, uhat)
uz[uhat[:, :, 1:end-1, :].==0] .= 0 
uz_center = (cat(zeros(Nx,Ny,1), uz, dims=3) .+ cat(uz, zeros(Nx,Ny,1), dims=3))/2 # center in z direction
uz_center = (uz_center[1:end,:,:] .+ vcat(uz_center[2:end,:,:],uz_center[1:1,:,:]))./2 # uz at center in the x dimension

vz = deriv(zC, v) 
vz[v[:, :, 1:end-1, :].==0] .= 0 
vz_center = (cat(zeros(Nx,Ny,1), vz, dims=3) .+ cat(vz, zeros(Nx,Ny,1), dims=3))/2 # center in z direction
vz_center = (vz_center[1:end,:,:] .+ vcat(vz_center[2:end,:,:],vz_center[1:1,:,:]))./2 # vz at center in the x dimension

S² = uz_center.^2 .+ vz_center.^2 # S² at cell centers

what_cen = (what[:, :, 1:end-1, 1] .+ what[:, :, 2:end, 1]) ./ 2 # what at center
# w should also be eliminated at unstable mixing layers 
what_cen[what_cen.==0] .= NaN          # mask out zero values at topography
what_cen[abs.(what_cen).<1e-5] .= NaN  # mask out small values
# what_cen[isnan.(Bz_center)] .= NaN # eliminate points where w is negative because we don't want to consider unstable mixing layers, as the wave dispersion relation no longer holds


Bz = deriv(zC, B)
Bz[b[:, :, 1:end-1, :].==0] .= 0 
Bz_center = (cat(zeros(Nx,Ny,1), Bz, dims=3) .+ cat(Bz, zeros(Nx,Ny,1), dims=3))/2 # average Bz at cell centers
# Bz_center[Bz_center.==0] .= NaN # 
# Bz_center[Bz_center.<0] .= NaN # eliminate points where Bz is negative because we don't want to consider unstable mixing layers, as the wave dispersion relation no longer holds
# Bz_center = Bz_center[:,:,:,1] # drop dimension


# take the interpolation of what_cen to hab
bin_edge = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];


# interpolation method: could be problematic near the bottom, so we will use binning instead

# interpolated_what = zeros(Nx, Ny, length(bin_center), 1) # interpolated_what is the mean what at each bin center
# interpolated_ε = zeros(Nx, Ny, length(bin_center), 1) # interpolated_what is the mean what at each bin center
# interpolated_∇κ∇B = zeros(Nx, Ny, length(bin_center), 1) # interpolated_what is the mean what at each bin center
# interpolated_S² = zeros(Nx, Ny, length(bin_center), 1) # interpolated_what is the mean what at each bin center
# interpolated_Bz = zeros(Nx, Ny, length(bin_center), 1)
# interpolated_uz_center = zeros(Nx, Ny, length(bin_center), 1) # interpolated_uz_center is the mean uz at each bin center
# interpolated_vz_center = zeros(Nx, Ny, length(bin_center), 1) # interpolated_vz_center is the mean vz at each bin center

# @time interpolated_what[:,:,:],_ =  interpolate_z_dimension(what_cen, hab, bin_edge)
# @time interpolated_ε[:,:,:],_ =  interpolate_z_dimension(ε, hab, bin_edge)
# @time interpolated_∇κ∇B[:,:,:],_ =  interpolate_z_dimension(∇κ∇B, hab, bin_edge)
# @time interpolated_S²[:,:,:],_ =  interpolate_z_dimension(S², hab, bin_edge)
# @time interpolated_uz_center[:,:,:],_ =  interpolate_z_dimension(uz_center, hab, bin_edge)
# @time interpolated_vz_center[:,:,:],_ =  interpolate_z_dimension(vz_center, hab, bin_edge)
# @time interpolated_Bz[:,:,:],_ =  interpolate_z_dimension(Bz_center, hab, bin_edge)

what_cen_hab = zeros(Nx, Ny, length(bin_center), 1) # what_cen_hab is the mean what at each bin center
ε_hab = zeros(Nx, Ny, length(bin_center), 1) # ε_hab is the mean ε at each bin center
∇κ∇B_hab = zeros(Nx, Ny, length(bin_center), 1) # ∇κ∇B_hab is the mean ∇κ∇B at each bin center
S²_hab = zeros(Nx, Ny, length(bin_center), 1) # S²_hab is the mean S² at each bin center
uz_center_hab = zeros(Nx, Ny, length(bin_center), 1) # uz_center_hab is the mean uz at each bin center
vz_center_hab = zeros(Nx, Ny, length(bin_center), 1) # vz_center_hab is the mean vz at each bin center
Bz_hab = zeros(Nx, Ny, length(bin_center), 1) # Bz_hab is the mean Bz at each bin center

band_width = 4.0 # bin width in meters
@time what_cen_hab, _ = weighted_bin_z_dimension(what_cen, hab, bin_center, band_width)
@time ε_hab, _ = weighted_bin_z_dimension(ε, hab, bin_center, band_width)
@time ∇κ∇B_hab, _ = weighted_bin_z_dimension(∇κ∇B, hab, bin_center, band_width)
@time S²_hab, _ = weighted_bin_z_dimension(S², hab, bin_center, band_width)
@time uz_center_hab, _ = weighted_bin_z_dimension(uz_center, hab, bin_center, band_width)
@time vz_center_hab, _ = weighted_bin_z_dimension(vz_center, hab, bin_center, band_width)
@time Bz_hab, _ = weighted_bin_z_dimension(Bz_center, hab, bin_center, band_width)

# z_upper = argmin(abs.(bin_edge.- 180))
z_upper = argmin(abs.(bin_edge.- 40))
z_lower = argmin(abs.(bin_edge.- 40))
what_nearbottom_avg = nanmean(what_cen_hab[:,:,z_lower:z_upper], dim=3) # average over x and y
ε_nearbottom_avg = nanmean(ε_hab[:,:,z_lower:z_upper], dim=3) # average over x and y
∇κ∇B_nearbottom_avg = nanmean(∇κ∇B_hab[:,:,z_lower:z_upper], dim=3) # average over x and y
S²_nearbottom_avg = nanmean(S²_hab[:,:,z_lower:z_upper], dim=3) # average over x and y
Bz_nearbottom_avg = nanmean(Bz_hab[:,:,z_lower:z_upper], dim=3) # average over x and y
uz_center_nearbottom_avg = nanmean(uz_center_hab[:,:,z_lower:z_upper], dim=3) # average over x and y
vz_center_nearbottom_avg = nanmean(vz_center_hab[:,:,z_lower:z_upper], dim=3) # average over x and y



# plot pcolors of these variables
using PyPlot
# Create a 3x2 subplot layout (with the last spot empty)
fig, ax = subplots(2, 3, figsize=(14, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Create contour levels for bathymetry
contour_levels = range(minimum(z_interp), maximum(z_interp), length=10)

# 1. Plot vertical velocity (what)
ax1 = ax[1, 1]
im1 = ax1.pcolormesh(xC, yC, what_nearbottom_avg[:, :, 1]', cmap="RdBu_r", 
                    norm=matplotlib.colors.Normalize(vmin=-0.01, vmax=0.01))
c1 = ax1.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax1.clabel(c1, inline=true, fontsize=8, fmt="%.0f")
cbar1 = colorbar(im1, ax=ax1)
ax1.set_title("Near Bottom Average Vertical Velocity", fontsize=12)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
cbar1.set_label(L"$\overline{w}$ (m s$^{-1}$)")
ax1.set_aspect("equal")  # Set equal aspect ratio

# 2. Plot dissipation rate (ε)
ax2 = ax[1, 2]
im2 = ax2.pcolormesh(xC, yC, ε_nearbottom_avg[:, :, 1]', 
                    norm=matplotlib.colors.LogNorm(vmin=1e-8, vmax=1e-6), 
                    cmap="Oranges")
c2 = ax2.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax2.clabel(c2, inline=true, fontsize=8, fmt="%.0f")
cbar2 = colorbar(im2, ax=ax2)
ax2.set_title("Near Bottom Average Dissipation Rate", fontsize=12)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
cbar2.set_label(L"$\overline{\varepsilon}$ (m$^2$ s$^{-3}$)")
ax2.set_aspect("equal")  # Set equal aspect ratio

# 3. Plot diapycnal mixing (∇κ∇B)
ax3 = ax[1, 3]
# Get the mixing data
mix_data = ∇κ∇B_nearbottom_avg[:, :, 1]'
vmin = nanminimum(mix_data)
vmax = 3e-9

# Create a colormap that's blue->white for negative, white->red for positive
neg_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("neg", ["blue", "white"])
pos_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pos", ["white", "red"])

# Find the position of zero in the range from vmin to vmax
if vmin < 0 && vmax > 0
    zero_pos = -vmin / (vmax - vmin)
    neg_colors = neg_cmap(LinRange(0, 1, Int(round(256*zero_pos))))
    pos_colors = pos_cmap(LinRange(0, 1, 256-Int(round(256*zero_pos))))
    colors = vcat(neg_colors, pos_colors)
    custom_cmap = matplotlib.colors.ListedColormap(colors)
    im3 = ax3.pcolormesh(xC, yC, mix_data, cmap=custom_cmap, vmin=vmin, vmax=vmax)
else
    # Fallback if we don't have both positive and negative values
    im3 = ax3.pcolormesh(xC, yC, mix_data, cmap="RdBu_r")
end

c3 = ax3.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax3.clabel(c3, inline=true, fontsize=8, fmt="%.0f")
cbar3 = colorbar(im3, ax=ax3)
ax3.set_title("Near Bottom Average Diapycnal Mixing", fontsize=12)
ax3.set_xlabel("x (m)")
ax3.set_ylabel("y (m)")
cbar3.set_label(L"$\overline{\nabla \cdot \kappa \nabla B}$ (m s$^{-3}$)")
ax3.set_aspect("equal")  # Set equal aspect ratio

# 4. Plot shear squared (S²)
ax4 = ax[2, 1]
im4 = ax4.pcolormesh(xC, yC, S²_nearbottom_avg[:, :, 1]', 
                    norm=matplotlib.colors.LogNorm(vmin=1e-8, vmax=1e-6), 
                    cmap="plasma")
c4 = ax4.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax4.clabel(c4, inline=true, fontsize=8, fmt="%.0f")
cbar4 = colorbar(im4, ax=ax4)
ax4.set_title("Near Bottom Average Shear Squared", fontsize=12)
ax4.set_xlabel("x (m)")
ax4.set_ylabel("y (m)")
cbar4.set_label(L"$\overline{S^2}$ (s$^{-2}$)")
ax4.set_aspect("equal")  # Set equal aspect ratio

# 5. Plot stratification (Bz) with custom colormap
ax5 = ax[2, 2]
# Create custom colormap for Bz with fixed limits
Bz_data = Bz_nearbottom_avg[:, :, 1]'
# Set explicit limits with upper limit at 1.5e-5
vmin = nanminimum(Bz_data)
vmax = 3e-6  # Fixed upper limit for better visualization of positive values

# Create a colormap that's blue->white for negative, white->red for positive
neg_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("neg", ["blue", "white"])
pos_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pos", ["white", "red"])

# Find the position of zero in the range from vmin to vmax
if vmin < 0 && vmax > 0
    zero_pos = -vmin / (vmax - vmin)
    neg_colors = neg_cmap(LinRange(0, 1, Int(round(256*zero_pos))))
    pos_colors = pos_cmap(LinRange(0, 1, 256-Int(round(256*zero_pos))))
    colors = vcat(neg_colors, pos_colors)
    custom_cmap = matplotlib.colors.ListedColormap(colors)
    im5 = ax5.pcolormesh(xC, yC, Bz_data, cmap=custom_cmap, vmin=vmin, vmax=vmax)
else
    # Fallback if we don't have both positive and negative values
    im5 = ax5.pcolormesh(xC, yC, Bz_data, cmap="RdBu_r", vmax=vmax)
end
c5 = ax5.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax5.clabel(c5, inline=true, fontsize=8, fmt="%.0f")
cbar5 = colorbar(im5, ax=ax5)
ax5.set_title("Near Bottom Average Stratification", fontsize=12)
ax5.set_xlabel("x (m)")
ax5.set_ylabel("y (m)")
cbar5.set_label(L"$\overline{B_z}$ (s$^{-2}$)")
ax5.set_aspect("equal")  # Set equal aspect ratio

# Plot just the bathymetry in the empty subplot
ax6 = ax[2, 3]
bathy = ax6.contourf(x_interp, y_interp, z_interp', levels=contour_levels, cmap="terrain")
c6 = ax6.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.8)
ax6.clabel(c6, inline=true, fontsize=9, fmt="%.0f")
cbar6 = colorbar(bathy, ax=ax6)
ax6.set_title("Bathymetry", fontsize=12)
ax6.set_xlabel("x (m)")
ax6.set_ylabel("y (m)")
cbar6.set_label("Depth (m)")
ax6.set_aspect("equal")  # Set equal aspect ratio

# Add a main title
plt.suptitle("Near Bottom Averaged Properties (0-40m HAB)", fontsize=16, y=0.98)

# Adjust the layout to accommodate the equal aspect ratio
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
plt.savefig("output/tilt/near_bottom_averages_bins_452-462avg.png", dpi=150, bbox_inches="tight")



###### plot pcolor of what, ∇κ∇B, ε, Bz, uz_center, vz_center

# Create a 2x3 subplot layout for the six variables
using PyPlot
close("all")
fig, ax = subplots(2, 3, figsize=(14, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Create contour levels for bathymetry
contour_levels = range(minimum(z_interp), maximum(z_interp), length=10)

# 1. Plot vertical velocity (what)
ax1 = ax[1, 1]
im1 = ax1.pcolor(xC, yC, what_nearbottom_avg[:, :, 1]', cmap="RdBu_r", 
                    norm=matplotlib.colors.Normalize(vmin=-0.01, vmax=0.01))
c1 = ax1.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax1.clabel(c1, inline=true, fontsize=8, fmt="%.0f")
cbar1 = colorbar(im1, ax=ax1)
ax1.set_title("Near Bottom Average Vertical Velocity", fontsize=12)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
cbar1.set_label(L"$\overline{w}$ (m s$^{-1}$)")
ax1.set_aspect("equal")

# 2. Plot diapycnal mixing (∇κ∇B)
ax2 = ax[1, 2]
mix_data = ∇κ∇B_nearbottom_avg[:, :, 1]'
vmax = 8e-10
vmin = -vmax

# Use a SymLogNorm for symmetric logarithmic scaling
norm = matplotlib.colors.SymLogNorm(linthresh=1e-10, linscale=1, vmin=vmin, vmax=vmax, base=10)

# Create a colormap that's blue->white for negative, white->red for positive
im2 = ax2.pcolor(xC, yC, mix_data, cmap="RdBu_r", norm=norm)

c2 = ax2.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax2.clabel(c2, inline=true, fontsize=8, fmt="%.0f")
cbar2 = colorbar(im2, ax=ax2)
ax2.set_title("Near Bottom Average Diapycnal Mixing", fontsize=12)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
cbar2.set_label(L"$\overline{\nabla \cdot \kappa \nabla B}$ (m s$^{-3}$)")
ax2.set_aspect("equal")

# 3. Plot dissipation rate (ε)
ax3 = ax[1, 3]
im3 = ax3.pcolor(xC, yC, ε_nearbottom_avg[:, :, 1]', 
                    norm=matplotlib.colors.LogNorm(vmin=1e-8, vmax=1e-6), 
                    cmap="Oranges")
c3 = ax3.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax3.clabel(c3, inline=true, fontsize=8, fmt="%.0f")
cbar3 = colorbar(im3, ax=ax3)
ax3.set_title("Near Bottom Average Dissipation Rate", fontsize=12)
ax3.set_xlabel("x (m)")
ax3.set_ylabel("y (m)")
cbar3.set_label(L"$\overline{\varepsilon}$ (m$^2$ s$^{-3}$)")
ax3.set_aspect("equal")

# 4. Plot stratification (Bz)
ax4 = ax[2, 1]
Bz_data = Bz_nearbottom_avg[:, :, 1]'
vmin = nanminimum(Bz_data)
vmax = 3e-6

# Create a colormap that's blue->white for negative, white->red for positive
neg_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("neg", ["blue", "white"])
pos_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pos", ["white", "red"])

# Find the position of zero in the range from vmin to vmax
if vmin < 0 && vmax > 0 && abs(vmin) > 1e-8
    zero_pos = -vmin / (vmax - vmin)
    neg_colors = neg_cmap(LinRange(0, 1, Int(round(256*zero_pos))))
    pos_colors = pos_cmap(LinRange(0, 1, 256-Int(round(256*zero_pos))))
    colors = vcat(neg_colors, pos_colors)
    custom_cmap = matplotlib.colors.ListedColormap(colors)
    im4 = ax4.pcolor(xC, yC, Bz_data, cmap=custom_cmap, vmin=vmin, vmax=vmax)
else
    im4 = ax4.pcolor(xC, yC, Bz_data, cmap="RdBu_r", vmax=vmax)
end

c4 = ax4.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax4.clabel(c4, inline=true, fontsize=8, fmt="%.0f")
cbar4 = colorbar(im4, ax=ax4)
ax4.set_title("Near Bottom Average Stratification", fontsize=12)
ax4.set_xlabel("x (m)")
ax4.set_ylabel("y (m)")
cbar4.set_label(L"$\overline{B_z}$ (s$^{-2}$)")
ax4.set_aspect("equal")

# 5. Plot vertical shear of u (uz_center)
ax5 = ax[2, 2]
im5 = ax5.pcolor(xC, yC, uz_center_nearbottom_avg[:, :, 1]', cmap="RdBu_r",
                    norm=matplotlib.colors.Normalize(vmin=-0.002, vmax=0.002))
c5 = ax5.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax5.clabel(c5, inline=true, fontsize=8, fmt="%.0f")
cbar5 = colorbar(im5, ax=ax5)
ax5.set_title("Near Bottom Average Vertical Shear of u", fontsize=12)
ax5.set_xlabel("x (m)")
ax5.set_ylabel("y (m)")
cbar5.set_label(L"$\partial \overline{u} / \partial z$ (s$^{-1}$)")
ax5.set_aspect("equal")

# 6. Plot vertical shear of v (vz_center)
ax6 = ax[2, 3]
im6 = ax6.pcolor(xC, yC, vz_center_nearbottom_avg[:, :, 1]', cmap="RdBu_r",
                    norm=matplotlib.colors.Normalize(vmin=-0.002, vmax=0.002))
c6 = ax6.contour(x_interp, y_interp, z_interp', levels=contour_levels, colors="k", linewidths=0.5, alpha=0.7)
ax6.clabel(c6, inline=true, fontsize=8, fmt="%.0f")
cbar6 = colorbar(im6, ax=ax6)
ax6.set_title("Near Bottom Average Vertical Shear of v", fontsize=12)
ax6.set_xlabel("x (m)")
ax6.set_ylabel("y (m)")
cbar6.set_label(L"$\partial \overline{v} / \partial z$ (s$^{-1}$)")
ax6.set_aspect("equal")

# Add a main title
plt.suptitle("Near Bottom Averaged Properties (0-40m HAB)", fontsize=16, y=0.98)

# Adjust the layout to accommodate the equal aspect ratio
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
plt.savefig("output/tilt/near_bottom_averages_six_variables_bins_452-462_0-40m.png", dpi=150, bbox_inches="tight")



# Create figure
using PyPlot
using Printf
# Set global font parameters
close("all") # close all previous figures

# Create figure with specified size
fig, ax = plt.subplots(figsize=(10, 6))
what_nearbottom_flat = what_cen_hab[:,:,1:4][:]
ε_nearbottom_flat = ε_hab[:,:,1:4][:]
∇κ∇B_nearbottom_flat = ∇κ∇B_hab[:,:,1:4][:]
# Flatten arrays to 1D for scatter plot
valid_indices = .!isnan.(what_nearbottom_flat) .& .!isnan.(∇κ∇B_nearbottom_flat) .& .!isnan.(ε_nearbottom_flat)
x = what_nearbottom_flat[valid_indices] 
y = ∇κ∇B_nearbottom_flat[valid_indices]
colors = ε_nearbottom_flat[valid_indices]

# Create scatter plot with more impressive colormap
scatter_plot = ax.scatter(x, y, c=colors, alpha=0.5, cmap="plasma", 
                         s=10, edgecolors="k", linewidths=0.1,
                         norm=matplotlib.colors.LogNorm())

axhline(y=0, color="black", linestyle="-", linewidth=1.5)
# Add zero line for vertical velocity
axvline(x=0, color="black", linestyle="-", linewidth=1.5)

# Add regression line with more striking color
p = [ones(length(x)) x] \ y
x_range = range(minimum(x), maximum(x), length=100)
pl=ax.plot(x_range, p[1] .+ p[2] .* x_range, color="#00CED1", linewidth=2.5, 
          label="Regression Line", linestyle="-", zorder=3)

# Format coefficients properly for display
slope_str = @sprintf("%.2e", p[2])
intercept_str = @sprintf("%.2e", p[1])

# Add equation to plot with specific fontsize
ax.legend(pl,["y = $slope_str x + $intercept_str"], loc="lower left", fontsize=15)

# Add labels and title with specific fontsizes
ax.set_xlabel(L"\mathrm{Near~bottom~average}~\overline{ŵ} ~\mathrm{[m~ s⁻¹]}", fontsize=18)
ax.set_ylabel(L"\mathrm{Near~bottom~average}~-∇⋅\overline{\mathcal{B}} ~\mathrm{[10⁻⁹~×~m~s⁻³]}", fontsize=18)
ax.set_yticks((-0.5:0.5:2)*1e-9) 
ax.set_yticklabels(("-0.5","0","0.5","1","1.5","2")) # Set y-tick labels with specific fontsize
# Set tick label sizes
ax.tick_params(axis="both", labelsize=16)

# Add colorbar with proper label and improved style
cbar = plt.colorbar(scatter_plot)
cbar.set_label(L"\mathrm{Near~ bottom ~average}~ε̄ ~\mathrm{[m²~s⁻³]}", fontsize=18)
cbar.ax.tick_params(labelsize=16)

# Set colorbar limits for better visibility
cbar.mappable.set_clim(1e-8, 1e-5)

# Improve overall appearance
ax.grid(true, linestyle="--", alpha=0.3)
fig.set_facecolor("white")

plt.tight_layout()
plt.savefig("output/tilt/downwelling_vs_upwelling_relationship.png", dpi=100)


## same relationship but with joint pdf
using StatsBase
using PyPlot

close("all") # close all previous figures

# Flatten arrays for joint PDF
what_flat = vec(what_nearbottom_avg[:,:,1])
mix_flat = vec(∇κ∇B_nearbottom_avg[:,:,1])

# Remove NaNs for valid joint PDF
valid = .!isnan.(what_flat) .& .!isnan.(mix_flat)
what_flat = what_flat[valid]
mix_flat = mix_flat[valid]

# Define bin edges
n_bins_x = 100
n_bins_y = 100
what_edges = range(minimum(what_flat), maximum(what_flat), length=n_bins_x+1)
mix_edges = range(minimum(mix_flat), maximum(mix_flat), length=n_bins_y+1)

# Compute 2D histogram
h_2d = StatsBase.fit(Histogram, (what_flat, mix_flat), (what_edges, mix_edges))
pdf_2d = h_2d.weights ./ sum(h_2d.weights)

# Bin centers for plotting
what_centers = (what_edges[1:end-1] .+ what_edges[2:end]) ./ 2
mix_centers = (mix_edges[1:end-1] .+ mix_edges[2:end]) ./ 2

# Plot joint PDF as a 2D density with logarithmic color scale
fig, ax = plt.subplots(figsize=(10, 7))
pcm = ax.pcolormesh(what_centers, mix_centers, pdf_2d', cmap="plasma", 
                    norm=matplotlib.colors.LogNorm(vmin=1e-8, vmax=maximum(pdf_2d)), shading="auto")
cb = plt.colorbar(pcm, ax=ax)
cb.set_label(L"\mathrm{Joint~PDF}", fontsize=18)
cb.ax.tick_params(labelsize=16)

ax.set_xlabel(L"\mathrm{Near~bottom~average}~\overline{ŵ} ~\mathrm{[m~ s^{-1}]}", fontsize=18)
ax.set_ylabel(L"\mathrm{Near~bottom~average}~-∇⋅\overline{\mathcal{B}} ~\mathrm{[m~s^{-3}]}", fontsize=18)
ax.tick_params(axis="both", labelsize=16)
ax.grid(true, linestyle="--", alpha=0.3)
ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
ax.axvline(0, color="black", linestyle="-", linewidth=1.5)
ax.set_title("Joint PDF of Near Bottom Vertical Velocity and Diapycnal Mixing", fontsize=16)

plt.tight_layout()
plt.savefig("output/tilt/joint_pdf_vertical_velocity_diapycnal_mixing_log.png", dpi=100)



## plot the relationship between turbulence diffusive flux and dissipation
using StatsBase
using PyPlot

close("all") # close all previous figures

# Flatten arrays for joint PDF
ε_flat = vec(ε_nearbottom_avg[:,:,1])
mix_flat = vec(∇κ∇B_nearbottom_avg[:,:,1])

# Remove NaNs for valid joint PDF
valid = .!isnan.(ε_flat) .& .!isnan.(mix_flat)
ε_flat = ε_flat[valid]
mix_flat = mix_flat[valid]

# Define bin edges
n_bins_x = 100
n_bins_y = 100
ε_edges = range(minimum(ε_flat), maximum(ε_flat), length=n_bins_x+1)
mix_edges = range(minimum(mix_flat), maximum(mix_flat), length=n_bins_y+1)

# Compute 2D histogram
h_2d = StatsBase.fit(Histogram, (ε_flat, mix_flat), (ε_edges, mix_edges))
pdf_2d = h_2d.weights ./ sum(h_2d.weights)

# Bin centers for plotting
ε_centers = (ε_edges[1:end-1] .+ ε_edges[2:end]) ./ 2
mix_centers = (mix_edges[1:end-1] .+ mix_edges[2:end]) ./ 2

# Plot joint PDF as a 2D density with logarithmic color scale
fig, ax = plt.subplots(figsize=(10, 7))
pcm = ax.pcolormesh(ε_centers, mix_centers, pdf_2d', cmap="plasma", 
                    norm=matplotlib.colors.LogNorm(vmin=1e-8, vmax=maximum(pdf_2d)), shading="auto")
cb = plt.colorbar(pcm, ax=ax)
cb.set_label(L"\mathrm{Joint~PDF}", fontsize=18)
cb.ax.tick_params(labelsize=16)

ax.set_xlabel(L"\mathrm{Near~bottom~average}~\overline{ε} ~\mathrm{[m~ s^{-1}]}", fontsize=18)
ax.set_ylabel(L"\mathrm{Near~bottom~average}~-∇⋅\overline{\mathcal{B}} ~\mathrm{[m~s^{-3}]}", fontsize=18)
ax.tick_params(axis="both", labelsize=16)
ax.grid(true, linestyle="--", alpha=0.3)
ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
ax.axvline(0, color="black", linestyle="-", linewidth=1.5)
ax.set_title("Joint PDF of Near Bottom Dissipation and Diapycnal Mixing", fontsize=16)

plt.tight_layout()
plt.savefig("output/tilt/joint_pdf_dissipation_diapycnal_mixing_log.png", dpi=100)



## relationship between vertical velocity and stratification



# Flatten arrays for joint PDF
w_flat = vec(what_cen_hab[:,:,1:4].*mask_flanks[:,:])
Bz_flat = vec(Bz_hab[:,:,1:4].*mask_flanks[:,:])
# w_flat = vec(what_nearbottom_avg[:,:,1])
# Bz_flat = vec(Bz_nearbottom_avg[:,:,1])

# Remove NaNs for valid joint PDF
valid = .!isnan.(w_flat) .& .!isnan.(Bz_flat)
w_flat = w_flat[valid]
Bz_flat = Bz_flat[valid]

# Define bin edges
n_bins_x = 200
n_bins_y = 200
w_edges = range(minimum(w_flat), maximum(w_flat), length=n_bins_x+1)
Bz_edges = range(minimum(Bz_flat), maximum(Bz_flat), length=n_bins_y+1)
# mix_edges = range(minimum(mix_flat), maximum(mix_flat), length=n_bins_y+1)

# Compute 2D histogram
h_2d = StatsBase.fit(Histogram, (w_flat, Bz_flat), (w_edges, Bz_edges))
pdf_2d = h_2d.weights ./ sum(h_2d.weights)

# Bin centers for plotting
w_centers = (w_edges[1:end-1] .+ w_edges[2:end]) ./ 2
Bz_centers = (Bz_edges[1:end-1] .+ Bz_edges[2:end]) ./ 2

# Plot joint PDF as a 2D density with logarithmic color scale
using PyPlot
using StatsBase
close("all") # close all previous figures
vmin = 1e-5  # or 1e-8, depending on your data
vmax = nanmaximum(pdf_2d)
N_levels = 300  # or more for even smoother contours

levels = 10 .^ range(log10(vmin), log10(vmax), length=N_levels)
pdf_2d[pdf_2d .< 1e-5] .= NaN  # Mask values below vmin for better visualization
fig, ax = plt.subplots(figsize=(12, 5))
pcm = ax.contourf(
    w_centers, Bz_centers, pdf_2d',
    levels=levels,
    cmap="plasma",
    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    alpha=0.9
)
cb = plt.colorbar(pcm, ax=ax)
cb.set_label(L"\mathrm{Joint~PDF}", fontsize=18)
cb.ax.tick_params(labelsize=16)
ax.set_ylim(-1e-6, 0.4e-5)
ax.set_xlabel(L"\mathrm{Near~bottom~average}~\overline{ŵ} ~\mathrm{[m~ s^{-1}]}", fontsize=18)
ax.set_ylabel(L"\mathrm{Near~bottom~average}~\overline{B_z} ~\mathrm{[s^{-2}]}", fontsize=18)
ax.tick_params(axis="both", labelsize=16)
ax.grid(true, linestyle="--", alpha=0.3)
ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
ax.axvline(0, color="black", linestyle="-", linewidth=1.5)
ax.set_title("Joint PDF of Near Bottom ŵ and Stratification", fontsize=16)
ax.set_xlim(-0.01, 0.01)
ax.set_ylim(-.5e-6, 0.4e-5)  
plt.tight_layout()
plt.savefig("output/tilt/joint_pdf_vertical_velocity_stratification_log.png", dpi=100)


##### Plot the relationship between vertical velocity and stratification 
using PyPlot
using Printf
# Set global font parameters
close("all") # close all previous figures

# Create figure with specified size
fig, ax = plt.subplots(figsize=(10, 6))

# Flatten arrays to 1D for scatter plot
valid_indices = .!isnan.(what_nearbottom_avg) .& .!isnan.(Bz_nearbottom_avg) .& .!isnan.(ε_nearbottom_avg)
x = what_nearbottom_avg[valid_indices] 
y = Bz_nearbottom_avg[valid_indices]
colors = ε_nearbottom_avg[valid_indices]

# Create scatter plot with more impressive colormap
scatter_plot = ax.scatter(x, y, c=colors, alpha=0.7, cmap="plasma", 
                         s=50, edgecolors="k", linewidths=0.3,
                         norm=matplotlib.colors.LogNorm())

axhline(y=0, color="black", linestyle="-", linewidth=1.5)
# Add zero line for vertical velocity
axvline(x=0, color="black", linestyle="-", linewidth=1.5)

# Add regression line with more striking color
p = [ones(length(x)) x] \ y
x_range = range(minimum(x), maximum(x), length=100)
pl=ax.plot(x_range, p[1] .+ p[2] .* x_range, color="#00CED1", linewidth=2.5, 
          label="Regression Line", linestyle="-", zorder=3)

# Format coefficients properly for display
slope_str = @sprintf("%.2e", p[2])
intercept_str = @sprintf("%.2e", p[1])

# Add equation to plot with specific fontsize
ax.legend(pl,["y = $slope_str x + $intercept_str"], loc="lower left", fontsize=15)

# Add labels and title with specific fontsizes
ax.set_xlabel(L"\mathrm{Near~bottom~average}~\overline{ŵ} ~\mathrm{[m~ s⁻¹]}", fontsize=18)
ax.set_ylabel(L"\mathrm{Near~bottom~stratification}~\overline{B_z} ~\mathrm{[s⁻²]}", fontsize=18)
ax.set_ylim(-1e-6, 0.5e-5) # Set y-limits for better visibility
# Set tick label sizes
ax.tick_params(axis="both", labelsize=16)

# Add colorbar with proper label and improved style
cbar = plt.colorbar(scatter_plot)
cbar.set_label(L"\mathrm{Near~ bottom ~average}~ε̄ ~\mathrm{[m²~s⁻³]}", fontsize=18)
cbar.ax.tick_params(labelsize=16)

# Set colorbar limits for better visibility
cbar.mappable.set_clim(1e-8, 1e-5)

# Improve overall appearance
ax.grid(true, linestyle="--", alpha=0.3)
fig.set_facecolor("white")

plt.tight_layout()
savefig("output/tilt/vertical_velocity_vs_stratification.png", dpi=100)


# this plot tries to understand why there is stripes of upwelling near the bottom
using NCDatasets
# Load bathymetry data
file_w = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=456.0_analysis_round=all_threeD.nc"
ds_w = Dataset(file_w, "r")
zC = ds_w["z_aac"][:]
xC = ds_w["x_caa"][:]
yC = ds_w["y_aca"][:]
w = ds_w["what"][:, :, 30, 1] # vertical velocity at cell centers
t = ds_w["time"][1]
# Create a pcolor plot of the w slice
using PyPlot
close("all")

fig, ax = plt.subplots(figsize=(10, 8))

# Create pcolor plot
im = ax.pcolor(xC, yC, w', cmap="RdBu_r", 
               norm=matplotlib.colors.Normalize(vmin=-0.01, vmax=0.01))

# Add contour lines for bathymetry
contour_levels = range(minimum(z_interp), maximum(z_interp), length=100)
c = ax.contour(x_interp, y_interp, z_interp', levels=contour_levels, 
               colors="k", linewidths=0.5, alpha=0.7)
ax.clabel(c, inline=true, fontsize=8, fmt="%.0f")

# Add colorbar and labels
cbar = colorbar(im, ax=ax)
ax.set_title("Vertical Velocity at z = #{zC[20]:.1f}m", fontsize=14)
ax.set_xlabel("x (m)", fontsize=12)
ax.set_ylabel("y (m)", fontsize=12)
cbar.set_label(L"w \; \mathrm{(m \; s^{-1})}", fontsize=12)

# Set aspect ratio to equal for proper visualization
ax.set_aspect("equal")

# Add grid lines
ax.grid(true, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("output/tilt/vertical_velocity_slice_z$(Int(round(zC[30]))).png", dpi=150)