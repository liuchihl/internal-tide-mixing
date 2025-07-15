# this script plots the joint PDF of mean w and topography steepness (criticality)
using NCDatasets
using Statistics
function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end

using ImageFiltering

"""
    smooth_bathymetry(z_interp, Le, Δx)

Smooth bathymetry with NaN-handling using Gaussian filter.

Arguments:
- `z_interp`: 2D bathymetry array, may contain NaNs
- `Le`: tidal excursion length (meters)
- `Δx`: grid spacing (meters)

Returns:
- Smoothed bathymetry array (NaNs preserved where originally present)
"""
function smooth_bathymetry(z_interp::AbstractMatrix, Le::Real, Δx::Real)
    σ = Le / (2.355 * Δx)             # Convert Le to σ in grid units
    kernel = Kernel.gaussian((σ, σ))  # Isotropic 2D Gaussian

    # Validity mask: 1 for real values, 0 for NaNs
    valid = .!isnan.(z_interp)
    mask = Float64.(valid)

    # Replace NaNs with 0s to not contribute to sum
    z_filled = ifelse.(valid, z_interp, 0.0)

    # Filter both data and mask with periodic boundary
    filtered_z = imfilter(z_filled, kernel, "circular")
    filtered_mask = imfilter(mask, kernel, "circular")

    # Normalize: filtered_z / filtered_mask
    z_smoothed = filtered_z ./ filtered_mask

    # Reassign NaNs to their original locations
    z_smoothed[.!valid] .= NaN

    return z_smoothed
end
include("functions/mmderiv.jl") # for mmderiv function
include("functions/interpolation_z_dimension.jl")


# compute mean w

tᶠ = 456.0
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

# # Combined running sums
# # first define the variables using the final timestep of tf-0.5 chunk 
# # (for example, the first timestep of 452.5 chunk is 452.0833, but we want 452.0 and it is at the final timestep  in tf-0.5 chunk)
# what_sum = ds_3D_0["what"][:, :, :, end:end]
# B_sum = ds_3D_0["B"][:, :, :, end:end]
b = ds_verification["b"][:, :, :, 1:1]

# # Count total timesteps processed
# total_steps = 1   # already have one timestep from the first dataset

# # Process first dataset
# Nt_first = length(ds_3D_first["time"][:])
# for n in 1:Nt_first
#     # Read data for this timestep
#     what_sum .+= ds_3D_first["what"][:, :, :, n:n]
#     B_sum .+= ds_3D_first["B"][:, :, :, n:n]
#     # Free memory by explicitly clearing variables if needed
#     GC.gc()
#     global total_steps += 1
#     println(total_steps)
# end

# # Process second dataset
# Nt_second = length(ds_3D_second["time"][:])
# for n in 1:Nt_second
#     # Read data for this timestep
#     what_sum .+= ds_3D_second["what"][:, :, :, n:n]
#     B_sum .+= ds_3D_second["B"][:, :, :, n:n]
#     global total_steps += 1

#     # Free memory by explicitly clearing variables if needed
#     GC.gc()
# end
# # Calculate averages
# B = B_sum ./ total_steps
# what = what_sum ./ total_steps

#### load data of 4 TP average
ds = Dataset("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=456.0_4tidal_periods_avg.nc")
B = ds["B"][:,:,:,1] # B at cell centers
what = ds["what"][:,:,:,1] # what at cell centers
zC = ds["z_aac"][:]



Bz = deriv(zC, B)
Bz[b[:, :, 1:end-1, :].==0] .= 0 
Bz_center = (cat(zeros(Nx,Ny,1), Bz, dims=3) .+ cat(Bz, zeros(Nx,Ny,1), dims=3))/2 # average Bz at cell centers
Bz_center[Bz_center.==0] .= NaN # 
Bz_center[Bz_center.<0] .= NaN # eliminate points where Bz is negative because we don't want to consider unstable mixing layers, as the wave dispersion relation no longer holds
Bz_center = Bz_center[:,:,:,1] # drop dimension

what_cen = (what[:, :, 1:end-1, 1] .+ what[:, :, 2:end, 1]) ./ 2 # what at center
# w should also be eliminated at unstable mixing layers 
what_cen[what_cen.==0] .= NaN          # mask out zero values at topography
what_cen[abs.(what_cen).<1e-5] .= NaN  # mask out small values
what_cen[isnan.(Bz_center)] .= NaN # eliminate points where w is negative because we don't want to consider unstable mixing layers, as the wave dispersion relation no longer holds

# finally mask Bz_center by what_cen
Bz_center[isnan.(what_cen)] .= NaN # mask Bz_center
# Clear the intermediate sums to free memory
what_sum = nothing
B_sum = nothing
GC.gc()
# take the interpolation of what_cen to hab
bin_edge = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
interpolated_what = zeros(Nx, Ny, length(bin_center), 1) # interpolated_what is the mean what at each bin center
interpolated_Bz = zeros(Nx, Ny, length(bin_center), 1) # interpolated_what is the mean what at each bin center
@time interpolated_what[:,:,:],_ =  interpolate_z_dimension(what_cen, hab, bin_edge)
@time interpolated_Bz[:,:,:],_ =  interpolate_z_dimension(Bz_center, hab, bin_edge)
# bottom to 50 hab=50m
z_upper = argmin(abs.(bin_edge.- 60))
what_nearbottom_avg = nanmean(interpolated_what[:,:,2:z_upper], dim=3) # average over x and y
Bz_nearbottom_avg = nanmean(interpolated_Bz[:,:,2:z_upper], dim=3) # average over x and y


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

# smooth the topography based on the tidal excursion length scale with the Gaussian filter 
U₀ = 0.025  # tidal amplitude
ω₀ = 1.4e-4  # tidal frequency
ℓₑ = U₀ * 2π / ω₀ # tidal excursion length scale
# smooth the topography with a Gaussian filter
Δx = 30.0    # horizontal grid spacing in meters
z_smoothed = smooth_bathymetry(z_interp, ℓₑ, Δx)


# calculate the criticality 
# parameters
f = -5.3e-5    # Coriolis frequency
# N² = 1e-6      # buoyancy frequency
N² = Bz_nearbottom_avg
# Calculate slope angle with NaN handling for invalid points
tanα = zeros(size(N²))
valid_points = (N² .> ω₀^2)  # Points where N² > ω₀²
tanα[valid_points] .= sqrt.((ω₀^2 - f^2) ./ (N²[valid_points] .- ω₀^2))
tanα[.!valid_points] .= NaN  # Set invalid points to NaN
hx = mmderiv(xC, z_smoothed[:,:])
hy = mmderiv(yC, z_smoothed[:,:]')'
tanβ = sqrt.(hx.^2 .+ hy.^2)
# criticality (steepness parameter)
γ = tanβ ./ tanα
# since locations with N² < ω₀² is set to NaN vertical velocity should also be masked
what_nearbottom_avg[.!valid_points] .= NaN

# plot the criticality and the vertical velocity near bottom
using PyPlot
using Statistics

# Create a figure with two subplots
close("all")
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# 1. Plot the criticality map
ax1 = ax[1]
pcm1 = ax1.pcolor(xC, yC, γ', cmap="RdBu_r", shading="auto")
cbar1 = fig.colorbar(pcm1, ax=ax1, label="Criticality parameter (γ)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_title("Topographic Criticality Parameter")
pcm1.set_clim(0,2)
ax1.grid(true, linestyle="-", alpha=1, color="black")  # Add grid to first plot

# 2. Plot the vertical velocity near bottom
ax2 = ax[2]
# Use symlog norm for better visualization of both small and large values
import PyPlot: matplotlib
# Create symmetric log norm with linear region between -linthresh and linthresh
linthresh = 0.001  # Threshold below which scale becomes linear
# what_smooth = smooth_bathymetry(what_nearbottom_avg, ℓₑ, Δx) # use the smoothed vertical velocity 

pcm2 = ax2.pcolor(xC, yC, what_nearbottom_avg[:,:,1]', 
                 cmap="RdBu_r", 
                 shading="auto")
pcm2.set_clim(-0.005, 0.005)  # Set color limits for better visibility                
# pcm2 = ax2.pcolor(xC, yC, what_nearbottom_avg[:,:,1]', 
#                  norm=matplotlib.colors.SymLogNorm(linthresh=linthresh, 
#                                                   linscale=1.0,
#                                                   vmin=-0.02, 
#                                                   vmax=0.02),
#                  cmap="RdBu_r", 
#                  shading="auto")
cbar2 = fig.colorbar(pcm2, ax=ax2, label="Average vertical velocity from hab=0-60 m")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
# ax2.set_title("Near-Bottom Vertical Velocity (SymLog scale)")
ax2.grid(true, linestyle="-", alpha=1, color="black")  # Add grid to second plot
plt.tight_layout()
savefig(string("output/tilt/criticality_vs_velocity_tf=",tᶠ,".png"), dpi=300)

# # Create a scatter plot of criticality vs vertical velocity
# fig2, ax = plt.subplots(figsize=(10, 8))
# scatter(γ, what_nearbottom_avg[:,:,1], alpha=0.5, s=5)
# axvline(x=1, color="r", linestyle="--", linewidth=2, label="Critical slope (γ=1)")
# axhline(y=0, color="k", linestyle="-", linewidth=1)
# xlabel("Criticality parameter (γ)")
# ylabel("Vertical velocity near bottom (m/s)")
# title("Relationship between Criticality and Near-Bottom Vertical Velocity")
# grid(true, alpha=0.3)
# legend()

# # Add trend line
# valid = .!isnan.(what_nearbottom_avg[:,:,1]) .& .!isnan.(γ)
# if sum(valid) > 0
#     γ_valid = γ[valid]
#     w_valid = what_nearbottom_avg[:,:,1][valid]
    
#     # Group by criticality ranges and calculate mean velocities
#     γ_bins = range(0, maximum(γ_valid), length=20)
#     γ_centers = (γ_bins[1:end-1] + γ_bins[2:end]) ./ 2
#     w_means = zeros(length(γ_centers))
    
#     for i in 1:length(γ_centers)
#         in_bin = (γ_valid .>= γ_bins[i]) .& (γ_valid .< γ_bins[i+1])
#         if sum(in_bin) > 0
#             w_means[i] = mean(w_valid[in_bin])
#         else
#             w_means[i] = NaN
#         end
#     end
    
#     plot(γ_centers, w_means, "r-", linewidth=2, label="Mean trend")
#     legend()
# end

# tight_layout()
# savefig(string("output/tilt/criticality_velocity_scatter_tf=",tᶠ,".png"), dpi=300)
# println("Relationship plots saved!")


# joint PDF of mean w and criticality 
using StatsBase
using Statistics
using PyPlot
# Flatten arrays for the joint histogram
γ_flat = vec(γ)
# w_flat = vec(what_nearbottom_avg[:, :, 1])  # Get the first element of the 3D array

# Use what_nearbottom_avg which has the same dimensions as γ
w_flat = vec(what_nearbottom_avg[:, :, 1])

# Remove NaN values
valid_idx = .!isnan.(w_flat) .& .!isnan.(γ_flat)
γ_valid = γ_flat[valid_idx]
w_valid = w_flat[valid_idx]

# Define bin edges for both variables
γ_bins = range(0, maximum(γ_valid) + 0.1, length=100)
w_bins = range(minimum(w_valid) - 1e-6, maximum(w_valid) + 1e-6, length=100)

# Compute the 2D histogram
hist2d = StatsBase.fit(Histogram, (γ_valid, w_valid), (γ_bins, w_bins))

# Normalize to get PDF
pdf_values = normalize(hist2d, mode=:pdf).weights

# For plotting later - convert bin edges to centers
γ_centers = (γ_bins[1:end-1] .+ γ_bins[2:end]) ./ 2
w_centers = (w_bins[1:end-1] .+ w_bins[2:end]) ./ 2

# Plotting the joint PDF
# Create a figure for the joint PDF
close("all")
fig, ax = subplots(figsize=(10, 6))

# Plot joint PDF with improved colormap (swapped x and y)
pcm = pcolor(w_centers, γ_centers, pdf_values,  # Swapped w and γ
             cmap="plasma",  # Professional colormap - better for density visualization
             shading="auto", 
             norm=matplotlib.colors.LogNorm())
cb = colorbar(pcm, ax=ax, label="PDF")
# Set larger tick labels for both the colorbar and the main plot axes
cb.ax.tick_params(labelsize=16)  # Increase colorbar tick label size
cb.ax.set_ylabel("PDF", fontsize=16)  # Increase the size of colorbar label
ax.tick_params(labelsize=16)     # Increase main plot tick label size
# Add lines for γ=1 (critical slope) with improved styling (now horizontal line)
axhline(y=1, color="black", linestyle="-", linewidth=2, label="Critical slope (γ=1)")

# Add zero line for vertical velocity (now vertical line)
axvline(x=0, color="black", linestyle="-", linewidth=1)

# Add trend line with improved styling
valid = .!isnan.(what_nearbottom_avg[:,:,1]) .& .!isnan.(γ)
if sum(valid) > 0
    γ_valid = γ[valid]
    w_valid = what_nearbottom_avg[:,:,1][valid]
    
    # Group by vertical velocity ranges and calculate mean criticality
    γ_means = zeros(length(w_centers))
    
    # Group by criticality ranges and calculate mean vertical velocities
    w_means = zeros(length(γ_centers))
    
    for i in 1:length(γ_centers)
        in_bin = (γ_valid .>= γ_bins[i]) .& (γ_valid .< γ_bins[i+1])
        if sum(in_bin) > 0
            w_means[i] = mean(w_valid[in_bin])
        else
            w_means[i] = NaN
        end
    end
    
    # Plot mean w for each γ bin
    plot(w_means, γ_centers, color="red", linewidth=2.5, label="Mean trend")
    # legend(frameon=true, framealpha=0.9, fontsize=14, loc="upper left")
end

# Labels and title (swapped)
ylabel("Slope criticality (γ)", fontsize=14)
xlabel(L"\mathrm{Near~bottom~average}~ \overline{ŵ} ~ \mathrm{[m~s⁻¹]}", fontsize=14)
grid(true, linestyle="--", alpha=0.7)

# Set axis limits if needed (swapped)
ylim(minimum(γ_centers), 4)
xlim(minimum(w_centers), maximum(w_centers))

tight_layout()
savefig(string("output/tilt/joint_pdf_criticality_w_tf=",tᶠ,"_z_upper=",bin_edge[z_upper],"_4TPavg.png"), dpi=300)


# Compute skewness of vertical velocity for different criticality ranges
using StatsBase

# Define criticality bins
# Define criticality bins with finer resolution, similar to joint PDF
γ_step = 0.1
γ_max_val = maximum(γ[.!isnan.(γ)])
γ_ranges = [(i, i+γ_step) for i in 0:γ_step:min(3.0, γ_max_val-γ_step)]
skewness_values = Float64[]
γ_centers = Float64[]
data_counts = Int[]
γ_labels = String[]

for (i, (γ_min, γ_max)) in enumerate(γ_ranges)
    # Find points within this criticality range
    mask = (γ .>= γ_min) .& (γ .<= γ_max)
    w_values = what_nearbottom_avg[mask]
    
    # Remove NaNs
    w_values = w_values[.!isnan.(w_values)]
    
    # Compute skewness if we have enough data points
    if length(w_values) > 30
        skew = skewness(w_values)
        push!(skewness_values, skew)
        label = "$(γ_min)-$(γ_max)"
        push!(γ_labels, label)
        println("$(γ_min)-$(γ_max) | $(round(skew, digits=4)) | $(length(w_values))")
    else
        println("$(γ_min)-$(γ_max) | insufficient data | $(length(w_values))")
    end
end

# Plot skewness vs criticality
close("all")
fig, ax = plt.subplots(figsize=(10, 6))
γ_centers = [(r[1] + r[2])/2 for r in γ_ranges][1:length(skewness_values)]
bar(γ_centers, skewness_values, width=0.2, alpha=0.7)
xlabel("Criticality parameter (γ)")
ylabel("Skewness of vertical velocity")
title("Skewness of near-bottom vertical velocity by criticality")
grid(true, alpha=0.3, linestyle="--")
ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
ax.axvline(x=1, color="r", linestyle="--", alpha=0.7, label="Critical slope (γ=1)")
legend()
tight_layout()
savefig(string("output/tilt/w_skewness_by_criticality_tf=", tᶠ, ".png"), dpi=300)
println("Skewness analysis completed and figure saved.")



using PyPlot

# Create the plot
close("all")
fig = figure(figsize=(12, 9))
pcolor(xC, yC, γ', cmap="viridis", shading="auto")
# Create a custom colormap that changes at criticality=1
cmap = plt.cm.coolwarm  # Use a diverging colormap
norm = plt.Normalize(vmin=0, vmax=2)  # Set range with 1 in the middle
im = pcolor(xC, yC, γ', cmap=cmap, norm=norm, shading="auto")
cb = colorbar(im, label="Criticality parameter γ")

# Add a horizontal line in the colorbar at γ=1
ax_cb = cb.ax
ax_cb.axhline(y=(1-0)/(2-0), color='k', linestyle='-', linewidth=1)
# ax_cb.text(2.5, (1-0)/(2-0), "γ=1", va='center', ha='left', fontsize=9)

xlabel("x (m)")
ylabel("y (m)")
title("Topographic criticality parameter")
tight_layout()

# Save the figure
savefig("output/tilt/criticality_map.png", dpi=300)
println("Figure saved as criticality_map.png")