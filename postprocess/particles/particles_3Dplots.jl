# cd("scratch/bcpi/cliu28/internal-tide-mixing/")
# run this first part 
using MAT
using NCDatasets
using NaNStatistics
using Interpolations
using CairoMakie
using ColorSchemes
using Printf
using Statistics
# load topography and transform it into Cartesian coordinates
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic")
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo, 1))
y_topo_lin = range(y_topo[1], y_topo[end], size(z_topo, 2))
close(file)
# high-resolution grids
Nx = 500
Ny = 1000
θ = 0.0036
x_interp = range(x_topo[1], x_topo[end], length=Nx)
y_interp = range(y_topo[1], y_topo[end], length=Ny)

# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp .- minimum(z_interp)

# create an extended topography
dx_interp = x_interp[2] - x_interp[1]
dy_interp = y_interp[2] - y_interp[1]
Lx_interp = x_interp[end]
Ly_interp = y_interp[end]

x_interp_ext = vcat(x_interp[3Nx÷4:Nx] .- Lx_interp .- dx_interp, x_interp, x_interp .+ Lx_interp .+ dx_interp)
y_interp_ext = vcat(y_interp, y_interp .+ Ly_interp .+ dy_interp)
z_interp_ext = vcat(z_interp[3Nx÷4:Nx, :], z_interp, z_interp)
z_interp_ext = vcat(z_interp_ext', z_interp_ext')'

## transform the topography into Cartesian coordinates
# create a meshgrid
X_interp_ext = repeat(x_interp_ext, 1, length(y_interp_ext))
Y_interp_ext = repeat(y_interp_ext', length(x_interp_ext), 1)
Z_interp_ext = z_interp_ext
# Transform topography to Cartesian coordinates
X_cart_interp_ext = X_interp_ext .* cos(θ) .- Z_interp_ext .* sin(θ)
Y_cart_interp_ext = Y_interp_ext  # y-coordinate is unchanged
Z_cart_interp_ext = X_interp_ext .* sin(θ) .+ Z_interp_ext .* cos(θ)

simname = "tilt"
z_center_particle = 1000  # Center height of particles
# output_file = string("output/", simname, "/concatenated_particle_data_z", z_center_particle, "_early.nc")
# @info "Loading concatenated data for animation..."
# ds = NCDataset(output_file, "r")
# x_cart_full = ds["x_cartesian"][:, :]
# y_cart_full = ds["y_cartesian"][:, :]
# z_cart_full = ds["z_cartesian"][:, :]
# B_full = ds["buoyancy"][:, :]
# time_full = ds["time"][:]

output_file = string("output/", simname, "/concatenated_particle_data_z", z_center_particle, "_all.nc")
@info "Loading concatenated data for animation..."
ds = NCDataset(output_file, "r")
x_cart_full = ds["main_x_cartesian"][:, :]
y_cart_full = ds["main_y_cartesian"][:, :]
z_cart_full = ds["main_z_cartesian"][:, :]
B_full = ds["main_buoyancy"][:, :]
time_full = ds["main_time"][:]

b_begin = B_full[:, 1]  # Initial z positions
b₀ = median(b_begin)         # median particle buoyancy

###########

valid_indices = findall(b_begin .<= b₀)
x_0_down = x_cart_full[valid_indices, :]
y_0_down = y_cart_full[valid_indices, :]
z_0_down = z_cart_full[valid_indices, :]
B_0_down = B_full[valid_indices, :]
B_pert_0_down = B_0_down[:, :] .- B_0_down[:, 1]  # Remove the initial buoyancy offset
all_mean_buoyancy_0_down = vec(mean(B_pert_0_down, dims=1))
all_median_buoyancy_0_down = vec(median(B_pert_0_down, dims=1))
# only includes particles above mean buoyancy
valid_indices = findall(b_begin .>= b₀)
x_0_up = x_cart_full[valid_indices, :]
y_0_up = y_cart_full[valid_indices, :]
z_0_up = z_cart_full[valid_indices, :]
B_0_up = B_full[valid_indices, :]
B_pert_0_up = B_0_up[:, :] .- B_0_up[:, 1]  # Remove the initial buoyancy offset
all_mean_buoyancy_0_up = vec(mean(B_pert_0_up, dims=1))
all_median_buoyancy_0_up = vec(median(B_pert_0_up, dims=1))

# 

close(ds)

B_0_up = nothing
B_0_down = nothing
# Clean up memory
GC.gc()




## plot 3 panels (1x3): 
# 1: is t=0, showing initial particle distribution with upper and lower half, using different colors
# 2: shows the t=458.0 for the upper
# 3: shows the t=458.0 for the lower
# Clean up memory
GC.gc()
# create a 4-panel figure for topography and particle trajectories in terms of buoyancy perturbation
# Create terrain colormap
full_cmap = ColorSchemes.terrain.colors
custom_cmap = full_cmap[1:floor(Int, 1 * length(full_cmap))]
# Create a properly structured figure with 1 row, 4 columns (3 plots + 1 colorbar)
fig = CairoMakie.Figure(resolution=(1200, 400))

# Define the four evenly distributed time steps
n_particles, n_time_steps = size(x_cart_full)
subsample = 1:1:n_particles÷2

# Calculate tidal period
T_tidal = 2 * pi / 1.4e-4

# Set consistent camera angle for all subplots
camera_azimuth = 1.7π
camera_elevation = 0.15π

# Create the three subplots in positions [1,1], [1,2], [1,3]
ax1 = Axis3(fig[1, 1],
    xlabel="x̂ [km]",
    ylabel="y [km]",
    zlabel="ẑ [m]",
    title="t = 0 (Initial)",
    aspect=(1, 1.4, 0.6),
    limits=((0, 30e3), (0, 40e3), (0, 1500))
)

ax2 = Axis3(fig[1, 2],
    xlabel="x̂ [km]",
    ylabel="y [km]",
    zlabel="ẑ [m]",
    title="t = $(round(time_full[end]/T_tidal, digits=1))T (Lower half)",
    aspect=(1, 1.4, 0.6),
    limits=((0, 30e3), (0, 40e3), (0, 1500))
)

ax3 = Axis3(fig[1, 3],
    xlabel="x̂ [km]",
    ylabel="y [km]",
    zlabel="ẑ [m]",
    title="t = $(round(time_full[end]/T_tidal, digits=1))T (Upper half)",
    aspect=(1, 1.4, 0.6),
    limits=((0, 30e3), (0, 40e3), (0, 1500))
)
y_extend = 1347
# Set consistent camera angles
for ax in [ax1, ax2, ax3]
    ax.azimuth = camera_azimuth
    ax.elevation = camera_elevation
    ax.xticks = ([0, 10, 20, 30] .* 1e3, ["0", "10", "20", "30"])
    ax.yticks = ([0, 10, 20, 30, 40] .* 1e3, ["0", "10", "20", "30", "40"])

    surface!(ax, X_cart_interp_ext[:, 1:y_extend], Y_cart_interp_ext[:, 1:y_extend], Z_cart_interp_ext[:, 1:y_extend],
        colormap=:terrain,
        shading=NoShading,
        transparency=false,
        alpha=0.8,
        colorrange=(0, 1500))
end

# Plot data for each subplot
# Panel 1: Initial distribution (both upper and lower)
scatter!(ax1,
    x_0_down[subsample, 1],
    y_0_down[subsample, 1],
    z_0_down[subsample, 1],
    color=:purple,
    markersize=5,
    alpha=0.7,
    label="Lower half"
)

scatter!(ax1,
    x_0_up[subsample, 1],
    y_0_up[subsample, 1],
    z_0_up[subsample, 1],
    color=:orange,
    markersize=5,
    alpha=0.7,
    label="Upper half"
)

# Panel 2: Final state - lower half with buoyancy perturbation
particles2 = scatter!(ax2,
    x_0_down[subsample, end],
    y_0_down[subsample, end],
    z_0_down[subsample, end],
    color=B_pert_0_down[subsample, end],
    markersize=5,
    colormap=reverse(cgrad(:RdYlGn)),
    colorrange=(-2e-5, 2e-5)
)

# Panel 3: Final state - upper half with buoyancy perturbation  
particles3 = scatter!(ax3,
    x_0_up[subsample, end],
    y_0_up[subsample, end],
    z_0_up[subsample, end],
    color=B_pert_0_up[subsample, end],
    markersize=5,
    colormap=reverse(cgrad(:RdYlGn)),
    colorrange=(-2e-5, 2e-5)
)

# Add colorbar in the 4th column
Colorbar(fig[1, 4],
    particles3,  # Use the last scatter plot as reference
    label="10⁻⁵ ΔBₚ [m s⁻²]",
    width=8,
    height=Relative(0.6),
    ticks=([-2e-5, -1e-5, 0, 1e-5, 2e-5], ["-2", "-1", "0", "1", "2"]),
)

# Adjust column spacing
colgap!(fig.layout, 3, 50)

# Save the figure
save(string("output/", simname, "/3D_particle_trajectories_3panels_z_center=", z_center_particle, "_Bpert_differentregions.png"), fig)

@info "3-panel 3D particle trajectory plot saved"
# Final cleanup
GC.gc()


###### plot pdf of the particles in the x-y plane as the first panel,
# second panel is the mean buoyancy change over time in x-y plane in the lower half
# third panel is buoyancy change dB/dt over time in x-y plane in the upper half
# Create a 3-panel figure for particle distribution and buoyancy evolution

using StatsBase

# 1. Flatten all particle positions over all time steps
x_all_flat = vec(x_cart_full)
y_all_flat = vec(y_cart_full)
x_0_down_flat = vec(x_0_down)
y_0_down_flat = vec(y_0_down)
x_0_up_flat = vec(x_0_up)
y_0_up_flat = vec(y_0_up)

# 2. Define bin edges using X_cart_interp_ext and Y_cart_interp_ext
xedges = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=400)
yedges = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=500)

#for initial PDF
xedges_init = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=300)
yedges_init = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=350)


# 3. Compute 2D histogram (counts)
h_all = StatsBase.fit(Histogram, (x_all_flat, y_all_flat), (xedges, yedges))
h_down = StatsBase.fit(Histogram, (x_0_down_flat, y_0_down_flat), (xedges, yedges))
h_up = StatsBase.fit(Histogram, (x_0_up_flat, y_0_up_flat), (xedges, yedges))

# 4. Normalize to get PDF (probability density)
pdf_all = h_all.weights ./ sum(h_all.weights)
pdf_down = h_down.weights ./ sum(h_down.weights)
pdf_up = h_up.weights ./ sum(h_up.weights)

## calculate the initial contour of the particle distribution 
x_all_flat_init = vec(x_cart_full[:,1])
y_all_flat_init = vec(y_cart_full[:,1])
h_all_init = StatsBase.fit(Histogram, (x_all_flat_init, y_all_flat_init), (xedges_init, yedges_init))
pdf_all_init = h_all_init.weights ./ sum(h_all_init.weights)
# Find the 95% contour threshold for the initial PDF
pdf_flat = vec(pdf_all_init)
sorted_pdf = sort(pdf_flat, rev=true)
cumsum_pdf = cumsum(sorted_pdf)
idx_95 = findfirst(cumsum_pdf .>= 0.95)
threshold_95 = sorted_pdf[idx_95]


### Calculate mean buoyancy change over time in the lower half
dBdt_down = (B_0_down[:, 2:end] .- B_0_down[:, 1:end-1])./(time_full[2]-time_full[1])  # Change in buoyancy over time
dBdt_up = (B_0_up[:, 2:end] .- B_0_up[:, 1:end-1])./(time_full[2]-time_full[1])  # Change in buoyancy over time

using StatsBase

# Prepare bin edges
# xedges = X_cart_interp_ext[:,1]
# yedges = Y_cart_interp_ext[1,:]

# Prepare accumulator arrays
nx, ny = length(xedges)-1, length(yedges)-1
sum_dBdt_down = zeros(nx, ny)
count_dBdt_down = zeros(nx, ny)

# Loop over all particles and time steps (for lower half)
for p in 1:size(x_0_down, 1)
    for t in 1:size(x_0_down, 2)-1
        x = x_0_down[p, t]
        y = y_0_down[p, t]
        val = dBdt_down[p, t]
        # Find bin indices
        ix = searchsortedfirst(xedges, x) - 1
        iy = searchsortedfirst(yedges, y) - 1
        if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny
            sum_dBdt_down[ix, iy] += val
            count_dBdt_down[ix, iy] += 1
        end
    end
    # @info "Processed particle $p of $(size(x_0_down, 1)) in lower half"
end
# 4. Compute mean dBdt in each bin (avoid division by zero)
mean_dBdt_down = fill(NaN, nx, ny)
for ix in 1:nx, iy in 1:ny
    if count_dBdt_down[ix, iy] > 60
        mean_dBdt_down[ix, iy] = sum_dBdt_down[ix, iy] / count_dBdt_down[ix, iy]
    end
end


# Prepare accumulator arrays
nx, ny = length(xedges)-1, length(yedges)-1
sum_dBdt_up = zeros(nx, ny)
count_dBdt_up = zeros(nx, ny)

# Loop over all particles and time steps (for upper half)
for p in 1:size(x_0_up, 1)
    for t in 1:size(x_0_up, 2)-1
        x = x_0_up[p, t]
        y = y_0_up[p, t]
        val = dBdt_up[p, t]
        # Find bin indices
        ix = searchsortedfirst(xedges, x) - 1
        iy = searchsortedfirst(yedges, y) - 1
        if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny
            sum_dBdt_up[ix, iy] += val
            count_dBdt_up[ix, iy] += 1
        end
    end
end
# Compute mean dBdt in each bin (avoid division by zero)
mean_dBdt_up = fill(NaN, nx, ny)
for ix in 1:nx, iy in 1:ny
    if count_dBdt_up[ix, iy] > 60  # filter out bins with too few particles
        mean_dBdt_up[ix, iy] = sum_dBdt_up[ix, iy] / count_dBdt_up[ix, iy]
    end
end

using PyPlot

# Prepare bin centers for plotting
xcenters = 0.5 .* (xedges[1:end-1] + xedges[2:end]) ./ 1e3  # km
ycenters = 0.5 .* (yedges[1:end-1] + yedges[2:end]) ./ 1e3  # km
# Prepare extended bin centers for initial PDF
xcenters_init = 0.5 .* (xedges_init[1:end-1] + xedges_init[2:end]) ./ 1e3  # km
ycenters_init = 0.5 .* (yedges_init[1:end-1] + yedges_init[2:end]) ./ 1e3  # km

# Prepare topography for contours (crop to plotting region if needed)
topo_x = X_cart_interp_ext[:, 1:1000] ./ 1e3  # km
topo_y = Y_cart_interp_ext[:, 1:1000] ./ 1e3  # km
topo_z = Z_cart_interp_ext[:, 1:1000]         # m

close("all")
fig, axs = subplots(1, 3, figsize=(16, 5), constrained_layout=true)

# Set global font sizes
PyPlot.rc("font", size=16)          # controls default text sizes
PyPlot.rc("axes", titlesize=18)     # fontsize of the axes title
PyPlot.rc("axes", labelsize=16)     # fontsize of the x and y labels
PyPlot.rc("xtick", labelsize=14)    # fontsize of the tick labels
PyPlot.rc("ytick", labelsize=14)    # fontsize of the tick labels
PyPlot.rc("legend", fontsize=14)    # legend fontsize

# 1. Particle PDF
pdf_all[pdf_all.==0] .= NaN
pcm_all = axs[1].contourf(xcenters, ycenters, pdf_all',level=[0:0.5:3].*1e-5, cmap="afmhot_r", shading="auto", alpha=1, edgecolors="black", linewidth=1)
cont = axs[1].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[1].contour(xcenters_init, ycenters_init, pdf_all_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[1].set_title("Particle PDF", fontsize=18)
axs[1].set_xlabel("x̂ [km]", fontsize=16)
axs[1].set_ylabel("y [km]", fontsize=16)
axs[1].set_ylim(0, 40)
axs[1].set_xlim(0, 30)
axs[1].tick_params(axis="both", labelsize=14)
# 2. Mean dB/dt (lower half)
pcm = axs[2].pcolormesh(xcenters, ycenters, mean_dBdt_down', cmap="coolwarm", shading="auto", vmin=-1e-9, vmax=1e-9)
cont = axs[2].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[2].contour(xcenters_init, ycenters_init, pdf_all_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[2].set_title("Lower half", fontsize=18)
axs[2].set_xlabel("x̂ [km]", fontsize=16)
# axs[2].set_ylabel("y [km]", fontsize=16)
axs[2].set_ylim(0, 40)
axs[2].set_xlim(0, 30)
axs[2].tick_params(axis="both", labelsize=14)
axs[2].set_yticklabels([])  # Turn off y-tick labels

# 3. Mean dB/dt (upper half)
pcm = axs[3].pcolormesh(xcenters, ycenters, mean_dBdt_up', cmap="coolwarm", shading="auto", vmin=-1e-9, vmax=1e-9)
cont = axs[3].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[3].contour(xcenters_init, ycenters_init, pdf_all_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
cbar = fig.colorbar(pcm, ax=axs[3], label="Net Bₚ change [m s⁻³]", extend="both")
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Net Bₚ change [m s⁻³]", fontsize=16)
axs[3].set_title("Upper half", fontsize=18)
axs[3].set_xlabel("x̂ [km]", fontsize=16)
# axs[3].set_ylabel("y [km]", fontsize=16)
axs[3].set_ylim(0, 40)
axs[3].set_xlim(0, 30)
axs[3].tick_params(axis="both", labelsize=14)
axs[3].set_yticklabels([])  # Turn off y-tick labels

fig.savefig("output/$(simname)/combined_pdf_dBdt_panels_lessbins.png", dpi=100)
close(fig)





####### plot pdf using hab to bin  

###### plot pdf of the particles in the x-y plane as the first panel,
# second panel is the mean buoyancy change over time in x-y plane in the lower half
# third panel is buoyancy change dB/dt over time in x-y plane in the upper half
# Create a 3-panel figure for particle distribution and buoyancy evolution
using StatsBase
using Interpolations

# Compute height above bottom (hab) for each particle at each time
filename_field = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_field = Dataset(filename_field, "r")
zC = ds_field["zC"][:]
Nz = length(zC)
close(ds_field)

# Prepare hab grid: [Nx_ext, Ny_ext, Nz]
hab_grid = Array{Float64}(undef, size(X_cart_interp_ext,1), size(Y_cart_interp_ext,2), Nz)
for k in 1:Nz
    hab_grid[:,:,k] = zC[k] .- Z_cart_interp_ext
end

# Interpolator for hab
z_levels = zC
itp_hab = interpolate((x_interp_ext, y_interp_ext, z_levels), hab_grid, Gridded(Linear()))

GC.gc()
n_particles, n_times = size(x_cart_full)
hab_particles = Array{Float64}(undef, n_particles, n_times)
for p in 1:n_particles
    for t in 1:n_times
        x = x_cart_full[p, t]
        y = y_cart_full[p, t]
        z = z_cart_full[p, t]
        if x_interp_ext[1] ≤ x ≤ x_interp_ext[end] &&
           y_interp_ext[1] ≤ y ≤ y_interp_ext[end] &&
           z_levels[1] ≤ z ≤ z_levels[end]
            hab_particles[p, t] = itp_hab(x, y, z)
        else
            hab_particles[p, t] = NaN
        end
    end
end

# Bin edges for x, y
xedges = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=300)
yedges = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=350)

# For initial PDF
xedges_init = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=300)
yedges_init = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=350)

# Flatten all particle positions over all time steps
x_all_flat = vec(x_cart_full)
y_all_flat = vec(y_cart_full)

# 2D histogram for PDF
h_all = StatsBase.fit(Histogram, (x_all_flat, y_all_flat), (xedges, yedges))
pdf_all = h_all.weights ./ sum(h_all.weights)

# Initial PDF for contour
x_all_flat_init = vec(x_cart_full[:,1])
y_all_flat_init = vec(y_cart_full[:,1])
h_all_init = StatsBase.fit(Histogram, (x_all_flat_init, y_all_flat_init), (xedges_init, yedges_init))
pdf_all_init = h_all_init.weights ./ sum(h_all_init.weights)
pdf_flat = vec(pdf_all_init)
sorted_pdf = sort(pdf_flat, rev=true)
cumsum_pdf = cumsum(sorted_pdf)
idx_95 = findfirst(cumsum_pdf .>= 0.95)
threshold_95 = sorted_pdf[idx_95]

# Bin centers for plotting
xcenters = 0.5 .* (xedges[1:end-1] + xedges[2:end]) ./ 1e3  # km
ycenters = 0.5 .* (yedges[1:end-1] + yedges[2:end]) ./ 1e3  # km
xcenters_init = 0.5 .* (xedges_init[1:end-1] + xedges_init[2:end]) ./ 1e3
ycenters_init = 0.5 .* (yedges_init[1:end-1] + yedges_init[2:end]) ./ 1e3

# Topography for contours
topo_x = X_cart_interp_ext[:, 1:1000] ./ 1e3
topo_y = Y_cart_interp_ext[:, 1:1000] ./ 1e3
topo_z = Z_cart_interp_ext[:, 1:1000]

# Compute mean dB/dt for upper and lower hab regions
hab_edges = [0, 300, 1500]  # [lower, upper]
nx, ny = length(xedges)-1, length(yedges)-1
sum_dBdt_upper = zeros(nx, ny)
count_dBdt_upper = zeros(nx, ny)
sum_dBdt_lower = zeros(nx, ny)
count_dBdt_lower = zeros(nx, ny)

for p in 1:n_particles
    for t in 1:(n_times-1)
        x = x_cart_full[p, t]
        y = y_cart_full[p, t]
        hab = hab_particles[p, t]
        dB = (B_full[p, t+1] - B_full[p, t]) / (time_full[t+1] - time_full[t])
        ix = searchsortedfirst(xedges, x) - 1
        iy = searchsortedfirst(yedges, y) - 1
        if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny && !isnan(hab)
            if hab ≥ hab_edges[2] && hab < hab_edges[3]
                sum_dBdt_upper[ix, iy] += dB
                count_dBdt_upper[ix, iy] += 1
            elseif hab ≥ hab_edges[1] && hab < hab_edges[2]
                sum_dBdt_lower[ix, iy] += dB
                count_dBdt_lower[ix, iy] += 1
            end
        end
    end
end

mean_dBdt_upper = fill(NaN, nx, ny)
mean_dBdt_lower = fill(NaN, nx, ny)
for ix in 1:nx, iy in 1:ny
    if count_dBdt_upper[ix, iy] > 10
        mean_dBdt_upper[ix, iy] = sum_dBdt_upper[ix, iy] / count_dBdt_upper[ix, iy]
    end
    if count_dBdt_lower[ix, iy] > 10
        mean_dBdt_lower[ix, iy] = sum_dBdt_lower[ix, iy] / count_dBdt_lower[ix, iy]
    end
end

using PyPlot

close("all")
fig, axs = subplots(1, 3, figsize=(16, 5), constrained_layout=true)

PyPlot.rc("font", size=16)
PyPlot.rc("axes", titlesize=18)
PyPlot.rc("axes", labelsize=16)
PyPlot.rc("xtick", labelsize=14)
PyPlot.rc("ytick", labelsize=14)
PyPlot.rc("legend", fontsize=14)

# 1. Particle PDF
pdf_all[pdf_all.==0] .= NaN
pcm_all = axs[1].contourf(xcenters, ycenters, pdf_all', level=[0:0.5:3] .* 1e-5, cmap="afmhot_r", shading="auto", alpha=1, edgecolors="black", linewidth=1)
cont = axs[1].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[1].contour(xcenters_init, ycenters_init, pdf_all_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[1].set_title("Particle PDF", fontsize=18)
axs[1].set_xlabel("x̂ [km]", fontsize=16)
axs[1].set_ylabel("y [km]", fontsize=16)
axs[1].set_ylim(0, 40)
axs[1].set_xlim(0, 30)
axs[1].tick_params(axis="both", labelsize=14)

# 2. Mean dB/dt (lower half, hab 0-200m)
pcm = axs[2].pcolormesh(xcenters, ycenters, mean_dBdt_lower', cmap="coolwarm", shading="auto", vmin=-1e-9, vmax=1e-9)
cont = axs[2].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[2].contour(xcenters_init, ycenters_init, pdf_all_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[2].set_title("HAB 0-$(hab_edges[2]) m", fontsize=18)
axs[2].set_xlabel("x̂ [km]", fontsize=16)
axs[2].set_ylim(0, 40)
axs[2].set_xlim(0, 30)
axs[2].tick_params(axis="both", labelsize=14)
axs[2].set_yticklabels([])

# 3. Mean dB/dt (upper half, hab 200m-top)
pcm = axs[3].pcolormesh(xcenters, ycenters, mean_dBdt_upper', cmap="coolwarm", shading="auto", vmin=-1e-9, vmax=1e-9)
cont = axs[3].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[3].contour(xcenters_init, ycenters_init, pdf_all_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
cbar = fig.colorbar(pcm, ax=axs[3], label="Net Bₚ change [m s⁻³]", extend="both")
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Net Bₚ change [m s⁻³]", fontsize=16)
axs[3].set_title("HAB $(hab_edges[2])-1500 m", fontsize=18)
axs[3].set_xlabel("x̂ [km]", fontsize=16)
axs[3].set_ylim(0, 40)
axs[3].set_xlim(0, 30)
axs[3].tick_params(axis="both", labelsize=14)
axs[3].set_yticklabels([])

fig.savefig("output/$(simname)/combined_pdf_dBdt_panels_hab_$(hab_edges[2]).png", dpi=100)
close(fig)








###### Create a 4-panel figure for topography and particle trajectories
fig = CairoMakie.Figure(resolution=(1000, 700))

# Define the four evenly distributed time steps
n_particles, n_time_steps = size(x_cart_full)
time_indices = [1, div(n_time_steps, 3), div(2 * n_time_steps, 3), n_time_steps]

# Calculate tidal period (assuming 12.42 hours for M2 tidal period)
T_tidal = 2 * pi / 1.4e-4  # Tidal period in seconds

# Set consistent camera angle for all subplots
camera_azimuth = 1.6π
camera_elevation = 0.15π

# Create the four subplots
axes = []
for i in 1:4
    ax = Axis3(fig[div(i - 1, 2)+1, mod(i - 1, 2)+1],
        xlabel="x̂ [km]",
        ylabel="y [km]",
        zlabel="ẑ [m]",
        aspect=(3.5, 4, 2),
        limits=((0, 30e3), (0, 40e3), (0, 1500))
    )

    # Set consistent camera angle
    ax.azimuth = camera_azimuth
    ax.elevation = camera_elevation

    push!(axes, ax)
end

# Create terrain colormap
full_cmap = ColorSchemes.terrain.colors
custom_cmap = full_cmap[1:floor(Int, 1 * length(full_cmap))]

# MEMORY OPTIMIZATION: Subsample particles and trails
particle_subsample = 1:1:n_particles  # Use every 5th particle
trail_subsample = 1:10:n_particles    # Use every 10th particle for trails
trail_length = 8

# Plot terrain and particles for each time step
for (subplot_idx, time_idx) in enumerate(time_indices)
    ax = axes[subplot_idx]

    # Plot the terrain surface (use alpha to make it less memory intensive)
    surface!(ax, X_cart_interp_ext[:, 1:1000], Y_cart_interp_ext[:, 1:1000], Z_cart_interp_ext[:, 1:1000],
        colormap=:terrain,
        shading=NoShading,
        transparency=false,
        alpha=0.8,
        colorrange=(0, 1500))

    # OPTIMIZED: Plot only subsampled particles
    particles = scatter!(ax,
        x_cart_full[particle_subsample, time_idx],
        y_cart_full[particle_subsample, time_idx],
        z_cart_full[particle_subsample, time_idx],
        color=B_full[particle_subsample, time_idx],
        markersize=5,
        colormap=reverse(cgrad(:RdBu)),
        colorrange=(0.001, 0.0013)
    )

    # OPTIMIZED: Create trails using vectorized approach
    # start_idx = max(1, time_idx - trail_length)
    # if start_idx < time_idx
    #     # Pre-allocate trail arrays
    #     trail_x = Vector{Float64}()
    #     trail_y = Vector{Float64}()
    #     trail_z = Vector{Float64}()
    #     trail_colors = Vector{Float64}()

    #     # Build trail data efficiently
    #     for i in trail_subsample
    #         for j in start_idx:(time_idx-1)
    #             # Add line segment points
    #             push!(trail_x, x_cart_full[i, j])
    #             push!(trail_y, y_cart_full[i, j])
    #             push!(trail_z, z_cart_full[i, j])
    #             push!(trail_colors, B_full[i, time_idx])

    #             push!(trail_x, x_cart_full[i, j+1])
    #             push!(trail_y, y_cart_full[i, j+1])
    #             push!(trail_z, z_cart_full[i, j+1])
    #             push!(trail_colors, B_full[i, time_idx])
    #         end
    #     end

    #     # Create single trail object instead of many
    #     if !isempty(trail_x)
    #         linesegments!(ax, 
    #             Point3f.(zip(trail_x, trail_y, trail_z)),
    #             color = trail_colors,
    #             colormap = :YlOrRd_9,
    #             colorrange = (0.001, 0.0013),
    #             linewidth = 1.5,
    #             alpha = 0.6)
    #     end
    # end

    # Calculate time in tidal periods
    current_time_seconds = time_full[time_idx]
    tidal_periods = current_time_seconds / T_tidal

    # Set title with time in tidal periods
    ax.title = @sprintf("t = %.2f tidal period", tidal_periods)
    ax.xticks = ([0, 10, 20, 30] .* 1e3, ["0", "10", "20", "30"])
    ax.yticks = ([0, 10, 20, 30, 40] .* 1e3, ["0", "10", "20", "30", "40"])
    @info "Plotted subplot $subplot_idx at time step $time_idx (t = %.2f T)" tidal_periods

    # MEMORY CLEANUP: Force garbage collection after each subplot
    GC.gc()
end

# Add thin colorbars to the right of the fourth subplot
# Create a layout for colorbars
cbar_layout = GridLayout(2, 1, height=Relative(0.6))
fig[1:2, 3] = cbar_layout

# Terrain colorbar
Colorbar(cbar_layout[1, 1],
    label="Terrain Elevation [m]",
    colormap=custom_cmap,
    limits=(0, 1500),
    width=15)

# Particle buoyancy colorbar
Colorbar(cbar_layout[2, 1],
    label="Particle Buoyancy [m s⁻²]",
    colormap=reverse(cgrad(:RdBu)),
    limits=(0.001, 0.0013),
    width=15)

# Adjust layout to give more space to the plots
colgap!(fig.layout, 1, 10)
colgap!(fig.layout, 2, 20)
rowgap!(fig.layout, 1, 5)

# Add overall title
# Label(fig[0, 1:2], "3D Particle Trajectories at Different Tidal Phases", 
#       fontsize = 18, tellwidth = false)

# Save the figure
save(string("output/", simname, "/3D_particle_trajectories_4panels_z_center=", z_center_particle, ".png"), fig)

@info "4-panel 3D particle trajectory plot saved"

# Final cleanup
GC.gc()







# create a 4-panel figure for topography and particle trajectories in terms of buoyancy perturbation

fig = CairoMakie.Figure(resolution=(1000, 700))

# Define the four evenly distributed time steps
n_particles, n_time_steps = size(x_cart_full)
time_indices = [1, div(n_time_steps, 3), div(2 * n_time_steps, 3), n_time_steps]

# Calculate tidal period (assuming 12.42 hours for M2 tidal period)
T_tidal = 2 * pi / 1.4e-4  # Tidal period in seconds

# Set consistent camera angle for all subplots
camera_azimuth = 1.6π
camera_elevation = 0.15π

# Create the four subplots
axes = []
for i in 1:4
    ax = Axis3(fig[div(i - 1, 2)+1, mod(i - 1, 2)+1],
        xlabel="x̂ [km]",
        ylabel="y [km]",
        zlabel="ẑ [m]",
        aspect=(3.5, 4, 2),
        limits=((0, 30e3), (0, 40e3), (0, 1500))
    )

    # Set consistent camera angle
    ax.azimuth = camera_azimuth
    ax.elevation = camera_elevation

    push!(axes, ax)
end

# Create terrain colormap
full_cmap = ColorSchemes.terrain.colors
custom_cmap = full_cmap[1:floor(Int, 1 * length(full_cmap))]

# MEMORY OPTIMIZATION: Subsample particles and trails
particle_subsample = 1:1:n_particles  # Use every 5th particle
trail_length = 8

# Plot terrain and particles for each time step
for (subplot_idx, time_idx) in enumerate(time_indices)
    ax = axes[subplot_idx]

    # Plot the terrain surface (use alpha to make it less memory intensive)
    surface!(ax, X_cart_interp_ext[:, 1:1000], Y_cart_interp_ext[:, 1:1000], Z_cart_interp_ext[:, 1:1000],
        colormap=:terrain,
        shading=NoShading,
        transparency=false,
        alpha=0.8,
        colorrange=(0, 1500))
    particles = scatter!(ax,
        x_0_down[:, time_idx],
        y_0_down[:, time_idx],
        z_0_down[:, time_idx],
        color=B_pert_0_down[:, time_idx],
        markersize=5,
        colormap=reverse(cgrad(:RdYlGn)),
        colorrange=(-1e-5, 1e-5)
    )
    # particles = scatter!(ax, 
    #         x_0_up[:, time_idx], 
    #         y_0_up[:, time_idx], 
    #         z_0_up[:, time_idx],
    #         color = B_pert_0_up[:, time_idx],
    #         markersize = 5,
    #         colormap = reverse(cgrad(:RdYlGn)),
    #         colorrange = (-1e-5, 1e-5)
    #       )    
    # Calculate time in tidal periods
    current_time_seconds = time_full[time_idx]
    tidal_periods = current_time_seconds / T_tidal

    # Set title with time in tidal periods
    ax.title = @sprintf("t = %.2f tidal period", tidal_periods)
    ax.xticks = ([0, 10, 20, 30] .* 1e3, ["0", "10", "20", "30"])
    ax.yticks = ([0, 10, 20, 30, 40] .* 1e3, ["0", "10", "20", "30", "40"])
    @info "Plotted subplot $subplot_idx at time step $time_idx (t = %.2f T)" tidal_periods

    # MEMORY CLEANUP: Force garbage collection after each subplot
    GC.gc()
end

# Add thin colorbars to the right of the fourth subplot
# Create a layout for colorbars
cbar_layout = GridLayout(2, 1, height=Relative(0.6))
fig[1:2, 3] = cbar_layout

# Terrain colorbar
Colorbar(cbar_layout[1, 1],
    label="Terrain Elevation [m]",
    colormap=custom_cmap,
    limits=(0, 1500),
    width=15)

# Particle buoyancy colorbar
Colorbar(cbar_layout[2, 1],
    label="ΔB_p [m s⁻²]",
    colormap=reverse(cgrad(:RdYlGn)),
    limits=(-1e-5, 1e-5),
    width=15)

# Adjust layout to give more space to the plots
colgap!(fig.layout, 1, 10)
colgap!(fig.layout, 2, 20)
rowgap!(fig.layout, 1, 5)

# Add overall title
# Label(fig[0, 1:2], "3D Particle Trajectories at Different Tidal Phases", 
#       fontsize = 18, tellwidth = false)

# Save the figure
save(string("output/", simname, "/3D_particle_trajectories_4panels_z_center=", z_center_particle, "_Bperturbation_down.png"), fig)

@info "4-panel 3D particle trajectory plot saved"

# Final cleanup
GC.gc()