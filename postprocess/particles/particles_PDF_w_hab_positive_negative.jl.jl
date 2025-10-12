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

x_interp_ext = vcat(x_interp[2Nx÷5:Nx] .- Lx_interp .- dx_interp, x_interp, x_interp .+ Lx_interp .+ dx_interp, x_interp .+ 2Lx_interp .+ 2dx_interp)
y_interp_ext = vcat(y_interp, y_interp .+ Ly_interp .+ dy_interp)
z_interp_ext = vcat(z_interp[2Nx÷5:Nx, :], z_interp, z_interp, z_interp)
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

###########

positive_indices = findall(B_full[:,end] .> b_begin)
negative_indices = findall(B_full[:,end] .< b_begin)
x_0_positive = x_cart_full[positive_indices, :]
y_0_positive = y_cart_full[positive_indices, :]
z_0_positive = z_cart_full[positive_indices, :]
B_0_positive = B_full[positive_indices, :]
B_pert_0_positive = B_0_positive[:, :] .- B_0_positive[:, 1]  # Remove the initial buoyancy offset
mean_buoyancy_0_positive = vec(mean(B_pert_0_positive, dims=1))
median_buoyancy_0_positive = vec(median(B_pert_0_positive, dims=1))

x_0_negative = x_cart_full[negative_indices, :]
y_0_negative = y_cart_full[negative_indices, :]
z_0_negative = z_cart_full[negative_indices, :]
B_0_negative = B_full[negative_indices, :]
B_pert_0_negative = B_0_negative[:, :] .- B_0_negative[:, 1]  # Remove the initial buoyancy offset
mean_buoyancy_0_negative = vec(mean(B_pert_0_negative, dims=1))
median_buoyancy_0_negative = vec(median(B_pert_0_negative, dims=1))

close(ds)

# B_0_positive = nothing
# B_0_negative = nothing
# Clean up memory
GC.gc()



###### plot pdf of the particles in the x-y plane as the first panel,
# second panel is the mean buoyancy change over time in x-y plane in the upper portion
# third panel is buoyancy change dZ/dt over time in x-y plane in the lower portion
# Create a 3-panel figure for particle distribution and buoyancy evolution

using StatsBase

# 1. Flatten all particle positions over all time steps
x_flat = vec(x_cart_full)
y_flat = vec(y_cart_full)
x_0_negative_flat = vec(x_0_negative)
y_0_negative_flat = vec(y_0_negative)
x_0_positive_flat = vec(x_0_positive)
y_0_positive_flat = vec(y_0_positive)

# 2. Define bin edges using X_cart_interp_ext and Y_cart_interp_ext
xedges = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=400)
yedges = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=500)

#for initial PDF
xedges_init = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=300)
yedges_init = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=350)


# 3. Compute 2D histogram (counts)
h_all = StatsBase.fit(Histogram, (x_flat, y_flat), (xedges, yedges))

# 4. Normalize to get PDF (probability density)
pdf_all = h_all.weights ./ sum(h_all.weights)

## calculate the initial contour of the particle distribution 
x_flat_init = vec(x_cart_full[:,1])
y_flat_init = vec(y_cart_full[:,1])
h_init = StatsBase.fit(Histogram, (x_flat_init, y_flat_init), (xedges_init, yedges_init))
pdf_init = h_init.weights ./ sum(h_init.weights)
# Find the 95% contour threshold for the initial PDF
pdf_flat = vec(pdf_init)
sorted_pdf = sort(pdf_flat, rev=true)
cumsum_pdf = cumsum(sorted_pdf)
idx_95 = findfirst(cumsum_pdf .>= 0.95)
threshold_95 = sorted_pdf[idx_95]


### Calculate mean buoyancy change over time
w_neg = (z_0_negative[:, 2:end] .- z_0_negative[:, 1:end-1])./(time_full[2]-time_full[1])  # Change in buoyancy over time
w_pos = (z_0_positive[:, 2:end] .- z_0_positive[:, 1:end-1])./(time_full[2]-time_full[1])  # Change in buoyancy over time

using StatsBase
# Prepare accumulator arrays
nx, ny = length(xedges)-1, length(yedges)-1
sum_w_neg = zeros(nx, ny)
count_w_neg = zeros(nx, ny)

# Loop over all particles and time steps (for lower half)
for p in 1:size(x_0_negative, 1)
    for t in 1:size(x_0_negative, 2)-1
        x = x_0_negative[p, t]
        y = y_0_negative[p, t]
        val = w_neg[p, t]
        # Find bin indices
        ix = searchsortedfirst(xedges, x) - 1
        iy = searchsortedfirst(yedges, y) - 1
        if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny
            sum_w_neg[ix, iy] += val
            count_w_neg[ix, iy] += 1
        end
    end
    # @info "Processed particle $p of $(size(x_0_negative, 1)) in lower half"
end
# 4. Compute mean w in each bin (avoid division by zero)
mean_w_neg = fill(NaN, nx, ny)
for ix in 1:nx, iy in 1:ny
    if count_w_neg[ix, iy] > 60
        mean_w_neg[ix, iy] = sum_w_neg[ix, iy] / count_w_neg[ix, iy]
    end
end


# Prepare accumulator arrays
nx, ny = length(xedges)-1, length(yedges)-1
sum_w_pos = zeros(nx, ny)
count_w_pos = zeros(nx, ny)

# Loop over all particles and time steps (for upper half)
for p in 1:size(x_0_positive, 1)
    for t in 1:size(x_0_positive, 2)-1
        x = x_0_positive[p, t]
        y = y_0_positive[p, t]
        val = w_pos[p, t]
        # Find bin indices
        ix = searchsortedfirst(xedges, x) - 1
        iy = searchsortedfirst(yedges, y) - 1
        if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny
            sum_w_pos[ix, iy] += val
            count_w_pos[ix, iy] += 1
        end
    end
end
# Compute mean w in each bin (avoid division by zero)
mean_w_pos = fill(NaN, nx, ny)
for ix in 1:nx, iy in 1:ny
    if count_w_pos[ix, iy] > 60  # filter out bins with too few particles
        mean_w_pos[ix, iy] = sum_w_pos[ix, iy] / count_w_pos[ix, iy]
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
PyPlot.rc("font", size=20)          # controls default text sizes
PyPlot.rc("axes", titlesize=20)     # fontsize of the axes title
PyPlot.rc("axes", labelsize=20)     # fontsize of the x and y labels
PyPlot.rc("xtick", labelsize=20)    # fontsize of the tick labels
PyPlot.rc("ytick", labelsize=20)    # fontsize of the tick labels
PyPlot.rc("legend", fontsize=20)    # legend fontsize

# 1. Particle PDF

pdf_all[pdf_all.==0] .= NaN
levels_log = collect(10 .^ LinRange(-6, -3, 100))

pcm_all = axs[1].contourf(
    xcenters, ycenters, pdf_all',
    levels=levels_log,
    cmap="turbo",
    norm=PyPlot.matplotlib.colors.LogNorm(vmin=levels_log[1], vmax=levels_log[end]),
    shading="auto", alpha=1, edgecolors="black", linewidth=1
)
cont = axs[1].contour(
    cat(topo_x,topo_x,dims=2),
    cat(topo_y,topo_y[:,end].+topo_y,dims=2),
    cat(topo_z,topo_z,dims=2),
    levels=10:200:1500, colors="k", linewidths=0.5
)

# pcm_all = axs[1].contourf(xcenters, ycenters, pdf_all', levels=range(1e-8, stop=10^(-3.4), length=13), vmin=1e-5, vmax=10^(-3.4), cmap="afmhot_r", shading="auto", alpha=1, edgecolors="black", linewidth=1)
# cont = axs[1].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[1].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[1].set_title("All particles", fontsize=20)
axs[1].set_xlabel("x̂ [km]", fontsize=20)
axs[1].set_ylabel("y [km]", fontsize=20)
axs[1].set_ylim(0, 45)
axs[1].set_xlim(-8.5, 44)
axs[1].tick_params(axis="both", labelsize=20)
axs[1].set_aspect("equal", adjustable="box")
axs[1].set_xticks([0, 10, 20, 30, 40])

# 2. Mean w (negative buoyancy change)
pcm = axs[2].pcolormesh(xcenters, ycenters, mean_w_neg', cmap="coolwarm", shading="auto", vmin=-1e-2, vmax=1e-2)
cont = axs[2].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[2].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[2].set_title(L"\in B^p(t_f)<B^p(t_0)", fontsize=20)
axs[2].set_xlabel("x̂ [km]", fontsize=20)
# axs[2].set_ylabel("y [km]", fontsize=16)
axs[2].set_ylim(0, 45)
axs[2].set_xlim(-8.5, 44)
axs[2].tick_params(axis="both", labelsize=20)
axs[2].set_yticklabels([])  # Turn off y-tick labels
axs[2].set_aspect("equal", adjustable="box")
axs[2].set_xticks([0, 10, 20, 30, 40])

# 3. Mean w (positive)
pcm = axs[3].pcolormesh(xcenters, ycenters, mean_w_pos', cmap="coolwarm", shading="auto", vmin=-1e-2, vmax=1e-2)
cont = axs[3].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[3].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
cbar = fig.colorbar(pcm, ax=axs, extend="both", shrink=0.8)
cbar.ax.tick_params(labelsize=20)
cbar.set_label("wᵖ [m s⁻¹]", fontsize=20)
axs[3].set_title(L"\in B^p(t_f)>B^p(t_0)", fontsize=20)
axs[3].set_xlabel("x̂ [km]", fontsize=20)
# axs[3].set_ylabel("y [km]", fontsize=16)
axs[3].set_ylim(0, 45)
axs[3].set_xlim(-8.5, 44)
axs[3].set_xticks([0, 10, 20, 30, 40])
axs[3].tick_params(axis="both", labelsize=20)
axs[3].set_yticklabels([])  # Turn off y-tick labels
axs[3].set_aspect("equal", adjustable="box")

fig.savefig("output/$(simname)/combined_pdf_w_panels_pos_neg_portion.png", dpi=100)
println("Figure saved to output/$(simname)/combined_pdf_w_panels_pos_neg_portion.png")
close(fig)





####### plot pdf using hab to bin  
# plot the positive particles
using StatsBase
using Interpolations
# Compute height above bottom (hab) for each particle at each time
filename_field = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_field = Dataset(filename_field, "r")
zC = ds_field["zC"][:]
Nz = length(zC)
close(ds_field)

# Build a 2D interpolator for bottom elevation
itp_z = Interpolations.interpolate((x_interp_ext, y_interp_ext), Z_cart_interp_ext, Gridded(Interpolations.Linear()))
GC.gc()
n_particles, n_times = size(x_0_positive)
hab_particles = Array{Float64}(undef, n_particles, n_times)
for p in 1:n_particles
    for t in 1:n_times
        x = x_0_positive[p, t]
        y = y_0_positive[p, t]
        z = z_0_positive[p, t]
        # Interpolate bottom elevation at (x, y)
        if x_interp_ext[1] ≤ x ≤ x_interp_ext[end] &&
           y_interp_ext[1] ≤ y ≤ y_interp_ext[end]
            z_bottom = itp_z(x, y)
            hab_particles[p, t] = z - z_bottom
        else
            hab_particles[p, t] = NaN
        end
    end
end
GC.gc()
# Bin edges for x, y
xedges = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=300)
yedges = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=350)

# For initial PDF
xedges_init = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=300)
yedges_init = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=350)

# Flatten all particle positions over all time steps
x_flat = vec(x_0_positive)
y_flat = vec(y_0_positive)

# 2D histogram for PDF
h_pos = StatsBase.fit(Histogram, (x_flat, y_flat), (xedges, yedges))
# 4. Normalize to get PDF (probability density)
pdf_pos = h_pos.weights ./ sum(h_pos.weights)
# Initial PDF for contour
x_flat_init = vec(x_0_positive[:,1])
y_flat_init = vec(y_0_positive[:,1])
h_init = StatsBase.fit(Histogram, (x_flat_init, y_flat_init), (xedges_init, yedges_init))
pdf_init = h_init.weights ./ sum(h_init.weights)
pdf_flat = vec(pdf_init)
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

# Compute mean dZ/dt for positive particles in different hab ranges
hab_edges = [0, 200, 1000]  # [lower, upper]
nx, ny = length(xedges)-1, length(yedges)-1
sum_w_upper = zeros(nx, ny)
count_w_upper = zeros(nx, ny)
sum_w_lower = zeros(nx, ny)
count_w_lower = zeros(nx, ny)

for p in 1:n_particles
    for t in 1:(n_times-1)
        x = x_0_positive[p, t]
        y = y_0_positive[p, t]
        hab = hab_particles[p, t]
        dZ = (z_0_positive[p, t+1] - z_0_positive[p, t]) / (time_full[t+1] - time_full[t])
        ix = searchsortedfirst(xedges, x) - 1
        iy = searchsortedfirst(yedges, y) - 1
        if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny && !isnan(hab)
            if hab ≥ hab_edges[2] && hab < hab_edges[3]   # upper portion
                sum_w_upper[ix, iy] += dZ
                count_w_upper[ix, iy] += 1
            elseif hab ≥ hab_edges[1] && hab < hab_edges[2]   # lower portion
                sum_w_lower[ix, iy] += dZ
                count_w_lower[ix, iy] += 1
            end
        end
    end
end

mean_w_upper = fill(NaN, nx, ny)
mean_w_lower = fill(NaN, nx, ny)
for ix in 1:nx, iy in 1:ny
    if count_w_upper[ix, iy] > 10
        mean_w_upper[ix, iy] = sum_w_upper[ix, iy] / count_w_upper[ix, iy]
    end
    if count_w_lower[ix, iy] > 10
        mean_w_lower[ix, iy] = sum_w_lower[ix, iy] / count_w_lower[ix, iy]
    end
end

using PyPlot

close("all")
fig, axs = subplots(1, 3, figsize=(16, 5), constrained_layout=true)

# Set global font sizes for consistency
PyPlot.rc("font", size=20)
PyPlot.rc("axes", titlesize=20)
PyPlot.rc("axes", labelsize=20)
PyPlot.rc("xtick", labelsize=20)
PyPlot.rc("ytick", labelsize=20)
PyPlot.rc("legend", fontsize=20)

# 1. Particle PDF
pdf_pos[pdf_pos.==0] .= NaN
levels_log = collect(10 .^ LinRange(-6, -3, 100))

pcm_all = axs[1].contourf(
    xcenters, ycenters, pdf_pos',
    levels=levels_log,
    cmap="turbo",
    norm=PyPlot.matplotlib.colors.LogNorm(vmin=levels_log[1], vmax=levels_log[end]),
    shading="auto", alpha=1, edgecolors="black", linewidth=1
)
cont = axs[1].contour(
    cat(topo_x,topo_x,dims=2),
    cat(topo_y,topo_y[:,end].+topo_y,dims=2),
    cat(topo_z,topo_z,dims=2),
    levels=10:200:1500, colors="k", linewidths=0.5
)
axs[1].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[1].set_title(L"∈ ~B^p(t_f)>B^p(t_0)", fontsize=20)
axs[1].set_xlabel("x̂ [km]", fontsize=20)
axs[1].set_ylabel("y [km]", fontsize=20)
axs[1].set_ylim(0, 45)
axs[1].set_xlim(-8.5, 44)
axs[1].tick_params(axis="both", labelsize=20)
axs[1].set_aspect("equal", adjustable="box")
axs[1].set_xticks([0, 10, 20, 30, 40])
cbar = fig.colorbar(pcm_all, ax=axs, label="PDF", extend="both", shrink=0.8)
cbar.ax.tick_params(labelsize=20)
cbar.set_label("PDF", fontsize=20)
cbar.ax.set_yscale("log")  # Ensure log scale for colorbar

# 2. Mean dZ/dt (lower half, hab 0-200m)
pcm = axs[2].pcolormesh(xcenters, ycenters, mean_w_lower', cmap="coolwarm", shading="auto", vmin=-1e-2, vmax=1e-2)
cont = axs[2].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[2].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[2].set_title("HAB 0-$(hab_edges[2]) m", fontsize=20)
axs[2].set_xlabel("x̂ [km]", fontsize=20)
axs[2].set_ylim(0, 45)
axs[2].set_xlim(-8.5, 44)
axs[2].tick_params(axis="both", labelsize=20)
axs[2].set_yticklabels([])
axs[2].set_aspect("equal", adjustable="box")
axs[2].set_xticks([0, 10, 20, 30, 40])
# 3. Mean dZ/dt (upper half, hab 200m-top)
pcm = axs[3].pcolormesh(xcenters, ycenters, mean_w_upper', cmap="coolwarm", shading="auto", vmin=-1e-2, vmax=1e-2)
cont = axs[3].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[3].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")

axs[3].set_title("HAB $(hab_edges[2])-$(hab_edges[3]) m", fontsize=20)
axs[3].set_xlabel("x̂ [km]", fontsize=20)
axs[3].set_ylim(0, 45)
axs[3].set_xlim(-8.5, 44)
axs[3].tick_params(axis="both", labelsize=20)
axs[3].set_yticklabels([])
axs[3].set_aspect("equal", adjustable="box")
axs[3].set_xticks([0, 10, 20, 30, 40])

fig.savefig("output/$(simname)/combined_pdf_w_panels_hab_$(hab_edges[2])_pos_portion.png", dpi=100)
println("Figure saved to output/$(simname)/combined_pdf_w_panels_hab_$(hab_edges[2])_pos_portion.png")
close(fig)






####### plot pdf using hab to bin  

# plot the negative particles
using StatsBase
using Interpolations
# Compute height above bottom (hab) for each particle at each time
filename_field = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_field = Dataset(filename_field, "r")
zC = ds_field["zC"][:]
Nz = length(zC)
close(ds_field)

# Build a 2D interpolator for bottom elevation
itp_z = Interpolations.interpolate((x_interp_ext, y_interp_ext), Z_cart_interp_ext, Gridded(Interpolations.Linear()))
GC.gc()
n_particles, n_times = size(x_0_negative)
hab_particles = Array{Float64}(undef, n_particles, n_times)
for p in 1:n_particles
    for t in 1:n_times
        x = x_0_negative[p, t]
        y = y_0_negative[p, t]
        z = z_0_negative[p, t]
        # Interpolate bottom elevation at (x, y)
        if x_interp_ext[1] ≤ x ≤ x_interp_ext[end] &&
           y_interp_ext[1] ≤ y ≤ y_interp_ext[end]
            z_bottom = itp_z(x, y)
            hab_particles[p, t] = z - z_bottom
        else
            hab_particles[p, t] = NaN
        end
    end
end
GC.gc()
# Bin edges for x, y
xedges = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=300)
yedges = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=350)

# For initial PDF
xedges_init = range(minimum(X_cart_interp_ext[:,1]), maximum(X_cart_interp_ext[:,1]), length=300)
yedges_init = range(minimum(Y_cart_interp_ext[1,:]), maximum(Y_cart_interp_ext[1,:]), length=350)

# Flatten all particle positions over all time steps
x_flat = vec(x_0_negative)
y_flat = vec(y_0_negative)

GC.gc()

# 2D histogram for PDF
h_neg = StatsBase.fit(Histogram, (x_flat, y_flat), (xedges, yedges))
pdf_neg = h_neg.weights ./ sum(h_neg.weights)

# Initial PDF for contour
x_flat_init = vec(x_0_negative[:,1])
y_flat_init = vec(y_0_negative[:,1])
h_init = StatsBase.fit(Histogram, (x_flat_init, y_flat_init), (xedges_init, yedges_init))
pdf_init = h_init.weights ./ sum(h_init.weights)
pdf_flat = vec(pdf_init)
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

# Compute mean dZ/dt for positive particles in different hab ranges
hab_edges = [0, 200, 1000]  # [lower, upper]
nx, ny = length(xedges)-1, length(yedges)-1
sum_w_upper = zeros(nx, ny)
count_w_upper = zeros(nx, ny)
sum_w_lower = zeros(nx, ny)
count_w_lower = zeros(nx, ny)

for p in 1:n_particles
    for t in 1:(n_times-1)
        x = x_0_negative[p, t]
        y = y_0_negative[p, t]
        hab = hab_particles[p, t]
        dZ = (z_0_negative[p, t+1] - z_0_negative[p, t]) / (time_full[t+1] - time_full[t])
        ix = searchsortedfirst(xedges, x) - 1
        iy = searchsortedfirst(yedges, y) - 1
        if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny && !isnan(hab)
            if hab ≥ hab_edges[2] && hab < hab_edges[3]   # upper portion
                sum_w_upper[ix, iy] += dZ
                count_w_upper[ix, iy] += 1
            elseif hab ≥ hab_edges[1] && hab < hab_edges[2]   # lower portion
                sum_w_lower[ix, iy] += dZ
                count_w_lower[ix, iy] += 1
            end
        end
    end
end

mean_w_upper = fill(NaN, nx, ny)
mean_w_lower = fill(NaN, nx, ny)
for ix in 1:nx, iy in 1:ny
    if count_w_upper[ix, iy] > 10
        mean_w_upper[ix, iy] = sum_w_upper[ix, iy] / count_w_upper[ix, iy]
    end
    if count_w_lower[ix, iy] > 10
        mean_w_lower[ix, iy] = sum_w_lower[ix, iy] / count_w_lower[ix, iy]
    end
end

using PyPlot

close("all")
fig, axs = subplots(1, 3, figsize=(16, 5), constrained_layout=true)

# Set global font sizes for consistency
PyPlot.rc("font", size=20)
PyPlot.rc("axes", titlesize=20)
PyPlot.rc("axes", labelsize=20)
PyPlot.rc("xtick", labelsize=20)
PyPlot.rc("ytick", labelsize=20)
PyPlot.rc("legend", fontsize=20)

# 1. Particle PDF (negative particles)
pdf_neg[pdf_neg.==0] .= NaN
levels_log = collect(10 .^ LinRange(-6, -3, 100))

pcm_all = axs[1].contourf(
    xcenters, ycenters, pdf_neg',
    levels=levels_log,
    cmap="turbo",
    norm=PyPlot.matplotlib.colors.LogNorm(vmin=levels_log[1], vmax=levels_log[end]),
    shading="auto", alpha=1, edgecolors="black", linewidth=1
)
cont = axs[1].contour(
    cat(topo_x,topo_x,dims=2),
    cat(topo_y,topo_y[:,end].+topo_y,dims=2),
    cat(topo_z,topo_z,dims=2),
    levels=10:200:1500, colors="k", linewidths=0.5
)
axs[1].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[1].set_title(L"∈ ~B^p(t_f)<B^p(t_0)", fontsize=20)
axs[1].set_xlabel("x̂ [km]", fontsize=20)
axs[1].set_ylabel("y [km]", fontsize=20)
axs[1].set_ylim(0, 45)
axs[1].set_xlim(-8.5, 44)
axs[1].tick_params(axis="both", labelsize=20)
axs[1].set_aspect("equal", adjustable="box")
axs[1].set_xticks([0, 10, 20, 30, 40])
# cbar = fig.colorbar(pcm_all, ax=axs, label="PDF", extend="both", shrink=0.8)
# cbar.ax.tick_params(labelsize=20)
# cbar.set_label("PDF", fontsize=20)
# cbar.ax.set_yscale("log")  # Ensure log scale for colorbar

# 2. Mean dZ/dt (lower half, hab 0-40m)
pcm = axs[2].pcolormesh(xcenters, ycenters, mean_w_lower', cmap="coolwarm", shading="auto", vmin=-1e-2, vmax=1e-2)
cont = axs[2].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[2].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
axs[2].set_title("HAB 0-$(hab_edges[2]) m", fontsize=20)
axs[2].set_xlabel("x̂ [km]", fontsize=20)
axs[2].set_ylim(0, 45)
axs[2].set_xlim(-8.5, 44)
axs[2].tick_params(axis="both", labelsize=20)
axs[2].set_yticklabels([])
axs[2].set_aspect("equal", adjustable="box")
axs[2].set_xticks([0, 10, 20, 30, 40])

# 3. Mean dZ/dt (upper half, hab 40-200m)
pcm = axs[3].pcolormesh(xcenters, ycenters, mean_w_upper', cmap="coolwarm", shading="auto", vmin=-1e-2, vmax=1e-2)
cont = axs[3].contour(cat(topo_x,topo_x,dims=2), cat(topo_y,topo_y[:,end].+topo_y,dims=2), cat(topo_z,topo_z,dims=2), levels=10:200:1500, colors="k", linewidths=0.5)
axs[3].contour(xcenters_init, ycenters_init, pdf_init', levels=[threshold_95], colors="green", linewidths=1.5, linestyles="-")
cbar3 = fig.colorbar(pcm, ax=axs, extend="both", shrink=0.8)
cbar3.ax.tick_params(labelsize=20)
cbar3.set_label("10⁻⁹ × wᵖ [m s⁻³]", fontsize=20)
axs[3].set_title("HAB $(hab_edges[2])-$(hab_edges[3]) m", fontsize=20)
axs[3].set_xlabel("x̂ [km]", fontsize=20)
axs[3].set_ylim(0, 45)
axs[3].set_xlim(-8.5, 44)
axs[3].tick_params(axis="both", labelsize=20)
axs[3].set_yticklabels([])
axs[3].set_aspect("equal", adjustable="box")
axs[3].set_xticks([0, 10, 20, 30, 40])

fig.savefig("output/$(simname)/combined_pdf_w_panels_hab_$(hab_edges[2])_neg_portion.png", dpi=100)
println("Figure saved to output/$(simname)/combined_pdf_w_panels_hab_$(hab_edges[2])_neg_portion.png")
close(fig)