using NCDatasets
using NaNStatistics
using Printf
using Statistics
using CairoMakie
using ColorSchemes
#### Plot the mean buoyancy evolution with standard deviation bands
simname = "tilt"
z_center_particle = 1000  # Center height of particles
output_file = string("output/", simname, "/concatenated_particle_data_z", z_center_particle, "_all.nc")
ds = NCDataset(output_file, "r")
x_cart_full = ds["main_x_cartesian"][:,:]
# y_cart_full = ds["main_y_cartesian"][:,:]
z_cart_full = ds["main_z_cartesian"][:, :]
B_full = ds["main_buoyancy"][:, :]
time_full = ds["main_time"][:]
close(ds)

# find particles lower than 1000 m

b_begin = B_full[:, 1]  # Initial buoyancy positions
z_begin = z_cart_full[:, 1]  # Initial z positions
x_begin = x_cart_full[:, 1]  # Initial x positions
b_final = B_full[:, end]  # Initial buoyancy positions
z_final = z_cart_full[:, end]  # Initial z positions
x_final = x_cart_full[:, end]  # Initial x positions
###########
positive_B_indices = findall(B_full[:, end] .> b_begin)
negative_B_indices = findall(B_full[:, end] .< b_begin)
positive_Z_indices = findall(z_cart_full[:, end] .> z_begin)
negative_Z_indices = findall(z_cart_full[:, end] .< z_begin)
positive_X_indices = findall(x_cart_full[:, end] .> x_begin)
negative_X_indices = findall(x_cart_full[:, end] .< x_begin)

x_0_positive = x_cart_full[positive_X_indices, :]
z_0_positive = z_cart_full[positive_Z_indices, :]
B_0_positive = B_full[positive_B_indices, :]
B_pert_0_positive = B_0_positive[:, :] .- B_0_positive[:, 1]  # Remove the initial buoyancy offset
x_pert_0_positive = x_0_positive[:, :] .- x_0_positive[:, 1]  # Remove the initial x offset
z_pert_0_positive = z_0_positive[:, :] .- z_0_positive[:, 1]  # Remove the initial buoyancy offset
median_buoyancy_0_positive = vec(median(B_pert_0_positive, dims=1))
# std_buoyancy_0_positive = vec(std(B_pert_0_positive, dims=1))
median_z_0_positive = vec(median(z_pert_0_positive, dims=1))
median_x_0_positive = vec(median(x_pert_0_positive, dims=1))
# std_z_0_positive = vec(std(z_pert_0_positive, dims=1))

x_0_negative = x_cart_full[negative_X_indices, :]
z_0_negative = z_cart_full[negative_Z_indices, :]
B_0_negative = B_full[negative_B_indices, :]
B_pert_0_negative = B_0_negative[:, :] .- B_0_negative[:, 1]  # Remove the initial buoyancy offset
x_pert_0_negative = x_0_negative[:, :] .- x_0_negative[:, 1]  # Remove the initial x offset
z_pert_0_negative = z_0_negative[:, :] .- z_0_negative[:, 1]  # Remove the initial buoyancy offset
median_buoyancy_0_negative = vec(median(B_pert_0_negative, dims=1))
# std_buoyancy_0_negative = vec(std(B_pert_0_negative, dims=1))
median_z_0_negative = vec(median(z_pert_0_negative, dims=1))
median_x_0_negative = vec(median(x_pert_0_negative, dims=1))
# std_z_0_negative = vec(std(z_pert_0_negative, dims=1))

# include all particles
x_full_all = x_cart_full[:, :]
z_full_all = z_cart_full[:, :]
B_full_all = B_full[:, :]
B_pert_all = B_full_all[:, :] .- B_full_all[:, 1]  # Remove the initial buoyancy offset
B_full_all=nothing
B_full=nothing
GC.gc()
x_pert_all = x_full_all[:, :] .- x_full_all[:, 1]  # Remove the initial x offset
x_full_all=nothing
x_cart_full=nothing
GC.gc()
z_pert_all = z_full_all[:, :] .- z_full_all[:, 1]  # Remove the initial z offset
z_full_all=nothing
z_cart_full=nothing
GC.gc()
median_buoyancy_0_all = vec(median(B_pert_all, dims=1))
# std_buoyancy_0_all = vec(std(B_pert_all, dims=1))
median_z_0_all = vec(median(z_pert_all, dims=1))
# std_z_0_all = vec(std(z_pert_all, dims=1))
median_x_0_all = vec(median(x_pert_all, dims=1))
# std_x_0_all = vec(std(x_pert_all, dims=1))

# calculate the ratio of the number of particles that increased buoyancy to the number of decreased buoyancy at each timesteps
number_pos = zeros(Int, size(B_pert_all, 2))
number_negative = zeros(Int, size(B_pert_all, 2))
for i in 1:size(B_pert_all, 2)
        number_pos[i] = length(findall(B_pert_all[:, i] .> 0))
        number_negative[i] = length(findall(B_pert_all[:, i] .< 0))
end
ratio_B = number_pos[2:end] ./ number_negative[2:end]

# calculate the ratio of the number of particles that increased height to the number of decreased height at each timesteps
number_pos = zeros(Int, size(z_pert_all, 2))
number_negative = zeros(Int, size(z_pert_all, 2))
for i in 1:size(z_pert_all, 2)
        number_pos[i] = length(findall(z_pert_all[:, i] .> 0))
        number_negative[i] = length(findall(z_pert_all[:, i] .< 0))
end
ratio_Z = number_pos[2:end] ./ number_negative[2:end]

# calculate the ratio of the number of particles that increased x position to the number of decreased x position at each timesteps
number_pos = zeros(Int, size(x_pert_all, 2))
number_negative = zeros(Int, size(x_pert_all, 2))
for i in 1:size(x_pert_all, 2)
        number_pos[i] = length(findall(x_pert_all[:, i] .> 0))
        number_negative[i] = length(findall(x_pert_all[:, i] .< 0))
end
ratio_X = number_pos[2:end] ./ number_negative[2:end]


using PyPlot

# Plot the ratio of positive to negative particles over time
close("all")
plt.figure(figsize=(6, 3))
plt.plot((time_full[2:end] .- time_full[1]) ./ (2*pi/1.4e-4), ratio, color="purple", linewidth=2)
plt.xlabel("Time since particle release [Tidal period]", fontsize=16)
plt.ylabel(L"N_+^p~/~N_-^p", fontsize=16)
# plt.title("Ratio of Positive to Negative Particles Over Time", fontsize=16)
plt.grid(true, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.xlim(0, (time_full[end]-time_full[1]) / (2*pi/1.4e-4))  # Set x-axis limit to max time in hours
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
output_file = string("output/", simname, "/ratio_positive_negative_z", z_center_particle, ".png")
@info "Saving ratio plot to $output_file"
plt.savefig(output_file, bbox_inches="tight")
plt.close()

# b_begin = B_full[:, 1]  # Initial z positions
# # z_final = z_cart_full[:, 1]  # Initial z positions
# b_all = maximum(b_begin)   # include all particles
# b₀ = median(b_begin)         # median particle buoyancy
# b1 = b₀ - std(b_begin)     # 1 std below mean buoyancy

# Filter all arrays to keep only valid particles
# include all particles

# only includes particles below 1 std below mean buoyancy
# valid_indices = findall(b_begin .<= b1)
# B_full_1 = B_full[valid_indices, :]
# B_pert_1 = B_full_1[:,:] .- B_full_1[:,1]  # Remove the initial buoyancy offset
# all_mean_buoyancy_1 = vec(mean(B_pert_1, dims=1))
# all_median_buoyancy_1 = vec(median(B_pert_1, dims=1))
# all_std_buoyancy_1 = vec(std(B_pert_1, dims=1))
# # only includes particles below mean buoyancy
# valid_indices = findall(b_begin .<= b₀)
# B_full_0 = B_full[valid_indices, :]
# B_pert_0 = B_full_0[:,:] .- B_full_0[:,1]  # Remove the initial buoyancy offset
# all_mean_buoyancy_0 = vec(mean(B_pert_0, dims=1))
# all_median_buoyancy_0 = vec(median(B_pert_0, dims=1))
# all_std_buoyancy_0 = vec(std(B_pert_0, dims=1))
# # only includes particles above mean buoyancy
# valid_indices = findall(b_begin .>= b₀)
# B_full_0_up = B_full[valid_indices, :]
# B_pert_0_up = B_full_0_up[:,:] .- B_full_0_up[:,1]  # Remove the initial buoyancy offset
# all_mean_buoyancy_0_up = vec(mean(B_pert_0_up, dims=1))
# all_median_buoyancy_0_up = vec(median(B_pert_0_up, dims=1))
# all_std_buoyancy_0_up = vec(std(B_pert_0_up, dims=1))


# B_pert_1=nothing
# B_pert_0=nothing
# B_full_1=nothing
# B_full_0=nothing
# B_full_all=nothing
# B_full_0_up=nothing
# B_pert_0_up=nothing
# Clean up memory
GC.gc()


# Create the figure
fig = CairoMakie.Figure(resolution=(900, 600), fontsize=16)
ax = Axis(fig[1, 1],
        xlabel="Time (hours)",
        ylabel="Buoyancy (m/s²)",
        title="Median Particle Buoyancy Evolution",
        xminorticksvisible=true,
        yminorticksvisible=true,
        xminorticks=IntervalsBetween(5),
        yminorticks=IntervalsBetween(4))

# Plot mean buoyancy with standard deviation band
band!(ax, time_full, median_buoyancy_0_all .- std_buoyancy_0_all, median_buoyancy_0_all .+ std_buoyancy_0_all,
        color=:blue,
        alpha=0.3,
        label="± 1 Standard Deviation")

lines!(ax, time_full, median_buoyancy_0_all,
        linewidth=3,
        color=:blue,
        label="Mean Buoyancy")
# Plot individual particle trajectories with high transparency
# for i in 1:1000:size(B_full, 1)
#     lines!(ax, time_full, B_full[i, :],
#         linewidth=1,
#         color=(:gray, 0.1))  # Very transparent gray
# end

# Add legend
axislegend(ax, position=:lt)

# Save the figure
output_file = string("output/", simname, "/median_buoyancy_evolution_z", z_center_particle, ".png")
@info "Saving figure to $output_file"
save(output_file, fig)



# pdf time series
using PyPlot
using StatsBase  # For histogram functions

# Calculate PDF of B_pert at each time step
n_timesteps = length(time_full)
n_bins = 400

# Create consistent bin edges for all timesteps
bin_edges = range(-4e-5, 4e-5, length=n_bins + 1)
bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2  # Midpoints of bin edges

# Initialize PDF data array
pdf_data = zeros(n_bins, n_timesteps)

for t in 1:n_timesteps
        # Get buoyancy perturbations at this time
        b_pert_t = B_pert_all[:, t]

        # Create histogram with consistent bins
        h = StatsBase.fit(Histogram, b_pert_t, bin_edges)

        # Normalize to get PDF
        bin_width = step(bin_edges)
        ind = findall(h.weights .< 100)
        h.weights[ind] .= 0  # Set weights below threshold to zero
        pdf_data[:, t] = h.weights ./ (sum(h.weights) * bin_width)
end
pdf_data[pdf_data .< 1e4] .= NaN  # Set  small values to NaN for better visualization
# B_pert_all = nothing  # Clean up memory
GC.gc()
# Create pcolor plot
cmap = plt.get_cmap("managua").copy()
cmap.set_under("white")
close("all")
PyPlot.figure(figsize=(10, 5))
# Use original pdf_data without log transformation
plt.pcolor((time_full[:].-time_full[1])/(2*pi/1.4e-4), bin_centers * 1e5, pdf_data/1e4, cmap="managua", shading="nearest")
cbar = plt.colorbar(label="PDF")
cbar.ax.set_ylabel("10⁴ × PDF", fontsize=18)  # Set label font size
cbar.ax.tick_params(labelsize=18)  # Major tick labels
cbar.ax.tick_params(which="minor", labelsize=18)  # Minor tick labels
# cbar._set_scale("linear")  # Set colorbar to log scale
plt.xlabel("Time since particle release [Tidal period]", fontsize=18)
plt.ylabel("ΔBₚ [10⁻⁵ m s⁻²]", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim(451,451.5)
plt.clim(1,10)
plt.ylim(-3,3)
# Plot median lines
plt.plot((time_full[:].-time_full[1])/(2*pi/1.4e-4), median_buoyancy_0_all * 1e5, color="red", linewidth=3, label="all")
plt.plot((time_full[:].-time_full[1])/(2*pi/1.4e-4), median_buoyancy_0_negative * 1e5, color="yellow", linewidth=2.5, linestyle="-.", label=L"B^p(t_f) < B^p(t_0)")
plt.plot((time_full[:].-time_full[1])/(2*pi/1.4e-4), median_buoyancy_0_positive * 1e5, color="lime", linewidth=2.5, linestyle="--", label=L"B^p(t_f) > B^p(t_0)")

plt.axhline(0, color="black", linewidth=1, linestyle="-")  # Horizontal line at 0
plt.legend(fontsize=16, loc="lower left", facecolor=(0.9, 0.9, 0.9), edgecolor="black")
# plt.grid(true, which="both", linestyle="--", linewidth=0.5)
# Save figure
output_file = string("output/", simname, "/buoyancy_pdf_evolution_z_different_cutoff_", z_center_particle, ".png")
@info "Saving PDF evolution figure to $output_file"
plt.savefig(output_file, bbox_inches="tight")



### Combined PDF and Ratio Plot for Publication

using PyPlot
using StatsBase  # For histogram functions

# Calculate PDF of B_pert at each time step
n_timesteps = length(time_full)
n_bins = 400

# Create consistent bin edges for all timesteps
bin_edges = range(-4e-5, 4e-5, length=n_bins + 1)
bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2  # Midpoints of bin edges

# Initialize PDF data array
pdf_data = zeros(n_bins, n_timesteps)

for t in 1:n_timesteps
        b_pert_t = B_pert_all[:, t]
        h = StatsBase.fit(Histogram, b_pert_t, bin_edges)
        bin_width = step(bin_edges)
        ind = findall(h.weights .< 100)
        h.weights[ind] .= 0
        pdf_data[:, t] = h.weights ./ (sum(h.weights) * bin_width)
end
pdf_data[pdf_data .< 1e4] .= NaN
GC.gc()

# Prepare time axis for both plots
time_tidal = (time_full[:] .- time_full[1]) / (2*pi/1.4e-4)
time_tidal_ratio = (time_full[2:end] .- time_full[1]) / (2*pi/1.4e-4)
# Create figure with two panels using PyPlot's subplots
fig, axarr = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw=Dict("height_ratios" => [2, 1]), sharex=true,constrained_layout=true)

# Panel (a): PDF pcolor plot (top)
ax_pdf = axarr[1]
cmap = plt.get_cmap("managua").copy()
cmap.set_under("white")
pcm = ax_pdf.pcolor(time_tidal, bin_centers * 1e5, pdf_data/1e4, cmap="managua", shading="nearest")
# Place colorbar outside the right of panel a to avoid squeezing
cbar = fig.colorbar(pcm, ax=ax_pdf, label="PDF", pad=0.02, fraction=0.046, location="right", extend="up")
cbar.ax.set_ylabel("10⁴ × PDF", fontsize=18)
cbar.ax.tick_params(labelsize=16)
cbar.ax.tick_params(which="minor", labelsize=16)
# cbar._set_scale("linear")
ax_pdf.set_ylabel(L"\Delta B_p ~\mathrm{[10⁻⁵~ m~ s⁻²]}", fontsize=18)
ax_pdf.tick_params(labelsize=16)
ax_pdf.grid(true, linestyle="-", alpha=0.75)
ax_pdf.set_xlim(0, time_tidal[end])
ax_pdf.set_ylim(-3, 3)
pcm.set_clim(1, 10)
ax_pdf.axhline(0, color="black", linewidth=1, linestyle="-")

# Superimpose median lines
ax_pdf.plot(time_tidal, median_buoyancy_0_all * 1e5, color="#0072B2", linewidth=3, label="All", zorder=10)  # Blue
ax_pdf.plot(time_tidal, median_buoyancy_0_negative * 1e5, color="#D55E00", linewidth=2.5, linestyle="-", label=L"$\in B^p(t_f) < B^p(t_0)$", zorder=10)  # Orange
ax_pdf.plot(time_tidal, median_buoyancy_0_positive * 1e5, color="#009E73", linewidth=2.5, linestyle="-", label=L"$\in B^p(t_f) > B^p(t_0)$", zorder=10)  # Green
ax_pdf.legend(fontsize=16, loc="lower left", frameon=false)
ax_pdf.text(0.02, 0.92, "(a)", transform=ax_pdf.transAxes, fontsize=16, va="top", ha="left")

# Panel (b): Ratio plot (bottom)
ax_ratio = axarr[2]
ax_ratio.plot(time_tidal_ratio, ratio, color="purple", linewidth=2.5, label=L"$N_+^p~/~N_-^p$")
ax_ratio.set_ylabel(L"$N_+^p~/~N_-^p$", fontsize=18)
ax_ratio.grid(true, linestyle="-", alpha=0.75)
ax_ratio.tick_params(labelsize=16)
ax_ratio.set_xlabel("Time since particles release [Tidal period]", fontsize=18)
ax_ratio.set_xlim(0, time_tidal[end])
# ax_ratio.legend(fontsize=16, loc="upper right", facecolor=(0.9, 0.9, 0.9), edgecolor="black")
ax_ratio.text(0.02, 0.92, "(b)", transform=ax_ratio.transAxes, fontsize=16, va="top", ha="left")

# Save combined figure
output_file = string("output/", simname, "/combined_pdf_ratio_evolution_z", z_center_particle, ".png")
@info "Saving combined PDF and ratio figure to $output_file"
plt.savefig(output_file, bbox_inches="tight")
plt.close()





### A new version of the plot including ΔBᵖ and Δzᵖ: Combine PDF and Ratio Plot and scatterplot of the two for Publication

using PyPlot
using StatsBase  # For histogram functions

# Calculate PDF of B_pert at each time step
n_timesteps = length(time_full)
n_bins = 400

# Create consistent bin edges for all timesteps
bin_edges_B = range(-4e-5, 4e-5, length=n_bins + 1)
bin_centers_B = (bin_edges_B[1:end-1] .+ bin_edges_B[2:end]) ./ 2  # Midpoints of bin edges
bin_edges_Z = range(-700, 700, length=n_bins + 1)
bin_centers_Z = (bin_edges_Z[1:end-1] .+ bin_edges_Z[2:end]) ./ 2  # Midpoints of bin edges
bin_edges_X = range(-4000, 40000, length=n_bins + 1)
bin_centers_X = (bin_edges_X[1:end-1] .+ bin_edges_X[2:end]) ./ 2  # Midpoints of bin edges
# Initialize PDF data array
pdf_data_B = zeros(n_bins, n_timesteps)
pdf_data_Z = zeros(n_bins, n_timesteps)
pdf_data_X = zeros(n_bins, n_timesteps)

for t in 1:n_timesteps
        b_pert_t = B_pert_all[:, t]
        z_pert_t = z_pert_all[:,t]
        x_pert_t = x_pert_all[:,t]
        h_b = StatsBase.fit(Histogram, b_pert_t, bin_edges_B)
        h_z = StatsBase.fit(Histogram, z_pert_t, bin_edges_Z)
        h_x = StatsBase.fit(Histogram, x_pert_t, bin_edges_X)
        bin_width_B = step(bin_edges_B)
        bin_width_Z = step(bin_edges_Z)
        bin_width_X = step(bin_edges_X)
        ind = findall(h_b.weights .< 100)
        h_b.weights[ind] .= 0
        pdf_data_B[:, t] = h_b.weights ./ (sum(h_b.weights) * bin_width_B)
        ind = findall(h_z.weights .< 100)
        h_z.weights[ind] .= 0
        pdf_data_Z[:, t] = h_z.weights ./ (sum(h_z.weights) * bin_width_Z)
        ind = findall(h_x.weights .< 100)
        h_x.weights[ind] .= 0
        pdf_data_X[:, t] = h_x.weights ./ (sum(h_x.weights) * bin_width_X)
end
pdf_data_B[pdf_data_B .< 1e4] .= NaN
pdf_data_Z[pdf_data_Z .< nanmaximum(pdf_data_Z)*0.002] .= NaN
# pdf_data_X[pdf_data_X .< 1e4] .= NaN
GC.gc()

# Prepare time axis for both plots
time_tidal = (time_full[:] .- time_full[1]) / (2*pi/1.4e-4)
time_tidal_ratio = (time_full[2:end] .- time_full[1]) / (2*pi/1.4e-4)
# Create figure with two panels using PyPlot's subplots
close("all")
fig, axarr = plt.subplots(3, 2, figsize=(12, 8), gridspec_kw=Dict("height_ratios" => [5, 4, 4]), sharex=false,constrained_layout=true)

# Panel (a): PDF pcolor plot (top)
ax_pdf = axarr[1]
cmap = plt.get_cmap("managua").copy()
cmap.set_under("white")
pcm = ax_pdf.pcolor(time_tidal, bin_centers_B * 1e5, pdf_data_B/1e4, cmap="managua", shading="nearest")
# Place colorbar outside the right of panel a to avoid squeezing
cbar = fig.colorbar(pcm, ax=ax_pdf, label="PDF", pad=0.02, fraction=0.046, location="right", extend="max")
cbar.ax.set_ylabel("10⁴ × PDF", fontsize=18)
cbar.ax.tick_params(labelsize=16)
cbar.ax.tick_params(which="minor", labelsize=16)
# cbar._set_scale("linear")
ax_pdf.set_ylabel(L"\Delta B_p ~\mathrm{[10⁻⁵~ m~ s⁻²]}", fontsize=18)
ax_pdf.tick_params(labelsize=16)
ax_pdf.grid(true, linestyle="-", alpha=0.75)
ax_pdf.set_xlim(0, time_tidal[end])
ax_pdf.set_ylim(-3, 3)
pcm.set_clim(1, 10)
ax_pdf.axhline(0, color="black", linewidth=1, linestyle="-")

# Superimpose median lines
ax_pdf.plot(time_tidal, median_buoyancy_0_all * 1e5, color="#0072B2", linewidth=3, label="All", zorder=10)  # Blue
ax_pdf.plot(time_tidal, median_buoyancy_0_negative * 1e5, color="#D55E00", linewidth=2.5, linestyle="-", label=L"$\in B^p(t_f) < B^p(t_0)$", zorder=10)  # Orange
ax_pdf.plot(time_tidal, median_buoyancy_0_positive * 1e5, color="#009E73", linewidth=2.5, linestyle="-", label=L"$\in B^p(t_f) > B^p(t_0)$", zorder=10)  # Green
ax_pdf.legend(fontsize=14, loc="lower left", frameon=false)
ax_pdf.text(0.02, 0.92, "(a)", transform=ax_pdf.transAxes, fontsize=16, va="top", ha="left")

# Panel (b): PDF pcolor plot
ax_pdf_Z = axarr[4]
cmap = plt.get_cmap("managua").copy()
cmap.set_under("white")
pcm = ax_pdf_Z.pcolor(time_tidal, bin_centers_Z, pdf_data_Z, cmap="managua", shading="nearest")
# Place colorbar outside the right of panel a to avoid squeezing
cbar = fig.colorbar(pcm, ax=ax_pdf_Z, label="PDF", pad=0.02, fraction=0.046, location="right", extend="max")
cbar.ax.set_ylabel("PDF", fontsize=18)
cbar.ax.tick_params(labelsize=16)
cbar.ax.tick_params(which="minor", labelsize=16)
# cbar._set_scale("linear")
ax_pdf_Z.set_ylabel(L"\Delta Z_p ~\mathrm{[m]}", fontsize=18)
ax_pdf_Z.tick_params(labelsize=16)
ax_pdf_Z.grid(true, linestyle="-", alpha=0.75)
ax_pdf_Z.set_xlim(0, time_tidal[end])
ax_pdf_Z.set_ylim(-200, 200)
pcm.set_clim(0, nanmaximum(pdf_data_Z)*0.05)
ax_pdf_Z.axhline(0, color="black", linewidth=1, linestyle="-")

# Superimpose median lines
ax_pdf_Z.plot(time_tidal, median_z_0_all, color="#0072B2", linewidth=3, label="All", zorder=10)  # Blue
ax_pdf_Z.plot(time_tidal, median_z_0_negative, color="#D55E00", linewidth=2.5, linestyle="-", label=L"$\in Z^p(t_f) < Z^p(t_0)$", zorder=10)  # Orange
ax_pdf_Z.plot(time_tidal, median_z_0_positive, color="#009E73", linewidth=2.5, linestyle="-", label=L"$\in Z^p(t_f) > Z^p(t_0)$", zorder=10)  # Green
ax_pdf_Z.legend(fontsize=14, loc="lower left", frameon=false)
ax_pdf_Z.text(0.02, 0.92, "(b)", transform=ax_pdf_Z.transAxes, fontsize=16, va="top", ha="left")



# Panel (c): Ratio plot (bottom)
ax_ratio = axarr[2]
ax_ratio.plot(time_tidal_ratio, ratio_B, color="purple", linewidth=2.5, label=L"$N_+^p~/~N_-^p$")
ax_ratio.set_ylabel(L"$N_+^p~/~N_-^p$", fontsize=18)
ax_ratio.grid(true, linestyle="-", alpha=0.75)
ax_ratio.tick_params(labelsize=16)
ax_ratio.set_xlabel(L"$t-t_0$ [Tidal period]", fontsize=18)
ax_ratio.set_xlim(0, time_tidal[end])
ax_ratio.axhline(1, color="black", linewidth=1, linestyle="--")
# ax_ratio.set_ylim(-0.1, maximum(ratio_B))

# ax_ratio.legend(fontsize=16, loc="upper right", facecolor=(0.9, 0.9, 0.9), edgecolor="black")
ax_ratio.text(0.02, 0.92, "(c)", transform=ax_ratio.transAxes, fontsize=16, va="top", ha="left")

# Panel (d): Ratio plot (bottom)
ax_ratio_Z = axarr[5]
ax_ratio_Z.plot(time_tidal_ratio, ratio_Z, color="purple", linewidth=2.5, label=L"$N_+^p~/~N_-^p$")
ax_ratio_Z.set_ylabel(L"$N_+^p~/~N_-^p$", fontsize=18)
ax_ratio_Z.grid(true, linestyle="-", alpha=0.75)
ax_ratio_Z.tick_params(labelsize=16)
ax_ratio_Z.set_xlabel(L"$t-t_0$ [Tidal period]", fontsize=18)
ax_ratio_Z.set_xlim(0, time_tidal[end])
ax_ratio_Z.set_ylim(-0.1, maximum(ratio_Z))
ax_ratio_Z.axhline(1, color="black", linewidth=1, linestyle="--")
# ax_ratio.legend(fontsize=16, loc="upper right", facecolor=(0.9, 0.9, 0.9), edgecolor="black")
ax_ratio_Z.text(0.02, 0.92, "(d)", transform=ax_ratio_Z.transAxes, fontsize=16, va="top", ha="left")


# Panel (e): ΔBᵖ vs ΔZᵖ scatter plot
ax_relation = axarr[3]
ix = 1:1:size(z_pert_all, 1)  # sample indices for particles
it = size(z_pert_all, 2):size(z_pert_all, 2)  # time index

z_vals = z_pert_all[ix, it]
b_vals = B_pert_all[ix, it] * 1e5
x_vals = x_pert_all[ix, it]

sc = ax_relation.scatter(z_vals, b_vals, c=x_vals, cmap="gist_rainbow_r", alpha=0.5, edgecolors="none", s=8)
cb = plt.colorbar(sc, ax=ax_relation)
cb.set_label("ΔXᵖ [m]", fontsize=18)  # Increase colorbar label font size
cb.ax.tick_params(labelsize=18)
ax_relation.set_ylabel(L"\Delta B_p ~\mathrm{[10⁻⁵~ m~ s⁻²]}", fontsize=18)
ax_relation.set_xlabel(L"\Delta Z^p ~\mathrm{[m]}", fontsize=18)
ax_relation.grid(true, linestyle="--", alpha=0.5)
ax_relation.tick_params(labelsize=16)
ax_relation.set_xlim(minimum(z_pert_all), maximum(z_pert_all))
ax_relation.text(0.02, 0.92, "(e)", transform=ax_relation.transAxes, fontsize=18, va="top", ha="left")
ax_relation.axhline(0, color="black", linewidth=1, linestyle="-")
ax_relation.axvline(0, color="black", linewidth=1, linestyle="-")

axarr[6].axis("off")
# Panel (f): ΔBᵖ vs ΔXᵖ scatter plot
# ax_relation_X = axarr[6]
# ax_relation_X.scatter(x_pert_all[1:1000:end, 2:end][:], B_pert_all[1:1000:end, 2:end][:], color="purple", alpha=0.3, edgecolors="none")
# ax_relation_X.set_ylabel(L"$\Delta B^p$", fontsize=18)
# ax_relation_X.set_xlabel(L"$\Delta X^p$", fontsize=18)
# ax_relation_X.grid(true, linestyle="--", alpha=0.5)
# ax_relation_X.tick_params(labelsize=16)
# ax_relation_X.set_xlim(minimum(x_pert_all), maximum(x_pert_all))
# ax_relation_X.text(0.02, 0.92, "(f)", transform=ax_relation_X.transAxes, fontsize=18, va="top", ha="left")


# Save combined figure
output_file = string("output/", simname, "/combined_pdf_ratio_evolution_Z_B_z", z_center_particle, "_final.png")
@info "Saving combined PDF and ratio figure to $output_file"
plt.savefig(output_file, bbox_inches="tight")
plt.close()




### Same as above but plot pdf of ΔBᵖ, ΔZᵖ with a threshold

# compute 2D PDF
using StatsBase

# Select time index and flatten arrays
it = size(z_pert_all, 2)  # last time index
ind1 = x_final[:] .< 1e4  # only include particles that does not pass 15 km in x direction
ind2 = 38e3 .> x_final[:] .> 35e3  # only include particles that move pass 15 km in x direction
z_flat_before_sill = vec(z_pert_all[ind1, it])
b_flat_before_sill = vec(B_pert_all[ind1, it] * 1e5)
z_flat_after_sill = vec(z_pert_all[ind2, it])
b_flat_after_sill = vec(B_pert_all[ind2, it] * 1e5)

# Define bin edges for z and b
n_bins_z = 100
n_bins_b = 100
z_edges_after_sill = range(minimum(z_flat_after_sill), maximum(z_flat_after_sill), length=n_bins_z+1)
b_edges_after_sill = range(minimum(b_flat_after_sill), maximum(b_flat_after_sill), length=n_bins_b+1)
z_edges_before_sill = range(minimum(z_flat_before_sill), maximum(z_flat_before_sill), length=n_bins_z+1)
b_edges_before_sill = range(minimum(b_flat_before_sill), maximum(b_flat_before_sill), length=n_bins_b+1)

# Compute 2D histogram
h_2d_after_sill = StatsBase.fit(Histogram, (z_flat_after_sill, b_flat_after_sill), (z_edges_after_sill, b_edges_after_sill))
h_2d_before_sill = StatsBase.fit(Histogram, (z_flat_before_sill, b_flat_before_sill), (z_edges_before_sill, b_edges_before_sill))
# Normalize to get PDF
pdf_2d_after_sill = h_2d_after_sill.weights ./ sum(h_2d_after_sill.weights)
pdf_2d_before_sill = h_2d_before_sill.weights ./ sum(h_2d_before_sill.weights)

# Bin centers for plotting
z_centers_after_sill = (z_edges_after_sill[1:end-1] .+ z_edges_after_sill[2:end]) ./ 2
b_centers_after_sill = (b_edges_after_sill[1:end-1] .+ b_edges_after_sill[2:end]) ./ 2
z_centers_before_sill = (z_edges_before_sill[1:end-1] .+ z_edges_before_sill[2:end]) ./ 2
b_centers_before_sill = (b_edges_before_sill[1:end-1] .+ b_edges_before_sill[2:end]) ./ 2

using PyPlot

# Calculate PDF of B_pert at each time step
n_timesteps = length(time_full)
n_bins = 400

# Create consistent bin edges for all timesteps
bin_edges_B = range(-4e-5, 4e-5, length=n_bins + 1)
bin_centers_B = (bin_edges_B[1:end-1] .+ bin_edges_B[2:end]) ./ 2  # Midpoints of bin edges
bin_edges_Z = range(-700, 700, length=n_bins + 1)
bin_centers_Z = (bin_edges_Z[1:end-1] .+ bin_edges_Z[2:end]) ./ 2  # Midpoints of bin edges
bin_edges_X = range(-4000, 40000, length=n_bins + 1)
bin_centers_X = (bin_edges_X[1:end-1] .+ bin_edges_X[2:end]) ./ 2  # Midpoints of bin edges
# Initialize PDF data array
pdf_data_B = zeros(n_bins, n_timesteps)
pdf_data_Z = zeros(n_bins, n_timesteps)
pdf_data_X = zeros(n_bins, n_timesteps)

for t in 1:n_timesteps
        b_pert_t = B_pert_all[:, t]
        z_pert_t = z_pert_all[:,t]
        x_pert_t = x_pert_all[:,t]
        h_b = StatsBase.fit(Histogram, b_pert_t, bin_edges_B)
        h_z = StatsBase.fit(Histogram, z_pert_t, bin_edges_Z)
        h_x = StatsBase.fit(Histogram, x_pert_t, bin_edges_X)
        bin_width_B = step(bin_edges_B)
        bin_width_Z = step(bin_edges_Z)
        bin_width_X = step(bin_edges_X)
        ind = findall(h_b.weights .< 100)
        h_b.weights[ind] .= 0
        pdf_data_B[:, t] = h_b.weights ./ (sum(h_b.weights) * bin_width_B)
        ind = findall(h_z.weights .< 100)
        h_z.weights[ind] .= 0
        pdf_data_Z[:, t] = h_z.weights ./ (sum(h_z.weights) * bin_width_Z)
        ind = findall(h_x.weights .< 100)
        h_x.weights[ind] .= 0
        pdf_data_X[:, t] = h_x.weights ./ (sum(h_x.weights) * bin_width_X)
end
pdf_data_B[pdf_data_B .< 1e4] .= NaN
pdf_data_Z[pdf_data_Z .< nanmaximum(pdf_data_Z)*0.002] .= NaN
# pdf_data_X[pdf_data_X .< 1e4] .= NaN
GC.gc()

# Prepare time axis for both plots
time_tidal = (time_full[:] .- time_full[1]) / (2*pi/1.4e-4)
time_tidal_ratio = (time_full[2:end] .- time_full[1]) / (2*pi/1.4e-4)
# Create figure with two panels using PyPlot's subplots
close("all")
fig, axarr = plt.subplots(3, 2, figsize=(12, 8), gridspec_kw=Dict("height_ratios" => [5, 4, 5]), sharex=false,constrained_layout=true)

# Panel (a): PDF pcolor plot (top)
ax_pdf = axarr[1]
cmap = plt.get_cmap("managua").copy()
cmap.set_under("white")
pcm = ax_pdf.pcolor(time_tidal, bin_centers_B * 1e5, pdf_data_B/1e4, cmap="managua", shading="nearest")
# Place colorbar outside the right of panel a to avoid squeezing
cbar = fig.colorbar(pcm, ax=ax_pdf, label="PDF", pad=0.02, fraction=0.046, location="right", extend="max")
cbar.ax.set_ylabel("10⁴ × PDF", fontsize=18)
cbar.ax.tick_params(labelsize=16)
cbar.ax.tick_params(which="minor", labelsize=16)
# cbar._set_scale("linear")
ax_pdf.set_ylabel(L"\Delta B_p ~\mathrm{[10⁻⁵~ m~ s⁻²]}", fontsize=18)
ax_pdf.tick_params(labelsize=16)
ax_pdf.grid(true, linestyle="-", alpha=0.75)
ax_pdf.set_xlim(0, time_tidal[end])
ax_pdf.set_ylim(-3, 3)
pcm.set_clim(1, 10)
ax_pdf.axhline(0, color="black", linewidth=1, linestyle="-")

# Superimpose median lines
ax_pdf.plot(time_tidal, median_buoyancy_0_all * 1e5, color="#0072B2", linewidth=3, label="All", zorder=10)  # Blue
ax_pdf.plot(time_tidal, median_buoyancy_0_negative * 1e5, color="#D55E00", linewidth=2.5, linestyle="-", label=L"$\in B^p(t_f) < B^p(t_0)$", zorder=10)  # Orange
ax_pdf.plot(time_tidal, median_buoyancy_0_positive * 1e5, color="#009E73", linewidth=2.5, linestyle="-", label=L"$\in B^p(t_f) > B^p(t_0)$", zorder=10)  # Green
ax_pdf.legend(fontsize=14, loc="lower left", frameon=false)
ax_pdf.text(0.02, 0.92, "(a)", transform=ax_pdf.transAxes, fontsize=16, va="top", ha="left")

# Panel (b): PDF pcolor plot
ax_pdf_Z = axarr[4]
cmap = plt.get_cmap("managua").copy()
cmap.set_under("white")
pcm = ax_pdf_Z.pcolor(time_tidal, bin_centers_Z, pdf_data_Z, cmap="managua", shading="nearest")
# Place colorbar outside the right of panel a to avoid squeezing
cbar = fig.colorbar(pcm, ax=ax_pdf_Z, label="PDF", pad=0.02, fraction=0.046, location="right", extend="max")
cbar.ax.set_ylabel("PDF", fontsize=18)
cbar.ax.tick_params(labelsize=16)
cbar.ax.tick_params(which="minor", labelsize=16)
# cbar._set_scale("linear")
ax_pdf_Z.set_ylabel(L"\Delta Z^p ~\mathrm{[m]}", fontsize=18)
ax_pdf_Z.tick_params(labelsize=16)
ax_pdf_Z.grid(true, linestyle="-", alpha=0.75)
ax_pdf_Z.set_xlim(0, time_tidal[end])
ax_pdf_Z.set_ylim(-200, 200)
pcm.set_clim(0, nanmaximum(pdf_data_Z)*0.05)
ax_pdf_Z.axhline(0, color="black", linewidth=1, linestyle="-")

# Superimpose median lines
ax_pdf_Z.plot(time_tidal, median_z_0_all, color="#0072B2", linewidth=3, label="All", zorder=10)  # Blue
ax_pdf_Z.plot(time_tidal, median_z_0_negative, color="#D55E00", linewidth=2.5, linestyle="-", label=L"$\in Z^p(t_f) < Z^p(t_0)$", zorder=10)  # Orange
ax_pdf_Z.plot(time_tidal, median_z_0_positive, color="#009E73", linewidth=2.5, linestyle="-", label=L"$\in Z^p(t_f) > Z^p(t_0)$", zorder=10)  # Green
ax_pdf_Z.legend(fontsize=14, loc="lower left", frameon=false)
ax_pdf_Z.text(0.02, 0.92, "(b)", transform=ax_pdf_Z.transAxes, fontsize=16, va="top", ha="left")



# Panel (c): Ratio plot (bottom)
ax_ratio = axarr[2]
ax_ratio.plot(time_tidal_ratio, ratio_B, color="purple", linewidth=2.5, label=L"$N_+^p~/~N_-^p$")
ax_ratio.set_ylabel(L"$N_+^p~/~N_-^p$", fontsize=18)
ax_ratio.grid(true, linestyle="-", alpha=0.75)
ax_ratio.set_yscale("log")
ax_ratio.tick_params(labelsize=18)
ax_ratio.tick_params(labelsize=16)
ax_ratio.set_xlabel(L"$t-t_0$ [Tidal period]", fontsize=18)
ax_ratio.set_xlim(0, time_tidal[end])
ax_ratio.set_ylim(0.5, 10)
ax_ratio.axhline(1, color="black", linewidth=1, linestyle="--")
# ax_ratio.set_ylim(-0.1, maximum(ratio_B))

# ax_ratio.legend(fontsize=16, loc="upper right", facecolor=(0.9, 0.9, 0.9), edgecolor="black")
ax_ratio.text(0.02, 0.92, "(c)", transform=ax_ratio.transAxes, fontsize=16, va="top", ha="left")

# Panel (d): Ratio plot (bottom)
ax_ratio_Z = axarr[5]
ax_ratio_Z.plot(time_tidal_ratio, ratio_Z, color="purple", linewidth=2.5, label=L"$N_+^p~/~N_-^p$")
ax_ratio_Z.set_ylabel(L"$N_+^p~/~N_-^p$", fontsize=18)
ax_ratio_Z.set_yscale("log")
ax_ratio_Z.grid(true, linestyle="-", alpha=0.75)
ax_ratio_Z.tick_params(labelsize=16)
ax_ratio_Z.set_xlabel(L"$t-t_0$ [Tidal period]", fontsize=18)
ax_ratio_Z.set_xlim(0, time_tidal[end])
ax_ratio_Z.set_ylim(-0.1, maximum(ratio_Z))
ax_ratio_Z.axhline(1, color="black", linewidth=1, linestyle="--")
# ax_ratio.legend(fontsize=16, loc="upper right", facecolor=(0.9, 0.9, 0.9), edgecolor="black")
ax_ratio_Z.text(0.1, 0.92, "(d)", transform=ax_ratio_Z.transAxes, fontsize=16, va="top", ha="left")


# Panel (e): ΔBᵖ vs ΔZᵖ 2D PDF plot (before sill)
ax_relation_before_sill = axarr[3]
pcm = ax_relation_before_sill.pcolormesh(z_centers_before_sill, b_centers_before_sill, pdf_2d_before_sill', cmap="inferno", shading="auto", norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=maximum(pdf_2d_before_sill)*0.1))
# cb = plt.colorbar(pcm, ax=ax_relation_before_sill)
# cb.set_label("PDF", fontsize=20)
# cb.ax.tick_params(labelsize=18)
ax_relation_before_sill.set_ylabel(L"\Delta B^p ~\mathrm{[10⁻⁵~ m~ s⁻²]}", fontsize=18)
ax_relation_before_sill.set_xlabel(L"\Delta Z^p ~\mathrm{[m]}", fontsize=18)
ax_relation_before_sill.grid(true, linestyle="--", alpha=0.5)
ax_relation_before_sill.tick_params(labelsize=16)
ax_relation_before_sill.set_xlim(minimum(z_centers_before_sill), maximum(z_centers_before_sill))
ax_relation_before_sill.set_ylim(minimum(b_centers_before_sill), maximum(b_centers_before_sill))
ax_relation_before_sill.set_ylim(-60, 60)
pcm.set_clim(1e-5, 1e-2)
ax_relation_before_sill.text(0.02, 0.92, L"(e) ∈ $X^p(t_f)$ < 15 km", transform=ax_relation_before_sill.transAxes, fontsize=18, va="top", ha="left")
ax_relation_before_sill.axhline(0, color="black", linewidth=1, linestyle="-")
ax_relation_before_sill.axvline(0, color="black", linewidth=1, linestyle="-")

# Panel (f): ΔBᵖ vs ΔZᵖ 2D PDF plot (after sill)
ax_relation_after_sill = axarr[6]
pcm = ax_relation_after_sill.pcolormesh(z_centers_after_sill, b_centers_after_sill, pdf_2d_after_sill', cmap="inferno", shading="auto", norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=maximum(pdf_2d_after_sill)*0.1))
cb = plt.colorbar(pcm, ax=ax_relation_after_sill)
cb.set_label("PDF", fontsize=18)
cb.ax.tick_params(labelsize=18)
ax_relation_after_sill.set_ylabel(L"\Delta B^p ~\mathrm{[10⁻⁵~ m~ s⁻²]}", fontsize=18)
ax_relation_after_sill.set_xlabel(L"\Delta Z^p ~\mathrm{[m]}", fontsize=18)
ax_relation_after_sill.grid(true, linestyle="--", alpha=0.5)
ax_relation_after_sill.tick_params(labelsize=16)
ax_relation_after_sill.set_xlim(minimum(z_centers_before_sill), maximum(z_centers_before_sill))
# ax_relation_after_sill.set_ylim(minimum(b_centers_after_sill), maximum(b_centers_after_sill))
ax_relation_after_sill.set_ylim(-60, 60)
pcm.set_clim(1e-5, 1e-2)
ax_relation_after_sill.text(0.02, 0.92, L"(f) ∈ $X^p(t_f)$ > 15 km", transform=ax_relation_after_sill.transAxes, fontsize=18, va="top", ha="left")
ax_relation_after_sill.axhline(0, color="black", linewidth=1, linestyle="-")
ax_relation_after_sill.axvline(0, color="black", linewidth=1, linestyle="-")


# Save combined figure
output_file = string("output/", simname, "/combined_pdf_ratio_evolution_Z_B_z", z_center_particle, "_final_pdf_no_sill_for_fig_f_35-38km.png")
@info "Saving combined PDF and ratio figure to $output_file"
plt.savefig(output_file, bbox_inches="tight")
plt.close()




## compute the correlation coefficient between ΔBᵖ and ΔZᵖ at the final time step wrt x position
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
x_interp = range(x_topo[1],x_topo[end], length=500)
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=1000)

Nx = 500
Ny = 1000
Nz = 250
θ = 0.0036
using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)

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

# compute the correlation coefficient between z and b at each location

x_bins = range(minimum(x_final), maximum(x_final), length=150)
corr_coef = zeros(size(x_bins)).*NaN
for i in 1:length(x_bins)-4
    # Get the bin edges
    x_bin_edges = (x_bins[i], x_bins[i+3])
    # Get the indices of particles in this bin
    ind = findall((x_final .> x_bin_edges[1]) .& (x_final .< x_bin_edges[2]))
    # Compute the correlation coefficient if there are enough points
    if length(ind) > 60
        corr_coef[i] = cor(z_pert_all[ind,end], B_pert_all[ind,end])
        # println("Correlation coefficient for bin $(i): $corr_coef")
    end
end
using PyPlot
fig, ax1 = subplots(figsize=(10, 4))  # Create a figure with a specified size and a primary axis

# Plot the correlation coefficient on the primary y-axis
ax1.plot(x_bins*1e-3, corr_coef, alpha=1, label=L"$r_{\Delta B^p, \Delta Z^p}$", color="red")
ax1.labelsize = 16
ax1.set_xlabel(L"$\hat{x}$ [km]", fontsize=16)
ax1.set_ylabel(L"$r_{\Delta B^p, \Delta Z^p}$", color="red", fontsize=16)
ax1.tick_params(axis="y", labelcolor="red", labelsize=14)  # Increase ticklabel size
ax1.tick_params(axis="x", labelsize=14)  # Increase ticklabel size for x-axis
ax1.set_xlim(0, 40)
# Create a secondary y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.fill_between(nanmean(X_cart_interp_ext[:, 333:666], dim=2) * 1e-3, nanmean(Z_cart_interp_ext[:, 333:666], dim=2), color="gray", alpha=0.5, label="Topography")
ax2.labelsize = 16
ax2.set_ylabel("Mean topography [m]", color="gray", fontsize=16)
ax2.tick_params(axis="y", labelcolor="gray", labelsize=14)  # Increase ticklabel size
ax2.set_ylim(0, 1200)
# Save the figure
savefig("output/tilt/correlation_coefficient_z_B_with_topography.png", bbox_inches="tight")
println("Saved correlation coefficient plot with topography to output/tilt/correlation_coefficient_z_B_with_topography.png")



# Create the figure: plot x position vs time
# Calculate mean and std for x and z positions
# all_mean_x = vec(mean(x_cart_full, dims=1))
# all_std_x = vec(std(x_cart_full, dims=1))
# all_mean_z = vec(mean(z_cart_full, dims=1))
# all_std_z = vec(std(z_cart_full, dims=1))

# # Create figure with 2 subplots
# fig = CairoMakie.Figure(resolution=(900, 900), fontsize=16)

# # X position subplot
# ax1 = Axis(fig[1, 1],
#           xlabel="Time (hours)",
#           ylabel="x position (m)",
#           title="Mean Particle X Position",
#           xminorticksvisible=true,
#           yminorticksvisible=true,
#           xminorticks=IntervalsBetween(5),
#           yminorticks=IntervalsBetween(4))

# band!(ax1, time_full, all_mean_x .- all_std_x, all_mean_x .+ all_std_x,
#       color=:blue, alpha=0.3, label="± 1 Standard Deviation")
# lines!(ax1, time_full, all_mean_x,
#       linewidth=3, color=:blue, label="Mean X Position")
# axislegend(ax1, position=:lt)

# # Z position subplot
# ax2 = Axis(fig[2, 1],
#           xlabel="Time (hours)",
#           ylabel="z position (m)",
#           title="Mean Particle Z Position",
#           xminorticksvisible=true,
#           yminorticksvisible=true,
#           xminorticks=IntervalsBetween(5),
#           yminorticks=IntervalsBetween(4))

# band!(ax2, time_full, all_mean_z .- all_std_z, all_mean_z .+ all_std_z,
#       color=:red, alpha=0.3, label="± 1 Standard Deviation")
# lines!(ax2, time_full, all_mean_z,
#       linewidth=3, color=:red, label="Mean Z Position")
# axislegend(ax2, position=:lt)

# # Save the figure
# output_file = string("output/", simname, "/mean_position_evolution_z", z_center_particle, ".png")
# @info "Saving figure to $output_file"
# save(output_file, fig)
