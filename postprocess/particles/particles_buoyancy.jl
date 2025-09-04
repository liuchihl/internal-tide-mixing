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
# x_cart_full = ds["x_cartesian"][:,:]
# y_cart_full = ds["y_cartesian"][:,:]
z_cart_full = ds["main_z_cartesian"][:, :]
B_full = ds["main_buoyancy"][:, :]
time_full = ds["main_time"][:]
close(ds)

# find particles lower than 1000 m

b_begin = B_full[:, 1]  # Initial z positions

###########
positive_indices = findall(B_full[:, end] .> b_begin)
negative_indices = findall(B_full[:, end] .< b_begin)
z_0_positive = z_cart_full[positive_indices, :]
B_0_positive = B_full[positive_indices, :]
B_pert_0_positive = B_0_positive[:, :] .- B_0_positive[:, 1]  # Remove the initial buoyancy offset
median_buoyancy_0_positive = vec(median(B_pert_0_positive, dims=1))
std_buoyancy_0_positive = vec(std(B_pert_0_positive, dims=1))

z_0_negative = z_cart_full[negative_indices, :]
B_0_negative = B_full[negative_indices, :]
B_pert_0_negative = B_0_negative[:, :] .- B_0_negative[:, 1]  # Remove the initial buoyancy offset
median_buoyancy_0_negative = vec(median(B_pert_0_negative, dims=1))
std_buoyancy_0_negative = vec(std(B_pert_0_negative, dims=1))

# include all particles
B_full_all = B_full[:, :]
B_pert_all = B_full_all[:, :] .- B_full_all[:, 1]  # Remove the initial buoyancy offset
median_buoyancy_0_all = vec(median(B_pert_all, dims=1))
std_buoyancy_0_all = vec(std(B_pert_all, dims=1))



# calculate the ratio of positive to negative particles at each timesteps
number_pos = zeros(Int, size(B_pert_all, 2))
number_negative = zeros(Int, size(B_pert_all, 2))
for i in 1:size(B_pert_all, 2)
        number_pos[i] = length(findall(B_pert_all[:, i] .> 0))
        number_negative[i] = length(findall(B_pert_all[:, i] .< 0))
end
ratio = number_pos[2:end] ./ number_negative[2:end]
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
cbar._set_scale("linear")  # Set colorbar to log scale
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
cbar._set_scale("linear")
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
