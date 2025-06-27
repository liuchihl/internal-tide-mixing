# this script analyzes the diapycanl velocity
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using Interpolations
using Printf
include("functions/bins_2d.jl")
include("functions/mmderiv.jl")

# load data
tᶠ = 453.0
θ = 3.6e-3
if θ == 3.6e-3
    simname = "tilt"
else
    simname = "flat"
end

base_folder = "/scratch/bcpi/cliu28/internal-tide-mixing/"
filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_analysis_round=all_threeD.nc")
ds_field = Dataset(filename_field, "r")
filename_b_verification = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_b = Dataset(filename_b_verification, "r")

# grids
zC = ds_field["z_aac"][:];
zF = ds_field["z_aaf"][:];
Nz = length(zC[:]);
xC = ds_field["x_caa"][:];
xF = ds_field["x_faa"][:];
Nx = length(xC[:]);
dx = xF[end] - xF[end-1];
yC = ds_field["y_aca"][:];
yF = ds_field["y_afa"][:];
Ny = length(yC[:]);
dy = yF[end] - yF[end-1];


b = ds_b["b"][:, :, :, 1]
time = ds_field["time"][:]
Nt = length(time)

Bx = zeros(Nx, Ny, Nz);
By = zeros(Nx, Ny, Nz);
Bz = zeros(Nx, Ny, Nz);
ẽ = zeros(Nx, Ny, Nz)

for tt in 1:Nt
    ∇κ∇B = ds_field["∇κ∇B"][:, :, :, tt]
    ∇κ∇B[b.==0] .= NaN
    B = ds_field["B"][:, :, :, tt]

    # Initialize temporary arrays for derivatives
    Bx = zeros(size(B))
    By = zeros(size(B))
    Bz = zeros(size(B))

    for j in 1:Ny
        Bx[:, j, :] = mmderiv(xC, B[:, j, :])
        Bz[:, j, :] = mmderiv(zC, B[:, j, :]')'
    end

    for i in 1:Nx
        By[i, :, :] = mmderiv(yC, B[i, :, :])
    end

    # Add to ∇B_mag using the global scope
    ∇B_mag = sqrt.(Bx .^ 2 + By .^ 2 + Bz .^ 2)
    global ẽ += ∇κ∇B ./ ∇B_mag
    @info @sprintf("%.2f%%", 100 * tt / Nt)
end
ẽ_avg = ẽ ./ Nt
# Save ẽ to a NetCDF file
output_dir = string(base_folder, "output/", simname, "/")

output_filename = string(output_dir, "diapycnal_velocity_tᶠ=", tᶠ, "_theta=", θ, ".nc")
ds_save = Dataset(output_filename, "c")

# Define dimensions
defDim(ds_save, "xC", length(xC))
defDim(ds_save, "yC", length(yC))
defDim(ds_save, "zC", length(zC))

# Define variables
v1 = defVar(ds_save, "ẽ_avg", Float64, ("xC", "yC", "zC"))
v1.attrib["long_name"] = "10 tidal period average diapycnal velocity"
v1.attrib["units"] = "m/s"
v1[:, :, :] = ẽ_avg

v2 = defVar(ds_save, "xC", Float64, ("xC",))
v2[:] = xC[:]

v3 = defVar(ds_save, "yC", Float64, ("yC",))
v3[:] = yC[:]

v4 = defVar(ds_save, "zC", Float64, ("zC",))
v4[:] = zC[:]

close(ds_save)
@info "Saved diapycnal velocity to $output_filename"



## read the data
# Read the saved diapycnal velocity data

# Function to load data for a given case
function load_diapycnal_velocity(tᶠ, θ)
    if θ == 3.6e-3
        simname = "tilt"
    else
        simname = "flat"
    end
    output_filename = string("output/", simname, "/diapycnal_velocity_tᶠ=", tᶠ, "_theta=", θ, ".nc")
    ds_e = Dataset(output_filename, "r")
    ẽ = ds_e["ẽ_avg"][:, :, :]
    xC = ds_e["xC"][:]
    yC = ds_e["yC"][:]
    zC = ds_e["zC"][:]
    return ẽ, xC, yC, zC
    close(ds_e)
end


# Load tilt case (θ = 3.6e-3)
tᶠ = 452.5
ẽ1, xC, yC, zC = load_diapycnal_velocity(tᶠ, 3.6e-3)
tᶠ = 453.0
ẽ2, xC, yC, zC = load_diapycnal_velocity(tᶠ, 3.6e-3)
ẽ_tilt = (ẽ1 .+ ẽ2) ./ 2

## plot heatmap x-z slices
# Define slice positions
slice_positions = [0.25, 0.5, 0.75]
y_indices = [findmin(abs.(yC .- (pos * yC[end]))) for pos in slice_positions]
y_indices = [idx[2] for idx in y_indices]  # Extract just the index

# Create a figure with three subplots in a row
fig = Figure(size=(1200, 300))

# Find the appropriate color range for consistent visualization
max_val = nanmaximum(abs.(filter(isfinite, ẽ_tilt))) * 0.01

# Labels for the subplots
panel_labels = ["(a)", "(b)", "(c)"]

for (i, (pos, y_idx)) in enumerate(zip(slice_positions, y_indices))
    # Create axis with appropriate labels
    if i == 1
        title = L"y = 0.25 L_y"
    elseif i == 2
        title = L"y = 0.5 L_y"
    else
        title = L"y = 0.75 L_y"
    end

    ax = Axis(fig[1, i],
        xlabel="x [km]",
        ylabel="z [m]",
        title=title,
    )

    # Plot heatmap
    hm = heatmap!(ax, xC*1e-3, zC, ẽ_tilt[:, y_idx, :],
        colormap=:balance,
        colorrange=(-max_val, max_val),
        nan_color=:gray)
    
    # Add panel label to the top left corner
    text!(ax, 0.05, 0.95, text=panel_labels[i], 
          align=(:left, :top), 
          space=:relative, 
          fontsize=18, 
          font=:bold)

    # Add colorbar only to the last plot
    if i == length(slice_positions)
        Colorbar(fig[1, i+1], hm, label=L"\mathbf{ē}~ \mathrm{[m~s^{-1}]}",
            width=20, height=200, labelsize=14, ticklabelsize=12)
    end
end

# Save the figure
save("output/tilt/diapycnal_velocity_cross_sections.png", fig, px_per_unit=2)

## plot hab
include("functions/bins.jl")
bin_edge = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
bin_mask = hab
dx = 30
dy = 30

ẽ_tilt_avg, bin_center = bins(ẽ_tilt, bin_edge, bin_mask, dx=dx, dy=dy, z_face=zF[:], normalize=true)
# Create a figure for the profile plot
fig = Figure(size=(600, 500))
ax = Axis(fig[1, 1], 
    xlabel="Diapycnal Velocity [m s⁻¹]", 
    ylabel="Height Above Bottom [m]")

# Plot the profile
lines!(ax, ẽ_tilt_avg[:,1], bin_center, linewidth=3, color=:royalblue)

# Add grid lines for better readability
ax.xgridvisible = true
ax.ygridvisible = true
ax.limits = (nothing,nothing), (minimum(bin_center), 500)
# Add zero line for reference
vlines!(ax, 0, minimum(bin_center), maximum(bin_center), 
        linestyle=:dash, color=:black, alpha=0.5)

# Save the figure
save("output/tilt/diapycnal_velocity_profile.png", fig, px_per_unit=2)

# Display the figure
fig



## plotting the hab
include("functions/bins.jl")
bin_edge = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
bin_mask = hab
dx = 30
dy = 30
# grids
base_folder = "/scratch/bcpi/cliu28/internal-tide-mixing/"
filename_field = string(base_folder, "output/tilt/internal_tide_theta=", 0.0036, "_Nx=500_Nz=250_tᶠ=", 460, "_threeD_Bbudget.nc")
ds_field = Dataset(filename_field, "r")

zF = ds_field["zF"][:];
z_face = zF
dx = 30
dy = 30
@time ẽ_tilt_avg, bin_center = bins(ẽ_tilt, bin_edge, bin_mask, dx=dx, dy=dy, z_face=z_face, normalize=true)
@time ẽ_flat_avg, _ = bins(ẽ_flat, bin_edge, bin_mask, dx=dx, dy=dy, z_face=z_face, normalize=true)

# Use PyPlot for plotting
using PyPlot

# Set figure size and create figure
PyPlot.figure(figsize=(10, 6))

# Plot both datasets
plt.plot(ẽ_tilt_avg, bin_center, "r-", linewidth=2, label="Tilt Case (θ = 3.6e-3)")
plt.plot(ẽ_flat_avg, bin_center, "b--", linewidth=2, label="Flat Case (θ = 0)")

# Add grid, labels and title
grid(true, alpha=0.3)
xlabel("Diapycnal Velocity (m/s)", fontsize=14)
ylabel("Height Above Bottom (m)", fontsize=14)
title("Diapycnal Velocity vs. Height Above Bottom", fontsize=16)

# Add legend
legend(fontsize=12)

# Set tick label size
tick_params(axis="both", which="major", labelsize=12)

# Adjust layout
tight_layout()

# Save figure
savefig("output/diapycnal_velocity_hab_both_cases.png", dpi=300, bbox_inches="tight")


# Find common color range for proper comparison
max_val = max(
    nanmaximum(abs.(filter(isfinite, ẽ_flat))),
    nanmaximum(abs.(filter(isfinite, ẽ_tilt)))
) * 0.01

# Create a figure with two subplots side by side
fig = Figure(size=(1200, 500))

# Flat case subplot
ax1 = Axis(fig[1, 1],
    xlabel="x (m)",
    ylabel="z (m)",
    title="Flat Case (θ = 0)")

# Tilt case subplot
ax2 = Axis(fig[1, 2],
    xlabel="x (m)",
    ylabel="z (m)",
    title="Tilt Case (θ = 3.6e-3)")

# Plot the heatmaps with the same colorrange
hm1 = heatmap!(ax1, xC_flat, zC_flat, ẽ_flat[:, length(yC_flat)÷2, :],
    colormap=:balance,
    colorrange=(-max_val, max_val),
    nan_color=:gray)

hm2 = heatmap!(ax2, xC_tilt, zC_tilt, ẽ_tilt[:, length(yC_tilt)÷2, :],
    colormap=:balance,
    colorrange=(-max_val, max_val),
    nan_color=:gray)

# Add colorbar
Colorbar(fig[1, 3], hm2, label="ẽ (m/s)")

# Save the figure
save("output/diapycnal_velocity_comparison.png", fig)

# # Display the figure
# fig
