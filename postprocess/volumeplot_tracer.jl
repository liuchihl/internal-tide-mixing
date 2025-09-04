using NCDatasets
using Statistics
using LinearAlgebra
using MAT
using PyPlot
using Printf
using PyCall
using Interpolations
# Load the data - you'll need to define these variables
# Assuming you have a single NetCDF file or know how to load your data
# You'll need to define: xC, yC, zC, and your data source

# Create animation
frame_rate = 12

# Initialize PyPlot
fig = figure(figsize=(14, 8))  # Wider figure for better layout

# Import necessary matplotlib tools for 3D plotting
mplot3d = pyimport("mpl_toolkits.mplot3d")
cm = pyimport("matplotlib.cm")
colors = pyimport("matplotlib.colors")

# Define meshgrid function for Julia
function meshgrid(x, y, z=nothing)
    if z === nothing
        # 2D case
        nx, ny = length(x), length(y)
        X = zeros(nx, ny)
        Y = zeros(nx, ny)

        for i in axes(x, 1), j in axes(y, 1)
            X[i, j] = x[i]
            Y[i, j] = y[j]
        end

        return X, Y
    else
        # 3D case
        nx, ny, nz = length(x), length(y), length(z)
        X = zeros(nx, ny, nz)
        Y = zeros(nx, ny, nz)
        Z = zeros(nx, ny, nz)

        for i in axes(x, 1), j in axes(y, 1), k in axes(z, 1)
            X[i, j, k] = x[i]
            Y[i, j, k] = y[j]
            Z[i, j, k] = z[k]
        end

        return X, Y, Z
    end
end

# Create an array of frames for the animation
frames = []
output_dir = "output/tilt/"
# all_times = [451.5, 452.0, 452.5, 453.0, 453.5, 454.0, 454.5, 455.0, 455.5, 456.0, 456.5, 457.0, 457.5, 458.0, 458.5, 459.0, 459.5, 460.0, 460.5, 461.0, 461.5, 462.0]  # Example times
# all_times = [461.5, 462.0]  # Example times
all_times = [458.5, 459.0, 459.5, 460.0, 460.5, 461.0, 461.5, 462.0]  # Example times

# Update n_frames to match all_times
n_frames = length(all_times)
@info "Creating animation with $(n_frames) frames..."

# Function to get filename for a given time
function get_data_file(time_val)
    return string(output_dir, "internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=", time_val, "_analysis_round=all_threeD.nc")
end

# Initialize coordinate arrays (load from first file to get grid)
first_file = get_data_file(all_times[1])
ds_temp = Dataset(first_file, "r")
xC = ds_temp["x_caa"][:]
yC = ds_temp["y_aca"][:]
zC = ds_temp["z_aac"][:]
close(ds_temp)

# load topography and transform it into Cartesian coordinates
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic")
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo, 1))
y_topo_lin = range(y_topo[1], y_topo[end], size(z_topo, 2))
close(file)
Nx = 500
Ny = 1000
θ = 0.0036
# high-resolution grids
x_interp = range(x_topo[1], x_topo[end], length=Nx)
y_interp = range(y_topo[1], y_topo[end], length=Ny)

# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp .- minimum(z_interp)
# Initialize frame counter
global global_frame_idx = 1


# Loop through files
for file_idx in 1:length(all_times)
    current_file_time = all_times[file_idx]
    data_file = get_data_file(current_file_time)

    # @info "Processing frame $frame_idx/$n_frames - Time: $current_time"
    @info "Looking for file: $data_file"

    # Check if file exists
    if !isfile(data_file)
        @warn "Data file not found: $data_file"
        continue
    end

    # Open the dataset for this time
    ds = Dataset(data_file, "r")

    # Load times from this file
    times_in_file = ds["time"][:]
    @info "Times in file: $(times_in_file)"

    timestep_indices = 1:1:length(times_in_file)
    for time_idx in timestep_indices
        @info "Processing time index: $time_idx, actual time: $(times_in_file[time_idx])"

        # Load the tracer data for this time step
        current_tracer = ds["c"][:, :, :, time_idx]

        clf() # Clear the figure

        # Create subplot with smaller colorbar
        ax = subplot2grid((1, 12), (0, 0), colspan=10, projection="3d")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        # Filter out very small values to improve visualization
        current_tracer_filtered = copy(current_tracer)


        # Get min/max for colormap scaling
        valid_values = filter(!isnan, current_tracer_filtered)


        # Add topography surface (after scatter, but forced visually behind)
        topo_ds = 4
        X_topo, Y_topo = meshgrid(x_interp[1:topo_ds:end], y_interp[1:topo_ds:end])
        Z_topo = z_interp[1:topo_ds:end, 1:topo_ds:end]

        X_topo_cart = X_topo .* cos(θ) .- Z_topo .* sin(θ)
        Y_topo_cart = Y_topo
        Z_topo_cart = X_topo .* sin(θ) .+ Z_topo .* cos(θ)

        # Proper lighting using LightSource
        ls = colors.LightSource(azdeg=315, altdeg=25)
        rgb = ls.shade(Z_topo_cart; cmap=cm.get_cmap("terrain"),
            vert_exag=1.0, blend_mode="soft")

        topo_surface = ax.plot_surface(
            X_topo_cart, Y_topo_cart, Z_topo_cart;
            facecolors=rgb,           # use shaded colors
            linewidth=0,
            antialiased=true,
            shade=false,              # we provide pre-shaded facecolors
            alpha=0.2,               # transparent so tracer is visible
            zorder=0
        )
        # Push the whole surface behind other 3D artists in mplot3d's sort
        try
            topo_surface.set_zsort("min")   # sort by min depth → tends to sit behind points
            topo_surface.set_zorder(0)
        catch
        end

        # Optional: wireframe overlay to enhance 3D relief
        ax.plot_wireframe(
            X_topo_cart, Y_topo_cart, Z_topo_cart;
            rstride=8, cstride=8,
            color="k", alpha=0.15, linewidth=0.4, zorder=1
        )



        # Create a downsampled version for performance
        downsample = 3
        xc_ds = xC[1:downsample:end]
        yc_ds = yC[1:downsample:end]
        zc_ds = zC[1:downsample:end]
        tracer_ds = current_tracer_filtered[1:downsample:end, 1:downsample:end, 1:downsample:end]

        # Create 3D meshgrid for tracer data
        X, Y, Z = meshgrid(xc_ds, yc_ds, zc_ds)

        # Transform tracer coordinates to Cartesian if needed
        X_cart = X .* cos(θ) .- Z .* sin(θ)
        Y_cart = Y
        Z_cart = X .* sin(θ) .+ Z .* cos(θ)

        # Reshape for scatter plot
        x_flat = reshape(X_cart, :)
        y_flat = reshape(Y_cart, :)
        z_flat = reshape(Z_cart, :)
        tracer_flat = reshape(tracer_ds, :)

        # Filter out points with low tracer values
        threshold = 1e-6
        mask = tracer_flat .> threshold

        if sum(mask) > 0
            # Take log10 of the filtered tracer values
            log_tracer_values = log10.(tracer_flat[mask])

            # Set consistent color range for all frames
            vmin_log = -6.0
            vmax_log = -1.0

            # Clamp values to the specified range
            log_tracer_clamped = clamp.(log_tracer_values, vmin_log, vmax_log)

            # Normalize sizes based on log values
            # sizes = 10 .+ 40 .* (log_tracer_clamped .- vmin_log) ./ (vmax_log - vmin_log)

            # Plot the scatter points with colormap
            scatter3D = ax.scatter(
                x_flat[mask], y_flat[mask], z_flat[mask],
                c=log_tracer_clamped,
                alpha=0.8,
                cmap="rainbow",
                edgecolors="none",
                vmin=vmin_log,
                vmax=vmax_log,
                depthshade=false,   # keep points bright; don't darken when "behind"
                zorder=10,           # draw above the surface artist
                s=38,
                marker="o"
            )

            # Add colorbar
            cbar = colorbar(scatter3D, ax=ax, shrink=0.6)
            cbar.set_label("log₁₀(C)")

            # Add custom tick labels for better readability
            tick_positions = [-6, -5, -4, -3, -2, -1]
            cbar.set_ticks(tick_positions)
            tick_labels = ["-$(abs(Int(pos)))" for pos in tick_positions]
            cbar.set_ticklabels(tick_labels)

            # Set consistent colorbar limits
            scatter3D.set_clim(-6, -1)
        else
            @warn "No tracer points above threshold for frame $global_frame_idx"
        end




        # Set axis limits (adjust based on your transformed coordinates)
        ax.set_xlim([minimum(X_cart), maximum(X_cart)])
        ax.set_ylim([minimum(Y_cart), maximum(Y_cart)])
        ax.set_zlim([minimum(Z_cart), maximum(Z_cart)])

        # Set aspect ratio for better visualization
        ax.set_box_aspect([2, 4, 2])

        # Add timestamp
        suptitle(@sprintf("Time: %.2f hours", times_in_file[time_idx]), fontsize=16)

        # Update the figure
        tight_layout()
        # Add timestamp using the ACTUAL time from the file
        # suptitle(@sprintf("Time: %.2f", current_actual_time), fontsize=16)

        # Save with unique frame number
        frame_filename = joinpath(output_dir, "temp_frames", @sprintf("frame_%04d.png", global_frame_idx))
        mkpath(dirname(frame_filename))
        savefig(frame_filename, dpi=100, bbox_inches="tight")

        @info "Saved frame $global_frame_idx"

        # Increment global frame counter
        global global_frame_idx += 1
    end

    # Always close the dataset
    close(ds)
end
@info "Total frames created"

# Create animation from saved frames
animation = pyimport("matplotlib.animation")
# Update total frames count
total_frames = global_frame_idx - 1
@info "Total frames created: $total_frames"

# Update the animation section to use total_frames
temp_dir = joinpath(output_dir, "temp_frames")
frame_files = [joinpath(temp_dir, @sprintf("frame_%04d.png", i)) for i in 1:total_frames]
# Filter only existing files
existing_frame_files = filter(isfile, frame_files)
@info "Found $(length(existing_frame_files)) frame files"

fig_anim = figure(figsize=(14, 8))

function animate(frame_num)
    clf()
    julia_idx = frame_num + 1

    # Check bounds to avoid errors
    if julia_idx > length(existing_frame_files) || julia_idx < 1
        @warn "Frame index out of bounds: $julia_idx"
        return []
    end

    img = plt.imread(existing_frame_files[julia_idx])
    imshow(img)
    axis("off")
    return []
end
# Clean up temporary files
for file in existing_frame_files
    # This loop was incomplete - adding empty body
end

# Create the animation
anim = animation.FuncAnimation(fig_anim, animate, frames=length(existing_frame_files),
    interval=1000 / frame_rate, blit=true)

# Save the animation
writer = animation.FFMpegWriter(fps=frame_rate, bitrate=5000)
anim.save(joinpath(output_dir, "tracer_volume_center_only.mp4"), writer=writer)

# Clean up temporary files
# for file in existing_frame_files
#     isfile(file) && rm(file)
# end
# isdir(temp_dir) && rm(temp_dir, recursive=true)

# @info "Animation saved to $(joinpath(output_dir, "tracer_volume.mp4"))"