# After all the initial setup code...

# Create animation
n_frames = 2  # length(all_times)
frame_rate = 15

# Initialize PyPlot
fig = figure(figsize=(14, 8))  # Wider figure for better layout

# Import necessary matplotlib tools for 3D plotting
mplot3d = pyimport("mpl_toolkits.mplot3d")
cm = pyimport("matplotlib.cm")
colors = pyimport("matplotlib.colors")

# Define meshgrid function for Julia
function meshgrid(x, y, z)
    X = [x[i] for i in 1:length(x), j in 1:length(y), k in 1:length(z)]
    Y = [y[j] for i in 1:length(x), j in 1:length(y), k in 1:length(z)]
    Z = [z[k] for i in 1:length(x), j in 1:length(y), k in 1:length(z)]
    return X, Y, Z
end

@info "Creating animation with $(n_frames) frames..."

# Create an array of frames for the animation
frames = []

# Loop through frames
for frame_idx in 0:(n_frames-1)
    # Find which time chunk and time index this frame corresponds to
    current_time = all_times[frame_idx + 1]  # Convert 0-based to 1-based
    
    # Find the appropriate file and time index
    cumulative_times = 0
    local_time_idx = 1
    chunk_idx = 1
    current_tracer = nothing  # Initialize to avoid undefined variable error
    
    for (i, time_chunk) in enumerate(time_chunks)
        filename = get_filename(time_chunk)
        ds_temp = Dataset(filename, "r")
        n_times_in_chunk = length(ds_temp["time"][:])
        
        if frame_idx < cumulative_times + n_times_in_chunk
            chunk_idx = i
            local_time_idx = frame_idx - cumulative_times + 1
            
            # Load only the current time step
            current_tracer = ds_temp["c"][:, :, :, local_time_idx]
            close(ds_temp)
            break
        end
        
        cumulative_times += n_times_in_chunk
        close(ds_temp)
    end
    
    # Check if we successfully loaded data
    if current_tracer === nothing
        @error "Failed to load tracer data for frame $frame_idx"
        continue
    end
    
    clf() # Clear the figure
    
    # Create subplot
    ax = subplot2grid((1, 10), (0, 0), colspan=8, projection="3d")
    ax.set_title("Tracer at time=$(current_time)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    
    # Filter out very small values to improve visualization
    current_tracer_filtered = copy(current_tracer)
    
    # Check for valid data
    if all(isnan.(current_tracer_filtered)) || all(current_tracer_filtered .== 0)
        @warn "No valid tracer data for frame $frame_idx"
        continue
    end
    
    # Get min/max for colormap scaling
    valid_values = filter(!isnan, current_tracer_filtered)
    if isempty(valid_values)
        @warn "No valid tracer values for frame $frame_idx"
        continue
    end
    
    max_val = maximum(valid_values)
    min_val = minimum(valid_values)
    
    # Create a downsampled version for performance
    downsample = 4
    xc_ds = xC[1:downsample:end]
    yc_ds = yC[1:downsample:end]
    zc_ds = zC[1:downsample:end]
    tracer_ds = current_tracer_filtered[1:downsample:end, 1:downsample:end, 1:downsample:end]
    
    # Create 3D meshgrid
    X, Y, Z = meshgrid(xc_ds, yc_ds, zc_ds)
    
    # Reshape for scatter plot
    x_flat = reshape(X, :)
    y_flat = reshape(Y, :)
    z_flat = reshape(Z, :)
    tracer_flat = reshape(tracer_ds, :)
    
    # Filter out points with low tracer values
    threshold = max_val * 0   # only positive values
    mask = tracer_flat .> threshold
    
    if sum(mask) > 0  # Only plot if there are points to show
        # Normalize sizes for visualization
        norm = colors.Normalize(vmin=min_val, vmax=max_val)
        sizes = 10 .+ 50 .* norm.(tracer_flat[mask])
        
        # Plot the scatter points with colormap - use single alpha value
        scatter3D = ax.scatter(
            x_flat[mask], y_flat[mask], log10.(z_flat[mask]),
            c=tracer_flat[mask],
            s=sizes,
            alpha=0.7,
            cmap="plasma",
            edgecolors="none"
        )
        
        # Add a thin colorbar using subplot2grid
        cax = subplot2grid((1, 10), (0, 9), colspan=1)
        cbar = PyPlot.colorbar(scatter3D, cax=cax)
        cbar.set_label("Tracer concentration", fontsize=10)
    else
        @warn "No tracer points above threshold for frame $frame_idx"
    end
    
    # Set axis limits
    ax.set_xlim([minimum(xC), maximum(xC)])
    ax.set_ylim([minimum(yC), maximum(yC)])
    ax.set_zlim([minimum(zC), maximum(zC)])
    
    # Set aspect ratio for better visualization
    ax.set_box_aspect([2, 4, 1])
    
    # Add timestamp
    suptitle(@sprintf("Time: %.2f", current_time), fontsize=16)
    
    # Update the figure
    tight_layout()
    
    # Save the current figure for this frame
    frame = PyPlot.gcf()
    push!(frames, frame)
    
    @info "Processing frame $(frame_idx+1)/$(n_frames) - Time: $(current_time)"
end

# Create animation from saved frames
animation = pyimport("matplotlib.animation")

# Create proper frame objects for animation
animation_frames = []
for i in 1:length(frames)
    # Clear previous frame
    clf()
    # Copy content from the saved frame
    copyobj = copy(frames[i])
    # Add the frame to animation frames
    push!(animation_frames, [copyobj])
end

anim = animation.ArtistAnimation(fig, animation_frames, interval=1000/frame_rate, blit=false)

# Save animation
writer = animation.FFMpegWriter(fps=frame_rate, bitrate=5000)
anim.save(joinpath(output_dir, "tracer_volume.mp4"), writer=writer)
