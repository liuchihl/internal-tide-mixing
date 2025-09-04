using NCDatasets
using CairoMakie
using Printf
using Statistics
using MAT
using Interpolations

# Function to find the first valid point above bottom for each (x,y) location
function find_first_valid_points(data, mask, n_points=4)
    """
    Find the first n_points valid (non-NaN) points above bottom for each (x,y) location
    Returns arrays of size (nx, ny, n_points)
    """
    nx, ny, nz = size(data)
    result = fill(NaN, nx, ny, n_points)
    
    for i in 1:nx, j in 1:ny
        valid_count = 0
        for k in 1:nz
            if !isnan(data[i, j, k]) && mask[i, j, k] != 0
                valid_count += 1
                if valid_count <= n_points
                    result[i, j, valid_count] = data[i, j, k]
                end
                if valid_count == n_points
                    break
                end
            end
        end
    end
    
    return result
end
# Custom colormap for Bz that centers at zero
function custom_colormap(min_val=-5e-7, max_val=2e-6)
    # Calculate the zero point position in normalized space [0,1]
    zero_point = -min_val / (max_val - min_val)
    
    # Create arrays of colors for smooth transitions
    neg_colors = range(colorant"darkblue", colorant"lightblue", length=50)
    neg_colors = vcat(neg_colors, range(colorant"lightblue", colorant"white", length=50))
    pos_colors = range(colorant"white", colorant"yellow", length=50)
    pos_colors = vcat(pos_colors, range(colorant"yellow", colorant"darkred", length=50))
    
    # Combine the colors into a single array
    all_colors = vcat(neg_colors[1:end-1], pos_colors)
    
    # Create a colormap with proper scaling
    return cgrad(all_colors, zero_point)
end

# Load bathymetry data
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo,1))
y_topo_lin = range(y_topo[1], y_topo[end], size(z_topo,2))
close(file)

# Define grid parameters
Nx = 500
Ny = 1000
x_interp = range(x_topo[1], x_topo[end], length=Nx)
y_interp = range(y_topo[1], y_topo[end], length=Ny)

itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp .- minimum(z_interp)

# Create time array - these are the file time points
t_files = 452.0:0.5:461.0

# Load mask data once
filename_verification = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_mask = Dataset(filename_verification, "r")
b = ds_mask["b"][:, :, :, 1]
close(ds_mask)




# Set up the figure
fig = CairoMakie.Figure(resolution=(1200, 800))
layer_names = ["1st point above bottom", "2nd point above bottom", 
              "3rd point above bottom", "4th point above bottom"]
vmin = -5e-7
vmax = 2e-6
contour_levels = range(minimum(z_interp), maximum(z_interp), length=20)

# Create axes for each layer
axes = [Axis(fig[1, i]) for i in 1:4]

# Configure axes
for (i, ax) in enumerate(axes)
    ax.title = layer_names[i]
    ax.xlabel = "x (km)"
    i == 1 && (ax.ylabel = "y (km)")
    ax.aspect = DataAspect()
end

# Create title
title_obj = Label(fig[0, :], "", fontsize=20)

# Create initial heatmaps and contours
heatmaps = []
for i in 1:4
    # Create initial empty heatmap
    h = heatmap!(axes[i], [0, 1], [0, 1], zeros(2, 2),
                colormap=custom_colormap(vmin, vmax), colorrange=(vmin, vmax))
    push!(heatmaps, h)
    
    # Add contour lines for bathymetry
    contour!(axes[i], x_interp/1e3, y_interp/1e3, z_interp,
             levels=contour_levels, color=:black, linewidth=0.5, alpha=0.7)
end

# Add colorbar
Colorbar(fig[1, 5], heatmaps[1], label="dB/dz (s⁻²)")

# Collect all time points for animation
all_time_points = []
all_file_info = []

for tf in t_files
    filename = @sprintf("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=%.1f_analysis_round=all_threeD.nc", tf)
    
    try
        ds = Dataset(filename, "r")
        time_in_file = ds["time"][:]
        close(ds)
        
        # Add all time points from this file
        for (time_idx, t) in enumerate(time_in_file)
            push!(all_time_points, t)
            push!(all_file_info, (filename, time_idx))
        end
        
        println("File $filename contains $(length(time_in_file)) time points")
    catch e
        println("Error reading file $filename: $e")
    end
end

println("Total animation frames: $(length(all_time_points))")
# length(all_time_points)
# Create the animation
record(fig, "output/tilt/stratification_layers_animation.mp4", 1:length(all_time_points); framerate=8) do frame_idx
    t_current = all_time_points[frame_idx]
    filename, time_idx = all_file_info[frame_idx]
    
    title_obj.text = @sprintf("Stratification (Bz) at Different Levels Above Bottom (t = %.2f)", t_current/(2*pi/1.4e-4))
    println("Processing frame $frame_idx: t = $t_current (file: $(basename(filename)), time_idx: $time_idx)")
    
    try
        ds = Dataset(filename, "r")
        
        # Load coordinates and data for the specific time index
        zC = ds["z_aac"][:]
        xC = ds["x_caa"][:]./1e3
        yC = ds["y_aca"][:]./1e3
        B = ds["B"][:, :, :, time_idx]  # Use the specific time index
        
        close(ds)
        
        # Calculate Bz
        Bz = diff(B, dims=3) ./ reshape(diff(zC[:]), 1, 1, length(zC)-1)
        Bz[b[:, :, 1:end-1] .== 0] .= 0
        
        # Center Bz at cell centers
        Bz_center = (cat(zeros(Nx, Ny, 1), Bz, dims=3) .+ cat(Bz, zeros(Nx, Ny, 1), dims=3)) / 2
        
        # Find first 4 valid points above bottom
        Bz_layers = find_first_valid_points(Bz_center, b, 4)
        
        # Update each layer's plot
        for layer in 1:4
            # Update heatmap data
            heatmaps[layer][1] = xC
            heatmaps[layer][2] = yC
            heatmaps[layer][3] = Bz_layers[:, :, layer]
        end
    catch e
        println("Error processing frame $frame_idx: $e")
    end
end

println("Animation complete!")



### plot B 
fig = CairoMakie.Figure(resolution=(1200, 800))
layer_names = ["1st point above bottom", "2nd point above bottom", 
              "3rd point above bottom", "4th point above bottom"]
vmin = 0.001
vmax = 0.0012
contour_levels = range(minimum(z_interp), maximum(z_interp), length=20)

# Create axes for each layer
axes = [Axis(fig[1, i]) for i in 1:4]

# Configure axes
for (i, ax) in enumerate(axes)
    ax.title = layer_names[i]
    ax.xlabel = "x (km)"
    i == 1 && (ax.ylabel = "y (km)")
    ax.aspect = DataAspect()
end

# Create title
title_obj = Label(fig[0, :], "", fontsize=20)

# Create initial heatmaps and contours
heatmaps = []
for i in 1:4
    # Create initial empty heatmap
    h = heatmap!(axes[i], [0, 1], [0, 1], zeros(2, 2),
                colormap=custom_colormap(vmin, vmax), colorrange=(vmin, vmax))
    push!(heatmaps, h)
    
    # Add contour lines for bathymetry
    contour!(axes[i], x_interp/1e3, y_interp/1e3, z_interp,
             levels=contour_levels, color=:black, linewidth=0.5, alpha=0.7)
end

# Add colorbar
Colorbar(fig[1, 5], heatmaps[1], label="B (m s⁻²)")

# Collect all time points for animation
all_time_points = []
all_file_info = []

for tf in t_files
    filename = @sprintf("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=%.1f_analysis_round=all_threeD.nc", tf)
    
    try
        ds = Dataset(filename, "r")
        time_in_file = ds["time"][:]
        close(ds)
        
        # Add all time points from this file
        for (time_idx, t) in enumerate(time_in_file)
            push!(all_time_points, t)
            push!(all_file_info, (filename, time_idx))
        end
        
        println("File $filename contains $(length(time_in_file)) time points")
    catch e
        println("Error reading file $filename: $e")
    end
end

println("Total animation frames: $(length(all_time_points))")
# length(all_time_points)
# Create the animation
record(fig, "output/tilt/buoyancy_layers_animation.mp4", 1:length(all_time_points); framerate=8) do frame_idx
    t_current = all_time_points[frame_idx]
    filename, time_idx = all_file_info[frame_idx]
    
    title_obj.text = @sprintf("B at Different Levels Above Bottom (t = %.2f)", t_current/(2*pi/1.4e-4))
    println("Processing frame $frame_idx: t = $t_current (file: $(basename(filename)), time_idx: $time_idx)")
    
    try
        ds = Dataset(filename, "r")
        
        # Load coordinates and data for the specific time index
        zC = ds["z_aac"][:]
        xC = ds["x_caa"][:]./1e3
        yC = ds["y_aca"][:]./1e3
        B = ds["B"][:, :, :, time_idx]  # Use the specific time index
        
        close(ds)
        
        # Find first 4 valid points above bottom
        B_layers = find_first_valid_points(B, b, 4)
        
        # Update each layer's plot
        for layer in 1:4
            # Update heatmap data
            heatmaps[layer][1] = xC
            heatmaps[layer][2] = yC
            heatmaps[layer][3] = B_layers[:, :, layer]
        end
    catch e
        println("Error processing frame $frame_idx: $e")
    end
end

println("Animation complete!")



### b'
ds = Dataset("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=456.0_4tidal_periods_avg.nc")
B_avg = ds["B"][:,:,:,1] 

fig = CairoMakie.Figure(resolution=(1200, 800))
layer_names = ["1st point above bottom", "2nd point above bottom", 
              "3rd point above bottom", "4th point above bottom"]
vmin = -0.00003
vmax = 0.00003
contour_levels = range(minimum(z_interp), maximum(z_interp), length=20)

# Create axes for each layer
axes = [Axis(fig[1, i]) for i in 1:4]

# Configure axes
for (i, ax) in enumerate(axes)
    ax.title = layer_names[i]
    ax.xlabel = "x (km)"
    i == 1 && (ax.ylabel = "y (km)")
    ax.aspect = DataAspect()
end

# Create title
title_obj = Label(fig[0, :], "", fontsize=20)

# Create initial heatmaps and contours
heatmaps = []
for i in 1:4
    # Create initial empty heatmap
    h = heatmap!(axes[i], [0, 1], [0, 1], zeros(2, 2),
                colormap=custom_colormap(vmin, vmax), colorrange=(vmin, vmax))
    push!(heatmaps, h)
    
    # Add contour lines for bathymetry
    contour!(axes[i], x_interp/1e3, y_interp/1e3, z_interp,
             levels=contour_levels, color=:black, linewidth=0.5, alpha=0.7)
end

# Add colorbar
Colorbar(fig[1, 5], heatmaps[1], label="b' (m s⁻²)")

# Collect all time points for animation
all_time_points = []
all_file_info = []

for tf in t_files
    filename = @sprintf("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=%.1f_analysis_round=all_threeD.nc", tf)
    
    try
        ds = Dataset(filename, "r")
        time_in_file = ds["time"][:]
        close(ds)
        
        # Add all time points from this file
        for (time_idx, t) in enumerate(time_in_file)
            push!(all_time_points, t)
            push!(all_file_info, (filename, time_idx))
        end
        
        println("File $filename contains $(length(time_in_file)) time points")
    catch e
        println("Error reading file $filename: $e")
    end
end

println("Total animation frames: $(length(all_time_points))")
# length(all_time_points)
# Create the animation
record(fig, "output/tilt/buoyancy_per_layers_animation.mp4", 1:length(all_time_points); framerate=8) do frame_idx
    t_current = all_time_points[frame_idx]
    filename, time_idx = all_file_info[frame_idx]
    
    title_obj.text = @sprintf("b' (away from 4TP average) at Different Levels Above Bottom (t = %.2f)", t_current/(2*pi/1.4e-4))
    println("Processing frame $frame_idx: t = $t_current (file: $(basename(filename)), time_idx: $time_idx)")
    
    try
        ds = Dataset(filename, "r")
        
        # Load coordinates and data for the specific time index
        zC = ds["z_aac"][:]
        xC = ds["x_caa"][:]./1e3
        yC = ds["y_aca"][:]./1e3
        B = ds["B"][:, :, :, time_idx]  # Use the specific time index
        b_per = B .- B_avg
        close(ds)
        
        # Find first 4 valid points above bottom
        b_per_layers = find_first_valid_points(b_per, b, 4)
        
        # Update each layer's plot
        for layer in 1:4
            # Update heatmap data
            heatmaps[layer][1] = xC
            heatmaps[layer][2] = yC
            heatmaps[layer][3] = b_per_layers[:, :, layer]
        end
    catch e
        println("Error processing frame $frame_idx: $e")
    end
end

println("Animation complete!")



### w

fig = CairoMakie.Figure(resolution=(1200, 800))
layer_names = ["1st point above bottom", "2nd point above bottom", 
              "3rd point above bottom", "4th point above bottom"]
vmin = -0.008
vmax = 0.008
contour_levels = range(minimum(z_interp), maximum(z_interp), length=20)

# Create axes for each layer
axes_w = [Axis(fig[1, i]) for i in 1:4]

# Configure axes
for (i, ax) in enumerate(axes_w)
    ax.title = layer_names[i]
    ax.xlabel = "x (km)"
    i == 1 && (ax.ylabel = "y (km)")
    ax.aspect = DataAspect()
end

# Create title
title_obj = Label(fig[0, :], "", fontsize=20)

# Create initial heatmaps and contours
heatmaps = []
for i in 1:4
    # Create initial empty heatmap
    h = heatmap!(axes_w[i], [0, 1], [0, 1], zeros(2, 2),
                colormap=custom_colormap(vmin, vmax), colorrange=(vmin, vmax))
    push!(heatmaps, h)
    
    # Add contour lines for bathymetry
    contour!(axes_w[i], x_interp/1e3, y_interp/1e3, z_interp,
             levels=contour_levels, color=:black, linewidth=0.5, alpha=0.7)
end

# Add colorbar
Colorbar(fig[1, 5], heatmaps[1], label="w (m s⁻¹)")

# Collect all time points for animation
all_time_points = []
all_file_info = []

for tf in t_files
    filename = @sprintf("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=%.1f_analysis_round=all_threeD.nc", tf)
    
    try
        ds = Dataset(filename, "r")
        time_in_file = ds["time"][:]
        close(ds)
        
        # Add all time points from this file
        for (time_idx, t) in enumerate(time_in_file)
            push!(all_time_points, t)
            push!(all_file_info, (filename, time_idx))
        end
        
        println("File $filename contains $(length(time_in_file)) time points")
    catch e
        println("Error reading file $filename: $e")
    end
end

println("Total animation frames: $(length(all_time_points))")
# length(all_time_points)
# Create the animation
record(fig, "output/tilt/w_per_layers_animation.mp4", 1:length(all_time_points); framerate=8) do frame_idx
    t_current = all_time_points[frame_idx]
    filename, time_idx = all_file_info[frame_idx]
    
    title_obj.text = @sprintf("w at Different Levels Above Bottom (t = %.2f)", t_current/(2*pi/1.4e-4))
    println("Processing frame $frame_idx: t = $t_current (file: $(basename(filename)), time_idx: $time_idx)")
    
    try
        ds = Dataset(filename, "r")
        
        # Load coordinates and data for the specific time index
        zC = ds["z_aac"][:]
        xC = ds["x_caa"][:]./1e3
        yC = ds["y_aca"][:]./1e3
        what = ds["what"][:, :, :, time_idx]  # Use the specific time index
        what_cen = (what[:, :, 1:end-1, 1] .+ what[:, :, 2:end, 1]) ./ 2 # what at center
        what_cen[what_cen.==0] .= NaN          # mask out zero values at topography
        close(ds)
        
        # Find first 4 valid points above bottom
        what_layers = find_first_valid_points(what_cen, what, 4)
        
        # Update each layer's plot
        for layer in 1:4
            # Update heatmap data
            heatmaps[layer][1] = xC
            heatmaps[layer][2] = yC
            heatmaps[layer][3] = what_layers[:, :, layer]
        end
    catch e
        println("Error processing frame $frame_idx: $e")
    end
end

println("Animation complete!")



## Rig
function custom_colormap(min_val=-5e-7, max_val=2e-6)
    # Calculate the zero point position in normalized space [0,1]
    zero_point = -min_val / (max_val - min_val)
    
    # Create arrays of colors for smooth transitions
    neg_colors = range(colorant"darkblue", colorant"lightblue", length=50)
    neg_colors = vcat(neg_colors, range(colorant"lightblue", colorant"white", length=50))
    pos_colors = range(colorant"white", colorant"yellow", length=50)
    pos_colors = vcat(pos_colors, range(colorant"yellow", colorant"darkred", length=50))
    
    # Combine the colors into a single array
    all_colors = vcat(neg_colors[1:end-1], pos_colors)
    
    # Create a colormap with proper scaling
    return cgrad(all_colors, zero_point)
end


fig = CairoMakie.Figure(resolution=(1200, 800))
layer_names = ["1st point above bottom", "2nd point above bottom", 
              "3rd point above bottom", "4th point above bottom"]
vmin = -0.5
vmax = 0.5
contour_levels = range(minimum(z_interp), maximum(z_interp), length=20)

# Create axes for each layer
axes_Ri = [Axis(fig[1, i]) for i in 1:4]

# Configure axes
for (i, ax) in enumerate(axes_Ri)
    ax.title = layer_names[i]
    ax.xlabel = "x (km)"
    i == 1 && (ax.ylabel = "y (km)")
    ax.aspect = DataAspect()
end

# Create title
title_obj = Label(fig[0, :], "", fontsize=20)

# Create initial heatmaps and contours
heatmaps = []
for i in 1:4
    # Create initial empty heatmap
    h = heatmap!(axes_Ri[i], [0, 1], [0, 1], zeros(2, 2),
                colormap=custom_colormap(vmin, vmax), colorrange=(vmin, vmax))
    push!(heatmaps, h)
    
    # Add contour lines for bathymetry
    contour!(axes_Ri[i], x_interp/1e3, y_interp/1e3, z_interp,
             levels=contour_levels, color=:black, linewidth=0.5, alpha=0.7)
end

# Add colorbar
Colorbar(fig[1, 5], heatmaps[1], label="w (m s⁻¹)")

# Collect all time points for animation
all_time_points = []
all_file_info = []

for tf in t_files
    filename = @sprintf("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=%.1f_analysis_round=all_threeD.nc", tf)
    
    try
        ds = Dataset(filename, "r")
        time_in_file = ds["time"][:]
        close(ds)
        
        # Add all time points from this file
        for (time_idx, t) in enumerate(time_in_file)
            push!(all_time_points, t)
            push!(all_file_info, (filename, time_idx))
        end
        
        println("File $filename contains $(length(time_in_file)) time points")
    catch e
        println("Error reading file $filename: $e")
    end
end

println("Total animation frames: $(length(all_time_points))")
# length(all_time_points)
# Create the animation
record(fig, "output/tilt/Ri_per_layers_animation.mp4", 1:length(all_time_points); framerate=8) do frame_idx
    t_current = all_time_points[frame_idx]
    filename, time_idx = all_file_info[frame_idx]
    
    title_obj.text = @sprintf("Rig at Different Levels Above Bottom (t = %.2f)", t_current/(2*pi/1.4e-4))
    println("Processing frame $frame_idx: t = $t_current (file: $(basename(filename)), time_idx: $time_idx)")
    
    try
        ds = Dataset(filename, "r")
        
        # Load coordinates and data for the specific time index
        zC = ds["z_aac"][:]
        xC = ds["x_caa"][:]./1e3
        yC = ds["y_aca"][:]./1e3
        Rig = ds["Rig"][:, :, :, time_idx]  # Use the specific time index
        Rig[Rig.==0] .= NaN          # mask out zero values at topography
        close(ds)
        
        # Find first 4 valid points above bottom
        Rig_layers = find_first_valid_points(Rig, b, 4)
        
        # Update each layer's plot
        for layer in 1:4
            # Update heatmap data
            heatmaps[layer][1] = xC
            heatmaps[layer][2] = yC
            heatmaps[layer][3] = Rig_layers[:, :, layer]
        end
    catch e
        println("Error processing frame $frame_idx: $e")
    end
end

println("Animation complete!")





## plot average Bz
using NCDatasets
using CairoMakie
using MAT
using Printf
# Load the average data
ds = Dataset("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=456.0_4tidal_periods_avg.nc")
B_avg = ds["B"][:,:,:,1]
filename_verification = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_mask = Dataset(filename_verification, "r")
b = ds_mask["b"][:, :, :, 1]
close(ds_mask)
contour_levels = range(minimum(z_interp), maximum(z_interp), length=20)

# Calculate Bz
zC = ds["z_aac"][:]
Bz_avg = diff(B_avg, dims=3) ./ reshape(diff(zC), 1, 1, length(zC)-1)
Bz_avg[b[:, :, 1:end-1] .== 0] .= NaN

# Center Bz at cell centers
Bz_avg_center = (cat(zeros(Nx, Ny, 1), Bz_avg, dims=3) .+ cat(Bz_avg, zeros(Nx, Ny, 1), dims=3)) / 2

# Find first valid points above bottom
Bz_layers = find_first_valid_points(Bz_avg_center, b, 1)

# Create the figure
fig = CairoMakie.Figure(resolution=(1200, 800))
ax = Axis(fig[1, 1], 
          title="First Point Above Bottom - Average Stratification (Bz)", 
          xlabel="x (km)", 
          ylabel="y (km)", 
          aspect=DataAspect())

# Create heatmap with the same colormap/range as in animations
vmin = -5e-7
vmax = 2e-6
hm = heatmap!(ax, ds["x_caa"][:]./1e3, ds["y_aca"][:]./1e3, Bz_layers[:,:,1],
             colormap=custom_colormap(vmin, vmax), colorrange=(vmin, vmax))

# Add contour lines for bathymetry
contour!(ax, x_interp/1e3, y_interp/1e3, z_interp,
         levels=contour_levels, color=:black, linewidth=0.5, alpha=0.7)

# Add colorbar
Colorbar(fig[1, 2], hm, label="dB/dz (s⁻²)")

# Save figure
save("output/tilt/average_Bz_first_point.png", fig)

close(ds)

