

## save all particles positions and plot the buoyancy at each particle position
simname = "tilt"    # this is the only thing that needs to be changed
z_center_particle = 1000  # center of the particle layer
output_file = string("output/", simname, "/concatenated_particle_data_z", z_center_particle, ".nc")
ds = NCDataset(output_file, "r")

# Plot buoyancy vs time for all particles
fig = Figure(resolution=(900, 600), fontsize=16)
ax = Axis(fig[1, 1],
          xlabel="Time (hours)",
          ylabel="Total Buoyancy (B)",
          title="Total Buoyancy Evolution for Individual Particles")

# Constants
Lx = 15kilometers  # Domain length in x
Ly = 30kilometers  # Domain length in y
N = 1e-3           # Buoyancy frequency
θ = 0.0036         # Tilt angle for "tilt" case
ΔB = N^2 * Lx * sin(θ)  # Buoyancy increment per domain crossing

# Load background buoyancy field
B̄ = calculate_background_buoyancy(θ)

# Get grid coordinates
filename = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ[1], "_analysis_round=all_threeD.nc")
ds = Dataset(filename, "r")
zC = ds["z_aac"][:]
xC = ds["x_caa"][:]
close(ds)

# Initialize arrays to store all time series data
all_time_series = []
all_B_series = []
# Loop through all time snapshots and calculate all data
for (idx, time_value) in enumerate(tᶠ)
    @info "Processing time step: $time_value"
    
    ds = load_particle_data(; simname, z_center_particle=1000, tᶠ=time_value)
    time = ds["time"][:] / 3600  # Convert to hours
    b_perturbation = ds["b"][:,:]
    x_pos = ds["x"][:,:]
    y_pos = ds["y"][:,:]
    z_pos = ds["z"][:,:]
    
    n_particles, n_times = size(b_perturbation)
    
    # Initialize arrays for this time snapshot
    B_total = zeros(n_particles, n_times)
    unwrapped_x = zeros(n_particles, n_times)
    unwrapped_y = zeros(n_particles, n_times)
    unwrapped_z = zeros(n_particles, n_times)
    n_crossings = zeros(Int, n_particles, n_times)
    
    # Set initial unwrapped positions
    unwrapped_x[:, 1] = x_pos[:, 1]
    unwrapped_y[:, 1] = y_pos[:, 1]
    unwrapped_z[:, 1] = z_pos[:, 1]
    
    # Compute unwrapped positions and total buoyancy over time
    for i in 1:n_particles
        for j in 1:(n_times-1)
            # Calculate displacement between consecutive time steps
            dx = x_pos[i, j+1] - x_pos[i, j]
            dy = y_pos[i, j+1] - y_pos[i, j]
            dz = z_pos[i, j+1] - z_pos[i, j]
            
            # Handle periodic crossing in x direction
            if dx > 0.5 * Lx  # Moved left-to-right across boundary
                dx -= Lx
                dz -= Lx * sin(θ)  # Account for tilt
                n_crossings[i, j+1] = n_crossings[i, j] - 1
            elseif dx < -0.5 * Lx  # Moved right-to-left across boundary
                dx += Lx
                dz += Lx * sin(θ)  # Account for tilt
                n_crossings[i, j+1] = n_crossings[i, j] + 1
            else
                n_crossings[i, j+1] = n_crossings[i, j]  # No crossing
            end
            
            # Handle periodic crossing in y direction
            if dy > 0.5 * Ly  # Moved across boundary in y-direction
                dy -= Ly
            elseif dy < -0.5 * Ly
                dy += Ly
            end
            
            # Update unwrapped positions
            unwrapped_x[i, j+1] = unwrapped_x[i, j] + dx
            unwrapped_y[i, j+1] = unwrapped_y[i, j] + dy
            unwrapped_z[i, j+1] = unwrapped_z[i, j] + dz
        end
        
        # Calculate total buoyancy for all time steps
        for j in 1:n_times
            # Grid indices for background buoyancy
            ind_x = argmin(abs.(xC[:] .- x_pos[i, j]))
            ind_z = argmin(abs.(zC[:] .- z_pos[i, j]))
            
            # Compute total buoyancy: background + perturbation + domain crossing increment
            B_total[i, j] = (B̄[ind_x, ind_z] + 
                            b_perturbation[i, j] + 
                            n_crossings[i, j] * ΔB)
        end
    end
    
    # Store the time series and data for this snapshot
    push!(all_time_series, time)
    push!(all_B_series, B_total)
    
    # Calculate and save mean and standard deviation
    mean_B = nanmean(B_total, dims=1)[1,:]
    std_B = nanstd(B_total, dims=1)[1,:]
    
    # Save snapshot data to individual files
    snapshot_file = string("output/", simname, "/particle_buoyancy_snapshot_z", z_center_particle, "_t", time_value, ".nc")
    
    # Create the NetCDF file for this snapshot
    ds_snapshot = NCDataset(snapshot_file, "c")
    
    # Define dimensions
    ds_snapshot.dim["time"] = length(time)
    ds_snapshot.dim["particle"] = n_particles
    
    # Create variables
    time_var = defVar(ds_snapshot, "time", Float64, ("time",))
    time_var[:] = time
    time_var.attrib["units"] = "hours"
    
    b_var = defVar(ds_snapshot, "total_buoyancy", Float64, ("particle", "time"))
    b_var[:,:] = B_total
    b_var.attrib["units"] = "m/s²"
    
    x_var = defVar(ds_snapshot, "x", Float64, ("particle", "time"))
    x_var[:,:] = unwrapped_x
    x_var.attrib["units"] = "m"
    
    y_var = defVar(ds_snapshot, "y", Float64, ("particle", "time"))
    y_var[:,:] = unwrapped_y
    y_var.attrib["units"] = "m"
    
    z_var = defVar(ds_snapshot, "z", Float64, ("particle", "time"))
    z_var[:,:] = unwrapped_z
    z_var.attrib["units"] = "m"
    
    mean_b_var = defVar(ds_snapshot, "mean_buoyancy", Float64, ("time",))
    mean_b_var[:] = mean_B
    mean_b_var.attrib["units"] = "m/s²"
    
    std_b_var = defVar(ds_snapshot, "std_buoyancy", Float64, ("time",))
    std_b_var[:] = std_B
    std_b_var.attrib["units"] = "m/s²"
    
    # Add global attributes
    ds_snapshot.attrib["title"] = "Particle Buoyancy Data"
    ds_snapshot.attrib["simulation"] = simname
    ds_snapshot.attrib["z_center_particle"] = z_center_particle
    ds_snapshot.attrib["time_step"] = time_value
    
    # Close the file
    close(ds_snapshot)
    @info "Saved snapshot data to $snapshot_file"
end






#### Plot the mean buoyancy evolution with standard deviation bands
simname = "tilt"
z_center_particle = 1000

# List all files in the output directory
base_dir = string("output/", simname, "/")
all_files = readdir(base_dir, join=true)

# Filter files that match the pattern for particle buoyancy snapshots
snapshot_files = filter(f -> 
    occursin("particle_buoyancy_snapshot_z$(z_center_particle)_t", f) && 
    endswith(f, ".nc"), 
    all_files)

# If no files are found, print a diagnostic message
if isempty(snapshot_files)
    @warn "No snapshot files found matching pattern 'particle_buoyancy_snapshot_z$(z_center_particle)_t*.nc' in $base_dir"
    # List a few files from the directory to check what's available
    @info "Sample files in directory: $(all_files[1:min(5, length(all_files))])"
end
# Initialize arrays to store the merged data
all_times = Float64[]
all_mean_buoyancy = Float64[]
all_std_buoyancy = Float64[]

# Load and combine data from all snapshots
for file in snapshot_files
    ds = Dataset(file,"r")
    # Extract the time values and convert to hours if needed
    times = ds["time"][2:end]
    mean_B = ds["mean_buoyancy"][2:end]
    std_B = ds["std_buoyancy"][2:end]
    close(ds)
    
    # Append to the combined arrays
    append!(all_times, times)
    append!(all_mean_buoyancy, mean_B)
    append!(all_std_buoyancy, std_B)
end

# Sort by time to ensure proper ordering
sorted_indices = sortperm(all_times)
all_times = all_times[sorted_indices]
all_mean_buoyancy = all_mean_buoyancy[sorted_indices]
all_std_buoyancy = all_std_buoyancy[sorted_indices]

# Create the figure
fig = CairoMakie.Figure(resolution=(900, 600), fontsize=16)
ax = Axis(fig[1, 1],
            xlabel="Time (hours)",
            ylabel="Buoyancy (m/s²)",
            title="Mean Particle Buoyancy Evolution",
            xminorticksvisible=true,
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(5),
            yminorticks=IntervalsBetween(4))

# Plot mean buoyancy with standard deviation band
band!(ax, all_times, all_mean_buoyancy .- all_std_buoyancy, all_mean_buoyancy .+ all_std_buoyancy,
        color=(Makie.RGB(0.4, 0.6, 0.8), 0.3),
        label="± 1 Standard Deviation")

lines!(ax, all_times, all_mean_buoyancy,
        linewidth=3,
        color=:blue,
        label="Mean Buoyancy")

# Add legend
axislegend(ax, position=:lt)

# Save the figure
output_file = string("output/", simname, "/mean_buoyancy_evolution_z", z_center_particle, ".png")
@info "Saving figure to $output_file"
save(output_file, fig)
