# Computing tracer-weighted buoyancy B_c as volume integral of tracer*B divided by volume integral of tracer

# Loading required packages
using Oceananigans
using Oceananigans.Units
using CairoMakie
using NCDatasets: Dataset  # Explicitly import Dataset from NCDatasets
using Statistics
using NaNStatistics
using Printf
using Dates

# General function to calculate tracer-weighted values for any field
function tracer_weighted_values(c_field, target_field, cell_volumes; 
                               chunk_size=50, tracer_threshold=1e-12)
    """
    Calculate tracer-weighted values for any field.
    
    Parameters:
    - c_field: 3D tracer concentration field [Nx, Ny, Nz]
    - target_field: 3D field to be weighted [Nx, Ny, Nz]
    - cell_volumes: 1D array of cell volumes [Nz]
    - chunk_size: number of vertical levels to process at once
    - tracer_threshold: minimum tracer value to consider
    
    Returns:
    - tracer-weighted value (scalar)
    """
    
    Nx, Ny, Nz = size(c_field)
    
    # Initialize accumulators
    field_weighted_sum = 0.0
    total_tracer_volume = 0.0
    
    # Process in chunks to save memory
    for k_start in 1:chunk_size:Nz
        k_end = min(k_start + chunk_size - 1, Nz)
        
        # Process this chunk
        for k in k_start:k_end
            cell_vol = cell_volumes[k]
            
            # Get slices for this level
            c_slice = c_field[:,:,k]
            target_slice = target_field[:,:,k]
            
            # Create mask for cells with tracer
            tracer_mask = c_slice .> tracer_threshold
            
            if any(tracer_mask)
                # Calculate weighted sums
                tracer_weights = c_slice[tracer_mask] * cell_vol
                field_weighted_sum += sum(tracer_weights .* target_slice[tracer_mask])
                total_tracer_volume += sum(tracer_weights)
            end
        end
        
        # Force garbage collection for chunks
        GC.gc()
    end
    
    # Calculate tracer-weighted average
    if total_tracer_volume > tracer_threshold
        return field_weighted_sum / total_tracer_volume
    else
        return NaN
    end
end

# Rest as normal code for flexibility:

# Define time range to process
tᶠ_start = 452.0
tᶠ_end = 457.0
tᶠ_step = 0.5
tᶠ_list = collect(tᶠ_start:tᶠ_step:tᶠ_end)

# Initialize arrays to store results
all_times_tilt = Float64[]
all_times_flat = Float64[]
all_w_tracer_tilt = Float64[]
all_w_tracer_flat = Float64[]
all_B_weighted_tilt = Float64[]
all_B_weighted_flat = Float64[]

println("Processing $(length(tᶠ_list)) files from tᶠ=$tᶠ_start to tᶠ=$tᶠ_end")

# Initialize grid variables
zC = nothing
zF = nothing
Nx = Ny = Nz = 0
dx = dy = 0.0
cell_volumes = nothing

# Process each file
for (file_idx, tᶠ) in enumerate(tᶠ_list)
    println("Processing file $file_idx/$(length(tᶠ_list)): tᶠ=$tᶠ")
    
    # Process both cases
    for simname in ["tilt"]
        θ = simname == "tilt" ? 0.0036 : 0.0
        filename = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_analysis_round=all_threeD.nc")
        
        try
            ds = Dataset(filename, "r")
            
            # Get grid information (only once)
            if file_idx == 1 && simname == "tilt"
                zC = ds["z_aac"][:]
                zF = ds["z_aaf"][:]
                Nx, Ny, Nz, _ = size(ds["c"])
                
                # Grid parameters
                H = 2.25kilometers
                Lx = 15kilometers
                Ly = 30kilometers
                dx = Lx / Nx
                dy = Ly / Ny
                dz = diff(zF)
                
                # Pre-calculate cell volumes
                cell_volumes = zeros(Nz)
                for k in 1:Nz
                    cell_volumes[k] = dx * dy * dz[k]
                end
            end
            
            time_file = ds["time"][:]
            Nt = length(time_file)
            w_tracer_file = zeros(Nt)
            B_weighted_file = zeros(Nt)

            # Process each time step
            for t in 1:Nt
                # Load current time step data
                c_field = ds["c"][:,:,:,t]
                B_field = ds["B"][:,:,:,t]
                omega_field = ds["∇κ∇B"][:,:,:,t]
                
                # Compute gradient magnitude using your exact method
                grad_B_magnitude = zeros(Nx, Ny, Nz)
                for k in 1:Nz
                    B_slice = B_field[:,:,k]
                    
                    # X-direction gradient using periodic boundaries
                    B_face_x = diff(vcat(B_slice[end:end,:], B_slice), dims=1) ./ dx
                    grad_B_x = (vcat(B_face_x[2:end, :], B_face_x[1:1, :]) .+ B_face_x) ./ 2
                    
                    # Y-direction gradient using periodic boundaries
                    B_face_y = diff(hcat(B_slice[:,end:end], B_slice), dims=2) ./ dy
                    grad_B_y = (hcat(B_face_y[:,2:end], B_face_y[:,1:1]) .+ B_face_y) ./ 2
                    
                    # Z-direction gradient
                    if k > 1 && k < Nz
                        grad_B_z_above = (B_field[:,:,k+1] - B_slice) / (zC[k+1] - zC[k])
                        grad_B_z_below = (B_slice - B_field[:,:,k-1]) / (zC[k] - zC[k-1])
                        grad_B_z = (grad_B_z_above .+ grad_B_z_below) ./ 2
                    elseif k == 1
                        grad_B_z = zeros(Nx, Ny)
                    elseif k == Nz
                        grad_B_z = fill(1e-6, Nx, Ny)
                    end
                    
                    grad_B_magnitude[:,:,k] = sqrt.(grad_B_x.^2 + grad_B_y.^2 + grad_B_z.^2)
                end
                
                # Compute tracer-weighted values
                omega_weighted = tracer_weighted_values(c_field, omega_field, cell_volumes)
                grad_B_weighted = tracer_weighted_values(c_field, grad_B_magnitude, cell_volumes)
                B_weighted = tracer_weighted_values(c_field, B_field, cell_volumes)
                
                # Compute diapycnal velocity
                if grad_B_weighted > 1e-12
                    w_tracer_file[t] = 2 * omega_weighted / grad_B_weighted
                else
                    w_tracer_file[t] = NaN
                end
                
                # Store B_weighted in an array over time
                B_weighted_file[t] = B_weighted     
                
                # Clear memory
                c_field = nothing
                B_field = nothing
                omega_field = nothing
                grad_B_magnitude = nothing
                B_magnitude = nothing
                GC.gc()
                println("Processed timestep $t of $Nt for $simname")
            end
            
            # Store results
            if simname == "tilt"
                append!(all_times_tilt, time_file)
                append!(all_w_tracer_tilt, w_tracer_file)
                append!(all_B_weighted_tilt, B_weighted_file)
            else
                append!(all_times_flat, time_file)
                append!(all_w_tracer_flat, w_tracer_file)
                append!(all_B_weighted_flat, B_weighted_file)
            end
            
            close(ds)
            
        catch e
            println("Error processing $simname tᶠ=$tᶠ: $e")
            continue
        end
    end
    
    GC.gc()
end

# Sort results by time
sort_indices_tilt = sortperm(all_times_tilt)
all_times_tilt = all_times_tilt[sort_indices_tilt]
all_w_tracer_tilt = all_w_tracer_tilt[sort_indices_tilt]
all_B_weighted_tilt = all_B_weighted_tilt[sort_indices_tilt]

sort_indices_flat = sortperm(all_times_flat)
all_times_flat = all_times_flat[sort_indices_flat]
all_w_tracer_flat = all_w_tracer_flat[sort_indices_flat]
all_B_weighted_flat = all_B_weighted_flat[sort_indices_flat]

# Save as NetCDF files
println("Saving results...")

# Save tilt case
output_filename_tilt = "output/tracer_weighted_vertical_velocity_tilt.nc"
Dataset(output_filename_tilt, "c") do ds
    # Define dimensions
    defDim(ds, "time", length(all_times_tilt))
    
    # Define variables
    time_var = defVar(ds, "time", Float64, ("time",))
    w_tracer_var = defVar(ds, "w_tracer", Float64, ("time",))
    
    # Add attributes
    time_var.attrib["long_name"] = "Time"
    time_var.attrib["units"] = "seconds"
    time_var.attrib["description"] = "Time since simulation start"
    
    w_tracer_var.attrib["long_name"] = "Tracer-weighted vertical velocity"
    w_tracer_var.attrib["units"] = "m/s"
    w_tracer_var.attrib["description"] = "Vertical velocity computed as 2*omega/|∇B| where omega is ∇κ∇B"
    w_tracer_var.attrib["formula"] = "w = 2 * <c*∇κ∇B> / <c*|∇B|>"

    # Add B-weighted variable
    B_weighted_var = defVar(ds, "B_weighted", Float64, ("time",))
    B_weighted_var.attrib["long_name"] = "Tracer-weighted buoyancy"
    B_weighted_var.attrib["units"] = "m/s²"
    B_weighted_var.attrib["description"] = "Buoyancy weighted by tracer concentration"  
    B_weighted_var.attrib["formula"] = "B_c = <c*B> / <c>"
    
    # Global attributes
    ds.attrib["title"] = "Tracer-weighted vertical velocity for tilt case"
    ds.attrib["created"] = string(now())
    ds.attrib["case"] = "tilt"
    ds.attrib["time_range"] = "tᶠ = $tᶠ_start to $tᶠ_end with step $tᶠ_step"
    ds.attrib["total_time_points"] = length(all_times_tilt)
    
    # Write data
    time_var[:] = all_times_tilt
    w_tracer_var[:] = all_w_tracer_tilt
    B_weighted_var[:] = all_B_weighted_tilt
end

println("Tilt results saved to: $output_filename_tilt")

# Save flat case
# output_filename_flat = "output/tracer_weighted_vertical_velocity_flat.nc"
# Dataset(output_filename_flat, "c") do ds
#     # Define dimensions
#     defDim(ds, "time", length(all_times_flat))
    
#     # Define variables
#     time_var = defVar(ds, "time", Float64, ("time",))
#     w_tracer_var = defVar(ds, "w_tracer", Float64, ("time",))
    
#     # Add attributes
#     time_var.attrib["long_name"] = "Time"
#     time_var.attrib["units"] = "seconds"
#     time_var.attrib["description"] = "Time since simulation start"
    
#     w_tracer_var.attrib["long_name"] = "Tracer-weighted vertical velocity"
#     w_tracer_var.attrib["units"] = "m/s"
#     w_tracer_var.attrib["description"] = "Vertical velocity computed as 2*omega/|∇B| where omega is ∇κ∇B"
#     w_tracer_var.attrib["formula"] = "w = 2 * <c*∇κ∇B> / <c*|∇B|>"
    
#     # Global attributes
#     ds.attrib["title"] = "Tracer-weighted vertical velocity for flat case"
#     ds.attrib["created"] = string(now())
#     ds.attrib["case"] = "flat"
#     ds.attrib["time_range"] = "tᶠ = $tᶠ_start to $tᶠ_end with step $tᶠ_step"
#     ds.attrib["total_time_points"] = length(all_times_flat)
    
#     # Write data
#     time_var[:] = all_times_flat
#     w_tracer_var[:] = all_w_tracer_flat
# end

# println("Flat results saved to: $output_filename_flat")

# Create a combined NetCDF file with both cases
# combined_filename = "output/tracer_weighted_vertical_velocity_combined.nc"
# Dataset(combined_filename, "c") do ds
#     # Define dimensions
#     defDim(ds, "time_tilt", length(all_times_tilt))
#     defDim(ds, "time_flat", length(all_times_flat))
    
#     # Define variables for tilt case
#     time_tilt_var = defVar(ds, "time_tilt", Float64, ("time_tilt",))
#     w_tracer_tilt_var = defVar(ds, "w_tracer_tilt", Float64, ("time_tilt",))
    
#     # Define variables for flat case
#     time_flat_var = defVar(ds, "time_flat", Float64, ("time_flat",))
#     w_tracer_flat_var = defVar(ds, "w_tracer_flat", Float64, ("time_flat",))
    
#     # Add attributes
#     time_tilt_var.attrib["long_name"] = "Time - Tilt Case"
#     time_tilt_var.attrib["units"] = "seconds"
#     w_tracer_tilt_var.attrib["long_name"] = "Tracer-weighted vertical velocity - Tilt Case"
#     w_tracer_tilt_var.attrib["units"] = "m/s"
    
#     time_flat_var.attrib["long_name"] = "Time - Flat Case"
#     time_flat_var.attrib["units"] = "seconds"
#     w_tracer_flat_var.attrib["long_name"] = "Tracer-weighted vertical velocity - Flat Case"
#     w_tracer_flat_var.attrib["units"] = "m/s"
    
#     # Global attributes
#     ds.attrib["title"] = "Tracer-weighted vertical velocity comparison"
#     ds.attrib["created"] = string(now())
#     ds.attrib["time_range"] = "tᶠ = $tᶠ_start to $tᶠ_end with step $tᶠ_step"
    
#     # Write data
#     time_tilt_var[:] = all_times_tilt
#     w_tracer_tilt_var[:] = all_w_tracer_tilt
#     time_flat_var[:] = all_times_flat
#     w_tracer_flat_var[:] = all_w_tracer_flat
# end

# println("Combined results saved to: $combined_filename")

# Add plotting for vertical velocity
fig_w = Figure(resolution=(1500, 600), fontsize=20)

ax1_w = Axis(fig_w[1, 1], 
            xlabel="Time (hours)", 
            ylabel="Vertical Velocity (m/s)", 
            title="Tracer-weighted Vertical Velocity - Tilt Case",
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorticks = IntervalsBetween(5),
            yminorticks = IntervalsBetween(4))

ax2_w = Axis(fig_w[1, 2], 
            xlabel="Time (hours)", 
            ylabel="Vertical Velocity (m/s)", 
            title="Tracer-weighted Vertical Velocity - Flat Case",
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorticks = IntervalsBetween(5),
            yminorticks = IntervalsBetween(4))

# Plot vertical velocities
lines!(ax1_w, all_times_tilt / 3600, all_w_tracer_tilt, 
       linewidth=3, color=:red, label="w_tracer")
lines!(ax2_w, all_times_flat / 3600, all_w_tracer_flat, 
       linewidth=3, color=:blue, label="w_tracer")

# Add legends
axislegend(ax1_w, position=:lt)
axislegend(ax2_w, position=:lt)

# Save the figure
save("output/tracer_weighted_vertical_velocity_all_files.png", fig_w)

# Print summary statistics
println("\nSummary Statistics:")
println("Tilt case ($(length(all_times_tilt)) time points):")
println("  Time range: $(minimum(all_times_tilt)/3600) to $(maximum(all_times_tilt)/3600) hours")
println("  Mean w_tracer: $(mean(skipmissing(all_w_tracer_tilt))) m/s")
println("  Std w_tracer: $(std(skipmissing(all_w_tracer_tilt))) m/s")

println("Flat case ($(length(all_times_flat)) time points):")
println("  Time range: $(minimum(all_times_flat)/3600) to $(maximum(all_times_flat)/3600) hours")
println("  Mean w_tracer: $(mean(skipmissing(all_w_tracer_flat))) m/s")
println("  Std w_tracer: $(std(skipmissing(all_w_tracer_flat))) m/s")

# Function to read the saved NetCDF files (for future use)
function read_tracer_weighted_results(filename)
    Dataset(filename, "r") do ds
        return Dict(
            "time" => ds["time"][:],
            "w_tracer" => ds["w_tracer"][:]
        )
    end
end

println("Analysis complete!")