cd("/scratch/bcpi/cliu28/internal-tide-mixing/")
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using LinearAlgebra
using MAT
using Interpolations
using ColorSchemes

# define a function that loads particle data
function load_particle_data(; simname, z_center_particle=1000, tᶠ="457.0")
    θ = simname == "tilt" ? 0.0036 : 0
    
    filename_particles = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_analysis_round=all_particles_z=",z_center_particle,".nc")
    ds_particles = Dataset(filename_particles,"r")
    
    return Dict(
        "x" => ds_particles["x"][:,:],  # Load all particles, not just first 500
        "y" => ds_particles["y"][:,:],
        "z" => ds_particles["z"][:,:],
        "time" => ds_particles["time"][:],
        "b" => ds_particles["b"][:,:]
    )
end

function calculate_background_buoyancy(θ)
    θ = θ
    ĝ = (sin(θ), 0, cos(θ)) # the vertical unit vector in rotated coordinates
    N = 1e-3
    @inline ẑ(x, z, ĝ) = x*ĝ[1] + z*ĝ[3]
    @inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)

    # Create a background field
    B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

    # Setup grid
    H = 2.25kilometers # vertical extent
    Lx = 15kilometers # along-canyon extent
    Ly = 30kilometers # cross-canyon extent
    Nx = 500
    Ny = 1000
    Nz = 250
    
    # Bottom-intensified stretching for vertical grid
    z_faces(k) = - H * ((1 + ((Nz + 1 - k) / Nz - 1) / 1.2) * 
                        (1 - exp(-15 * (Nz + 1 - k) / Nz)) / (1 - exp(-15)) - 1)

    grid = RectilinearGrid(size=(Nx, Ny, Nz), 
            x = (0, Lx),
            y = (0, Ly), 
            z = z_faces,
            halo = (4, 4, 4),
            topology = (Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded))
            
    model = NonhydrostaticModel(
        grid = grid,
        background_fields = (; b=B̄_field),
        tracers = :b
    )
    
    return interior(compute!(Field(model.background_fields.tracers.b)))[:,1,:]
end

# Function to process and save a time period
function process_time_period(tᶠ_list, period_name, n_particles_to_load, simname, z_center_particle, θ, B̄, ΔB, xC, zC, Lx, Ly, N)
    @info "Processing $period_name period: $(tᶠ_list[1]) to $(tᶠ_list[end])"
    @info "Loading $n_particles_to_load particles per time step"
    
    # Load data for this period
    all_raw_x = []
    all_raw_y = []
    all_raw_z = []
    all_raw_b = []
    all_raw_time = []
    
    for tᶠ in tᶠ_list
        @info "Loading raw data for time $tᶠ"
        data = load_particle_data(; simname, z_center_particle=z_center_particle, tᶠ=tᶠ)
        
        push!(all_raw_x, data["x"][1:n_particles_to_load,:])
        push!(all_raw_y, data["y"][1:n_particles_to_load,:])
        push!(all_raw_z, data["z"][1:n_particles_to_load,:])
        push!(all_raw_b, data["b"][1:n_particles_to_load,:])
        push!(all_raw_time, data["time"][:])
    end
    
    # Concatenate data
    @info "Concatenating raw data for $period_name..."
    x_all = hcat(all_raw_x...)
    y_all = hcat(all_raw_y...)
    z_all = hcat(all_raw_z...)
    b_all = hcat(all_raw_b...)
    time_all = vcat(all_raw_time...)
    
    # Get dimensions
    n_particles, n_times = size(x_all)
    @info "Period $period_name: $n_particles particles × $n_times time steps"
    
    # Clear raw data to free memory
    all_raw_x = nothing
    all_raw_y = nothing
    all_raw_z = nothing
    all_raw_b = nothing
    all_raw_time = nothing
    GC.gc()
    
    # Create output file for this period
    output_file = string("output/", simname, "/concatenated_particle_data_z", z_center_particle, "_", period_name, ".nc")
    @info "Creating output file: $output_file"
    
    ds_output = NCDataset(output_file, "c")
    
    # Define dimensions
    ds_output.dim["particle"] = n_particles
    ds_output.dim["time"] = n_times
    
    # Create variables
    time_var = defVar(ds_output, "time", Float64, ("time",))
    time_var[:] = time_all
    time_var.attrib["units"] = "seconds"
    
    # Create output variables
    x_unwrap_var = defVar(ds_output, "x_unwrapped", Float64, ("particle", "time"))
    x_unwrap_var.attrib["units"] = "meters"
    
    y_unwrap_var = defVar(ds_output, "y_unwrapped", Float64, ("particle", "time"))
    y_unwrap_var.attrib["units"] = "meters"
    
    z_unwrap_var = defVar(ds_output, "z_unwrapped", Float64, ("particle", "time"))
    z_unwrap_var.attrib["units"] = "meters"
    
    x_cart_var = defVar(ds_output, "x_cartesian", Float64, ("particle", "time"))
    x_cart_var.attrib["units"] = "meters"
    
    y_cart_var = defVar(ds_output, "y_cartesian", Float64, ("particle", "time"))
    y_cart_var.attrib["units"] = "meters"
    
    z_cart_var = defVar(ds_output, "z_cartesian", Float64, ("particle", "time"))
    z_cart_var.attrib["units"] = "meters"
    
    b_var = defVar(ds_output, "buoyancy", Float64, ("particle", "time"))
    b_var.attrib["units"] = "m/s²"
    
    # Process in chunks
    chunk_size = 1000
    n_chunks = ceil(Int, n_particles / chunk_size)
    total_wrapping_issues = 0
    
    for chunk_idx in 1:n_chunks
        start_idx = (chunk_idx - 1) * chunk_size + 1
        end_idx = min(chunk_idx * chunk_size, n_particles)
        
        @info "Processing chunk $chunk_idx/$n_chunks: particles $start_idx to $end_idx"
        
        # Extract data for this chunk
        x_chunk = x_all[start_idx:end_idx, :]
        y_chunk = y_all[start_idx:end_idx, :]
        z_chunk = z_all[start_idx:end_idx, :]
        b_chunk = b_all[start_idx:end_idx, :]
        
        n_particles_chunk = size(x_chunk, 1)
        
        # Initialize unwrapped arrays for this chunk
        unwrapped_x_chunk = copy(x_chunk)
        unwrapped_y_chunk = copy(y_chunk)
        unwrapped_z_chunk = copy(z_chunk)
        
        # Process each particle in the chunk
        for i in 1:n_particles_chunk
            # Unwrap x-direction
            for j in 2:n_times
                dx = x_chunk[i, j] - x_chunk[i, j-1]
                
                # Detect and correct boundary crossing
                if dx > 0.5 * Lx
                    # Particle wrapped from right to left
                    dx = dx - Lx
                elseif dx < -0.5 * Lx
                    # Particle wrapped from left to right
                    dx = dx + Lx
                end
                
                # Build unwrapped position incrementally
                unwrapped_x_chunk[i, j] = unwrapped_x_chunk[i, j-1] + dx
            end
            
            # Unwrap y-direction
            for j in 2:n_times
                dy = y_chunk[i, j] - y_chunk[i, j-1]
                
                # Detect and correct boundary crossing
                if dy > 0.5 * Ly
                    # Particle wrapped from top to bottom
                    dy = dy - Ly
                elseif dy < -0.5 * Ly
                    # Particle wrapped from bottom to top
                    dy = dy + Ly
                end
                
                # Build unwrapped position incrementally
                unwrapped_y_chunk[i, j] = unwrapped_y_chunk[i, j-1] + dy
            end
            
            # Add tilt correction to z-direction based on x-crossings
            for j in 1:n_times
                # Calculate how many domain widths we've moved in x
                x_domains_moved = (unwrapped_x_chunk[i, j] - unwrapped_x_chunk[i, 1]) / Lx
                
                # Apply tilt correction
                unwrapped_z_chunk[i, j] = z_chunk[i, j] + x_domains_moved * Lx * sin(θ)
            end
        end
        
        # Calculate buoyancy for this chunk
        B_chunk = zeros(n_particles_chunk, n_times)
        
        for i in 1:n_particles_chunk
            for j in 1:n_times
                # Find grid indices using wrapped coordinates
                ind_x = argmin(abs.(xC[:] .- x_chunk[i, j]))
                ind_z = argmin(abs.(zC[:] .- z_chunk[i, j]))
                
                # Calculate number of crossings
                n_crossings_x = round(Int, (unwrapped_x_chunk[i, j] - x_chunk[i, j]) / Lx)
                
                # Compute total buoyancy
                B_chunk[i, j] = B̄[ind_x, ind_z] + b_chunk[i, j] + n_crossings_x * ΔB
            end
        end
        
        # Transform to Cartesian coordinates
        x_cart_chunk = unwrapped_x_chunk .* cos(θ) .- unwrapped_z_chunk .* sin(θ)
        y_cart_chunk = unwrapped_y_chunk
        z_cart_chunk = unwrapped_x_chunk .* sin(θ) .+ unwrapped_z_chunk .* cos(θ)
        
        # Debug: Check for wrapping issues in this chunk
        chunk_wrapping_issues = 0
        for i in 1:n_particles_chunk
            for j in 1:n_times-1
                dx = x_cart_chunk[i, j+1] - x_cart_chunk[i, j]
                if abs(dx) > 5000
                    chunk_wrapping_issues += 1
                    total_wrapping_issues += 1
                    if total_wrapping_issues <= 5  # Only show first 5 issues across all chunks
                        particle_global_id = start_idx + i - 1
                        @info "WRAPPING STILL PRESENT: Particle $particle_global_id (chunk $chunk_idx), time step $j to $(j+1)"
                        @info "  x_before: $(x_cart_chunk[i, j]), x_after: $(x_cart_chunk[i, j+1])"
                        @info "  Jump: $dx"
                    end
                end
            end
        end
        
        if chunk_wrapping_issues > 0
            @info "Chunk $chunk_idx: Found $chunk_wrapping_issues wrapping issues"
        end
        
        # Save this chunk to file
        x_unwrap_var[start_idx:end_idx, :] = unwrapped_x_chunk
        y_unwrap_var[start_idx:end_idx, :] = unwrapped_y_chunk
        z_unwrap_var[start_idx:end_idx, :] = unwrapped_z_chunk
        x_cart_var[start_idx:end_idx, :] = x_cart_chunk
        y_cart_var[start_idx:end_idx, :] = y_cart_chunk
        z_cart_var[start_idx:end_idx, :] = z_cart_chunk
        b_var[start_idx:end_idx, :] = B_chunk
        
        # Clear chunk data to free memory
        x_chunk = nothing
        y_chunk = nothing
        z_chunk = nothing
        b_chunk = nothing
        unwrapped_x_chunk = nothing
        unwrapped_y_chunk = nothing
        unwrapped_z_chunk = nothing
        B_chunk = nothing
        x_cart_chunk = nothing
        y_cart_chunk = nothing
        z_cart_chunk = nothing
        
        GC.gc()  # Force garbage collection
    end
    
    # Final wrapping check summary for this period
    if total_wrapping_issues == 0
        @info "✓ No wrapping issues found in $period_name data!"
    else
        @info "⚠ Found $total_wrapping_issues wrapping issues in $period_name data"
    end
    
    # Add global attributes
    ds_output.attrib["title"] = "Concatenated Particle Data - $period_name"
    ds_output.attrib["simulation"] = simname
    ds_output.attrib["z_center_particle"] = z_center_particle
    ds_output.attrib["theta"] = θ
    ds_output.attrib["Lx"] = Lx
    ds_output.attrib["Ly"] = Ly
    ds_output.attrib["time_period"] = period_name
    ds_output.attrib["time_range"] = "$(tᶠ_list[1]) to $(tᶠ_list[end])"
    ds_output.attrib["unwrapping_method"] = "robust_incremental_chunked"
    ds_output.attrib["chunk_size"] = chunk_size
    ds_output.attrib["total_particles"] = n_particles
    ds_output.attrib["total_time_steps"] = n_times
    
    close(ds_output)
    
    # Clear remaining arrays
    x_all = nothing
    y_all = nothing
    z_all = nothing
    b_all = nothing
    GC.gc()
    
    @info "✓ $period_name period saved successfully!"
    return total_wrapping_issues
end

# Main execution
simname = "tilt"    # this is the only thing that needs to be changed
z_center_particle = 1000

# Split time periods based on particle count change
tᶠ_early = ["451.5","452.0", "452.5", "453.0", "453.5", "454.0", "454.5", "455.0", "455.5", "456.0", "456.5", "457.0", "457.5", "458.0"]
tᶠ_late = ["458.5"]  # Add more time steps as needed

θ = simname == "tilt" ? 0.0036 : 0

Lx = 15kilometers  # Domain length in x
Nx = 500
Ny = 1000
Ly = 30kilometers  # Domain length in y
N = 1e-3           # Buoyancy frequency
Lz = 2.25kilometers  # Vertical extent (assuming Lz = H from your grid)

# background buoyancy
B̄ = calculate_background_buoyancy(θ)
ΔB = N^2 * Lx*sin(θ)      # Buoyancy increment per domain crossing

# Load grid data for buoyancy calculation
@info "Loading grid data for buoyancy calculation..."
filename = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ_early[1], "_analysis_round=all_threeD.nc")
ds = Dataset(filename,"r")
zC = ds["z_aac"][:];
xC = ds["x_caa"][:]; 
close(ds)

# Process both time periods
@info "=== Processing Early Period (451.5-458.0) ==="
early_issues = process_time_period(tᶠ_early, "early", 499774, simname, z_center_particle, θ, B̄, ΔB, xC, zC, Lx, Ly, N)

@info "=== Processing Late Period (458.5+) ==="
late_issues = process_time_period(tᶠ_late, "late", 499774 + 25000, simname, z_center_particle, θ, B̄, ΔB, xC, zC, Lx, Ly, N)

@info "=== SUMMARY ==="
@info "Early period: $early_issues wrapping issues"
@info "Late period: $late_issues wrapping issues" 
@info "All processing completed successfully!"
@info "Output files created:"
@info "  - concatenated_particle_data_z$(z_center_particle)_early.nc (451.5-458.0, 499774 particles)"
@info "  - concatenated_particle_data_z$(z_center_particle)_late.nc (458.5+, 524774 particles)"