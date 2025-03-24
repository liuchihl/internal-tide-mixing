# Computing tracer-weighted buoyancy B_c as volume integral of tracer*B divided by volume integral of tracer

# Loading required packages
using Oceananigans
using Oceananigans.Units
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using Printf

# Define function to load tracer concentration data from the dataset
function load_tracer_data(simname)
    θ = simname == "tilt" ? 0.0036 : 0.0
    tᶠ = "460"
    
    # Load the 3D tracer concentration dataset
    filename = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_B-c.nc")
    ds = Dataset(filename, "r")
    
    # Get grid information
    zC = ds["zC"][:]
    xC = ds["xC"][:]
    yC = ds["yC"][:] # Assuming y-coordinates are available in the dataset
    time = ds["time"][:]
    
    # Load tracer concentration (c) and buoyancy perturbation (b) fields
    # Note: Variable names might need adjustment based on actual dataset structure
    tracer_c = ds["c"][:,:,:,:] # Assuming format is [x,y,z,time]
    buoyancy_b = ds["b"][:,:,:,:] # Buoyancy perturbation field
    
    ds_info = Dict(
        "xC" => xC,
        "yC" => yC,
        "zC" => zC,
        "time" => time,
        "c" => tracer_c,
        "b" => buoyancy_b
    )
    
    return ds_info
end

# Function to calculate B_c (tracer-weighted buoyancy) over time
function calculate_B_c(simname, B̄_field)
    # Load 3D tracer data
    data = load_tracer_data(simname)
    
    # Get dimensions
    Nx = length(data["xC"])
    Ny = length(data["yC"])
    Nz = length(data["zC"])
    Nt = length(data["time"])
    
    # Grid parameters (from your original code)
    H = 2.25kilometers # vertical extent
    Lx = 15kilometers # along-canyon extent
    Ly = 30kilometers # cross-canyon extent
    θ = simname == "tilt" ? 0.0036 : 0.0

    # Calculate cell volumes (simplified if uniform grid)
    # For a non-uniform grid, this would need to be adjusted
    dx = Lx / Nx
    dy = Ly / Ny
    
    # For the vertically stretched grid, calculate dz at each level
    # This is an approximation based on your z_faces function
    dz = zeros(Nz)
    for k in 1:Nz
        z_top = z_faces(k+1)
        z_bottom = z_faces(k)
        dz[k] = abs(z_top - z_bottom)
    end
    
    # Function to calculate z-faces (from your original code)
    function z_faces(k)
        return -H * ((1 + ((Nz + 1 - k) / Nz - 1) / 1.2) * 
                    (1 - exp(-15 * (Nz + 1 - k) / Nz)) / (1 - exp(-15)) - 1)
    end
    
    # Initialize array to store B_c over time
    B_c_time_series = zeros(Nt)
    
    # Loop over time steps
    for t in 1:Nt
        numerator = 0.0  # ∫ c*B dV
        denominator = 0.0  # ∫ c dV
        
        # Loop over spatial grid
        for i in 1:Nx, j in 1:Ny, k in 1:Nz
            # Get tracer concentration at this point
            c_val = data["c"][i,j,k,t]
            
            # Calculate total buoyancy: background + perturbation
            # Background buoyancy depends on position
            x_val = data["xC"][i]
            z_val = data["zC"][k]
            
            # Get background buoyancy from the pre-computed field
            # Find closest indices in the background field
            ind_x = argmin(abs.(data["xC"] .- x_val))
            ind_z = argmin(abs.(data["zC"] .- z_val))
            
            # Background buoyancy + perturbation
            B_total = B̄_field[ind_x, ind_z] + data["b"][i,j,k,t]
            
            # Note: For the tilted case, we'd need to account for the periodic domain extension
            # similar to your particle calculations, but applied to the 3D field
            
            # For tilted case, account for buoyancy change across periodic boundaries
            if simname == "tilt"
                # In a full implementation, we would need to track domain crossings
                # This is a simplified version assuming no domain crossings in the eulerian field
                # A complete solution would require tracking the phase of waves
            end
            
            # Cell volume
            cell_volume = dx * dy * dz[k]
            
            # Add to integrals
            numerator += c_val * B_total * cell_volume
            denominator += c_val * cell_volume
        end
        
        # Calculate B_c for this time step
        if denominator > 0
            B_c_time_series[t] = numerator / denominator
        else
            B_c_time_series[t] = NaN  # Avoid division by zero
        end
    end
    
    return Dict(
        "time" => data["time"],
        "B_c" => B_c_time_series
    )
end

# Calculate B_c for both cases
B_c_tilt = calculate_B_c("tilt", B̄_tilt)
B_c_flat = calculate_B_c("flat", B̄_flat)

# Convert time to hours for plotting
tilt_time_hours = B_c_tilt["time"] / 3600
flat_time_hours = B_c_flat["time"] / 3600

# Create figure to compare all buoyancy measures
fig = Figure(resolution=(1500, 600), fontsize=20)

ax1 = Axis(fig[1, 1], 
          xlabel="Time (hours)", 
          ylabel="Buoyancy", 
          title="Tilt Case (θ=0.0036)",
          xminorticksvisible = true,
          yminorticksvisible = true,
          xminorticks = IntervalsBetween(5),
          yminorticks = IntervalsBetween(4))

ax2 = Axis(fig[1, 2], 
          xlabel="Time (hours)", 
          title="Flat Case (θ=0)",
          xminorticksvisible = true,
          yminorticksvisible = true,
          xminorticks = IntervalsBetween(5),
          yminorticks = IntervalsBetween(4))

# Set consistent scientific notation format for y-axis ticks
for ax in [ax1, ax2]
    ax.ytickformat = x -> [@sprintf("%.5f", v) for v in x]
end

# Plot for Tilt case
lines!(ax1, tilt_time_hours, tilt_mean_total_B, 
     linewidth=3, color=:red, label="Particle mean B")
lines!(ax1, tilt_time_hours, B_c_tilt["B_c"], 
     linewidth=3, color=:purple, linestyle=:dash, label="Tracer-weighted B_c")

# Plot for Flat case
lines!(ax2, flat_time_hours, flat_mean_total_B, 
     linewidth=3, color=:blue, label="Particle mean B")
lines!(ax2, flat_time_hours, B_c_flat["B_c"], 
     linewidth=3, color=:purple, linestyle=:dash, label="Tracer-weighted B_c")

# Add legends
axislegend(ax1, position=:lt)
axislegend(ax2, position=:lt)

# Save the figure
save("output/comparison_particle_vs_tracer_weighted_buoyancy.png", fig)

# Print statistics to compare the two measures
println("Tilt case:")
println("  Mean of particle-based buoyancy: $(mean(tilt_mean_total_B))")
println("  Mean of tracer-weighted buoyancy: $(mean(skipmissing(B_c_tilt["B_c"])))")
println("  Correlation: $(cor(tilt_mean_total_B, B_c_tilt["B_c"]))")

println("Flat case:")
println("  Mean of particle-based buoyancy: $(mean(flat_mean_total_B))")
println("  Mean of tracer-weighted buoyancy: $(mean(skipmissing(B_c_flat["B_c"])))")
println("  Correlation: $(cor(flat_mean_total_B, B_c_flat["B_c"]))")