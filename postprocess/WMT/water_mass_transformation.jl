#  This script calculates the water mass transformation
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using Interpolations
using MAT
using LinearAlgebra

function deriv(z,y)
   dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
   return dydz
end

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins_2d.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/mmderiv.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bin_stat_over_xy.jl")

# Parameters
tᶠ = 457.0
θ = 3.6e-3
if θ==3.6e-3
    simname = "tilt"
else 
    simname = "flat"
end
base_folder = "/scratch/bcpi/cliu28/internal-tide-mixing/"

# Define the time chunks to process (from 452.0 to 457.0)
time_chunks = [452.0, 452.5, 453.0, 453.5, 454.0, 454.5, 455.0, 455.5, 456.0, 456.5, 457.0]
# time_chunks = [452.0]
# Get grid information from the first file
filename_first = string(base_folder, "output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", time_chunks[1], "_analysis_round=all_threeD.nc")
ds_first = Dataset(filename_first, "r")

# Extract grid information
zC = ds_first["z_aac"][:]
zF = ds_first["z_aaf"][:]
xC = ds_first["x_caa"][:]
xF = ds_first["x_faa"][:]
yC = ds_first["y_aca"][:]
yF = ds_first["y_afa"][:]
Nz = length(zC)
Nx = length(xC)
dx = xF[end] - xF[end-1]
Ny = length(yC)
dy = yF[end] - yF[end-1]

close(ds_first)

# Load hab and verification data
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:,:,:]
close(ds_hab)

filename_b_verification = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_b = Dataset(filename_b_verification, "r")
b = ds_b["b"][:,:,:,1:1]
close(ds_b)

# Coordinate transformation
if simname == "tilt"
    Lx = (xF[end]+dx) * cos(θ)
    Lz = (xF[end]+dx) * sin(θ)
    x = xC * cos(θ) .- zC' * sin(θ)
    z = xC * sin(θ) .+ zC' * cos(θ)
    z_face = xF * sin(θ) .+ zF' * cos(θ)
    N = 1e-3
    ΔB = N^2*Lz 

    # Binning parameters
    ranges = 0:-1:-40
    bin_edge1 = 0:8:1500
    bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
    bin_edge2 = (0.1:0.05:0.9).*1e-3
    bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2
else # flat case
    z_face = zF
    bin_edge1 = 0:8:1500
    bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
    bin_edge2 = collect(range(0.0005353, 0.0016, length=41))
    bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2
end

# Function to process water mass transformation for a region
function process_wmt_region(region_name, y_indices=nothing)
    println("Processing WMT for region: ", region_name)
    
    # Initialize arrays to store all timesteps
    all_∇κ∇B = []
    all_∇κ∇B_b = []
    all_times = []
    
    # Process each time chunk
    for (i, tf_current) in enumerate(time_chunks)
        println("Processing chunk ending at: ", tf_current)
        
        # Create filenames for current chunk
        filename_3D = string(base_folder, "output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tf_current, "_analysis_round=all_threeD.nc")
        
        # Open datasets
        ds_3D = Dataset(filename_3D, "r")
        
        # Get number of timesteps
        Nt_current = length(ds_3D["time"][:])
        times_current = ds_3D["time"][:]
        
        # Determine which timesteps to process
        if i == 1
            # For first chunk, only take the final timestep
            timestep_range = Nt_current:Nt_current
        else
            # For subsequent chunks, take all timesteps
            timestep_range = 1:Nt_current
        end
        
        # Extract regional data based on y_indices
        if y_indices !== nothing
            bin_mask1 = hab[:, y_indices, :]
            b_region = b[:, y_indices, :, :]
        else
            bin_mask1 = hab
            b_region = b
        end
        
        # Process each timestep
        for n in timestep_range
            println("  Processing timestep ", n, " of ", Nt_current)
            
            # Extract data for current timestep
            if y_indices !== nothing
                ∇κ∇B = ds_3D["∇κ∇B"][:, y_indices, :, n:n]
                div_uB = ds_3D["div_uB"][:, y_indices, :, n:n]
                B = ds_3D["B"][:, y_indices, :, n:n]
            else
                ∇κ∇B = ds_3D["∇κ∇B"][:, :, :, n:n]
                div_uB = ds_3D["div_uB"][:, :, :, n:n]
                B = ds_3D["B"][:, :, :, n:n]
            end
            
            # Mask out regions where b == 0
            B[b_region .== 0] .= 0
            
            # Initialize accumulators
            int_∇κ∇B = zeros(length(bin_center1), length(bin_center2), 1) # [hab, buoyancy]
            int_∇κ∇B_b = zeros(length(bin_center2), 1) # [buoyancy]
            
            if simname == "tilt"
                # Process extended domain for tilt case
                for m in ranges
                    z̃_face = z_face .+ m*Lz
                    B̃ = B .+ m*ΔB
                    bin_mask2 = B̃
                    
                    f1, _, _ = bins_2d(∇κ∇B, bin_edge1, bin_edge2, bin_mask1, bin_mask2, dx=dx, dy=dy, z_face=z̃_face, normalize=false)
                    int_∇κ∇B += f1
                    f2, _ = bins(∇κ∇B, bin_edge2, bin_mask2, dx=dx, dy=dy, z_face=z̃_face, normalize=false)
                    int_∇κ∇B_b += f2
                    @info "m = $m, n= $n"  # Progress tracking
                end
            else
                # Process flat case
                bin_mask2 = B
                int_∇κ∇B, _, _ = bins_2d(∇κ∇B, bin_edge1, bin_edge2, bin_mask1, bin_mask2, dx=dx, dy=dy, z_face=z_face, normalize=false)
                int_∇κ∇B_b, _ = bins(∇κ∇B, bin_edge2, bin_mask2, dx=dx, dy=dy, z_face=z_face, normalize=false)
            end

            # Store results of buoyancy derivatives
            push!(all_∇κ∇B, int_∇κ∇B./diff(bin_edge2)[1])
            push!(all_∇κ∇B_b, int_∇κ∇B_b./diff(bin_edge2)[1])
            push!(all_times, times_current[n])
        end
        
        close(ds_3D)
        GC.gc()
    end
    
    # Combine results into arrays
    total_timesteps = length(all_times)
    ∇κ∇B_t = zeros(length(bin_center1), length(bin_center2), total_timesteps)
    ∇κ∇B_b_t = zeros(length(bin_center2), total_timesteps)
    
    for i in 1:total_timesteps
        ∇κ∇B_t[:, :, i] = all_∇κ∇B[i][:, :, 1]
        ∇κ∇B_b_t[:, i] = all_∇κ∇B_b[i][:, 1]
    end
    
    # Save results
    output_filename = string(base_folder, "output/", simname, "/WMT_", region_name, "_tᶠ=", tᶠ, "_#bintervals=",length(bin_edge2),"_instantaneous.nc")
    
    Dataset(output_filename, "c") do ds_create
        # Define dimensions
        defDim(ds_create, "z_TF", length(bin_center1))
        defDim(ds_create, "buoyancy", length(bin_center2))
        defDim(ds_create, "buoyancy_diff", length(bin_center2))
        defDim(ds_create, "time", total_timesteps)
        
        # Define global attributes
        ds_create.attrib["title"] = "2D histogram, binned by hab and buoyancy - instantaneous snapshots"
        ds_create.attrib["time_chunks"] = string(time_chunks)
        ds_create.attrib["total_timesteps"] = total_timesteps
        ds_create.attrib["region"] = region_name
        
        # Define variables
        v1 = defVar(ds_create, "∇κ∇B_t", Float64, ("z_TF", "buoyancy_diff", "time"))
        v1[:, :, :] = ∇κ∇B_t
        v2 = defVar(ds_create, "∇κ∇B_b_t", Float64, ("buoyancy_diff", "time"))
        v2[:, :] = ∇κ∇B_b_t
        v3 = defVar(ds_create, "bin_center1", Float64, ("z_TF",))
        v3[:] = bin_center1
        v4 = defVar(ds_create, "bin_center2", Float64, ("buoyancy",))
        v4[:] = bin_center2
        v5 = defVar(ds_create, "t", Float64, ("time",))
        v5[:] = all_times
        
        # Write attributes
        v1.attrib["units"] = "m³/s"
        v2.attrib["units"] = "m³/s"
        v3.attrib["units"] = "m"
        v4.attrib["units"] = "m/s²"
        v5.attrib["units"] = "s"
    end
    
    println("Saved WMT results for ", region_name, " to: ", output_filename)
end

# Process different regions based on y-coordinates
println("Processing WMT for all regions using instantaneous snapshots...")

# 1) Whole domain
process_wmt_region("total")

# 2) Canyon region (y between 10km and 20km)
north = argmin(abs.(yC[:] .- 20e3))
south = argmin(abs.(yC[:] .- 10e3))
process_wmt_region("canyon", south:north)

# 3) Flanks region (y < 10km and y > 20km)
process_wmt_region("flanks", vcat(1:south, north:length(yC)))

println("All WMT processing complete!")