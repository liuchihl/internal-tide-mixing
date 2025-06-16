using StatsBase

@views function bins_pdf(var, bin_edge, bin_var; dx, dy, z_face, n_pdf_bins=100, pdf_limits=nothing)
    # Calculate bin centers for the spatial binning (hab)
    bin_center = @. (bin_edge[1:end-1] + bin_edge[2:end]) / 2
    
    # Calculate volume elements
    z_diff = diff(z_face)
    ΔV = dx * dy * z_diff  # This might be [nz] or [nx, nz] depending on z_face shape
    
    # Determine dimensions
    nx, ny, nz = size(var)[1:3]
    nt = size(var, 4)
    n_hab_bins = length(bin_center)
    
    # Debug dimensions
    println("Input dimensions:")
    println("var: $(size(var))")
    println("bin_var: $(size(bin_var))")
    println("ΔV: $(size(ΔV))")
    
    # If pdf_limits not provided, determine from data
    if isnothing(pdf_limits)
        all_valid_values = Float64[]
        for l in 1:nt
            var_flat = reshape(view(var, :, :, :, l), :)
            bin_var_flat = reshape(view(bin_var, :, :, :), :)
            
            # Only use shared valid indices
            indices = 1:min(length(var_flat), length(bin_var_flat))
            valid_points = view(bin_var_flat, indices) .> 0
            
            append!(all_valid_values, view(var_flat, indices)[valid_points])
        end
        
        # Find the min and max, with some padding
        min_val = minimum(all_valid_values)
        max_val = maximum(all_valid_values)
        range = max_val - min_val
        pdf_limits = (min_val - 0.05*range, max_val + 0.05*range)
    end
    
    # Create edges for the PDF bins
    pdf_edges = LinRange(pdf_limits[1], pdf_limits[2], n_pdf_bins+1)
    pdf_centers = @. (pdf_edges[1:end-1] + pdf_edges[2:end]) / 2
    
    # Initialize PDF array: [hab bins, pdf bins]
    pdf_values = zeros(n_hab_bins, n_pdf_bins)
    
    # Collect all valid data across all timesteps
    all_var_valid = Float64[]
    all_bin_valid = Float64[]
    all_ΔV_valid = Float64[]
    
    for l in 1:nt
        # Ensure proper dimensionality
        if size(bin_var, 4) > 1
            bin_var_3d = view(bin_var, :, :, :, l)
        else
            bin_var_3d = view(bin_var, :, :, :)
        end
        
        var_3d = view(var, :, :, :, l)
        
        # Check dimensions
        if size(var_3d) != size(bin_var_3d)
            println("Warning: Dimension mismatch between var and bin_var at timestep $l")
            println("var_3d: $(size(var_3d)), bin_var_3d: $(size(bin_var_3d))")
            # Proceed with the minimum common indices
            min_nx = min(size(var_3d, 1), size(bin_var_3d, 1))
            min_ny = min(size(var_3d, 2), size(bin_var_3d, 2))
            min_nz = min(size(var_3d, 3), size(bin_var_3d, 3))
            var_3d = view(var_3d, 1:min_nx, 1:min_ny, 1:min_nz)
            bin_var_3d = view(bin_var_3d, 1:min_nx, 1:min_ny, 1:min_nz)
        end
        
        # Create properly sized volume array
        if length(size(ΔV)) == 1  # ΔV is [nz]
            nx_local, ny_local, nz_local = size(var_3d)
            if length(ΔV) >= nz_local
                ΔV_local = view(ΔV, 1:nz_local)
                ΔV_3D = ones(nx_local, ny_local, 1) .* reshape(ΔV_local, 1, 1, :)
            else
                println("Warning: ΔV size ($(length(ΔV))) is smaller than nz ($nz_local)")
                ΔV_3D = ones(nx_local, ny_local, 1) .* reshape(ΔV, 1, 1, :)
            end
        else  # ΔV is [nx, nz] or other format
            println("Warning: Complex ΔV format detected. Using uniform volume weighting.")
            ΔV_3D = ones(size(var_3d))
        end
        
        # Flatten arrays
        var_flat = reshape(var_3d, :)
        bin_var_flat = reshape(bin_var_3d, :)
        ΔV_flat = reshape(ΔV_3D, :)
        
        # Ensure all arrays have the same length
        min_length = min(length(var_flat), length(bin_var_flat), length(ΔV_flat))
        var_flat = var_flat[1:min_length]
        bin_var_flat = bin_var_flat[1:min_length]
        ΔV_flat = ΔV_flat[1:min_length]
        
        # Filter out points where bin_var is 0 (below topography)
        valid_points = bin_var_flat .> 0
        append!(all_var_valid, var_flat[valid_points])
        append!(all_bin_valid, bin_var_flat[valid_points])
        append!(all_ΔV_valid, ΔV_flat[valid_points])
    end
    
    # Compute PDF for each height bin using all valid data
    for i in 1:n_hab_bins
        # Get points in this height bin
        bin_start = bin_edge[i]
        bin_end = bin_edge[i+1]
        in_bin = (all_bin_valid .>= bin_start) .& (all_bin_valid .< bin_end)
        
        if sum(in_bin) > 0
            # Get values in this bin
            values_in_bin = all_var_valid[in_bin]
            volumes_in_bin = all_ΔV_valid[in_bin]
            
            # Compute weighted histogram (PDF)
            weights = Weights(volumes_in_bin)
            h = fit(Histogram, values_in_bin, weights, pdf_edges)
            
            # Normalize to get PDF
            total_weight = sum(h.weights)
            if total_weight > 0
                pdf_values[i, :] = h.weights ./ total_weight
            end
        end
    end
    
    return pdf_values, bin_center, pdf_centers
end

# Example usage:
# pdf_values, hab_centers, rig_centers = bin_pdf(Rig, hab_edges, hab_values; 
#                                              dx=dx, dy=dy, z_face=z_face,
#                                              n_pdf_bins=100, 
#                                              pdf_limits=(-10, 10))
# 
# # For plotting with PyPlot:
# using PyPlot
# 
# figure(figsize=(10, 6))
# contourf(rig_centers, hab_centers, pdf_values, 20, cmap="viridis")
# colorbar(label="PDF")
# xlabel("Richardson Number (Rig)")
# ylabel("Height Above Bottom (hab)")
# title("PDF of Richardson Number vs Height Above Bottom")