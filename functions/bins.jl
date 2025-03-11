using StatsBase

@views function bins(var, bin_edge, bin_var; dx, dy, z_face, normalize = false)
    # Calculate bin centers
    bin_center = @. (bin_edge[1:end-1] + bin_edge[2:end]) / 2
    
    # Calculate volume elements
    z_diff = diff(z_face)
    ΔV = dx * dy * z_diff  # This might be [nz] or [nx, nz] depending on z_face shape
    
    # Initialize output arrays
    nt = size(var, 4)
    integrand = zeros(length(bin_center), nt)
    norm_volume = normalize ? zeros(length(bin_center), nt) : nothing
    
    # Determine if the data is 2D (i.e., y dimension is 1)
    is_2D = size(var, 2) == 1
    
    # Get dimensions
    nx, ny, nz = size(var)[1:3]
    
    # Process each time step
    for l in 1:nt
        # Flatten spatial dimensions for this time step
        var_flat = reshape(view(var, :, :, :, l), :)
        bin_var_flat = reshape(view(bin_var, :, :, :), :)
        
        # Create properly sized volume array
        # First, create the correct sized ΔV array based on dimensions
        if length(size(ΔV)) == 1  # ΔV is [nz]
            # Need to replicate across x and y dimensions
            ΔV_3D = ones(nx, ny, 1) .* reshape(ΔV, 1, 1, :)
        else  # ΔV is [nx, nz]
            # Need to replicate across y dimension
            ΔV_3D = ones(1, ny, 1) .* reshape(ΔV, nx, 1, :)
        end
        
        # Flatten volume array to match other arrays
        ΔV_flat = reshape(ΔV_3D, :)
        
        # Print debug info
        # println("Dimensions: var_flat: $(size(var_flat)), bin_var_flat: $(size(bin_var_flat)), ΔV_flat: $(size(ΔV_flat))")
        
        # Filter out points where bin_var is 0 (below topography)
        valid_points = bin_var_flat .> 0
        var_valid = var_flat[valid_points]
        bin_valid = bin_var_flat[valid_points]
        ΔV_valid = ΔV_flat[valid_points]
        
        # Calculate histogram with weights, only using valid points
        weights = Weights(var_valid .* ΔV_valid)
        h = StatsBase.fit(Histogram, bin_valid, weights, bin_edge)
        integrand[:, l] = h.weights
        
        if normalize
            # Calculate volume histogram
            vol_weights = Weights(ΔV_valid)
            h_vol = StatsBase.fit(Histogram, bin_valid, vol_weights, bin_edge)
            norm_volume[:, l] = h_vol.weights
        end
    end
    
    if normalize
        # Avoid division by zero
        bin_mask = norm_volume .> 0
        integrand[bin_mask] ./= norm_volume[bin_mask]
    end
    
    return integrand, bin_center
end