using StatsBase

@views function bins(var, bin_edge, bin_var; dx, dy, z_face, normalize = false)
    """
    1D binning using StatsBase's histogram.
    
    Parameters:
    -----------
    var : Array
        Variable to bin (dimensions: x,y,z,t or x,1,z,t)
    bin_edge : Vector
        Bin edges for the variable
    bin_var : Array
        Variable for binning (same dimensions as var)
    dx, dy : Float64
        Grid spacing in x and y directions
    z_face : Matrix
        Z-face positions for volume calculation (dimensions: x,z+1)
    normalize : Bool
        Whether to normalize by volume
    
    Returns:
    --------
    integrand : Array
        Binned data
    bin_center : Vector
        Bin centers
    Chih-Lun Liu, Nov 12 2024
    """
    # Calculate bin centers
    bin_center = @. (bin_edge[1:end-1] + bin_edge[2:end]) / 2
    
    # Calculate volume elements
    z_diff = diff(z_face, dims=2)
    ΔV = dx * dy * z_diff  # Shape: [nx, nz]
    
    # Initialize output arrays
    nt = size(var, 4)
    integrand = zeros(length(bin_center), nt)
    norm_volume = normalize ? zeros(length(bin_center), nt) : nothing
    
    # Determine if the data is 2D (i.e., y dimension is 1)
    is_2D = size(var, 2) == 1
    
    # Pre-allocate arrays for the flattened data
    nx, ny, nz = size(var)[1:3]
    flat_length = nx * ny * nz
    
    # Create views for better performance
    var_view = view(var, :, :, :, :)
    bin_var_view = view(bin_var, :, :, :, :)
    
    # Process each time step
    for l in 1:nt
        # Flatten spatial dimensions
        var_flat = reshape(view(var_view, :, :, :, l), flat_length)
        bin_var_flat = reshape(view(bin_var_view, :, :, :, l), flat_length)
        
        # Create repeated ΔV array
        if is_2D
            ΔV_flat = repeat(vec(ΔV), ny)  # ny is 1 in this case
        else
            ΔV_flat = repeat(vec(ΔV), inner=ny)
        end
        
        # Calculate histogram with weights
        weights = Weights(var_flat .* ΔV_flat)
        h = StatsBase.fit(Histogram, bin_var_flat, weights, bin_edge)
        integrand[:, l] = h.weights
        
        if normalize
            # Calculate volume histogram
            vol_weights = Weights(ΔV_flat)
            h_vol = StatsBase.fit(Histogram, bin_var_flat, vol_weights, bin_edge)
            norm_volume[:, l] = h_vol.weights
        end
    end
    
    if normalize
        # Avoid division by zero
        bin_var = norm_volume .> 0
        integrand[bin_var] ./= norm_volume[bin_var]
    end
    
    return integrand, bin_center
end