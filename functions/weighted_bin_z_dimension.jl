@views function weighted_bin_z_dimension(var, hab, new_bin_centers, bandwidth=4.0)
    """
    Weighted binning where each data point contributes to nearby bins
    bandwidth: controls how far influence extends (in same units as hab)
    """
    dims = size(var)
    is_4d = length(dims) == 4
    
    if is_4d
        nx, ny, nz, nt = dims
    else
        nx, ny, nz = dims
        nt = 1
    end
    
    num_new_bins = length(new_bin_centers)
    T = eltype(var)
    
    # Only allocate final result array
    if is_4d
        binned_var = fill(T(NaN), nx, ny, num_new_bins, nt)
    else
        binned_var = fill(T(NaN), nx, ny, num_new_bins)
    end
    
    # Pre-calculate weight threshold distance to avoid unnecessary calculations
    max_distance = bandwidth * sqrt(-2 * log(0.01))  # distance where weight = 0.01
    
    # Process each spatial location
    for j in 1:ny, i in 1:nx, l in 1:nt
        # Temporary arrays for this column only
        weighted_vals = zeros(T, num_new_bins)
        weight_sums = zeros(Float64, num_new_bins)
        
        for k in 1:nz
            if is_4d
                val = var[i, j, k, l]
                hab_val = hab[i, j, k]
            else
                val = var[i, j, k]
                hab_val = hab[i, j, k]
            end
            
            if isnan(val) || isnan(hab_val)
                continue
            end
            
            # Only calculate weights for bins within reasonable distance
            for bin_idx in 1:num_new_bins
                distance = abs(hab_val - new_bin_centers[bin_idx])
                
                # Skip if too far away
                if distance > max_distance
                    continue
                end
                
                # Gaussian weighting
                weight = exp(-(distance^2) / (2 * bandwidth^2))
                
                weighted_vals[bin_idx] += val * weight
                weight_sums[bin_idx] += weight
            end
        end
        
        # Normalize and store results for this column
        for bin_idx in 1:num_new_bins
            if weight_sums[bin_idx] > 0
                if is_4d
                    binned_var[i, j, bin_idx, l] = weighted_vals[bin_idx] / weight_sums[bin_idx]
                else
                    binned_var[i, j, bin_idx] = weighted_vals[bin_idx] / weight_sums[bin_idx]
                end
            end
        end
    end
    
    return binned_var, new_bin_centers
end

# Illustration of usage:
# Calculate distances from height=10 to each bin center

# distances = [|10-4|, |10-12|, |10-20|, |10-28|] = [6, 2, 10, 18]

# # Calculate weights using Gaussian kernel (bandwidth=4)
# weights = exp.(-distances.^2 / (2*4^2))
# weights = [exp(-36/32), exp(-4/32), exp(-100/32), exp(-324/32)]
# weights = [0.32, 0.88, 0.04, 0.0001]

# # Contribution to each bin
# bin_1: 100 × 0.32 = 32
# bin_2: 100 × 0.88 = 88  
# bin_3: 100 × 0.04 = 4
# bin_4: 100 × 0.0001 ≈ 0