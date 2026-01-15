using NaNStatistics

@views function bin_z_dimension(var, hab, new_bin_edges)
    """
    Bin the z-dimension of a 3D or 4D matrix onto new bin edges using spatial binning.
    Handles cases where the y-dimension is missing (i.e., [nx, nz] or [nx, nz, nt]).
    """
    # Ensure var and hab have at least 3 dimensions: [nx, ny, nz] or [nx, ny, nz, nt]
    if ndims(var) == 2
        var = reshape(var, size(var,1), 1, size(var,2))
    end
    if ndims(hab) == 2
        hab = reshape(hab, size(hab,1), 1, size(hab,2))
    end
    if ndims(var) == 3 && size(var,3) == size(hab,3) && size(var,2) == size(hab,2)
        # [nx, ny, nz]
        nx, ny, nz = size(var)
        nt = 1
        is_4d = false
    elseif ndims(var) == 4
        nx, ny, nz, nt = size(var)
        is_4d = true
    else
        error("Input var must be [nx, nz], [nx, nz, nt], [nx, 1, nz], [nx, 1, nz, nt], [nx, ny, nz], or [nx, ny, nz, nt]")
    end

    # Calculate new bin centers
    bin_centers = @. (new_bin_edges[1:end-1] + new_bin_edges[2:end]) / 2
    num_new_bins = length(bin_centers)

    # Initialize output array with appropriate dimensions
    T = eltype(var)
    if is_4d
        binned_var = fill(T(NaN), nx, ny, num_new_bins, nt)
    else
        binned_var = fill(T(NaN), nx, ny, num_new_bins)
    end

    # Process each spatial location
    for j in 1:ny, i in 1:nx
        for l in 1:nt
            vals_per_bin = [T[] for _ in 1:num_new_bins]
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
                bin_idx = searchsortedfirst(new_bin_edges, hab_val) - 1
                if bin_idx >= 1 && bin_idx <= num_new_bins
                    push!(vals_per_bin[bin_idx], val)
                end
            end
            for bin_idx in 1:num_new_bins
                if !isempty(vals_per_bin[bin_idx])
                    if is_4d
                        binned_var[i, j, bin_idx, l] = mean(vals_per_bin[bin_idx])
                    else
                        binned_var[i, j, bin_idx] = mean(vals_per_bin[bin_idx])
                    end
                end
            end
        end
    end

    # If input was 2D, drop y dimension in output for convenience
    if ndims(var) == 3 && size(var,2) == 1
        binned_var = dropdims(binned_var; dims=2)
    elseif ndims(var) == 4 && size(var,2) == 1
        binned_var = dropdims(binned_var; dims=2)
    end

    return binned_var, bin_centers
end