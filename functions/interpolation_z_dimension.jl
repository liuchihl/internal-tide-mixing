using Interpolations
using NCDatasets
using NaNStatistics

@views function interpolate_z_dimension(var, hab, new_bin_edges)
    """
    Interpolate the z-dimension of a 3D or 4D matrix onto new bin edges.
    
    Parameters:
    -----------
    var : Array{T,3} or Array{T,4}
        Input 3D (x, y, z) or 4D matrix (x, y, z, t)
    hab : Array{T,3} or Array{T,2}
        Height above bottom coordinate
    new_bin_edges : Vector{Float64}
        New z coordinates for interpolation
    
    Returns:
    --------
    interpolated_var : Array{T,3} or Array{T,4}
        Interpolated matrix with new z dimension
    bin_centers : Vector{Float64}
        Centers of the new z bins
    """
    # Get dimensions and check dimensionality
    dims = size(var)
    is_4d = length(dims) == 4
    
    if is_4d
        nx, ny, nz, nt = dims
    else
        nx, ny, nz = dims
        nt = 1
    end
    
    # Calculate new bin centers
    bin_centers = @. (new_bin_edges[1:end-1] + new_bin_edges[2:end]) / 2
    num_new_bins = length(bin_centers)
    
    # Initialize output array with appropriate dimensions
    if is_4d
        interpolated_var = zeros(nx, ny, num_new_bins, nt)
    else
        interpolated_var = zeros(nx, ny, num_new_bins)
    end
    
    # Interpolate for each x, y, and t
    for l in 1:nt
        for j in 1:ny
            for i in 1:nx
                # Extract 1D slice for this x,y,t location
                if is_4d
                    slice = var[i, j, :, l]
                    hab_slice = hab[i, j, :]
                else
                    slice = var[i, j, :]
                    hab_slice = hab[i, j, :]
                end
                
                # Create interpolation object
                itp = LinearInterpolation(hab_slice, slice, extrapolation_bc=Interpolations.Flat())
                
                # Interpolate to new bin centers
                if is_4d
                    interpolated_var[i, j, :, l] = itp(bin_centers)
                else
                    interpolated_var[i, j, :] = itp(bin_centers)
                end
            end
        end
    end
    
    return interpolated_var, bin_centers
end
