using Interpolations
using NCDatasets
using NaNStatistics
@views function interpolate_z_dimension(var, hab, new_bin_edges)
    
    """
    Interpolate the z-dimension of a 4D matrix onto new bin edges.
    
    Parameters:
    -----------
    var : Array{T,4}
        Input 4D matrix (x, y, z, t)
    hab : Array{T,3}
        Height above bottom coordinate 
    new_bin_edges : Vector{Float64}
        New z coordinates for interpolation
    
    Returns:
    --------
    interpolated_var : Array{T,4}
        Interpolated 4D matrix with new z dimension
    bin_centers : Vector{Float64}
        Centers of the new z bins
    """
    # Get dimensions
    nx, ny, nz, nt = size(var)
    
    # Calculate new bin centers
    bin_centers = @. (new_bin_edges[1:end-1] + new_bin_edges[2:end]) / 2
    num_new_bins = length(bin_centers)
    
    # Initialize output array
    interpolated_var = zeros(nx, ny, num_new_bins, nt)
    
    # Interpolate for each x, y, and t
    for l in 1:nt
        for j in 1:ny
            for i in 1:nx
                # Extract 1D slice for this x,y,t location
                slice = var[i, j, :, l]
                
                # Create interpolation object
                itp = LinearInterpolation(hab[i,j,:], slice, extrapolation_bc=Interpolations.Flat())
                
                # Interpolate to new bin centers
                interpolated_var[i, j, :, l] = itp(bin_centers)
            end
        end
    end
    
    return interpolated_var, bin_centers
end




