using StatsBase

@views function bins_2d(var, bin_edge1, bin_edge2, bin_mask1, bin_mask2; dx, dy, z_face, normalize = false)
    # Calculate bin centers
    bin_center1 = @. (bin_edge1[1:end-1] + bin_edge1[2:end]) / 2
    bin_center2 = @. (bin_edge2[1:end-1] + bin_edge2[2:end]) / 2
    
    # Calculate volume elements
    z_diff = diff(z_face, dims=2)
    ΔV = dx .* dy .* z_diff  # Shape: [nx, nz]
    
    # Initialize output arrays
    nt = size(var, 4)
    integrand = zeros(length(bin_center1), length(bin_center2), nt)
    norm_volume = normalize ? zeros(length(bin_center1), length(bin_center2), nt) : nothing
    
    # Pre-allocate arrays for the flattened data
    nx, ny, nz = size(var)[1:3]
    flat_length = nx * ny * nz
    
    # Create views for better performance
    var_view = view(var, :, :, :, :)
    mask1_view = view(bin_mask1, :, :, :, :)
    mask2_view = view(bin_mask2, :, :, :, :)
    
    # Process each time step
    for l in 1:nt
        # Flatten spatial dimensions
        var_flat = reshape(view(var_view, :, :, :, l), flat_length)
        mask1_flat = reshape(view(mask1_view, :, :, :, l), flat_length)
        mask2_flat = reshape(view(mask2_view, :, :, :, l), flat_length)
        
        # Create repeated ΔV array
        ΔV_flat = repeat(vec(ΔV), inner=ny)
        
        # Create weights
        weights = Weights(var_flat .* ΔV_flat)
        
        # Calculate histogram
        h = StatsBase.fit(Histogram, (mask1_flat, mask2_flat), weights, 
               (bin_edge1, bin_edge2))
        
        integrand[:, :, l] = h.weights
        
        if normalize
            # Calculate volume histogram
            vol_weights = Weights(ΔV_flat)
            h_vol = StatsBase.fit(Histogram, (mask1_flat, mask2_flat),
                       vol_weights, (bin_edge1, bin_edge2))
            norm_volume[:, :, l] = h_vol.weights
        end
    end
    
    if normalize
        # Avoid division by zero
        mask = norm_volume .> 0
        integrand[mask] ./= norm_volume[mask]
    end
    
    return integrand, bin_center1, bin_center2
end


## using loops, but much slower!
# @views function bins_2d(var, bin_edge1, bin_edge2, bin_mask1, bin_mask2; dx, dy, z_face, normalize = false)
#     # This function bins data based on two variables.
#     # bin_edge1: edges for the first variable
#     # bin_edge2: edges for the second variable
#     # bin_mask1, bin_mask2: masks for the two variables

#     bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
#     bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2
#     integrand = zeros(length(bin_center1), length(bin_center2), size(var, 4))
#     norm_volume = zeros(length(bin_center1), length(bin_center2), size(var, 4))
#     z_diff = diff(z_face, dims=2)
#     ΔV = dx .* dy .* z_diff  # Assuming z_diff dimensions align with the last two dimensions of var

#     for l in 1:size(var, 4)
#         for m in 1:length(bin_center1)
#             for n in 1:length(bin_center2)
#                 sum_var = 0.0
#                 sum_vol = 0.0
#                 for k in 1:size(var, 3)
#                     for j in 1:size(var, 2)
#                         for i in 1:size(var, 1)
#                             if bin_edge1[m] <= bin_mask1[i, j, k, l] < bin_edge1[m+1] &&
#                                bin_edge2[n] <= bin_mask2[i, j, k, l] < bin_edge2[n+1]
#                                 count = var[i, j, k, l] * ΔV[i, k]
#                                 sum_var += count
#                                 if normalize
#                                     sum_vol += ΔV[i, k]
#                                 end
#                             end
#                         end
#                     end
#                 end
#                 integrand[m, n, l] = sum_var
#                 if normalize
#                     norm_volume[m, n, l] = sum_vol
#                 end
#             end
#         end
#     end

#     if normalize
#         return integrand ./ norm_volume, bin_center1, bin_center2
#     else
#         return integrand, bin_center1, bin_center2
#     end
# end


# test:
# slope = "tilt"
# θ = 0.0036
# timerange = "40-80"
# filename_field_budget = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg_Bbudget.nc")
# ds_budget = Dataset(filename_field_budget,"r")
# filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
# ds_field = Dataset(filename_field,"r")

# zC = ds_budget["zC"][:]; zF = ds_budget["zF"][:];
# Nz=length(zC[:]); 

# xC = ds_budget["xC"][:]; xF = ds_budget["xF"][:]; 
# Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

# yC = ds_budget["yC"][:]; yF = ds_budget["yF"][:]
# Ny=length(yC[:]);       dy = yF[end]-yF[end-1];

# Lx = (xF[end]+dx) * cos(θ)
# Lz = (xF[end]+dx) * sin(θ)
# x = xC * cos(θ) .- zC' * sin(θ)
# z = xC * sin(θ) .+ zC' * cos(θ)
# z_face = xF * sin(θ) .+ zF' * cos(θ)
# N = 1e-3
# ΔB = N^2*Lz 
# z_face = xF * sin(θ) .+ zF' * cos(θ)     # Cartesian coordinate

# t = ds_budget["time"][:];

# # load hab
# filename_hab = "output/hab.nc"
# ds_hab = Dataset(filename_hab,"r")
# hab = ds_hab["hab"][:,:,:];
# ∇κ∇B = ds_budget["∇κ∇B"][:,:,:,1:1];    # ∇⋅κ∇B: buoyancy flux divergence
# B = ds_field["B"][:,:,:,1:1];    # ∇⋅κ∇B: buoyancy flux divergence

# # var, bin_edge1, bin_edge2, bin_mask1, bin_mask2
# # inputs for the 2D binning
#     ranges = 0:-1:-40
#     var = ∇κ∇B[:,:,:,1]
#     bin_edge1 = 0:10:1500#0
#     bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
#     # bin_edge2 = (0.1:0.02:0.9).*1e-3  # Define the edges of the bins
#     bin_edge2 = (0.1:0.05:0.9).*1e-3  # Define the edges of the bins
#     bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2
#     bin_mask1 = hab
#     int_∇κ∇B = zeros(length(bin_edge1)-1,length(bin_edge2)-1,1)
#     # int_div_uB = zeros(length(bin_edge)-1,1)
#     # int_dBdt = zeros(length(bin_edge)-1,1)
        
#     for n in ranges
#         z̃_face = z_face .+ n*Lz     # extended domain
#         B̃ = B .+ n*ΔB
#         bin_mask2 = B̃
#         @time f1, _, _= bins_2d(var,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
#         int_∇κ∇B += f1
#         @show n
#     end