@views function bins(var, bin_edge, bin_mask; dx, dy, z_face, normalize = false)
    # This is a function for binning quantities, similar to what a histogram would do
    # the main goal is to use it for height above bottom calculations and water mass transformation
    # var: variable that has multi-dimensions, i.e., (x,y,z) or (x,y,z,t). 
    # bin_edge: edges of the bin, it's a vector.
    # bin_mask: could be B(x,y,z,t) or hab(x,y,z). Should have the same dimensional as var
    # in most cases, dx, dy are uniform
    # z_face: is a 2-D matrix
    # normalize: if true, the integrand is normalized by its volume, if false only computes integrand
    #+----------------------+
    # Chih-Lun Liu Nov 5 2024    
    
    bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
    integrand = zeros(length(bin_center), size(var, 4))
    norm_volume = zeros(length(bin_center), size(var, 4))
    z_diff = diff(z_face, dims=2)
    ΔV = dx .* dy .* z_diff  # Assuming z_diff dimensions align with the last two dimensions of var

    for l in 1:size(var, 4)
        for m in 1:length(bin_center)
            sum_var = 0.0
            sum_vol = 0.0
            for k in 1:size(var, 3)
                for j in 1:size(var, 2)
                    for i in 1:size(var, 1)
                        if bin_edge[m] <= bin_mask[i, j, k, l] < bin_edge[m+1]
                            count = var[i, j, k, l] * ΔV[i, k]
                            sum_var += count 
                            if normalize
                                sum_vol += ΔV[i, k]
                            end
                        end
                    end
                end
            end
            integrand[m, l] = sum_var
            if normalize
                norm_volume[m, l] = sum_vol 
            end
        end
    end

    if normalize
        return integrand ./ norm_volume, bin_center
    else
        return integrand, bin_center
    end
end

# broadcast approach, but much slower...
# function bins(var, bin_edge, bin_mask; dx, dy, z_face, normalize = false)
     
#     # Calculate the bin centers
#     bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
    
#     # Initialize the layer mask and integrand arrays
#     layer_mask = zeros(size(var))
#     integrand = zeros(length(bin_center), size(var,4))
#     norm_volume = zeros(length(bin_center), size(var,4))
    
#     # Calculate volume element ΔV 
#     z_diff = diff(z_face, dims=2)
#     ΔV = reshape(dx .* dy .* z_diff, (size(var, 1), 1, size(var, 3), 1))

#     # Loop over each bin center to calculate integrand
#     for i in 1:length(bin_center)
#             layer_mask .= bin_edge[i] .<= bin_mask .< bin_edge[i+1]
#             integrand[i, :] = nansum(var .* layer_mask .* ΔV, dims=(1, 2, 3))
#         if normalize == true
#             norm_volume[i,:] = nansum(layer_mask .* ΔV, dims=(1, 2, 3))
#         end
#     end

#     if normalize
#         return integrand./norm_volume, bin_center
#     else
#         return integrand, bin_center
#     end
# end


## HOW TO USE THIS FUNCTION, SEE THE TESTS BELOW:
#1 WMT
# using NCDatasets
# using NaNStatistics
# filename_field = "output/test_Bbudget/internal_tide_theta=0.0036_realtopo3D_Nx=10_Nz=5_0-1_threeD_timeavg.nc"
#     ds_field = Dataset(filename_field,"r")

# zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
# Nz=length(zC[:]); 
# xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
# Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

# yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
# Ny=length(yC[:]);       dy = yF[end]-yF[end-1];

# t = ds_field["time"][:];

# θ = 3.6e-3
# # for n in 
# n = 2
# b = ds_field["b"][:,:,:,n:n];          # buoyancy perturbation
# B = ds_field["B"][:,:,:,n:n];          # total buoyancy
# ∇κ∇B = ds_field["∇κ∇B"][:,:,:,n:n];    # ∇⋅κ∇B: buoyancy flux divergence
# ∇κ∇B[b.==0] .= NaN;
# B[b.==0] .= NaN;

# ## coordinate transformation from slope-coordinate to Cartesian coordinate
# Lx = (xF[end]+dx) * cos(θ)
# Lz = (xF[end]+dx) * sin(θ)
# x = xC * cos(θ) .- zC' * sin(θ)
# z = xC * sin(θ) .+ zC' * cos(θ)
# z_face = xF * sin(θ) .+ zF' * cos(θ)
# y = yC
# # extended coordinate
# x̃ = x; z̃ = z; z̃_face = z_face; B̃ = B;
# N = 1e-3
# ΔB = N^2*Lz    
# for n in -1:-1:-2
# x̃ = [x .+ n*Lx ; x̃ ; x .- n*Lx]
# z̃ = [z .+ n*Lz ; z̃ ; z .- n*Lz]
# z̃_face = [z_face .+ n*Lz ; z̃_face ; z_face .- n*Lz]
# B̃ = [B .+ n*ΔB ; B̃ ; B .- n*ΔB]
# end
# ∇κ∇B̃ = [∇κ∇B;∇κ∇B;∇κ∇B;∇κ∇B;∇κ∇B];
# bin_edge = (1:2:20)*1e-4  # Define the edges of the bins

# int_∇κ∇B̃,bin_center = bin_loop(∇κ∇B̃,bin_edge,B̃,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
# using PyPlot
# close(gcf())
# plot(collect(bin_center),int_∇κ∇B̃)
# gcf()


# ## second test: terrain-following averages
#     using NCDatasets
#     using NaNStatistics
#     slope = "tilt"
#     θ = 0.0036
#     timerange = "0-40"
#     filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
#     ds_field = Dataset(filename_field,"r")
  


#     # grids
#     zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
#     Nz=length(zC[:]); 
#     xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
#     θ=0.0036
#     z_face = xF * sin(θ) .+ zF' * cos(θ)     # Cartesian coordinate

#     Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

#     yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
#     Ny=length(yC[:]);       dy = yF[end]-yF[end-1];

#     t = ds_field["time"][:];

#     ds_hab = Dataset("output/hab.nc","r")
#     hab = ds_hab["hab"][:,:,:];
#     bin_mask = hab
#     bin_edge = 0:5:1500

#     ε = ds_field["ε"][:,:,:,40];
#     @time int_ε,bin_center = bin_loop(ε,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
#     @time int_interpolate = terrain_following_fast(hab, ε, bin_edge, Nx, Ny, Int(zF[end]))

#     using CairoMakie
#  fig = Figure()
#  ax = Axis(fig[1,1], xlabel="ε",ylabel="hab (m)", 
#             limits = (nothing, (0, 100)))
#     ln1=lines!(ax,log10.(int_ε[:]),bin_center[:],linewidth=2.5)
#     ln2=lines!(ax,log10.(int_interpolate[:]),bin_edge[:],linestyle=:dash,linewidth=2.5)
#     axislegend(ax,[ln1,ln2],["Bin","Interpolation"])
#     display(fig)
    
# save("output/tilt/test_binning.png",fig)