## calculate Thorpe scale

using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations


# function deriv(x,y)
# spl = interpolate(x, y, BSplineOrder(6))
# D1f = diff(spl, Derivative(1))   # change
#     return D1f.(x)
# end

function deriv(x,y)
    dydx =  diff(y[:,:,:,:],dims=3)./reshape(diff(x[:]),1,1,length(zC)-1)
    return dydx
 end

# function nice_divergent_levels(c, clim; nlevels=20)
#     levels = range(-clim, stop=clim, length=nlevels)
#     cmax = maximum(abs, c)
#     clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
#     return (-clim, clim), levels
# end

## load data
# filename_field = "output/internal_tide_3days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_xz.nc"
# ds_field = Dataset(filename_field,"r")
# bathy_data = "output/bathymetry.nc"
# Bathy = Dataset(bathy_data,"r")
# filename_slice = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_xz.nc"
filename_slice = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_50_60_xz.nc"
# filename_slice = "output/no_tilt/internal_tide_5days-theta=0_realtopo3D_Nx500_Nz250_slices_30_50_xz.nc"
ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_slice["zC"]; Nz=length(zC)
xC = ds_slice["xC"]; Nx=length(xC)
t = ds_slice["time"];

# load all data
B = ds_slice["B"].var;        B = B[:,:,:,:];
b = ds_slice["b"].var;        b = b[:,:,:,:];
B[b.==0] .= NaN


using Statistics

function thorpe_scale(depths, buoyancy_matrix)
    # Ensure inputs are arrays
    depths = collect(depths)
    buoyancy_matrix = collect(buoyancy_matrix)
    
    
    thorpe_scales = []

    for i in 1:1#Nx
        for j in 1:1#length(t)
        buoyancy = buoyancy_matrix[i, 1, :, j]
        
        # Remove NaNs from both depths and buoyancy
        inx = .!isnan(buoyancy)
        valid_depths = depths[inx]
        valid_buoyancy = buoyancy[inx]
        
        # Sort the buoyancy to obtain a stably stratified profile
        sorted_indices = sortperm(valid_buoyancy)
        sorted_buoyancy = valid_buoyancy[sorted_indices]
        sorted_depths = valid_depths[sorted_indices]
        
        # Calculate the Thorpe displacements
        thorpe_displacements = valid_depths .- sorted_depths

        # Filter out regions that are already stably stratified
        unstable_indices = findall(x -> x != 0, thorpe_displacements)
        if isempty(unstable_indices)
            push!(thorpe_scales, 0.0)  # If no unstable regions, Thorpe scale is 0
            continue
        end
        
        thorpe_displacements = thorpe_displacements[unstable_indices]

        # Calculate the root mean square of the displacements
        thorpe_scale = sqrt(mean(thorpe_displacements .^ 2))
        push!(thorpe_scales, thorpe_scale)
        end
    end 

    return thorpe_scales
end