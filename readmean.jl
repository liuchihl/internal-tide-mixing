# This script calculates the terrain-following averaged quantities by interpolation
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using BSplineKit
using Statistics
using LinearAlgebra
using Interpolations
using MAT
using Dierckx
# load field data, ideally we want to output avg values
# filename_field = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_fields_timeavg_10_30.nc"
filename_field = "output/no_tilt/internal_tide_5days-theta=0_realtopo3D_Nx500_Nz250_fields_timeavg_0_20.nc"
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC)
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC)
yC = ds_field["yC"]; Ny=length(yC)
t = ds_field["time"];

# B = ds_field["B"];
b = ds_field["b"];
uhat = ds_field["uhat"];
# ε = ds_field["ε"];
# χ = ds_field["χ"];



function deriv(x,y)
    spl = BSplineKit.interpolate(x, y, BSplineOrder(6))
    D1f = diff(spl, Derivative(1))   # change
        return D1f.(x)
end

# @time mapslices(x -> deriv(x,zC[:]), b[:,:,:,3], dims=3)

dbdz = zeros(size(b));
@time for i in 1:Nx
    for j in 1:Ny
        for k = 1:length(t[:])
        dbdz[i,j,:,k] = deriv(zC[:],vec(b[i,j,:,k]))
        end
    end
end



### load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"].var[:,:,:];

# terrain following average algorithms
@views function terrain_following_fast(hab, u, new_height, Nx, Ny)

    mean_values = zeros(length(new_height), size(u, 4))
    hab_interp = zeros(size(zC))
    for i in 1:Nx
        for j in 1:Ny
            hab_interp = hab[i, j, :]
            for tt = 1:size(u,4)
            itp = Interpolations.interpolate((hab_interp,), u[i, j, :, tt], Gridded(Linear()))
            itp_extrapolated = Interpolations.extrapolate(itp, Interpolations.Flat())
            
            # Directly accumulate the interpolated values into the mean array
            mean_values[:, tt] .+= itp_extrapolated.(new_height)
            end
        end
    end
    mean_values ./= (Nx * Ny)

    return mean_values
end

# call the function to get the terrain following averaged velocity (assuming u is a 4D matrix) 
new_height = 0:10:3000
 @time u_avg = terrain_following_fast(hab, uhat[:,:,:,:], new_height, Nx, Ny);
  dbdz_avg = terrain_following_fast(hab, dbdz[:,:,:,:], new_height, Nx, Ny);


# This creates a new NetCDF file 
# The mode "c" stands for creating a new file

ds_create = Dataset("output/supercritical_tilt/TF_avg.nc","c")
# Define the dimension
defDim(ds_create,"z_TF",length(new_height))

# Define a global attribute
ds_create.attrib["title"] = "Terrain-following averages"
# Define the variables
vv = defVar(ds_create,"Dissipation rate",Float64,("z_TF",))
# write a single column
vv[:] = ε_interp
# write attributes
vv.attrib["units"] = "m²/s"

# close(ds_create)
# close(ds_field)







# # find the grid that is above z_interp at x-y plane
# inx = zeros(Nx,Ny)  # Preallocate inx array to store the indices
# # create an array of indices that captures the frist element above the topography
# for i in 1:Nx
#    for j in 1:Ny
# inx[i,j] = findfirst(x -> x > z_interp[i,j], zC[:])
#    end
# end


# # using Plots, StatsBase

# function terrain_follow_average(ϕ)

# ## creating terrain-aligned horizontal average

# # Find the tallest point and use histogram to bin the vertical grids first
# binsize = ceil(maximum(diff(zF)));  # smallest binsize has to be larger than the maximun Δz
# row, col = findmax(z_interp)[2][1],findmax(z_interp)[2][2]
# h = fit(Histogram, zC[Int(inx[row,col]):end],[zC[Int(inx[row,col])]:binsize:maximum(zC);])
# bins = diff(h.edges[1])/2 .+ h.edges[1][1:end-1]   # central bin
# # preallocation
# temp = zeros(Nx,Ny,length(h.weights));
# u_TFM=zeros(1,length(h.weights))

# # loop over the entire x-y plane to get the depth histogram from the topography to the surface
# using Distributed
# Threads.@threads for k in 1:length(h.weights)
#     for i in 1:Nx
#         for j in 1:Ny
#             h = fit(Histogram, zC[Int(inx[i,j]):end],[zC[Int(inx[i,j])]:binsize:maximum(zC);])
#             window = 0; 
#             # lg and ug are the lowest and upmost grids within the bin.
#             # For example, if zF = 0, 0.5, 1, 1.5, 2, 2.5, 3, and bins = 0.1, 2.1, 4.1. Within the first bin, lg=0 and ug=2
#             lg = Int(inx[i,j])+window # lowest grid in the bin
#             ug = Int(inx[i,j])+window+h.weights[k]-1 #upmost grid in the bin
#             # temp[i,j,k] = sum(ϕ[i,j,lg:ug].*diff(zF)[lg:ug]) ./ (zF[lg]-zF[ug])
#             temp[i,j,k] = sum(ϕ[i,j,lg:ug].*diff(zF)[lg:ug]) ./ (zF[lg]-zF[ug])
#             window = window + h.weights[k]
#         end
#     end
# end


# ϕ̄ = vec(mean(temp,dims=(1,2))) 
# # uu = vec(mean(temp,dims=(1,2))) 
# # shift the bins to 0
# return ϕ̄, bins.-minimum(bins)    
# end

# ε_avg = terrain_follow_average(ε[:,:,:,73])
# χ_avg = terrain_follow_average(χ[:,:,:,73])
# # uhat_avg = terrain_follow_average(ε[:,:,:,73])

# # calculate without a loop
# # h = fit(Histogram, zC[Int(inx[i,j]):end],[zC[Int(inx[i,j])]:binsize:maximum(zC);],weights = )
# bins = 0:binsize:maximum(z_interp)

# h = fit.(Histogram, eachrow(z_interp), Ref(bins), weights = grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ*grid.Δzᵃᵃᶜ[1:end-4])
# hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]
# hh = hab .* (hab .> 0)    # mask hab



# u.*reshape(grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ*grid.Δzᵃᵃᶜ[1:end-4],1,1,Nz)
