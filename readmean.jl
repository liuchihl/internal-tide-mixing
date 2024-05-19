using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
# using BSplineKit
using LinearAlgebra
using Interpolations
using MAT
# load field data, ideally we want to output avg values
filename_field = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_fields.nc"
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC)
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC)
yC = ds_field["yC"]; Ny=length(yC)
t = ds_field["time"];

B = ds_field["B"];
uhat = ds_field["uhat"];
ε = ds_field["ε"];
χ = ds_field["χ"];



# load topography 
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=Ny)

using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)



### calculate the terrain following avg
# hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]
# hh = hab .* (hab .> 0)    # mask hab
# hh[hh.==0] .= NaN
# hab_interp = 0:20:3000 # define the desired vertical grids to be interpolated on
# itp = interpolate((xC,yC,zC), u[1:end-4,1:end-4,1:end-4], Gridded(Linear()))
# u_interp = itp(xC,yC,15:40:3000)
# new_height = 0:40:3000
# B_interp = zeros(Nx,Ny,length(new_height))
# hab_interp = zeros(Nx,Ny,Nz)
# # hab_interp = 0:20:3000

# for i in 1:Nx
#     for j in 1:Ny
# hab = zC .- z_interp[i,j]  # height above bottom [Nz]
# itp = interpolate((hab,), ε[i,j,:,6], Gridded(Linear()))
# itp_extrapolated = extrapolate(itp, Interpolations.Flat())
#         for inx in 1:length(new_height)
#             B_interp[i,j,inx] = itp_extrapolated(new_height[inx])
#         end
#     end
# end



using Interpolations
using Base.Threads: @threads

function terrain_following(new_height, inx, v1, v2, v3, v4, v5)
# new_height: desired hab
# inx: time step
# v1,v2,....: output variables such as u, v, w, etc.. 
    # preallocate

    v1_interp = zeros(Nx, Ny, length(new_height))
    v2_interp = zeros(Nx, Ny, length(new_height))
    v3_interp = zeros(Nx, Ny, length(new_height))
    v4_interp = zeros(Nx, Ny, length(new_height))
    v5_interp = zeros(Nx, Ny, length(new_height))
@threads for i in 1:Nx      # not really increasing speed
            for j in 1:Ny
                hab = zC .- z_interp[i,j]  # height above bottom [Nz]
                itp = interpolate((hab,), v1[i, j, :, inx], Gridded(Linear()))
                itp_extrapolated = extrapolate(itp, Interpolations.Flat())
                # broadcast the extrapolation function over it
                v1_interp[i, j, :] .= itp_extrapolated.(new_height)
                v2_interp[i, j, :] .= itp_extrapolated.(new_height)
                v3_interp[i, j, :] .= itp_extrapolated.(new_height)
                v4_interp[i, j, :] .= itp_extrapolated.(new_height)
                v5_interp[i, j, :] .= itp_extrapolated.(new_height)

            end
        end
    return dropdims(mean(v1_interp,dims=(1,2)) , dims=(1,2))
end

# call the function to calculate terrain-following average
new_height = 0:40:3000
inx = 6
ε_interp = terrain_following(ε, new_height, inx);


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

close(ds_create)
close(ds_field)







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
