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
using NaNStatistics
using LinearAlgebra
using Interpolations
using MAT
##
function deriv(z,y)
   dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
   return dydz
end

include("../functions/bins.jl")
include("../functions/mmderiv.jl")
slope = "tilt"
tᶠ = "10"
θ=3.6e-3

# load data
filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")
# filename_field_budget = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg_Bbudget.nc")
# ds_budget = Dataset(filename_field_budget,"r")
# filename_3D = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD.nc")
# ds_3D = Dataset(filename_3D,"r")

zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]); 

xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
z_face = xF * sin(θ) .+ zF' * cos(θ)     # Cartesian coordinate

t = ds_field["time"][:];

bin_edge = 0:5:1500#0
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2

# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];
bin_mask = hab

# preallocate
b_avg = zeros(length(bin_edge)-1,length(t))
Bz_avg = zeros(length(bin_edge)-1,length(t))
uhat_avg = zeros(length(bin_edge)-1,length(t))
u_avg = zeros(length(bin_edge)-1,length(t))
what_avg = zeros(length(bin_edge)-1,length(t))
# epsilon_avg = zeros(length(bin_edge)-1,length(t))
# chi_avg = zeros(length(bin_edge)-1,length(t))
# ∇κ∇B_avg = zeros(length(bin_edge)-1,length(t))
# div_uB_avg = zeros(length(bin_edge)-1,length(t))
# u_bar_∇B_bar_avg = zeros(length(bin_edge)-1,length(t))
# u_prime∇B_prime_avg = zeros(length(bin_edge)-1,length(t))
# dBdt_avg = zeros(length(bin_edge)-1,length(t)-1)
t_diff = diff(t)  # Precompute time differences
dB̄dx = zeros(Nx,Ny,Nz,1)
dB̄dz = zeros(Nx,Ny,Nz,1)
for n in 1:length(t)
     b = ds_field["b"][:,:,:,n:n];          # buoyancy perturbation
     B = ds_field["B"][:,:,:,n:n];          # total buoyancy
    # epsilon = ds_field["ε"][:,:,:,n:n];
     uhat = ds_field["uhat"][:,:,:,n:n];    # true u
     what = ds_field["what"][:,:,:,n:n];    # true w
    #  chi = ds_field["χ"][:,:,:,n:n];        # -κ|∇b|²
    #  ∇κ∇B = ds_budget["∇κ∇B"][:,:,:,n:n];    # ∇⋅κ∇B: buoyancy flux divergence
    #  div_uB = ds_budget["div_uB"][:,:,:,n:n];# ∇⋅uB: buoyancy flux divergence
     what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
     # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
     wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
     u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
     w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
    #  for k in 1:size(B,3)
    #  dB̄dx[:,:,k,1] = mmderiv(xC[:],B[:,:,k,1])
    #  end
    #  for i in 1:size(B,1)
    #  dB̄dz[i,:,:,1] = mmderiv(zC[:],B[i,:,:,1]')'
    #  end
    #  u_bar_∇B_bar = u.*dB̄dx .+ w.*dB̄dz
    #  u_prime∇B_prime = div_uB .- u_bar_∇B_bar
    if haskey(ds_field,"Bz")
        Bz = ds_field["Bz"][:,:,:,n:n];   
    else
        N = 1.e-3
        Bz =  deriv(zC,B);
        Bz[b[:,:,1:end-1,:].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
    end
    # interpolate Bz from faces to center cell
    using Interpolations
    # Interpolate each row
    Bz_center = zeros(size(Bz,1),size(Bz,2),length(zC),1)
    for i in 1:size(Bz,1)
        for j in 1:size(Bz,2)
              itp = linear_interpolation(zF[2:end-1], Bz[i,j,:,1], extrapolation_bc=Line())
            Bz_center[i,j,:,:] = itp(zC)
        end
    end
    # terrain following quantities:
  
    @time b_avg[:,n], _ = bins(b,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time Bz_avg[:,n], _ = bins(Bz_center,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time what_avg[:,n], _ = bins(what_cen,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time u_avg[:,n], _ = bins(u,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    # @time chi_avg[:,n], _ = bins(chi,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    # @time epsilon_avg[:,n], _ = bins(epsilon,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    # @time ∇κ∇B_avg[:,n], _ = bins(∇κ∇B,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    # @time div_uB_avg[:,n], _ = bins(div_uB,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    # @time u_bar_∇B_bar_avg[:,n], _ = bins(u_bar_∇B_bar,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    # @time u_prime∇B_prime_avg[:,n], _ = bins(u_prime∇B_prime,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    #  if n < length(t)
    #      B2 = ds_field["B"][:,:,:,n:n+1];          # total buoyancy with two timesteps
    #      dBdt = (B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ t_diff[n]
    #      @time dBdt_avg[:,n], _ = bins(dBdt,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    #  end 
    
end

close(ds_field)
close(ds_hab)

# This creates a new NetCDF file 
# The mode "c" stands for creating a new file

ds_create = Dataset(string("output/",slope,"/TF_avg_tᶠ=",tᶠ,"_bin.nc"),"c")
# Define the dimension
defDim(ds_create,"z_TF",length(bin_center))
defDim(ds_create,"t",length(t))
defDim(ds_create, "t_diff", length(t) - 1)  # Define a separate dimension for the derivative

# Define a global attribute
ds_create.attrib["title"] = "Terrain-following averages"
# Define the variables
v1 = defVar(ds_create,"b_avg",Float64,("z_TF","t"))
v1[:,:] = b_avg
v2 = defVar(ds_create,"Bz_avg",Float64,("z_TF","t"))
v2[:,:] = Bz_avg
v3 = defVar(ds_create,"u_avg",Float64,("z_TF","t"))
v3[:,:] = u_avg
v4 = defVar(ds_create,"what_avg",Float64,("z_TF","t"))
v4[:,:] = what_avg
# v5 = defVar(ds_create,"chi_avg",Float64,("z_TF","t"))
# v5[:,:] = chi_avg
# v6 = defVar(ds_create,"epsilon_avg",Float64,("z_TF","t"))
# v6[:,:] = epsilon_avg
# v6 = defVar(ds_create,"dBdt_avg",Float64,("z_TF","t_diff"))
# v6[:,:] = dBdt_avg
# v7 = defVar(ds_create,"∇κ∇B_avg",Float64,("z_TF","t"))
# v7[:,:] = ∇κ∇B_avg
# v8 = defVar(ds_create,"div_uB_avg",Float64,("z_TF","t"))
# v8[:,:] = div_uB_avg
# v9 = defVar(ds_create,"u_bar_∇B_bar_avg",Float64,("z_TF","t"))
# v9[:,:] = u_bar_∇B_bar_avg
# v10 = defVar(ds_create,"u_prime∇B_prime_avg",Float64,("z_TF","t"))
# v10[:,:] = u_prime∇B_prime_avg
v11 = defVar(ds_create,"bin_center",Float64,("z_TF",))
v11[:,1] = bin_center
v12 = defVar(ds_create,"t",Float64,("t",))
v12[:,1] = t   

# write attributes
v1.attrib["units"] = "m/s²"
v2.attrib["units"] = "1/s²"
v3.attrib["units"] = "m/s"
v4.attrib["units"] = "m/s"
# v5.attrib["units"] = ""
# v6.attrib["units"] = "m/s³"
# v6.attrib["units"] = "WKg"
# v7.attrib["units"] = "m/s³"
# v8.attrib["units"] = "m/s³"
# v9.attrib["units"] = "m/s³"
# v10.attrib["units"] = "m/s³"
v11.attrib["units"] = "m"
v12.attrib["units"] = "s"

close(ds_create)



# ds_create = Dataset(string("output/",slope,"/TF_avg_",timerange,"_Bz.nc"),"c")
# # Define the dimension
# defDim(ds_create,"z_TF",length(bin_center))
# defDim(ds_create,"t",length(t))

# # Define a global attribute
# ds_create.attrib["title"] = "Terrain-following averages Bz"
# # Define the variables
# v2 = defVar(ds_create,"Bz_avg",Float64,("z_TF","t"))
# v2[:,:] = Bz_avg
# v9 = defVar(ds_create,"bin_center",Float64,("z_TF",))
# v9[:,1] = bin_center
# v10 = defVar(ds_create,"t",Float64,("t",))
# v10[:,1] = t   

# # write attributes
# v2.attrib["units"] = "1/s²"
# v9.attrib["units"] = "m"
# v10.attrib["units"] = "s"

# close(ds_create)


include("plot_terrain_following.jl")