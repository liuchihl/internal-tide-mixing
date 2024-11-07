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

# terrain following average algorithms
@views function terrain_following_fast(hab, ϕ, new_height, Nx, Ny, Lz)
    mean_values = zeros(length(new_height), size(ϕ, 4))
    hab_interp = zeros(size(ϕ,3))
    value = zeros(Nx,Ny,length(new_height))
    for i in 1:Nx
        for j in 1:Ny
            hab_interp = hab[i, j, :]
            for tt = 1:size(ϕ,4)
            itp = Interpolations.interpolate((hab_interp,), ϕ[i, j, :, tt], Gridded(Interpolations.Linear()))
            itp_extrapolated = Interpolations.extrapolate(itp, Interpolations.Flat())
            h = new_height
            h = Float64.(h)
            h[h .+ z_interp[i,j] .> Lz] .= NaN
            value[i,j,:] = itp_extrapolated.(h)
            end
        end
    end
    mean_values = dropdims(nanmean(value,dims=(1,2)),dims=(1,2))
    return mean_values
end


slope = "tilt"
timerange = "40-80"
θ=3.6e-3

###################################
# load data
# filename_field = "output/no_tilt/internal_tide_5days-theta=0_realtopo3D_Nx500_Nz250_fields_timeavg_0_20.nc"
# ds_field_no_0_20 = Dataset(filename_field,"r")
filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
t = ds_field["time"][:];

new_height = 0:5:1500#0

# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"].var[:,:,:];
# preallocate
b_avg = zeros(length(new_height),length(t))
Bz_avg = zeros(length(new_height),length(t))
uhat_avg = zeros(length(new_height),length(t))
u_avg = zeros(length(new_height),length(t))
what_avg = zeros(length(new_height),length(t))
epsilon_avg = zeros(length(new_height),length(t))
chi_avg = zeros(length(new_height),length(t))
∇κ∇B_avg = zeros(length(new_height),length(t))



topo_file = "topo.mat"
file = matopen(topo_file)
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
y_interp = range(y_topo[1],y_topo[end], length=Ny)


# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp.-minimum(z_interp)




# θ = 0# 3.6e-3
for n in 1:length(t)
    b = ds_field["b"][:,:,:,n:n];          # buoyancy perturbation
    B = ds_field["B"][:,:,:,n:n];          # total buoyancy
    epsilon = ds_field["ε"][:,:,:,n:n];
    uhat = ds_field["uhat"][:,:,:,n:n];    # true u
    what = ds_field["what"][:,:,:,n:n];    # true w
    chi = ds_field["χ"][:,:,:,n:n];        # -κ|∇b|²
    ∇κ∇B = ds_field["∇κ∇B"][:,:,:,n:n];    # ∇⋅κ∇B: buoyancy flux divergence
    what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
    # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
    wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
    u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
    if haskey(ds_field,"Bz")
    Bz = ds_field["Bz"][:,:,:,n:n];   
    else
    N = 1.e-3
    Bz =  deriv(zC,B);
    Bz_bc = 0*ones(Nx,Ny,1,1);
    Bz = cat(Bz, Bz_bc, dims=3);
    end
    # mask out
    # Bz[b.==0] .= NaN
    # epsilon[uhat.==0] .= NaN;
    # chi[b.==0] .= NaN;
    # uhat[uhat.==0] .= NaN;
    # b[b.==0] .= NaN;
    # u[u.==0] .= NaN;
    # what[what.==0] .= NaN;


    # bz_no_0_20 =  deriv(zC,b_no_0_20);
    # bz_no_20_30 =  deriv(zC,b_no_20_30);
    # θ = 0; 
    # bz_bc_0_20 = -N^2*cos(θ).*ones(Nx,Ny,1,length(t_no_0_20));
    # bz_bc_20_30 = -N^2*cos(θ).*ones(Nx,Ny,1,length(t_no_20_30));

    # bz_no_0_20 = cat(bz_no_0_20, bz_bc_0_20, dims=3);
    # bz_no_20_30 = cat(bz_no_20_30, bz_bc_20_30, dims=3);

    

    ### terrain following quantities:

    # call the function to get the terrain following averaged values
    Lz = Int(zF[end])
    @time b_avg[:,n] = terrain_following_fast(hab, b[:,:,:,1], new_height, Nx, Ny,Lz);
    @time Bz_avg[:,n] = terrain_following_fast(hab, Bz[:,:,:,1], new_height, Nx, Ny ,Lz);
    # @time uhat_avg[:,n] = terrain_following_fast(hab, uhat[:,:,:,1], new_height, Nx, Ny, Lz);
    @time what_avg[:,n] = terrain_following_fast(hab, what[:,:,1:end-1,1], new_height, Nx, Ny,Lz);
    @time u_avg[:,n] = terrain_following_fast(hab, u[:,:,:,1], new_height, Nx, Ny,Lz);
    @time epsilon_avg[:,n] = terrain_following_fast(hab, epsilon[:,:,:,1], new_height, Nx, Ny,Lz);
    @time chi_avg[:,n] = terrain_following_fast(hab, chi[:,:,:,1], new_height, Nx, Ny,Lz);
    @time ∇κ∇B_avg[:,n] = terrain_following_fast(hab, ∇κ∇B[:,:,:,1], new_height, Nx, Ny,Lz);
 
end

close(ds_field)
close(ds_hab)

# This creates a new NetCDF file 
# The mode "c" stands for creating a new file

ds_create = Dataset(string("output/",slope,"/TF_avg_",timerange,".nc"),"c")
# Define the dimension
defDim(ds_create,"z_TF",length(new_height))
defDim(ds_create,"t",length(t))

# Define a global attribute
ds_create.attrib["title"] = "Terrain-following averages"
# Define the variables
v1 = defVar(ds_create,"b_avg",Float64,("z_TF","t"))
v1[:,:] = b_avg
v2 = defVar(ds_create,"Bz_avg",Float64,("z_TF","t"))
v2[:,:] = Bz_avg
v3 = defVar(ds_create,"∇κ∇B_avg",Float64,("z_TF","t"))
v3[:,:] = ∇κ∇B_avg
# v3 = defVar(ds_create,"uhat_avg",Float64,("z_TF","t"))
# v3[:,:] = uhat_avg
v4 = defVar(ds_create,"u_avg",Float64,("z_TF","t"))
v4[:,:] = u_avg
v5 = defVar(ds_create,"what_avg",Float64,("z_TF","t"))
v5[:,:] = what_avg
v6 = defVar(ds_create,"chi_avg",Float64,("z_TF","t"))
v6[:,:] = chi_avg
v7 = defVar(ds_create,"epsilon_avg",Float64,("z_TF","t"))
v7[:,:] = epsilon_avg
v8 = defVar(ds_create,"new_height",Float64,("z_TF",))
v8[:,1] = new_height
v9 = defVar(ds_create,"t",Float64,("t",))
v9[:,1] = t   

# write attributes
v1.attrib["units"] = "m/s²"
v2.attrib["units"] = "1/s²"
v3.attrib["units"] = "m/s³"
v4.attrib["units"] = "m/s"
v5.attrib["units"] = "m/s"
v6.attrib["units"] = ""
v7.attrib["units"] = "WKg"
v8.attrib["units"] = "m"
v9.attrib["units"] = "s"

close(ds_create)

include("plot_terrain_following.jl")