
tᶠ = 456.0
θ = 3.6e-3
if θ == 3.6e-3
    simname = "tilt"
else
    simname = "flat"
end

if tᶠ ≤ 10
    output_mode = "verification"
    Nt = 11      # number of time averages
    endtime = tᶠ
elseif tᶠ ≤ 450
    output_mode = "spinup"
    Nt = 4
    if tᶠ == 450
        endtime = ["$i" for i in 50:40:450]
        # endtime = tᶠ   # if you want to run only one case
    end
else
    output_mode = "analysis"
    Nt = 1    # only 1 tidal average
    endtime = tᶠ
end

# global total_steps
tᶠ_0 = tᶠ - 1
tᶠ_first = tᶠ - 0.5
tᶠ_second = tᶠ

# filename_3D_0 = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_0, "_analysis_round=all_threeD.nc")
filename_3D_first = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_first, "_analysis_round=all_threeD.nc")
# filename_3D_second = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_second, "_analysis_round=all_threeD.nc")
filename_verification = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", 10, "_threeD_timeavg.nc")
# ds_3D_0 = Dataset(filename_3D_0, "r")
ds_3D_first = Dataset(filename_3D_first, "r")
# ds_3D_second = Dataset(filename_3D_second, "r")
ds_verification = Dataset(filename_verification, "r")

# Get dimensions
zC = ds_3D_first["z_aac"][:]
zF = ds_3D_first["z_aaf"][:]
xC = ds_3D_first["x_caa"][:]
xF = ds_3D_first["x_faa"][:]
yC = ds_3D_first["y_aca"][:]
yF = ds_3D_first["y_afa"][:]
Nz = length(zC[:])
Nx = length(xC[:])
dx = xF[end] - xF[end-1]
Ny = length(yC[:])
dy = yF[end] - yF[end-1]
z_face = zF

# # Combined running sums
# # first define the variables using the final timestep of tf-0.5 chunk 
# # (for example, the first timestep of 452.5 chunk is 452.0833, but we want 452.0 and it is at the final timestep  in tf-0.5 chunk)
# what_sum = ds_3D_0["what"][:, :, :, end:end]
# B_sum = ds_3D_0["B"][:, :, :, end:end]
b = ds_verification["b"][:, :, :, 1:1]

# Count total timesteps processed
total_steps = 1   # already have one timestep from the first dataset

# Process first dataset
Nt_first = length(ds_3D_first["time"][:])
for n in 1:Nt_first
    # Read data for this timestep
    B_sum .+= ds_3D_first["B"][:, :, :, n:n]
    # Free memory by explicitly clearing variables if needed
    GC.gc()
    global total_steps += 1
    println(total_steps)
end

# Process second dataset
Nt_second = length(ds_3D_second["time"][:])
for n in 1:Nt_second
    # Read data for this timestep
    # what_sum .+= ds_3D_second["what"][:, :, :, n:n]
    B_sum .+= ds_3D_second["B"][:, :, :, n:n]
    global total_steps += 1

    # Free memory by explicitly clearing variables if needed
    GC.gc()
end
# Calculate averages
B = B_sum ./ total_steps
# what = what_sum ./ total_steps







using NCDatasets
using Statistics
function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end

include("functions/mmderiv.jl") # for mmderiv function
include("functions/interpolation_z_dimension.jl")


# bathymetry
using MAT
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

# compute mean w

tᶠ = 456.0
θ = 3.6e-3
if θ == 3.6e-3
    simname = "tilt"
else
    simname = "flat"
end

if tᶠ ≤ 10
    output_mode = "verification"
    Nt = 11      # number of time averages
    endtime = tᶠ
elseif tᶠ ≤ 450
    output_mode = "spinup"
    Nt = 4
    if tᶠ == 450
        endtime = ["$i" for i in 50:40:450]
        # endtime = tᶠ   # if you want to run only one case
    end
else
    output_mode = "analysis"
    Nt = 1    # only 1 tidal average
    endtime = tᶠ
end

# global total_steps
# tᶠ_0 = tᶠ - 1
tᶠ_first = tᶠ - 0.5
# tᶠ_second = tᶠ

# filename_3D_0 = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_0, "_analysis_round=all_threeD.nc")
filename_3D_first = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_first, "_analysis_round=all_threeD.nc")
# filename_3D_second = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_second, "_analysis_round=all_threeD.nc")
filename_verification = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", 10, "_threeD_timeavg.nc")
# ds_3D_0 = Dataset(filename_3D_0, "r")
ds_3D_first = Dataset(filename_3D_first, "r")
# ds_3D_second = Dataset(filename_3D_second, "r")
ds_verification = Dataset(filename_verification, "r")

# Get dimensions
zC = ds_3D_first["z_aac"][:]
zF = ds_3D_first["z_aaf"][:]
xC = ds_3D_first["x_caa"][:]
xF = ds_3D_first["x_faa"][:]
yC = ds_3D_first["y_aca"][:]
yF = ds_3D_first["y_afa"][:]
Nz = length(zC[:])
Nx = length(xC[:])
dx = xF[end] - xF[end-1]
Ny = length(yC[:])
dy = yF[end] - yF[end-1]
z_face = zF

b = ds_verification["b"][:, :, :, 1:1]
#### load data of 4 TP average
ds = Dataset("output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=456.0_4tidal_periods_avg.nc")
B = ds["B"][:,:,:,1] 
what = ds["what"][:,:,:,1] 
uhat = ds["uhat"][:,:,:,1] 
v = ds["v"][:,:,:,1] 
ε = ds["ε"][:,:,:,1] # ε at cell centers
∇κ∇B = ds["∇κ∇B"][:,:,:,1] # ∇κ∇B at cell centers
zC = ds["z_aac"][:]
vz = zeros(Nx, Ny, Nz) # vertical velocity at cell centers

uz = deriv(zC, uhat)
uz[uhat[:, :, 1:end-1, :].==0] .= 0 
uz_center = (cat(zeros(Nx,Ny,1), uz, dims=3) .+ cat(uz, zeros(Nx,Ny,1), dims=3))/2 # center in z direction
uz_center = (uz_center[1:end,:,:] .+ vcat(uz_center[2:end,:,:],uz_center[1:1,:,:]))./2 # uz at center in the x dimension

vz = deriv(zC, v) 
vz[v[:, :, 1:end-1, :].==0] .= 0 
vz_center = (cat(zeros(Nx,Ny,1), vz, dims=3) .+ cat(vz, zeros(Nx,Ny,1), dims=3))/2 # center in z direction
vz_center = (vz_center[1:end,:,:] .+ vcat(vz_center[2:end,:,:],vz_center[1:1,:,:]))./2 # vz at center in the x dimension

S² = uz_center.^2 .+ vz_center.^2 # S² at cell centers

what_cen = (what[:, :, 1:end-1, 1] .+ what[:, :, 2:end, 1]) ./ 2 # what at center
# w should also be eliminated at unstable mixing layers 
what_cen[what_cen.==0] .= NaN          # mask out zero values at topography
what_cen[abs.(what_cen).<1e-5] .= NaN  # mask out small values
# what_cen[isnan.(Bz_center)] .= NaN # eliminate points where w is negative because we don't want to consider unstable mixing layers, as the wave dispersion relation no longer holds


Bz = deriv(zC, B)
Bz[b[:, :, 1:end-1, :].==0] .= 0 
Bz_center = (cat(zeros(Nx,Ny,1), Bz, dims=3) .+ cat(Bz, zeros(Nx,Ny,1), dims=3))/2 # average Bz at cell centers
