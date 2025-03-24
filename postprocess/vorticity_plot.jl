using NCDatasets
using CairoMakie

include("functions/mmderiv.jl")

# load data
tᶠ = 460
θ = 3.6e-3
if θ==3.6e-3
simname = "tilt"
else 
simname = "flat"
end
base_folder = "/scratch/bcpi/cliu28/internal-tide-mixing/"
filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_Bbudget.nc")
ds_field = Dataset(filename_field,"r")
filename_3D_B = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_B-c.nc")
ds_3D_B = Dataset(filename_3D_B,"r")
filename_3D_Bbudget = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_Bbudget.nc")
ds_3D_Bbudget = Dataset(filename_3D_Bbudget,"r")
filename_b_verification = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_b = Dataset(filename_b_verification,"r")
# grids
zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]); 
xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
t = ds_3D["time"][:];

