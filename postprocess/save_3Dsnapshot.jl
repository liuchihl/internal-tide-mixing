using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets


filename_ver = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
filename_spinup = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=170_threeD_timeavg.nc"

ds_ver = Dataset(filename_ver,"r")
ds_spinup = Dataset(filename_spinup,"r")
ds_field = ds_ver  # Using ds_ver for the grid information
b_ver = ds_ver["b"][:,:,:,1]
B_spinup = ds_spinup["B"][:,:,:,4]
t = ds_spinup["time"][4]
# grids
zC = ds_spinup["zC"][:]; Nz=length(zC[:])
xC = ds_spinup["xC"][:]; Nx=length(xC[:])
yC = ds_spinup["yC"][:]; Ny=length(yC[:])

# Create a new NetCDF file
ds_save = Dataset("output/tilt/buoyancy_timeaverage_160-170.nc", "c")

# Define dimensions
defDim(ds_save, "xC", length(xC))
defDim(ds_save, "yC", length(yC))
defDim(ds_save, "zC", length(zC))

# Define variables
v1 = defVar(ds_save, "b", Float64, ("xC", "yC", "zC"))
v1[:,:,:] = b_ver

v2 = defVar(ds_save, "B", Float64, ("xC", "yC", "zC"))
v2[:,:,:] = B_spinup

v4 = defVar(ds_save, "xC", Float64, ("xC",))
v4[:] = xC[:]

v5 = defVar(ds_save, "yC", Float64, ("yC",))
v5[:] = yC[:]

v6 = defVar(ds_save, "zC", Float64, ("zC",))
v6[:] = zC[:]

close(ds_save)

file = "output/tilt/buoyancy_timeaverage_160-170.nc"
ds = Dataset(file, "r")


b = ds_field["b"][:,:,:,end];
B = ds_field["B"][:,:,:,end];

uhat = ds_field["uhat"][:,:,:,end];
v = ds_field["v"][:,:,:,end];
what = ds_field["what"][:,:,:,end];







# This creates a new NetCDF file 
# The mode "c" stands for creating a new file

ds_create = Dataset("output/higher_top_resolution/internal_tide_day10.nc","c")
# Define the dimension
defDim(ds_create,"xC",length(xC))
defDim(ds_create,"yC",length(yC))
defDim(ds_create,"zC",length(zC))
defDim(ds_create,"xF",length(xF))
defDim(ds_create,"yF",length(yF))
defDim(ds_create,"zF",length(zF))

# Define a global attribute
ds_create.attrib["title"] = "3D fields"
# Define the variables
v1 = defVar(ds_create,"b",Float64,("xC","yC","zC"))
v1[:,:,:] = b
v2 = defVar(ds_create,"B",Float64,("xC","yC","zC"))
v2[:,:,:] = B
v3 = defVar(ds_create,"uhat",Float64,("xF","yC","zC"))
v3[:,:,:] = uhat
v4 = defVar(ds_create,"v",Float64,("xC","yF","zC"))
v4[:,:,:] = v
v5 = defVar(ds_create,"what",Float64,("xC","yC","zF"))
v5[:,:,:] = what
v6 = defVar(ds_create,"xC",Float64,("xC",))
v6[:] = xC[:]
v7 = defVar(ds_create,"yC",Float64,("yC",))
v7[:] = yC[:]
v8 = defVar(ds_create,"zC",Float64,("zC",))
v8[:] = zC[:]
v9 = defVar(ds_create,"xF",Float64,("xF",))
v9[:] = xF[:]
v10 = defVar(ds_create,"yF",Float64,("yF",))
v10[:] = yF[:]
v11 = defVar(ds_create,"zF",Float64,("zF",))
v11[:] = zF[:]

# write attributes

v6.attrib["units"] = "m"
v6.attrib["comments"] = "Locations of the cell centers in the x-direction."
v7.attrib["units"] = "m"
v7.attrib["comments"] = "Locations of the cell centers in the y-direction."
v8.attrib["units"] = "m"
v8.attrib["comments"] = "Locations of the cell faces in the z-direction."
v9.attrib["units"] = "m"
v9.attrib["comments"] = "Locations of the cell faces in the x-direction"
v10.attrib["units"] = "m"
v10.attrib["comments"] = "Locations of the cell faces in the y-direction."
v11.attrib["units"] = "m"
v11.attrib["comments"] = "Locations of the cell faces in the z-direction."

v1.attrib["units"] = "m/s²"
v1.attrib["comments"] = "Buoyancy perturbation"

v2.attrib["units"] = "m/s²"
v2.attrib["comments"] = "Total buoyancy (background buoyancy + buoyancy perturbation)"

v3.attrib["units"] = "m/s"
v3.attrib["comments"] = "True zonal velocity"

v4.attrib["units"] = "m/s"
v4.attrib["comments"] = "Cross-slope velocity"

v5.attrib["units"] = "m/s"
v5.attrib["comments"] = "True vertical velocity"



close(ds_create)