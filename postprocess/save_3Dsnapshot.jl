using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets


filename_field = "output/higher_top_resolution/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_5-10_threeD.nc"
# ds_field_no_20_30 = Dataset(filename_field,"r")
# filename_field = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_fields_timeavg_50_60.nc"
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
yF = ds_field["yF"]; 
t = ds_field["time"][:];

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