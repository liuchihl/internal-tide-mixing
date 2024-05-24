# this script calculates the height above bottom grids. The result is a 3D matrix [Nx,Ny,Nz]
# reason for doing this is that it took a while to calculate hab, so it's best to calculate once and save it.

filename_field = "output/no_tilt/internal_tide_5days-theta=0_realtopo3D_Nx500_Nz250_fields_timeavg_0_20.nc"
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC)
xC = ds_field["xC"]; Nx=length(xC)
yC = ds_field["yC"]; Ny=length(yC)

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


# compute height above bottom
hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]

# save hab into NetCDF file 
# The mode "c" stands for creating a new file

ds_hab = Dataset("output/hab.nc","c")
# Define the dimensions
defDim(ds_hab,"xC",length(xC))
defDim(ds_hab,"yC",length(yC))
defDim(ds_hab,"zC",length(zC))

# Define a global attribute
ds_hab.attrib["title"] = "Height Above Bottom Coordinate (3D)"
# Define the variables
vv = defVar(ds_hab,"hab",Float64,("xC","yC","zC"))
# write a single column
vv[:,:,:] = hab
# write attributes
vv.attrib["units"] = "m"
