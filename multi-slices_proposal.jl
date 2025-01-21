using NCDatasets
using MAT
using Interpolations
using CairoMakie
using Printf

file_xy = "/scratch/bcpi/cliu28/internal-tide-mixing/output/higher_top_resolution/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_0-13_slices_xy.nc"
file_xz = "/scratch/bcpi/cliu28/internal-tide-mixing/output/higher_top_resolution/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_0-13_slices_xz.nc"
ds_xy = Dataset(file_xy,"r")
ds_xz = Dataset(file_xz,"r")

# set parameters
ω₀ = 1.4e-4
M₂_period = 2π/ω₀

# grids
zC = ds_xz["zC"]; Nz=length(zC)
zF = ds_xy["zF"]; Nz=length(zF)
xC = ds_xy["xC"]; Nx=length(xC)
yC = ds_xy["yC"]; Ny=length(yC)
t = ds_xz["time"];


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
y_interp = range(y_topo[1],y_topo[end], length=Ny)



# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp.-minimum(z_interp)


# load all data
Bxy = ds_xy["B"].var;        Bxy = Bxy[:,:,:,:];
bxy = ds_xy["b"].var;        bxy = bxy[:,:,:,:];
uhatxy = ds_xy["uhat"].var; uhatxy = uhatxy[:,:,:,:];
εxy = ds_xy["ε"].var;       εxy = εxy[:,:,:,:];
Bxz = ds_xz["B"].var;        Bxz = Bxz[:,:,:,:];
bxz = ds_xz["b"].var;        bxz = bxz[:,:,:,:];
uhatxz = ds_xz["uhat"].var; uhatxz = uhatxz[:,:,:,:];
εxz = ds_xz["ε"].var;       εxz = εxz[:,:,:,:];
# set topography to NaN
εxy[uhatxy.==0] .= NaN
uhatxy[uhatxy.==0] .= NaN
Bxy[bxy.==0] .= NaN
εxz[uhatxz.==0] .= NaN
uhatxz[uhatxz.==0] .= NaN
Bxz[bxz.==0] .= NaN


# save(string("output/",slope,"/vertical_avg_w_",timerange,".png"),fig)
