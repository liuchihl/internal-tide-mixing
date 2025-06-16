using NCDatasets
using NaNStatistics
# slope = "tilt"
# timerange = "80-120"
# θ=3.6e-3
simname = "tilt"
tᶠ=450
θ = simname == "tilt" ? 0.0036 : 0
# load data
filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")
filename_b = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",10, "_threeD_timeavg.nc")
ds_b = Dataset(filename_b,"r")
zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]);       dz = abs.(zF[1:end-1]-zF[2:end])

xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
Ly = yF[end]+dy
t = ds_field["time"][:];

if tᶠ == 450
    n = simname=="tilt" ? 1 : 3
end
uhat_tavg = nanmean(ds_field["uhat"][:,:,:,n:n+1],dim=(4))
uhat_tavg[uhat_tavg.==0].=NaN
what_tavg = nanmean(ds_field["what"][:,:,:,n:n+1],dim=(4))
what_tavg[what_tavg.==0].=NaN
uhat = nansum(uhat_tavg*dy,dim=2);    # true u (integral)
what = nansum(what_tavg*dy,dim=2);    # true w (integral)

b = ds_b["b"][:,:,:,1]
B_avg = nanmean(ds_field["B"][:,:,:,n:n+1],dim=(4))
B_avg[b.==0].=NaN
B = nanmean(B_avg,dim=2)

what_cen = (what[:,1:end-1] .+ what[:,2:end])./2 # what at center
# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:],what_cen[1:end-1,:]) .+ what_cen[:,:])./2
U = (uhat[:,:]*cos(θ) .+ wtemp*sin(θ)) # cross-slope velocity transport
W = (-uhat[:,:]*sin(θ) .+ wtemp*cos(θ))# slope-normal velocity transport
ψ = zeros(Nx,Nz)
for i in 1:Nx
    for j in 2:Nz
        ψ[i,j] = ψ[i,j-1] .- U[i,j-1]*dz[j]
    end
end
ψ[uhat.==0].=NaN
# U[U.==0].=NaN
# W[W.==0].=NaN
# B[b.==0].=NaN
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
z_interp_y = nanmean(z_interp,dim=2)



# This creates a new NetCDF file 
# The mode "c" stands for creating a new file

# ds_create = Dataset(string("output/",slope,"/streamfunction_",timerange,".nc"),"c")
# Define the dimension
# defDim(ds_create,"x",Nx)
# defDim(ds_create,"z",Nz)
# # Define a global attribute
# ds_create.attrib["title"] = "Transport Streamfunction"
# # Define the variables
# v1 = defVar(ds_create,"ψ",Float64,("x","z"))
# v1[:,:] = ψ
# v2 = defVar(ds_create,"U",Float64,("x","z"))
# v2[:,:] = U
# v3 = defVar(ds_create,"W",Float64,("x","z"))
# v3[:,:] = W
# v4 = defVar(ds_create,"xC",Float64,("x",))
# v4[:] = xC[:]
# v5 = defVar(ds_create,"zC",Float64,("z",))
# v5[:] = zC[:]

# # write attributes
# v1.attrib["units"] = "m³/s"
# v2.attrib["units"] = "m²/s"
# v3.attrib["units"] = "m²/s"
# v4.attrib["units"] = "m"
# v5.attrib["units"] = "m"
# close(ds_create)

## plot
    # filename = "output/$slope/streamfunction_$timerange.nc"
    # ds = NCDataset(filename)
    # ψ = ds["ψ"][:,:]
    # U = ds["U"][:,:]
    # W = ds["W"][:,:]
    # xC = ds["xC"][:]
    # zC = ds["zC"][:]
    using PyPlot

    close("all")
    PyPlot.rc("font", size=18)  # Set default font size for all text elements

    fig, ax = subplots(1, 1, figsize=(10, 8))

    c1 = ax.pcolor(xC[:], zC[:], ψ',cmap="coolwarm", vmin=-4e4, vmax=4e4)
    colorbar(c1,label="ψ(m³/s)",)
    c2 = ax.contour(xC[:], zC[:], ψ',levels=-1e7:5e3:1e7,colors=[63 43 43]./255,linewidth=0.8,linestyles="-")
    ax.quiver(xC[1:12:end],zC[1:6:end], U[1:12:end,1:6:end]', W[1:12:end,1:6:end]',               
                angles="xy", scale_units="xy")
                # , scale=1, width=0.001,
                # headwidth=5, headlength=.1, headaxislength=20,linewidth=1.0)
    c3 = ax.contour(xC,zC,B_avg[:,500,:]',colors="green",levels=0.0007:0.0001:0.0025,linewidth=0.8,linestyles="-")
    time_start = round(Int, t[n]/(2π/1.4e-4)-10)
    time_end = round(Int, t[n+1]/(2π/1.4e-4))
    title("$(time_start)-$(time_end) tidal average")   
    ax.set_facecolor("gray")
    xlabel("x (m)") 
    ylabel("z (m)")

    PyPlot.plot(xC[:],z_interp_y,linewidth=2.5,color="brown")
    savefig("output/$simname/streamfunction_yavg_$simname.png",dpi=200)



    
    # filename = "output/tilt/streamfunction_80-120.nc"
    # ds = NCDataset(filename)
    # using PyPlot

    # close("all")
    # PyPlot.rc("font", size=18)  # Set default font size for all text elements

    # fig, ax = subplots(1, 1, figsize=(10, 8))

    # c1 = ax.contourf(xC[:], zC[:], ψ',120,cmap="GnBu_r")
    # colorbar(c1,label="ψ(m³/s)")
    # c2 = ax.contour(xC[200:334], zC[:], ψ[200:334,:]',levels=-72000:4000:8000,colors=[63 43 43]./255,linewidth=0.8,linestyles="-")

    # PyPlot.quiver(xC[200:5:334],zC[1:6:end], U[200:5:334,1:6:end]', W[200:5:334,1:6:end]',               
    #             angles="xy", scale_units="xy")
    #             # , scale=1, width=0.001,
    #             # headwidth=5, headlength=.1, headaxislength=20,linewidth=1.0)
    # ax.set_facecolor("gray")
    # xlabel("x (m)")
    # ylabel("z (m)")
    # xlim(xC[200],xC[334])
    # ylim(500,1750)
    # PyPlot.plot(xC[:],z_interp_y,linewidth=2.5,color="brown")
    # savefig("output/tilt/streamfunction_yavg_zoomin.png",dpi=200)