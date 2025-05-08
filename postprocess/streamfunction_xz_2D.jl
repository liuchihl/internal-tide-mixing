using NCDatasets
using NaNStatistics
using MAT
using Interpolations
tᶠ=126
θ=0#3.6e-3
simname = "2D_notilt"
# load data
filename_field = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_tᶠ=",tᶠ,"_1TP_timeavg.nc")
ds_field = Dataset(filename_field,"r")
zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]);       dz = abs.(zF[1:end-1]-zF[2:end])

xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy=30#dy = yF[end]-yF[end-1]; 
# Ly = yF[end]+dy
t = ds_field["time"][:];
n=24 #30
    uhat = nansum(nanmean(ds_field["uhat"][:,:,:,n:n+100],dim=(4))*dy,dim=2);    # true u (integral)
    what = nansum(nanmean(ds_field["what"][:,:,:,n:n+100],dim=(4))*dy,dim=2);    # true w (integral)
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
U[U.==0].=NaN
W[W.==0].=NaN

# bathymetry
using MAT
# load topography 
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
z_topo = z_topo[:,Int(round((65/2)))]   # take the middle column
x_topo = read(file, "x_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)

# Interpolation object (caches coefficients and such)
itp = LinearInterpolation(x_topo_lin, z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x) for x in x_interp]
z_interp = z_interp.-minimum(z_interp)


## plot
    using PyPlot

    close("all")
    PyPlot.rc("font", size=18)  # Set default font size for all text elements

    fig, ax = subplots(1, 1, figsize=(14, 8))

    c1 = ax.contourf(xC[:], zC[:], ψ',120,cmap="GnBu_r")
    colorbar(c1,label="ψ(m³/s)")
    c2 = ax.contour(xC[:], zC[:], ψ',levels=-72:10:32,colors=[63 43 43]./255,linewidth=0.8,linestyles="-")

    PyPlot.quiver(xC[1:8:end],zC[1:6:end], U[1:8:end,1:6:end]', W[1:8:end,1:6:end]',               
                angles="xy", scale_units="xy")
                # , scale=1, width=0.001,
                # headwidth=5, headlength=.1, headaxislength=20,linewidth=1.0)
    time_start = round(Int, t[n]/(2π/1.4e-4))
    time_end = round(Int, t[n+100]/(2π/1.4e-4))
    title("$(time_start)-$(time_end) tidal periods")   
    ax.set_facecolor("gray")
    xlabel("x (m)") 
    ylabel("z (m)")
    # PyPlot.plot(xC[:],z_interp_y,linewidth=2.5,color="brown")
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