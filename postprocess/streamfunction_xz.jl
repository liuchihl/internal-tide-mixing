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



    
## plot both cases in one figure
using PyPlot
using NaNStatistics

function load_and_process_data(simname)
    tᶠ=450
    θ = simname == "tilt" ? 0.0036 : 0
    
    # Load data
    filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg.nc")
    ds_field = Dataset(filename_field,"r")
    filename_b = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",10, "_threeD_timeavg.nc")
    ds_b = Dataset(filename_b,"r")
    
    # Get grid information
    zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
    Nz=length(zC[:]);       dz = abs.(zF[1:end-1]-zF[2:end])
    xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
    yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
    Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
    t = ds_field["time"][:];
    
    # Time index
    n = simname=="tilt" ? 1 : 3
    
    # Process velocity fields
    uhat_tavg = nanmean(ds_field["uhat"][:,:,:,n:n+1],dim=(4))
    uhat_tavg[uhat_tavg.==0].=NaN
    what_tavg = nanmean(ds_field["what"][:,:,:,n:n+1],dim=(4))
    what_tavg[what_tavg.==0].=NaN
    uhat = nansum(uhat_tavg*dy,dim=2)    # true u (integral)
    what = nansum(what_tavg*dy,dim=2)    # true w (integral)
    
    # Process buoyancy
    b = ds_b["b"][:,:,:,1]
    B_avg = nanmean(ds_field["B"][:,:,:,n:n+1],dim=(4))
    B_avg[b.==0].=NaN
    B_avg = B_avg[:,500,:] # take the center slice
    # B = nanmean(B_avg,dim=2)
    
    # Calculate streamfunction
    what_cen = (what[:,1:end-1] .+ what[:,2:end])./2 # what at center
    wtemp = (vcat(what_cen[end:end,:],what_cen[1:end-1,:]) .+ what_cen[:,:])./2
    U = (uhat[:,:]*cos(θ) .+ wtemp*sin(θ)) # cross-slope velocity transport
    W = (-uhat[:,:]*sin(θ) .+ wtemp*cos(θ))# slope-normal velocity transport
    
    ψ = zeros(length(xC),Nz)
    for i in 1:length(xC)
        for j in 2:Nz
            ψ[i,j] = ψ[i,j-1] .- U[i,j-1]*dz[j]
        end
    end
    ψ[uhat.==0].=NaN
    
    # Get time information for title
    time_start = round(Int, t[n]/(2π/1.4e-4)-10)
    time_end = round(Int, t[n+1]/(2π/1.4e-4))
    
    return xC, zC, ψ, U, W, B_avg, time_start, time_end
end

# Process topography data
using MAT
using NCDatasets
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)

# Load and process both simulations
xC_tilt, zC_tilt, ψ_tilt, U_tilt, W_tilt, B_avg_tilt, time_start_tilt, time_end_tilt = load_and_process_data("tilt")
xC_flat, zC_flat, ψ_flat, U_flat, W_flat, B_avg_flat, time_start_flat, time_end_flat = load_and_process_data("flat")
U_tilt[U_tilt.==0].=NaN
W_tilt[W_tilt.==0].=NaN
U_flat[U_flat.==0].=NaN
W_flat[W_flat.==0].=NaN

# Interpolate topography
using Interpolations
# high-resolution grids for each case
Nx_tilt = length(xC_tilt)
Ny_tilt = length(xC_tilt) * 2  # Assuming Ny=2Nx

x_interp = range(x_topo[1],x_topo[end], length=Nx_tilt)
y_interp = range(y_topo[1],y_topo[end], length=Ny_tilt)

# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp.-minimum(z_interp)
z_interp_y = nanmean(z_interp,dim=2)

# Create the plot
close("all")
PyPlot.rc("font", size=20)  # Set default font size for all text elements

fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharex=true, constrained_layout=true)
# Plot for tilt case
c1 = ax[1].pcolor(xC_tilt[:], zC_tilt[:], ψ_tilt', cmap="coolwarm", vmin=-4e4, vmax=4e4)
c2 = ax[1].contour(xC_tilt[:], zC_tilt[:], ψ_tilt', levels=-1e7:2e4:1e7, colors=[63 43 43]./255, linewidth=0.8, linestyles="-")

arrow_scale = 0.3  # Adjust this value to make arrows appropriate size
q1 = ax[1].quiver(xC_tilt[1:12:end], zC_tilt[1:6:end], U_tilt[1:12:end,1:6:end]', W_tilt[1:12:end,1:6:end]',               
             angles="xy", scale_units="xy",scale=arrow_scale)
c3 = ax[1].contour(xC_tilt, zC_tilt, B_avg_tilt[:,:]', colors="green", levels=0.0007:0.0001:0.0025, linewidth=0.8, linestyles="-")
ax[1].plot(xC_tilt[:], z_interp_y, linewidth=2.5, color="brown")
# ax[1].set_title("Tilt Case: $(time_start_tilt)-$(time_end_tilt) tidal average")
ax[1].set_facecolor("gray")
ax[1].set_ylabel("z [m]")
ax[1].set_xticklabels([0:2:14])
ax[1].set_xlabel("x [km]")

# Add panel label (a)
ax[1].text(0.02, 0.98, "(a)", transform=ax[1].transAxes, fontsize=22, fontweight="normal", 
           verticalalignment="top", horizontalalignment="left", 
           bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"white", "alpha"=>0.8))

# Add reference vector on topography
# Calculate a reasonable reference velocity
ref_value = 600  # Use a more representative reference value
ax[1].quiver(xC_tilt[100], 60, ref_value/3, 0, 
             angles="xy", scale_units="xy", color="black", scale=arrow_scale, headwidth=1.6, headlength=2.5, headaxislength=2.5)
ax[1].quiver(xC_tilt[100], 60, 0, ref_value/20, 
             angles="xy", scale_units="xy", color="black", scale=arrow_scale, headwidth=1.6, headlength=2.5, headaxislength=2.5)
ax[1].text(xC_tilt[100]+850, 30, "$(round(ref_value/3, digits=0)) m² s⁻¹", fontsize=12)
ax[1].text(xC_tilt[100]-200, 180, "$(round(ref_value/20, digits=0)) m² s⁻¹", fontsize=12)

# Plot for flat case
c4 = ax[2].pcolor(xC_flat[:], zC_flat[:], ψ_flat', cmap="coolwarm", vmin=-4e4, vmax=4e4)
cb2 = colorbar(c4, ax=ax, label="ψ [m³ s⁻¹]", shrink=0.8, pad=0.02, extend="both")
c5 = ax[2].contour(xC_flat[:], zC_flat[:], ψ_flat', levels=-1e7:2e4:1e7, colors=[63 43 43]./255, linewidth=0.8, linestyles="-")
q2 = ax[2].quiver(xC_flat[1:12:end], zC_flat[1:6:end], U_flat[1:12:end,1:6:end]', W_flat[1:12:end,1:6:end]',               
             angles="xy", scale_units="xy", scale=arrow_scale)
c6 = ax[2].contour(xC_flat, zC_flat, B_avg_flat[:,:]', colors="green", levels=0.0007:0.0001:0.0025, linewidth=0.8, linestyles="-")
ax[2].plot(xC_flat[:], z_interp_y, linewidth=2.5, color="brown")
# ax[2].set_title("Flat Case: $(time_start_flat)-$(time_end_flat) tidal average")
ax[2].set_facecolor("gray")
# Set ticks and labels correctly - make sure they match in number
x_ticks = [0,5,10,15].* 1000  # Convert km to m for x-axis
x_labels = string.(0:5:15)[1:length(x_ticks)]  # Make sure labels match number of ticks
ax[2].set_xticks(x_ticks)
ax[2].set_xticklabels(x_labels)
ax[2].set_xlabel("x [km]")

ax[2].text(0.02, 0.98, "(b)", transform=ax[2].transAxes, fontsize=22, fontweight="normal", 
           verticalalignment="top", horizontalalignment="left", 
           bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"white", "alpha"=>0.8))

# Add legend for common elements
# ax[1].plot([], [], color="green", linestyle="-", label="Buoyancy contour")
# ax[1].plot([], [], color="brown", linestyle="-", linewidth=2.5, label="Topography")
# ax[1].legend(loc="upper right")

# tight_layout()
savefig("output/tilt/comparison_streamfunction_tilt_vs_flat.png", dpi=200, bbox_inches="tight")
