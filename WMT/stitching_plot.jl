# This script stitches the simulation together
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using Interpolations
##
function deriv(z,y)
   dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
   return dydz
end

# load data
    slope = "tilt"
    timerange = "80-120"
    θ=3.6e-3

    filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
    ds_field = Dataset(filename_field,"r")
    filename_field_budget = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg_Bbudget.nc")
    ds_budget = Dataset(filename_field_budget,"r")
    filename_3D = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD.nc")
    ds_3D = Dataset(filename_3D,"r")
  
    # grids
    zC = ds_3D["zC"][:]; zF = ds_3D["zF"][:];
    Nz=length(zC[:]); 
    xC = ds_3D["xC"][:]; xF = ds_3D["xF"][:]; 
    Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

    yC = ds_3D["yC"][:]; yF = ds_3D["yF"][:]
    Ny=length(yC[:]);       dy = yF[end]-yF[end-1];

    t = ds_3D["time"][:];


    θ = 3.6e-3
# for n in 
    n = 30
    b = ds_field["b"][:,1,:,n:n];          # buoyancy perturbation
    B = nanmean(ds_field["B"][:,1,:,n:n+10],dims=3);          # total buoyancy
    # B = ds_field["B"][:,500,:,n:n];          # total buoyancy
    # ∇κ∇B = ds_budget["∇κ∇B"][:,500,:,n:n];    # ∇⋅κ∇B: buoyancy flux divergence
    ∇κ∇B = nanmean(ds_budget["∇κ∇B"][:,1,:,n:n+10],dims=3);    # ∇⋅κ∇B: buoyancy flux divergence
    # div_uB = nanmean(ds_budget["div_uB"][:,500,:,30:40],dims=3);   
    ∇κ∇B[b.==0] .= NaN;
    # div_uB[b.==0] .= NaN;
    B[b.==0] .= NaN;

## coordinate transformation from slope-coordinate to Cartesian coordinate
    Lx = (xF[end]+dx) * cos(θ)
    Lz = (xF[end]+dx) * sin(θ)
    x = xC * cos(θ) .- zC' * sin(θ)
    z = xC * sin(θ) .+ zC' * cos(θ)
    z_face = xF * sin(θ) .+ zF' * cos(θ)
    y = yC
    # extended coordinate
    x̃ = x; z̃ = z; z̃_face = z_face; B̃ = B;
    N = 1e-3
    ΔB = N^2*Lz 

    range = -1:-1:-40
    for n in range
    x̃ = [x .+ n*Lx ; x̃]
    z̃ = [z .+ n*Lz ; z̃]
    z̃_face = [z_face .+ n*Lz ; z̃_face]
    B̃ = [B .+ n*ΔB ; B̃]
    end
    
    num_repeats = length(range)+1

    # Create the array of repeated matrices
    ∇κ∇B̃ = (∇κ∇B for _ in 1:num_repeats)
    ∇κ∇B̃ = vcat(∇κ∇B̃...)
##
    using PyPlot
    PyPlot.rc("font", size=14)  # Sets base font size
    mpl_colors = PyPlot.pyimport("matplotlib.colors")
    linthresh = 1.0
    
    norm_log = mpl_colors["SymLogNorm"](linthresh=linthresh, vmin=-1e-10, vmax=1e-10, base=10)
    fig = figure(figsize=(20, 4))  # 10 inches wide by 6 inches tall
    # ax = subplots()
    c = PyPlot.pcolor(x̃./1000, z̃, ∇κ∇B̃[:,:,1],cmap="seismic",norm=norm_log)
    cbar = colorbar(c, extend="both")
    cbar.set_label("∇⋅(κ∇B)")

    # PyPlot.contour(x̃./1000, z̃, B̃[:,:,1],colors="black",(0.05:0.1:3).*1e-3,linewidths=1)
    PyPlot.contour(x̃./1000, z̃, B̃[:,:,1],colors="black",(0.1:0.8:0.9).*1e-3,linewidths=2.3)
    for n in -19:-1:-20
    # PyPlot.plot([Lx*n,Lx*n]./1000, [-150,3000],linewidth=1.1,linestyle="-",color="green")
    PyPlot.plot([Lx*n,Lx*n]./1000, [-1500,2300],linewidth=1.5,linestyle="-",color="green")
    @show n, Lx*n./1000
    end

    ax = gca()
    # Set the background color of the axes to gray
    ax.set_facecolor("gray")
    
    xlabel("x (km)")
    ylabel("z (m)")
    ylim(-1000,2300)
    minorticks_on()
    tight_layout()

    savefig(string("output/",slope,"/xz_buoyancy_extend_",timerange,"_y=0_extend40_t=120.png"),dpi=100)

    # fig, axes = subplots(1,1, figsize=(10, 8))
    # close(gcf())
    # PyPlot.pcolor(xC[:],zC[:],B[:,1,:,1]')
    # gcf()
    # savefig(string("output/",slope,"/xz_buoyancy_extend_",timerange,".png"))




## MWT


    slope = "tilt"
    timerange = "40-80"
    θ=3.6e-3

    filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
    ds_field = Dataset(filename_field,"r")
    filename_field_budget = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg_Bbudget.nc")
    ds_budget = Dataset(filename_field_budget,"r")

    # grids
    zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
    Nz=length(zC[:]); 
    xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
    Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

    yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
    Ny=length(yC[:]);       dy = yF[end]-yF[end-1];

    t = ds_field["time"][:];


    θ = 3.6e-3
    # for n in 
    n = 40
    b = ds_field["b"][:,500,:,n:n];          # buoyancy perturbation
    B = nanmean(ds_field["B"][:,500,:,30:40],dims=3);          # total buoyancy
    ∇κ∇B = nanmean(ds_budget["∇κ∇B"][:,500,:,30:40],dims=3);    # ∇⋅κ∇B: buoyancy flux divergence
    div_uB = nanmean(ds_budget["div_uB"][:,500,:,30:40],dims=3);   
    ∇κ∇B[b.==0] .= NaN;
    div_uB[b.==0] .= NaN;
    B[b.==0] .= NaN;






    bin_edge = (1:2:40)*1e-4  # Define the edges of the bins

    int_∇κ∇B̃,bin_center = bins(∇κ∇B̃[:,:,1],bin_edge,B̃,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
    using PyPlot
    close(gcf())
    plot(collect(bin_center),int_∇κ∇B̃)
    gcf()

    