# This script calculates the terrain-following averaged quantities by interpolation
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using BSplineKit
using Statistics
using NaNStatistics
using LinearAlgebra
using Interpolations
using MAT
##
function deriv(z, y)
    dydz = diff(y[:,:,:,:], dims=3) ./ reshape(diff(z), 1, 1, length(z)-1)
    return dydz
 end

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/mmderiv.jl")

tᶠ = 452.5
θ = 3.6e-3
if θ==3.6e-3
    simname = "tilt"
else 
    simname = "flat"
end

if  tᶠ ≤ 10
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
    Nt = 1
    endtime = tᶠ
end


bin_edge = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];
bin_mask = hab

# preallocate
b_avg = zeros(length(bin_edge)-1,Nt)
Bz_avg = zeros(length(bin_edge)-1,Nt)
uhat_avg = zeros(length(bin_edge)-1,Nt)
u_avg = zeros(length(bin_edge)-1,Nt)
what_avg = zeros(length(bin_edge)-1,Nt)
τ_avg = zeros(length(bin_edge)-1,Nt)

if output_mode == "analysis"
∇κ∇B_avg = zeros(length(bin_edge)-1,Nt)
div_uB_avg = zeros(length(bin_edge)-1,Nt)
u_bar_∇B_bar_avg = zeros(length(bin_edge)-1,Nt)
u_prime∇B_prime_avg = zeros(length(bin_edge)-1,Nt)
dBdt_avg = zeros(length(bin_edge)-1,Nt)
χ_avg = zeros(length(bin_edge)-1,Nt)
ε_avg = zeros(length(bin_edge)-1,Nt)
end

if output_mode == "verification" 
    # verification and analysis mode
    filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg.nc")
    ds_field = Dataset(filename_field,"r")
    t = ds_field["time"][:];

    # grids
    zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
    Nz=length(zC[:]); 

    xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
    Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

    yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
    Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
    z_face = zF
    for n in 1:Nt
        b = ds_field["b"][:,:,:,n:n];          # buoyancy perturbation
        B = ds_field["B"][:,:,:,n:n];          # total buoyancy
        uhat = ds_field["uhat"][:,:,:,n:n];    # true u
        what = ds_field["what"][:,:,:,n:n];    # true w
        what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
        # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
        wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
        u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
        w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
        if haskey(ds_field,"Bz")
            Bz = ds_field["Bz"][:,:,:,n:n];   
        else
            N = 1.e-3
            Bz =  deriv(zC,B);
            Bz[b[:,:,1:end-1,:].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
        end
    # interpolate Bz from faces to center cell
    using Interpolations
    # Interpolate each row
    Bz_center = zeros(size(Bz,1),size(Bz,2),length(zC),1)
    for i in 1:size(Bz,1)
        for j in 1:size(Bz,2)
                itp = linear_interpolation(zF[2:end-1], Bz[i,j,:,1], extrapolation_bc=Line())
            Bz_center[i,j,:,:] = itp(zC)
        end
    end
    # terrain following quantities:
    @time b_avg[:,n], _ = bins(b,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time Bz_avg[:,n], _ = bins(Bz_center,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time what_avg[:,n], _ = bins(what_cen,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time u_avg[:,n], _ = bins(u,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    end
    close(ds_field)
    close(ds_hab)
    

    ds_create = Dataset(string("output/",simname,"/TF_avg_tᶠ=",endtime,"_",output_mode,".nc"),"c")
    # Define the dimension
    defDim(ds_create,"z_TF",length(bin_center))
    defDim(ds_create,"t",Nt-1)
        # Define a global attribute
    ds_create.attrib["title"] = "Terrain-following averages"
    # Define the variables
    v1 = defVar(ds_create,"b_avg",Float64,("z_TF","t"))
    v1[:,:] = b_avg[:,1:end]
    v2 = defVar(ds_create,"Bz_avg",Float64,("z_TF","t"))
    v2[:,:] = Bz_avg[:,1:end]
    v3 = defVar(ds_create,"u_avg",Float64,("z_TF","t"))
    v3[:,:] = u_avg[:,1:end]
    v4 = defVar(ds_create,"what_avg",Float64,("z_TF","t"))
    v4[:,:] = what_avg[:,1:end]
    v5 = defVar(ds_create,"bin_center",Float64,("z_TF",))
    v5[:,1] = bin_center
    v6 = defVar(ds_create,"t",Float64,("t",))
    v6[:,1] = t   
    
    # write attributes
    v1.attrib["units"] = "m/s²"
    v2.attrib["units"] = "1/s²"
    v3.attrib["units"] = "m/s"
    v4.attrib["units"] = "m/s"
    v5.attrib["units"] = "m"
    v6.attrib["units"] = "s"
    
    close(ds_create)
    
elseif output_mode == "analysis"
    # Load two consecutive half tidal periods for full cycle averaging
    tᶠ_first = tᶠ  # 452.5
    tᶠ_second = tᶠ + 0.5  # 453.0

    filename_3D_first = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ_first, "_analysis_round=all_threeD.nc")
    filename_3D_second = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ_second, "_analysis_round=all_threeD.nc")
    filename_verification = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",10, "_threeD_timeavg.nc")   
    filename_avg_first = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ_first, "_analysis_round=all_threeD_timeavg.nc")
    filename_avg_second = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ_second, "_analysis_round=all_threeD_timeavg.nc")
    ds_avg_first = Dataset(filename_avg_first,"r")
    ds_avg_second = Dataset(filename_avg_second,"r")
    t = ds_avg_first["time"][:];  # Use time from first dataset
    ds_3D_first = Dataset(filename_3D_first,"r")
    ds_3D_second = Dataset(filename_3D_second,"r")
    ds_verification = Dataset(filename_verification,"r")

    # grids
    zC = ds_avg_first["z_aac"][:]; zF = ds_avg_first["z_aaf"][:];
    xC = ds_avg_first["x_caa"][:]; xF = ds_avg_first["x_faa"][:]; 
    yC = ds_avg_first["y_aca"][:]; yF = ds_avg_first["y_afa"][:]
    Nz=length(zC[:]); 
    Nx=length(xC[:]);       dx = xF[end]-xF[end-1];
    Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
    z_face = zF

    t_diff = diff(t)  # Precompute time differences
    dB̄dx = zeros(Nx,Ny,Nz,1)
    dB̄dy = zeros(Nx,Ny,Nz,1)
    dB̄dz = zeros(Nx,Ny,Nz,1)
    n=1
    b = ds_verification["b"][:,:,:,n:n];          # buoyancy perturbation
    B_first = ds_avg_first["B"][:,:,:,n:n]
    B_second = ds_avg_second["B"][:,:,:,n:n]
    uhat_first = ds_avg_first["uhat"][:,:,:,n:n]
    uhat_second = ds_avg_second["uhat"][:,:,:,n:n]
    what_first = ds_avg_first["what"][:,:,:,n:n]
    what_second = ds_avg_second["what"][:,:,:,n:n]
    v_first = ds_avg_first["v"][:,:,:,n:n]
    v_second = ds_avg_second["v"][:,:,:,n:n]
    ∇κ∇B_first = ds_avg_first["∇κ∇B"][:,:,:,n:n]
    ∇κ∇B_second = ds_avg_second["∇κ∇B"][:,:,:,n:n]
    div_uB_first = ds_avg_first["div_uB"][:,:,:,n:n]
    div_uB_second = ds_avg_second["div_uB"][:,:,:,n:n]
    ε_first = ds_avg_first["ε"][:,:,:,n:n]
    ε_second = ds_avg_second["ε"][:,:,:,n:n]
    χ_first = ds_avg_first["χ"][:,:,:,n:n]
    χ_second = ds_avg_second["χ"][:,:,:,n:n]
    
    # Average the two half periods
    B = (B_first .+ B_second) ./ 2
    uhat = (uhat_first .+ uhat_second) ./ 2
    what = (what_first .+ what_second) ./ 2
    v = (v_first .+ v_second) ./ 2
    ∇κ∇B = (∇κ∇B_first .+ ∇κ∇B_second) ./ 2
    div_uB = (div_uB_first .+ div_uB_second) ./ 2
    ε = (ε_first .+ ε_second) ./ 2
    χ = (χ_first .+ χ_second) ./ 2

    what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
    # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
    wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
    u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
    w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
    for k in 1:size(B,3)
    dB̄dx[:,:,k,1] = mmderiv(xC[:], B[:,:,k,1])
    end

    for j in 1:size(B,1)
    dB̄dy[j,:,:,1] = mmderiv(yC[:], B[j,:,:,1])
    end

    for i in 1:size(B,1)
    dB̄dz[i,:,:,1] = mmderiv(zC[:], B[i,:,:,1]')'
    end
    u_bar_∇B_bar = u.*dB̄dx .+ v.*dB̄dy .+ w.*dB̄dz
    u_prime∇B_prime = div_uB .- u_bar_∇B_bar
    N = 1.e-3
    Bz =  deriv(zC,B);
    Bz[b[:,:,1:end-1,:].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0

    # interpolate Bz from faces to center cell
    using Interpolations
    # Interpolate each row
    Bz_center = zeros(size(Bz,1),size(Bz,2),length(zC),1)
    for i in 1:size(Bz,1)
        for j in 1:size(Bz,2)
            itp = linear_interpolation(zF[2:end-1], Bz[i,j,:,1], extrapolation_bc=Line())
            Bz_center[i,j,:,:] = itp(zC)
        end
    end
    # terrain following quantities:

    @time B_avg[:,n], _ = bins(B,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time Bz_avg[:,n], _ = bins(Bz_center,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time what_avg[:,n], _ = bins(what_cen,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time u_avg[:,n], _ = bins(u,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time ∇κ∇B_avg[:,n], _ = bins(∇κ∇B,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time div_uB_avg[:,n], _ = bins(div_uB,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time u_bar_∇B_bar_avg[:,n], _ = bins(u_bar_∇B_bar,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time u_prime∇B_prime_avg[:,n], _ = bins(u_prime∇B_prime,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time ε_avg[:,n], _ = bins(ε,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time χ_avg[:,n], _ = bins(χ,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)


    fname = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ-0.5, "_analysis_round=all_threeD.nc")
    ds_previous = Dataset(fname,"r")
    dBdt = (ds_3D_second["B"][:,:,:,end] .- ds_previous["B"][:,:,:,end]) ./ (ds_3D_second["time"][end].-ds_previous["time"][end])
    dBdt_mmderiv_first = mmderiv(ds_3D_first["time"][:],ds_3D_first["B"][50,50,:,:]')'
    dBdt_mmderiv_second = mmderiv(ds_3D_second["time"][:],ds_3D_second["B"][50,50,:,:]')'
    dBdt_mmderiv = mean(vcat(dBdt_mmderiv_first',dBdt_mmderiv_second')', dims=2)  # Average over the last dimension (time)
    # dBdt = (B2_second[:, :, :] .- B2_first[:, :, :]) ./ (ds_3D_first["time"][end].-ds_previous["time"][end])
    @time dBdt_avg[:,n], _ = bins(dBdt,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)

    rhs = mean(vcat(∇κ∇B_first[50,50,:,1]',∇κ∇B_second[50,50,:,1]')',dims=2).-mean(vcat(div_uB_first[50,50,:,1]', div_uB_second[50,50,:,1]')',dims=2)
    lhs_mmderiv = dBdt_mmderiv[:]
    lhs = dBdt[50,50,:]
    ∇κ∇B = zeros(Nx,Ny,length(zC),1)
    div_uB = zeros(Nx,Ny,length(zC),1)
    for i in 1:6
        temp1 = ds_3D_first["∇κ∇B"][:,:,:,i:i] 
        temp2 = ds_3D_first["div_uB"][:,:,:,i:i]
        ∇κ∇B .+= temp1
        div_uB .+= temp2
    end
    for i in 1:6
        temp1 = ds_3D_second["∇κ∇B"][:,:,:,i:i] 
        temp2 = ds_3D_second["div_uB"][:,:,:,i:i]
        ∇κ∇B .+= temp1
        div_uB .+= temp2
    end
    ∇κ∇B_avg = ∇κ∇B./12
    div_uB_avg = div_uB./12
    rhs_snapshot = .-∇κ∇B_avg[500,500,:,1] .- div_uB_avg[500,500,:,1]
    using PyPlot
    close("all")
    fig=PyPlot.figure(figsize=(10, 6))
    plt.plot(rhs,zC, label="rhs: ∇κ∇B - div_uB")
    plt.plot(rhs_snapshot,zC, label="rhs: ∇κ∇B - div_uB (snapshots average)")
    plt.plot(lhs_mmderiv, zC, label="lhs: dB/dt (mmderiv)",linestyle="--")
    plt.plot(lhs, zC, label="lhs: dB/dt",linestyle="--")
    legend()
    ylabel("z")
    savefig("output/tilt/budget_check.png")

    # Create a colormap plot comparing rhs and lhs
    close("all")
    fig = PyPlot.figure(figsize=(12, 10))
    diff_field = rhs.-lhs
    diff_field[lhs.==0] .= NaN
    rhs[lhs.==0] .= NaN
    lhs[lhs.==0] .= NaN
    
    # Top plot: Left-hand side (dBdt)
    ax1 = fig.add_subplot(311)
    pcm1 = ax1.pcolor(xC, zC, lhs[:,500,:]', cmap="RdBu_r", shading="auto")
    ax1.set_title("LHS: dB/dt")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("z [m]")
    cbar1 = fig.colorbar(pcm1, ax=ax1)
    cbar1.set_label("m/s³")

    # Middle plot: Right-hand side (∇κ∇B - div_uB)
    ax2 = fig.add_subplot(312)
    pcm2 = ax2.pcolor(xC, zC, rhs[:,500,:,1]', cmap="RdBu_r", shading="auto")
    ax2.set_title("RHS: ∇κ∇B - div_uB")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("z [m]")
    cbar2 = fig.colorbar(pcm2, ax=ax2)
    cbar2.set_label("m/s³")

    # Bottom plot: Difference (RHS - LHS)
    ax3 = fig.add_subplot(313)
    pcm3 = ax3.pcolor(xC, zC, diff_field[:,500,:]', cmap="RdBu_r", shading="auto")
    ax3.set_title("Difference: RHS - LHS")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("z [m]")
    cbar3 = fig.colorbar(pcm3, ax=ax3)
    cbar3.set_label("m/s³")

    plt.tight_layout()
    savefig("output/tilt/budget_comparison_pcolor.png", dpi=300)


    close(ds_field)
    close(ds_hab)

    # This creates a new NetCDF file 
    # The mode "c" stands for creating a new file

    ds_create = Dataset(string("output/",simname,"/TF_avg_tᶠ=",tᶠ,"_analysis.nc"),"c")
    # Define the dimension
    defDim(ds_create,"z_TF",length(bin_center))
    defDim(ds_create,"t",Nt)
    # defDim(ds_create, "t_diff", Nt - 1)  # Define a separate dimension for the derivative

    # Define a global attribute
    ds_create.attrib["title"] = "Terrain-following averages"
    # Define the variables
    # v1 = defVar(ds_create,"b_avg",Float64,("z_TF","t"))
    # v1[:,:] = b_avg
    v2 = defVar(ds_create,"Bz_avg",Float64,("z_TF","t"))
    v2[:,:] = Bz_avg
    v3 = defVar(ds_create,"u_avg",Float64,("z_TF","t"))
    v3[:,:] = u_avg
    v4 = defVar(ds_create,"what_avg",Float64,("z_TF","t"))
    v4[:,:] = what_avg
    v6 = defVar(ds_create,"dBdt_avg",Float64,("z_TF","t"))
    v6[:,:] = dBdt_avg
    v7 = defVar(ds_create,"∇κ∇B_avg",Float64,("z_TF","t"))
    v7[:,:] = ∇κ∇B_avg
    v8 = defVar(ds_create,"div_uB_avg",Float64,("z_TF","t"))
    v8[:,:] = div_uB_avg
    v9 = defVar(ds_create,"u_bar_∇B_bar_avg",Float64,("z_TF","t"))
    v9[:,:] = u_bar_∇B_bar_avg
    v10 = defVar(ds_create,"u_prime∇B_prime_avg",Float64,("z_TF","t"))
    v10[:,:] = u_prime∇B_prime_avg
    v11 = defVar(ds_create,"ε_avg",Float64,("z_TF","t"))
    v11[:,:] = ε_avg
    v12 = defVar(ds_create,"χ_avg",Float64,("z_TF","t"))
    v12[:,:] = χ_avg
    v13 = defVar(ds_create,"bin_center",Float64,("z_TF",))
    v13[:,1] = bin_center
    v14 = defVar(ds_create,"t",Float64,("t",))
    v14[:,1] = t   

    # write attributes
    # v1.attrib["units"] = "m/s²"
    v2.attrib["units"] = "1/s²"
    v3.attrib["units"] = "m/s"
    v4.attrib["units"] = "m/s"
    v6.attrib["units"] = "m/s³"
    v6.attrib["units"] = "WKg"
    v7.attrib["units"] = "m/s³"
    v8.attrib["units"] = "m/s³"
    v9.attrib["units"] = "m/s³"
    v10.attrib["units"] = "m/s³"
    v11.attrib["units"] = "m²/s³"
    v12.attrib["units"] = "m²/s³"
    v13.attrib["units"] = "m"
    v14.attrib["units"] = "s"

    close(ds_create)

else     # spinup mode: including multiple cases, i.e., 50:40:1010 TP
    @info "spinup"
    for (i,time) in enumerate(endtime)
        filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",endtime[i], "_threeD_timeavg.nc")
        filename_verification = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",10, "_threeD_timeavg.nc")
        @info (i,time)
        ds_field = Dataset(filename_field,"r")
        ds_verification = Dataset(filename_verification,"r")
        t = ds_field["time"][:];
        # grids
        zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
        Nz=length(zC[:]); 
        xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
        Nx=length(xC[:]);       dx = xF[end]-xF[end-1];
        yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
        Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
        z_face = zF

        # for topostrophy
        dHdy = zeros(size(bin_mask))
        dHdx = zeros(size(bin_mask))
        for k in 1:size(bin_mask,3)
            dHdy[:,:,k] = mmderiv(yC,bin_mask[:,:,k]')'
            dHdx[:,:,k] = mmderiv(xC,bin_mask[:,:,k])
        end
        for n in 1:Nt
            B = ds_field["B"][:,:,:,n:n];          # total buoyancy
            b = ds_verification["b"][:,:,:,n:n];# buoyancy perturbation
            uhat = ds_field["uhat"][:,:,:,n:n];    # true u
            what = ds_field["what"][:,:,:,n:n];    # true w
            what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
            # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
            wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
            u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
            w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
            u_cen = (u[1:end,:,:] .+ vcat(u[2:end,:,:],u[1:1,:,:]))./2 # u at center in the x dimension
            if haskey(ds_field,"v")
                v = ds_field["v"][:,:,:,n:n];    # true u
                τ = @. -(u_cen*dHdy - v*dHdx)
            end
            # topostrophy

            if haskey(ds_field,"Bz")
                Bz = ds_field["Bz"][:,:,:,n:n];   
            else
                N = 1.e-3
                Bz =  deriv(zC,B);
                Bz[b[:,:,1:end-1,:].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
            end
            # interpolate Bz from faces to center cell
            using Interpolations
            # Interpolate each row
            Bz_center = zeros(size(Bz,1),size(Bz,2),length(zC),1)
            for i in 1:size(Bz,1)
                for j in 1:size(Bz,2)
                    itp = linear_interpolation(zF[2:end-1], Bz[i,j,:,1], extrapolation_bc=Line())
                    Bz_center[i,j,:,:] = itp(zC)
                end
            end
            # terrain following quantities:    
            @time Bz_avg[:,n], _ = bins(Bz_center,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
            @time what_avg[:,n], _ = bins(what_cen,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
            @time u_avg[:,n], _ = bins(u,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)        
            if haskey(ds_field,"v")
            @time τ_avg[:,n], _ = bins(τ,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)        
            end
        end
 

        t_diff = diff(t)  # Precompute time differences
        ds_create = Dataset(string("output/",simname,"/TF_avg_tᶠ=",endtime[i],"_",output_mode,".nc"),"c")
        # Define the dimension
        defDim(ds_create,"z_TF",length(bin_center))
        defDim(ds_create,"t",Nt)
        defDim(ds_create, "t_diff", Nt - 1)  # Define a separate dimension for the derivative

        # Define a global attribute
        ds_create.attrib["title"] = "Terrain-following averages"
        # Define the variables
        v1 = defVar(ds_create,"b_avg",Float64,("z_TF","t"))
        v1[:,:] = b_avg
        v2 = defVar(ds_create,"Bz_avg",Float64,("z_TF","t"))
        v2[:,:] = Bz_avg
        v3 = defVar(ds_create,"u_avg",Float64,("z_TF","t"))
        v3[:,:] = u_avg
        v4 = defVar(ds_create,"what_avg",Float64,("z_TF","t"))
        v4[:,:] = what_avg
        if haskey(ds_field,"v")
            v5 = defVar(ds_create,"τ_avg",Float64,("z_TF","t"))
            v5[:,:] = τ_avg
        end
        v6 = defVar(ds_create,"bin_center",Float64,("z_TF",))
        v6[:,1] = bin_center
        v7 = defVar(ds_create,"t",Float64,("t",))
        v7[:,1] = t   

        # write attributes
        v1.attrib["units"] = "m/s²"
        v2.attrib["units"] = "1/s²"
        v3.attrib["units"] = "m/s"
        v4.attrib["units"] = "m/s"
        if haskey(ds_field,"v")
            v5.attrib["units"] = "m/s"
        end
        v6.attrib["units"] = "m"
        v7.attrib["units"] = "s"

        close(ds_create)
        close(ds_field)
    end
end

# include("plot_terrain_following.jl")