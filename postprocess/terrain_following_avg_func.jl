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

tᶠ = 10
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
    else
        # endtime = tᶠ
    end
else
    output_mode = "analysis"
    Nt = 20
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
epsilon_avg = zeros(length(bin_edge)-1,Nt)
chi_avg = zeros(length(bin_edge)-1,Nt)
∇κ∇B_avg = zeros(length(bin_edge)-1,Nt)
div_uB_avg = zeros(length(bin_edge)-1,Nt)
u_bar_∇B_bar_avg = zeros(length(bin_edge)-1,Nt)
u_prime∇B_prime_avg = zeros(length(bin_edge)-1,Nt)
dBdt_avg = zeros(length(bin_edge)-1,Nt-1)
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
    t = ds_field["time"][:];
    # grids
    zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
    Nz=length(zC[:]); 
    xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
    Nx=length(xC[:]);       dx = xF[end]-xF[end-1];
    yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
    Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
    z_face = zF

    t_diff = diff(t)  # Precompute time differences
    dB̄dx = zeros(Nx,Ny,Nz,1)
    dB̄dz = zeros(Nx,Ny,Nz,1)
    for n in 1:Nt
        b = ds_field["b"][:,:,:,n:n];          # buoyancy perturbation
        B = ds_field["B"][:,:,:,n:n];          # total buoyancy
        uhat = ds_field["uhat"][:,:,:,n:n];    # true u
        what = ds_field["what"][:,:,:,n:n];    # true w
        ∇κ∇B = ds_budget["∇κ∇B"][:,:,:,n:n];    # ∇⋅κ∇B: buoyancy flux divergence
        div_uB = ds_budget["div_uB"][:,:,:,n:n];# ∇⋅uB: buoyancy flux divergence
        what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
        # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
        wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
        u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
        w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
        for k in 1:size(B,3)
        dB̄dx[:,:,k,1] = mmderiv(xC[:],B[:,:,k,1])
        end
        for i in 1:size(B,1)
        dB̄dz[i,:,:,1] = mmderiv(zC[:],B[i,:,:,1]')'
        end
        u_bar_∇B_bar = u.*dB̄dx .+ w.*dB̄dz
        u_prime∇B_prime = div_uB .- u_bar_∇B_bar
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
        @time ∇κ∇B_avg[:,n], _ = bins(∇κ∇B,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
        @time div_uB_avg[:,n], _ = bins(div_uB,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
        @time u_bar_∇B_bar_avg[:,n], _ = bins(u_bar_∇B_bar,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
        @time u_prime∇B_prime_avg[:,n], _ = bins(u_prime∇B_prime,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
        if n < Nt
            B2 = ds_field["B"][:,:,:,n:n+1];          # total buoyancy with two timesteps
            dBdt = (B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ t_diff[n]
            @time dBdt_avg[:,n], _ = bins(dBdt,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
        end 

    end

    close(ds_field)
    close(ds_hab)

    # This creates a new NetCDF file 
    # The mode "c" stands for creating a new file

    ds_create = Dataset(string("output/",simname,"/TF_avg_tᶠ=",tᶠ,"_bin.nc"),"c")
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
    v6 = defVar(ds_create,"dBdt_avg",Float64,("z_TF","t_diff"))
    v6[:,:] = dBdt_avg
    v7 = defVar(ds_create,"∇κ∇B_avg",Float64,("z_TF","t"))
    v7[:,:] = ∇κ∇B_avg
    v8 = defVar(ds_create,"div_uB_avg",Float64,("z_TF","t"))
    v8[:,:] = div_uB_avg
    v9 = defVar(ds_create,"u_bar_∇B_bar_avg",Float64,("z_TF","t"))
    v9[:,:] = u_bar_∇B_bar_avg
    v10 = defVar(ds_create,"u_prime∇B_prime_avg",Float64,("z_TF","t"))
    v10[:,:] = u_prime∇B_prime_avg
    v11 = defVar(ds_create,"bin_center",Float64,("z_TF",))
    v11[:,1] = bin_center
    v12 = defVar(ds_create,"t",Float64,("t",))
    v12[:,1] = t   

    # write attributes
    v1.attrib["units"] = "m/s²"
    v2.attrib["units"] = "1/s²"
    v3.attrib["units"] = "m/s"
    v4.attrib["units"] = "m/s"
    v6.attrib["units"] = "m/s³"
    v6.attrib["units"] = "WKg"
    v7.attrib["units"] = "m/s³"
    v8.attrib["units"] = "m/s³"
    v9.attrib["units"] = "m/s³"
    v10.attrib["units"] = "m/s³"
    v11.attrib["units"] = "m"
    v12.attrib["units"] = "s"

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

