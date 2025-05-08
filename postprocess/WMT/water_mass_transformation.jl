#  This script calculates the water mass transformation
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
include("../../functions/bins_2d.jl")
include("../../functions/mmderiv.jl")

    # load data
tᶠ = 460
θ = 0#3.6e-3
if θ==3.6e-3
    simname = "tilt"
else 
    simname = "flat"
end
base_folder = "/scratch/bcpi/cliu28/internal-tide-mixing/"
filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_Bbudget-wb.nc")
ds_field = Dataset(filename_field,"r")
filename_3D_B = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_B-c.nc")
ds_3D_B = Dataset(filename_3D_B,"r")
filename_3D_Bbudget = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_Bbudget.nc")
ds_3D_Bbudget = Dataset(filename_3D_Bbudget,"r")
filename_b_verification = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_b = Dataset(filename_b_verification,"r")
# grids
zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]); 
xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
t = ds_3D_Bbudget["time"][:];


## coordinate transformation from simname-coordinate to Cartesian coordinate
Lx = (xF[end]+dx) * cos(θ)
Lz = (xF[end]+dx) * sin(θ)
x = xC * cos(θ) .- zC' * sin(θ)
z = xC * sin(θ) .+ zC' * cos(θ)
z_face = xF * sin(θ) .+ zF' * cos(θ)
N = 1e-3
ΔB = N^2*Lz 

# bin_edge = (0.1:0.02:0.9).*1e-3  # Define the edges of the bins
# bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2

# var, bin_edge1, bin_edge2, bin_mask1, bin_mask2
# inputs for the 2D binning

# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:]
ranges = 0:-1:-40
bin_edge1 = 0:7:1500#0
bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
bin_mask1 = hab

b = ds_b["b"][:,:,:,1:1]

if simname == "tilt"
    bin_edge2 = (0.1:0.02:0.9).*1e-3 
    bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2
    global int_∇κ∇B = zeros(length(bin_center1),length(bin_center2),1)
    global int_div_uB = zeros(length(bin_center1),length(bin_center2),1)
    ∇κ∇B_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))
    div_uB_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))
end
    
if simname == "tilt"
## 1) whole domain
    for n in 1:2:length(t)
        global ∇κ∇B = ds_3D_Bbudget["∇κ∇B"][:,:,:,n:n];   # ∇⋅κ∇B: buoyancy flux divergence
        global div_uB = ds_3D_Bbudget["div_uB"][:,:,:,n:n];
        B = ds_3D_B["B"][:,:,:,n:n];
        # ∇κ∇B[b.==0].=0
        # div_uB[b.==0].=0
        B[b.==0].=0

        # Reset accumulator arrays for each timestep
        fill!(int_∇κ∇B, 0)
        fill!(int_div_uB, 0)

        for m in ranges
            z̃_face = z_face .+ m*Lz     # extended domain
            B̃ = B .+ m*ΔB
            bin_mask2 = B̃
            @time f1, _, _= bins_2d(∇κ∇B,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
            global int_∇κ∇B += f1   # f1 is the integrand
            @time f2, _, _= bins_2d(div_uB,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
            global int_div_uB += f2  # f2 is the integrand
            @info "m = $m, n= $n"  # Progress tracking
        end
    ## the numerator is already a difference between integrand because the volume integral different leaves the thin layer
    ∇κ∇B_t[:,:,n] .= int_∇κ∇B./diff(bin_edge2)[1]
    div_uB_t[:,:,n] .= int_div_uB./diff(bin_edge2)[1]

    fill!(∇κ∇B, 0)
    fill!(div_uB, 0)
    fill!(B, 0)
        
    end
    close(ds_field)
    close(ds_3D_B)
    close(ds_3D_Bbudget)
    close(ds_b)
    close(ds_hab)

    ds_create = Dataset(string(base_folder, "output/",simname,"/WMT_total_tᶠ=",tᶠ,".nc"),"c")

    # Define the dimension
    defDim(ds_create,"z_TF",length(bin_center1))
    defDim(ds_create,"buoyancy",length(bin_center2))
    defDim(ds_create,"buoyancy_diff",length(bin_center2))
    defDim(ds_create,"time",length(t[:]))
    
    # Define a global attribute
    ds_create.attrib["title"] = "2D histogram, binned by hab and buoyancy"
    # Define the variables
    v1 = defVar(ds_create,"∇κ∇B_t",Float64,("z_TF","buoyancy_diff","time"))
    v1[:,:,:] = ∇κ∇B_t
    v2 = defVar(ds_create,"div_uB_t",Float64,("z_TF","buoyancy_diff","time"))
    v2[:,:,:] = div_uB_t
    v3 = defVar(ds_create,"bin_center1",Float64,("z_TF",))
    v3[:,1] = bin_center1
    v4 = defVar(ds_create,"bin_center2",Float64,("buoyancy",))
    v4[:,1] = bin_center2
    v5 = defVar(ds_create,"t",Float64,("time",))
    v5[:,1] = t[:]
    
    # write attributes
    v1.attrib["units"] = "m³/s"
    v2.attrib["units"] = "m³/s"
    v3.attrib["units"] = "m"
    v4.attrib["units"] = "m/s²"
    v5.attrib["units"] = "s"
    
    close(ds_create)
    

## 2) canyon
    filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_Bbudget-wb.nc")
    ds_field = Dataset(filename_field,"r")
    filename_3D_B = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_B-c.nc")
    ds_3D_B = Dataset(filename_3D_B,"r")
    filename_3D_Bbudget = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_Bbudget.nc")
    ds_3D_Bbudget = Dataset(filename_3D_Bbudget,"r")  ## coordinate transformation from simname-coordinate to Cartesian coordinate
    ds_b = Dataset(filename_b_verification,"r")

    Lx = (xF[end]+dx) * cos(θ)
    Lz = (xF[end]+dx) * sin(θ)
    x = xC * cos(θ) .- zC' * sin(θ)
    z = xC * sin(θ) .+ zC' * cos(θ)
    z_face = xF * sin(θ) .+ zF' * cos(θ)
    N = 1e-3
    ΔB = N^2*Lz 

    
    north = argmin(abs.(yC[:] .- 20e3))  
    south = argmin(abs.(yC[:] .- 10e3))  
    filename_hab = "output/hab.nc"
    ds_hab = Dataset(filename_hab,"r")
    hab = ds_hab["hab"][:,south:north,:];
    ranges = 0:-1:-40
    bin_edge1 = 0:7:1500#0
    bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
    bin_mask1 = hab
    # bin_edge2 = (0.1:0.02:0.9).*1e-3  # Define the edges of the bins
    bin_edge2 = (0.1:0.02:0.9).*1e-3  # Define the edges of the bins
    bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2
    global int_∇κ∇B = zeros(length(bin_center1),length(bin_center2),1)
    global int_div_uB = zeros(length(bin_center1),length(bin_center2),1)
    ∇κ∇B_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))
    div_uB_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))
    b = ds_b["b"][:,south:north,:,1:1]

    for n in 1:2:length(t)
        global ∇κ∇B = ds_3D_Bbudget["∇κ∇B"][:,south:north,:,n:n];   # ∇⋅κ∇B: buoyancy flux divergence
        global div_uB = ds_3D_Bbudget["div_uB"][:,south:north,:,n:n];
        B = ds_3D_B["B"][:,south:north,:,n:n];
        # ∇κ∇B[b.==0].=0
        # div_uB[b.==0].=0
        @time B[b.==0].=0

        # Reset accumulator arrays for each timestep
        fill!(int_∇κ∇B, 0)
        fill!(int_div_uB, 0)

        for m in ranges
            z̃_face = z_face .+ m*Lz     # extended domain
            B̃ = B .+ m*ΔB
            bin_mask2 = B̃
            @time f1, _, _= bins_2d(∇κ∇B,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
            global int_∇κ∇B += f1
            @time f2, _, _= bins_2d(div_uB,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
            global int_div_uB += f2
            @info "m = $m, n= $n"  # Progress tracking
        end
    ∇κ∇B_t[:,:,n] .= int_∇κ∇B./diff(bin_edge2)[1]
    div_uB_t[:,:,n] .= int_div_uB./diff(bin_edge2)[1]

    fill!(∇κ∇B, 0)
    fill!(div_uB, 0)
    fill!(B, 0)
        
    end
    close(ds_field)
    close(ds_3D_B)
    close(ds_3D_Bbudget)
    close(ds_b)
    close(ds_hab)

    ds_create = Dataset(string(base_folder, "output/",simname,"/WMT_canyon_tᶠ=",tᶠ,".nc"),"c")

    # Define the dimension
    defDim(ds_create,"z_TF",length(bin_center1))
    defDim(ds_create,"buoyancy",length(bin_center2))
    defDim(ds_create,"buoyancy_diff",length(bin_center2))
    defDim(ds_create,"time",length(t[:]))
    
    # Define a global attribute
    ds_create.attrib["title"] = "2D histogram, binned by hab and buoyancy"
    # Define the variables
    v1 = defVar(ds_create,"∇κ∇B_t",Float64,("z_TF","buoyancy_diff","time"))
    v1[:,:,:] = ∇κ∇B_t
    v2 = defVar(ds_create,"div_uB_t",Float64,("z_TF","buoyancy_diff","time"))
    v2[:,:,:] = div_uB_t
    v3 = defVar(ds_create,"bin_center1",Float64,("z_TF",))
    v3[:,1] = bin_center1
    v4 = defVar(ds_create,"bin_center2",Float64,("buoyancy",))
    v4[:,1] = bin_center2
    v5 = defVar(ds_create,"t",Float64,("time",))
    v5[:,1] = t[:]
    
    # write attributes
    v1.attrib["units"] = "m³/s"
    v2.attrib["units"] = "m³/s"
    v3.attrib["units"] = "m"
    v4.attrib["units"] = "m/s²"
    v5.attrib["units"] = "s"
    
    close(ds_create)
    

## 3) flanks
    base_folder = "/scratch/bcpi/cliu28/internal-tide-mixing/"
    filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_Bbudget-wb.nc")
    ds_field = Dataset(filename_field,"r")
    filename_3D_B = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_B-c.nc")
    ds_3D_B = Dataset(filename_3D_B,"r")
    filename_3D_Bbudget = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_Bbudget.nc")
    ds_3D_Bbudget = Dataset(filename_3D_Bbudget,"r")
    filename_b_verification = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
    ds_b = Dataset(filename_b_verification,"r")
## coordinate transformation from simname-coordinate to Cartesian coordinate
    Lx = (xF[end]+dx) * cos(θ)
    Lz = (xF[end]+dx) * sin(θ)
    x = xC * cos(θ) .- zC' * sin(θ)
    z = xC * sin(θ) .+ zC' * cos(θ)
    z_face = xF * sin(θ) .+ zF' * cos(θ)
    N = 1e-3
    ΔB = N^2*Lz 
    north = argmin(abs.(yC[:] .- 20e3))  
    south = argmin(abs.(yC[:] .- 10e3))  
    filename_hab = "output/hab.nc"
    ds_hab = Dataset(filename_hab,"r")
    hab = ds_hab["hab"][:,vcat(1:south,north:end),:];
    ranges = 0:-1:-40
    bin_edge1 = 0:7:1500#0
    bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
    bin_mask1 = hab
    bin_edge2 = (0.1:0.02:0.9).*1e-3  # Define the edges of the bins
    bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2
    global int_∇κ∇B = zeros(length(bin_center1),length(bin_center2),1)
    global int_div_uB = zeros(length(bin_center1),length(bin_center2),1)
    ∇κ∇B_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))
    div_uB_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))
    b = ds_b["b"][:,vcat(1:south,north:end),:,1:1]

    for n in 1:2:length(t)
        global ∇κ∇B = ds_3D_Bbudget["∇κ∇B"][:,vcat(1:south,north:end),:,n:n];   # ∇⋅κ∇B: buoyancy flux divergence
        global div_uB = ds_3D_Bbudget["div_uB"][:,vcat(1:south,north:end),:,n:n];
        B = ds_3D_B["B"][:,vcat(1:south,north:end),:,n:n];
        # ∇κ∇B[b.==0].=0
        # div_uB[b.==0].=0
        @time B[b.==0].=0

        # Reset accumulator arrays for each timestep
        fill!(int_∇κ∇B, 0)
        fill!(int_div_uB, 0)

        for m in ranges
            z̃_face = z_face .+ m*Lz     # extended domain
            B̃ = B .+ m*ΔB
            bin_mask2 = B̃
            @time f1, _, _= bins_2d(∇κ∇B,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
            global int_∇κ∇B += f1
            @time f2, _, _= bins_2d(div_uB,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
            global int_div_uB += f2
            @info "m = $m, n= $n"  # Progress tracking
        end
        ∇κ∇B_t[:,:,n] .= int_∇κ∇B./diff(bin_edge2)[1]
        div_uB_t[:,:,n] .= int_div_uB./diff(bin_edge2)[1]

        fill!(∇κ∇B, 0)
        fill!(div_uB, 0)
        fill!(B, 0)
    end

    close(ds_field)
    close(ds_3D_B)
    close(ds_3D_Bbudget)
    close(ds_b)
    close(ds_hab)

    ds_create = Dataset(string(base_folder, "output/",simname,"/WMT_flanks_tᶠ=",tᶠ,".nc"),"c")

    # Define the dimension
    defDim(ds_create,"z_TF",length(bin_center1))
    defDim(ds_create,"buoyancy",length(bin_center2))
    defDim(ds_create,"buoyancy_diff",length(bin_center2))
    defDim(ds_create,"time",length(t[:]))
    
    # Define a global attribute
    ds_create.attrib["title"] = "2D histogram, binned by hab and buoyancy"
    # Define the variables
    v1 = defVar(ds_create,"∇κ∇B_t",Float64,("z_TF","buoyancy_diff","time"))
    v1[:,:,:] = ∇κ∇B_t
    v2 = defVar(ds_create,"div_uB_t",Float64,("z_TF","buoyancy_diff","time"))
    v2[:,:,:] = div_uB_t
    v3 = defVar(ds_create,"bin_center1",Float64,("z_TF",))
    v3[:,1] = bin_center1
    v4 = defVar(ds_create,"bin_center2",Float64,("buoyancy",))
    v4[:,1] = bin_center2
    v5 = defVar(ds_create,"t",Float64,("time",))
    v5[:,1] = t[:]
    
    # write attributes
    v1.attrib["units"] = "m³/s"
    v2.attrib["units"] = "m³/s"
    v3.attrib["units"] = "m"
    v4.attrib["units"] = "m/s²"
    v5.attrib["units"] = "s"
    
    close(ds_create)
        
# flat case is a different problem, the buoyancy bins cannot be the same as tilted case, and the WMT result is not directly comparable
elseif simname == "flat"
    z_face = zF
    bin_edge1 = 0:7:1500#0
    bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
    bin_mask1 = hab   
    bin_edge2 = collect(range(0.0005353, 0.0016, length=41))  # Define the edges of the bins
    bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2
    ∇κ∇B_t = zeros(length(bin_center1), length(bin_center2), length(t))
    div_uB_t = zeros(length(bin_center1), length(bin_center2), length(t))
    b = ds_b["b"][:,:,:,1:1]

    ## 1) whole domain
    for n in 1:2:length(t)
        global ∇κ∇B = ds_3D_Bbudget["∇κ∇B"][:,:,:,n:n];   # ∇⋅κ∇B: buoyancy flux divergence
        global div_uB = ds_3D_Bbudget["div_uB"][:,:,:,n:n];
        B = ds_3D_B["B"][:,:,:,n:n];
        B[b.==0].=0 
    
        bin_mask2 = B
        @time int_∇κ∇B, _, _= bins_2d(∇κ∇B,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z_face,normalize=false)
        @time int_div_uB, _, _= bins_2d(div_uB,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z_face,normalize=false)
        ∇κ∇B_t[:,:,n] .= int_∇κ∇B./diff(bin_edge2)[1]
        div_uB_t[:,:,n] .= int_div_uB./diff(bin_edge2)[1]

        fill!(∇κ∇B, 0)
        fill!(div_uB, 0)
        fill!(B, 0)
        
    end

    close(ds_field)
    close(ds_3D_B)
    close(ds_3D_Bbudget)
    close(ds_b)
    close(ds_hab)

    ds_create = Dataset(string(base_folder, "output/",simname,"/WMT_total_tᶠ=",tᶠ,".nc"),"c")

    # Define the dimension
    defDim(ds_create,"z_TF",length(bin_center1))
    defDim(ds_create,"buoyancy",length(bin_center2))
    defDim(ds_create,"buoyancy_diff",length(bin_center2))
    defDim(ds_create,"time",length(t[:]))

    # Define a global attribute
    ds_create.attrib["title"] = "2D histogram, binned by hab and buoyancy"
    # Define the variables
    v1 = defVar(ds_create,"∇κ∇B_t",Float64,("z_TF","buoyancy_diff","time"))
    v1[:,:,:] = ∇κ∇B_t
    v2 = defVar(ds_create,"div_uB_t",Float64,("z_TF","buoyancy_diff","time"))
    v2[:,:,:] = div_uB_t
    v3 = defVar(ds_create,"bin_center1",Float64,("z_TF",))
    v3[:,1] = bin_center1
    v4 = defVar(ds_create,"bin_center2",Float64,("buoyancy",))
    v4[:,1] = bin_center2
    v5 = defVar(ds_create,"t",Float64,("time",))
    v5[:,1] = t[:]

    # write attributes
    v1.attrib["units"] = "m³/s"
    v2.attrib["units"] = "m³/s"
    v3.attrib["units"] = "m"
    v4.attrib["units"] = "m/s²"
    v5.attrib["units"] = "s"

    close(ds_create)


    ## 2) canyon
    filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_Bbudget-wb.nc")
    ds_field = Dataset(filename_field,"r")
    filename_3D_B = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_B-c.nc")
    ds_3D_B = Dataset(filename_3D_B,"r")
    filename_3D_Bbudget = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_Bbudget.nc")
    ds_3D_Bbudget = Dataset(filename_3D_Bbudget,"r")
    ds_b = Dataset(filename_b_verification,"r")

    north = argmin(abs.(yC[:] .- 20e3))  
    south = argmin(abs.(yC[:] .- 10e3))  
    filename_hab = "output/hab.nc"
    ds_hab = Dataset(filename_hab,"r")
    hab = ds_hab["hab"][:,south:north,:];
    z_face = zF
    ∇κ∇B_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))
    div_uB_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))

    for n in 1:2:length(t)
        global ∇κ∇B = ds_3D_Bbudget["∇κ∇B"][:,south:north,:,n:n];   # ∇⋅κ∇B: buoyancy flux divergence
        global div_uB = ds_3D_Bbudget["div_uB"][:,south:north,:,n:n];
        B = ds_3D_B["B"][:,south:north,:,n:n];
        b = ds_b["b"][:,south:north,:,1:1]
        B[b.==0].=0
        
        bin_edge1 = 0:7:1500
        bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
        bin_mask1 = hab
        bin_edge2 = collect(range(0.0005353, 0.0016, length=41))
        bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2

        bin_mask2 = B
        @time int_∇κ∇B, _, _= bins_2d(∇κ∇B,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z_face,normalize=false)
        @time int_div_uB, _, _= bins_2d(div_uB,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z_face,normalize=false)
        ∇κ∇B_t[:,:,n] .= int_∇κ∇B./diff(bin_edge2)[1]
        div_uB_t[:,:,n] .= int_div_uB./diff(bin_edge2)[1]

        fill!(∇κ∇B, 0)
        fill!(div_uB, 0)
        fill!(B, 0)
    end
    
    close(ds_field)
    close(ds_3D_B)
    close(ds_3D_Bbudget)
    close(ds_b)
    close(ds_hab)

    ds_create = Dataset(string(base_folder, "output/",simname,"/WMT_canyon_tᶠ=",tᶠ,".nc"),"c")

    # Define the dimension
    defDim(ds_create,"z_TF",length(bin_center1))
    defDim(ds_create,"buoyancy",length(bin_center2))
    defDim(ds_create,"buoyancy_diff",length(bin_center2))
    defDim(ds_create,"time",length(t[:]))

    # Define a global attribute
    ds_create.attrib["title"] = "2D histogram, binned by hab and buoyancy"
    # Define the variables
    v1 = defVar(ds_create,"∇κ∇B_t",Float64,("z_TF","buoyancy_diff","time"))
    v1[:,:,:] = ∇κ∇B_t
    v2 = defVar(ds_create,"div_uB_t",Float64,("z_TF","buoyancy_diff","time"))
    v2[:,:,:] = div_uB_t
    v3 = defVar(ds_create,"bin_center1",Float64,("z_TF",))
    v3[:,1] = bin_center1
    v4 = defVar(ds_create,"bin_center2",Float64,("buoyancy",))
    v4[:,1] = bin_center2
    v5 = defVar(ds_create,"t",Float64,("time",))
    v5[:,1] = t[:]

    # write attributes
    v1.attrib["units"] = "m³/s"
    v2.attrib["units"] = "m³/s"
    v3.attrib["units"] = "m"
    v4.attrib["units"] = "m/s²"
    v5.attrib["units"] = "s"

    close(ds_create)


    ## 3) flanks
    filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_Bbudget-wb.nc")
    ds_field = Dataset(filename_field,"r")
    filename_3D_B = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_B-c.nc")
    ds_3D_B = Dataset(filename_3D_B,"r")
    filename_3D_Bbudget = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_Bbudget.nc")
    ds_3D_Bbudget = Dataset(filename_3D_Bbudget,"r")
    ds_b = Dataset(filename_b_verification,"r")

    north = argmin(abs.(yC[:] .- 20e3))  
    south = argmin(abs.(yC[:] .- 10e3))  
    filename_hab = "output/hab.nc"
    ds_hab = Dataset(filename_hab,"r")
    hab = ds_hab["hab"][:,vcat(1:south,north:end),:];
    z_face = zF
    ∇κ∇B_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))
    div_uB_t = zeros(length(bin_center1),length(bin_center2),length(t[:]))

    for n in 1:2:length(t)
        global ∇κ∇B = ds_3D_Bbudget["∇κ∇B"][:,vcat(1:south,north:end),:,n:n];   # ∇⋅κ∇B: buoyancy flux divergence
        global div_uB = ds_3D_Bbudget["div_uB"][:,vcat(1:south,north:end),:,n:n];
        B = ds_3D_B["B"][:,vcat(1:south,north:end),:,n:n];
        b = ds_b["b"][:,vcat(1:south,north:end),:,1:1]
        B[b.==0].=0
        
        bin_edge1 = 0:7:1500
        bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
        bin_mask1 = hab
        bin_edge2 = collect(range(0.0005353, 0.0016, length=41))
        bin_center2 = (bin_edge2[1:end-1] .+ bin_edge2[2:end]) ./ 2

        bin_mask2 = B
        @time int_∇κ∇B, _, _= bins_2d(∇κ∇B,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z_face,normalize=false)
        @time int_div_uB, _, _= bins_2d(div_uB,bin_edge1,bin_edge2,bin_mask1,bin_mask2,dx=dx,dy=dy,z_face=z_face,normalize=false)
        ∇κ∇B_t[:,:,n] .= int_∇κ∇B./diff(bin_edge2)[1]
        div_uB_t[:,:,n] .= int_div_uB./diff(bin_edge2)[1]

        fill!(∇κ∇B, 0)
        fill!(div_uB, 0)
        fill!(B, 0)
    end
    
    close(ds_field)
    close(ds_3D_B)
    close(ds_3D_Bbudget)
    close(ds_b)
    close(ds_hab)

    ds_create = Dataset(string(base_folder, "output/",simname,"/WMT_flanks_tᶠ=",tᶠ,".nc"),"c")

    # Define the dimension
    defDim(ds_create,"z_TF",length(bin_center1))
    defDim(ds_create,"buoyancy",length(bin_center2))
    defDim(ds_create,"buoyancy_diff",length(bin_center2))
    defDim(ds_create,"time",length(t[:]))

    # Define a global attribute
    ds_create.attrib["title"] = "2D histogram, binned by hab and buoyancy"
    # Define the variables
    v1 = defVar(ds_create,"∇κ∇B_t",Float64,("z_TF","buoyancy_diff","time"))
    v1[:,:,:] = ∇κ∇B_t
    v2 = defVar(ds_create,"div_uB_t",Float64,("z_TF","buoyancy_diff","time"))
    v2[:,:,:] = div_uB_t
    v3 = defVar(ds_create,"bin_center1",Float64,("z_TF",))
    v3[:,1] = bin_center1
    v4 = defVar(ds_create,"bin_center2",Float64,("buoyancy",))
    v4[:,1] = bin_center2
    v5 = defVar(ds_create,"t",Float64,("time",))
    v5[:,1] = t[:]

    # write attributes
    v1.attrib["units"] = "m³/s"
    v2.attrib["units"] = "m³/s"
    v3.attrib["units"] = "m"
    v4.attrib["units"] = "m/s²"
    v5.attrib["units"] = "s"

    close(ds_create)

end

    # for n in range
    #     z̃_face = z_face .+ n*Lz     # extended domain
    #     B̃ = B .+ n*ΔB
    #     f1, _ = bins(∇κ∇B[:,:,:,1],bin_edge,B̃,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
    #     int_∇κ∇B += f1
    #     f2, _ = bins(div_uB[:,:,:,1],bin_edge,B̃,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
    #     int_div_uB += f2
    #     f3, _ = bins(dBdt[:,:,:,1],bin_edge,B̃,dx=dx,dy=dy,z_face=z̃_face,normalize=false)
    #     int_dBdt += f3
    #     @show n
    # end



# ## 
# using NCDatasets
#     file_WMT = "output/tilt/WMT_70-80.nc"
#     ds = Dataset(file_WMT,"r")
#     int_∇κ∇B = ds["int_∇κ∇B"]
#     int_div_uB = ds["int_div_uB"]
#     int_dBdt = ds["int_dBdt"]
#     bin_center = ds["bin_center"]
    
#     dBdt_b = diff(int_dBdt[:])./diff(bin_center[:])
#     div_uB_b = diff(int_div_uB[:])./diff(bin_center[:])
#     ∇κ∇B_b = diff(int_∇κ∇B[:])./diff(bin_center[:])
#     bin_edge = (0.1:0.02:0.9).*1e-3  # Define the edges of the bins

#     using PyPlot
#     close(gcf())
#     # Plot each variable
#     figure(figsize=(3.5,6.5),constrained_layout=true)
#     PyPlot.plot(dBdt_b, bin_edge[2:end-1], label=L"\frac{\partial}{\partial B}\iiint \frac{\partial B}{\partial t} dV", color="blue", linestyle="-")
#     PyPlot.plot(div_uB_b, bin_edge[2:end-1], label=L"\frac{\partial}{\partial B}\iiint{\bf{u}}\cdot\nabla B dV", color="red", linestyle="-")
#     PyPlot.plot(∇κ∇B_b, bin_edge[2:end-1], label=L"\frac{\partial}{\partial B}\iiint\nabla\cdot(\kappa\nabla B) dV", color="green", linestyle="-")
#     # PyPlot.plot(-div_uB_b.+∇κ∇B_b, bin_edge[2:end-1], label="", color="black", linestyle="--")
#     PyPlot.plot([0,0],[0.0001,0.0009], color="black", linestyle="-")
#     ylim(0.0001,0.0009)
#     gcf()
#     # tight_layout()

#     # Adding labels and title
#     ylabel("Buoyancy, B")
#     # ylabel("Value")
#     title("Water mass transformation")

#     # Add a legend
#     legend()

#     # Show the plot
#     show()
#     simname = "tilt"
#     tᶠ = "40-80"
## Visualize B field for flat case
# if simname == "flat"
#     # Load B field for visualization
#     B_final = ds_3D_B["B"][:,500,:,end]
    
#     using PyPlot
    
#     # Create a new figure
#     figure(figsize=(10, 5))
    
#     # Create x-z grid for plotting
#     x_plot = xC * cos(θ)
#     z_plot = zC
    
#     # Meshgrid for contour
#     contour_levels = 3.8e-6:0.0001:0.002636
#     # Add contour lines with labels
#    cs = plt.contour(xC, zC, B_final', levels=50, colors="black", linewidths=0.5)
#     clabel(cs, inline=1, fontsize=8, fmt="%.5f")
    
#     # Add colorbar and labels
#     xlabel("x (m)")
#     ylabel("z (m)")
#     title("Buoyancy field (flat case)")
    
#     # Save the figure
#     savefig(string(base_folder, "output/", simname, "/B_field_xz_slice_tf", tᶠ, ".png"), dpi=200)
    
#     println("Buoyancy field visualization saved.")
# end
