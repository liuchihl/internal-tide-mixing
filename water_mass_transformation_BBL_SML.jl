#  This script calculates the terrain-following averaged quantities by interpolation
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using LinearAlgebra
using Interpolations
##
function deriv(z,y)
   dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
   return dydz
end

# load data

    filename_field = "output/test_Bbudget/internal_tide_theta=0.0036_realtopo3D_Nx=10_Nz=5_0-1_threeD_timeavg.nc"
    ds_field = Dataset(filename_field,"r")

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
n = 2
    b = ds_field["b"][:,:,:,n:n];          # buoyancy perturbation
    B = ds_field["B"][:,:,:,n:n];          # total buoyancy
    ∇κ∇B = ds_field["∇κ∇B"][:,:,:,n:n];    # ∇⋅κ∇B: buoyancy flux divergence
    ∇κ∇B[b.==0] .= NaN;
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
    for n in -1:-1:-2
    x̃ = [x .+ n*Lx ; x̃ ; x .- n*Lx]
    z̃ = [z .+ n*Lz ; z̃ ; z .- n*Lz]
    z̃_face = [z_face .+ n*Lz ; z̃_face ; z_face .- n*Lz]
    B̃ = [B .+ n*ΔB ; B̃ ; B .- n*ΔB]
    end
    ∇κ∇B̃ = [∇κ∇B;∇κ∇B;∇κ∇B;∇κ∇B;∇κ∇B];

## figure
    # fig = Figure()
    # axis_kwargs_hm = (xlabel = "x (m)",
    #               ylabel = "z (m)",
    #               yminorticksvisible = true,
    #               limits = ((x̃[1],x̃[end]), (z̃[1], z̃[end]))
    #               )


    # ax_b = Axis(fig[1, 1]; title = "B (Total Buoyancy)", yminorticks = IntervalsBetween(5),axis_kwargs_hm...)
    # using ColorSchemes
    # U₀ = 0.025
    # hm_b = contour!(ax_b, x̃[:,1], z̃[1,:], B̃[:,1,:,1],
    #     colorrange = (minimum(filter(!isnan,B̃)),maximum(filter(!isnan,B̃))*0.8), colormap = :diverging_bwr_20_95_c54_n256,
    #     lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    #     # Colorbar(fig[1,2], hm_b )
    # display(fig)
    # save(string("output/",slope,"/hab_buoyancy_",timerange,".png"),fig)
    


    using PyPlot
    close(gcf())
    fig, axes = subplots(1,1, figsize=(10, 8))
    axes[:pcolor](x̃, z̃, ∇κ∇B̃[:,1,:,1])
    axes[:contour](x̃, z̃, B̃[:,1,:,1],colors="black",(0.0001:0.00005:0.002),linewidths=1,nan_color="gray")
    xlabel("x (m)")
    ylabel("z (m)")
    minorticks_on()

    gcf()

## bin the flux terms with respective to buoyancy
    bin_edge = (1:2:20)*1e-4  # Define the edges of the bins
    bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end])/2
    layer_mask = zeros(size(x̃,1),Ny,Nz,length(t))
    integrand = zeros(length(bin_center),length(t))

    ΔV = reshape(dx*dy*diff(z̃_face,dims=2), (50, 1, 5, 1))
    for i=1:length(bin_center)
        layer_mask .= bin_edge[i] .<= B̃[:,:,:,:] .< bin_edge[i+1]
        integrand[i,:] = nansum(∇κ∇B̃ .* layer_mask .* ΔV, dim = (1,2,3))
    end



    