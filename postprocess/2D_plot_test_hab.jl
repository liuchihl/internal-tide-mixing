### this script tests why 2D hab average has wiggles 
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations
using MAT

simname = "2D_tilt"
timerange = "0-500"
θ = 0.0036
## load data
filename_slice = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_",timerange, "_1TP_timeavg.nc")
ds = Dataset(filename_slice,"r")

# grids
zC = ds["zC"][:]; Nz=length(zC); 
zF = ds["zF"][:]; 
xF = ds["xF"][:];
xC = ds["xC"][:]; Nx=length(xC)
yC = ds["yC"][:]; Ny=length(yC)
t = ds["time"][:];

# load all data
B = ds["B"][:,:,:,:]; 
Bz = ds["Bz"][:,:,:,:];
b = ds["b"][:,:,:,:];
uhat = ds["uhat"][:,:,:,:];
what = ds["what"][:,:,:,:];
v = ds["v"][:,:,:,:];


# set topography to NaN
uhat[uhat.==0] .= NaN
what[what.==0] .= NaN
v[b.==0] .= NaN
B[b.==0] .= NaN
Bz[b.==0] .= NaN


simname = "2D_tilt"
timerange = "0-500"
θ = 0.0036
## load data
filename_tilt = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_",timerange, "_1TP_timeavg.nc")
ds_tilt = Dataset(filename_tilt,"r")

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
hab = ones(Nx,1,Nz).*reshape(zC,1,1,Nz) .- reshape(z_interp,Nx,1,1).*ones(1,1,Nz)  # height above bottom [Nx,1,Nz]

# mask it with buoyancy 
# Find height above bottom from buoyancy field
hab_mask = zeros(Nx,Ny,Nz)
w_data = ds_tilt["what"][:,:,:,1]  # get buoyancy data for first time step
for i in 1:Nx    # loop over x dimension
    for j in 1:Ny # loop over y dimension
        first_nonzero = findfirst(w_data[i,j,:] !== 0)  # find first index where b >= 0 in z dimension
        if first_nonzero !== nothing
            for k in first_nonzero:Nz
                hab_mask[i,j,k] = zF[k] - zF[first_nonzero]  # distance above topography
            end
        end
    end
end

# Create comparison figure
fig_compare = Figure(resolution=(1200, 600))

# Set up axes for heatmaps
ax_hab = Axis(fig_compare[1, 1], title="Height Above Bottom (Original)",
              xlabel="X", ylabel="Z")
ax_binvar = Axis(fig_compare[1, 2], title="Height Above Bottom (from Buoyancy)",
                 xlabel="X", ylabel="Z")
ax_line = Axis(fig_compare[2, 1:2], title="Comparison at selected x positions",
               xlabel="Height (m)", ylabel="Z")

# Plot heatmaps with same colorrange
max_val = maximum(filter(!isnan, vcat(hab, bin_var)))
min_val = minimum(filter(!isnan, vcat(hab, bin_var)))

hm1 = heatmap!(ax_hab, hab[:, 1, :].-bin_var[:,1,:], colorrange=(0, 1))
contour!(ax_hab, hab[:, 1, :], levels=0:0, linewidth=0.5, color=:black)
hm2 = heatmap!(ax_binvar, bin_var[:, 1, :], colorrange=(0, max_val))
contour!(ax_binvar, bin_var[:, 1, :], levels=0:0, linewidth=0.5, color=:black)
Colorbar(fig_compare[1, 3], hm1)

# Plot line comparisons at a few x positions
x_positions = 100
# Plot vertical profiles
lines!(ax_line, hab[x_positions, 1, :], zC, label="HAB x=100")
lines!(ax_line, bin_var[x_positions, 1, :], zC, linestyle=:dash, label="BinVar x=100") 

# Add horizontal slice axis
ax_horizontal = Axis(fig_compare[3, 1:2], title="Comparison at z=100m",
                    xlabel="X (m)", ylabel="Height (m)")

# Plot horizontal slices
z_index = 20
lines!(ax_horizontal, xC, hab[:,1,z_index].-bin_var[:,1,z_index], label="HAB")
# lines!(ax_horizontal, xC, bin_var[:,1,z_index], linestyle=:dash, label="BinVar")
# axislegend(ax_horizontal, position=:rt)

axislegend(ax_line, position=:rt)
save("output/2D_tilt/height_above_bottom_comparison.png", fig_compare)

# Create hab matrix using the bin_mask
hab = ones(Nx,1,Nz).*reshape(zC,1,1,Nz) .* bin_mask


include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")
bin_edge = 0:10:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2



Nt = length(t)
b_avg = zeros(length(bin_edge)-1,Nt)
Bz_avg = zeros(length(bin_edge)-1,Nt)
uhat_avg = zeros(length(bin_edge)-1,Nt)
u_avg = zeros(length(bin_edge)-1,Nt)
what_avg = zeros(length(bin_edge)-1,Nt)
τ_avg = zeros(length(bin_edge)-1,Nt)

n = 501
    b = ds_tilt["b"][:,:,:,n:n];           # buoyancy perturbation
    B = ds_tilt["B"][:,:,:,n:n];           # total buoyancy
    Bz = ds_tilt["Bz"][:,:,:,n:n];           # total buoyancy
    uhat = ds_tilt["uhat"][:,:,:,n:n];     # true u
    what = ds_tilt["what"][:,:,:,n:n];     # true w
    what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
    # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
    wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
    u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
    w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
    @time Bz_avg[:,n], _ = bins(Bz,bin_edge,bin_mask,dx=dx,dy=30,z_face=z_face,normalize=true)
    @time what_avg[:,n], _ = bins(what_cen,bin_edge,bin_mask,dx=dx,dy=30,z_face=z_face,normalize=true)
    @time u_avg[:,n], _ = bins(u,bin_edge,bin_mask,dx=dx,dy=30,z_face=z_face,normalize=true)        
    @time b_avg[:,n], _ = bins(b,bin_edge,bin_mask,dx=dx,dy=30,z_face=z_face,normalize=true)        


    # Compare results using different bin_mask inputs
    w_avg_interp, _ = bins(what_cen,bin_edge,hab,dx=dx,dy=30,z_face=z_face,normalize=true)
    w_avg_mask, _ = bins(what_cen,bin_edge,bin_mask,dx=dx,dy=30,z_face=z_face,normalize=true)
    # Create a line plot comparing the two methods
    fig = Figure(resolution = (500, 400))
    ax = Axis(fig[1, 1], 
              title = "",
              ylabel = "Height above bottom (m)",
              xlabel = "w average",
              limits = (nothing, (0, 500))
              )
    ax_com = Axis(fig[1, 2], 
              title = "Comparison of HAB methods",
              xlabel = "Height above bottom (m)",
              ylabel = "w average",
              limits = (nothing, (0, 500))
              )

    lines!(ax, w_avg_interp[:,1],bin_center, label="Original HAB")
    lines!(ax, w_avg_mask[:,1],bin_center, label="HAB from w mask", linestyle=:dash)
    axislegend(ax, position=:rt)  # right-top position

    lines!(ax_com, w_avg_interp[:,1].-w_avg_mask[:,1],bin_center, label="comparison")
  
    display(fig)
    save(string("output/",simname,"/hab_comparison_lineplot_",timerange,".png"), fig)
