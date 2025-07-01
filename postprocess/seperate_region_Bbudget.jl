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

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bin_stat_over_xy.jl")
###### load topography first
Nx = 500
Ny = 1000
Nz = 250
basefile = "/scratch/bcpi/cliu28/internal-tide-mixing"
cd(basefile)
using MAT
using Interpolations
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic")
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo, 1))
y_topo_lin = range(y_topo[1], y_topo[end], size(z_topo, 2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1], x_topo[end], length=Nx)
y_interp = range(y_topo[1], y_topo[end], length=Ny)


# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp .- minimum(z_interp)

###### Define functions
function deriv(z, y)
    dydz = diff(y[:, :, :, :], dims=3) ./ reshape(diff(z), 1, 1, length(z) - 1)
    return dydz
end

function create_masks(; sill_height_threshold=300, high_sill_threshold=830, w=w)
    # Create a mask for elevated bathymetry in the central region
    mask_bathymetry = zeros(Int, size(z_interp))
    mask_bathymetry[z_interp.>sill_height_threshold] .= 1
    mask_bathymetry[:, 18Ny÷30:Ny] .= 0
    mask_bathymetry[:, 1:7Ny÷30] .= 0

    # Read the vertical velocity data
    mask_downwelling = zeros(Int, size(z_interp))
    # Check each water column
    for i in 1:Nx
        for j in 1:Ny
            # If any point in the column meets the condition
            if any(w[i, j, :] .< -1e-2)
                # Mark the entire column
                mask_downwelling[i, j] = 1
            end
        end
    end

    # Combine both masks - sill is where we have both high bathymetry and strong downwelling
    mask_sill = mask_downwelling .* mask_bathymetry

    ## Create separate masks for the north and south sills
    mask_sill_bathymetry = zeros(Int, size(z_interp))
    mask_sill_bathymetry[z_interp.>high_sill_threshold] .= 1

    mask_north_bathymetry = copy(mask_sill_bathymetry)
    mask_north_bathymetry[:, 1:18Ny÷30] .= 0
    mask_south_bathymetry = copy(mask_sill_bathymetry)
    mask_south_bathymetry[:, Ny÷3:Ny] .= 0
    mask_flanks = mask_north_bathymetry .+ mask_south_bathymetry
    return mask_sill, mask_flanks
end

######

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/mmderiv.jl")

tᶠ = 454.0
θ = 3.6e-3
if θ == 3.6e-3
    simname = "tilt"
else
    simname = "flat"
end

if tᶠ ≤ 10
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
    Nt = 1    # only 1 tidal average
    endtime = tᶠ
end


bin_edge = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
bin_mask = hab

# preallocate
B_avg = zeros(length(bin_edge) - 1, Nt)
Bz_avg = zeros(length(bin_edge) - 1, Nt)
uhat_avg = zeros(length(bin_edge) - 1, Nt)
u_avg = zeros(length(bin_edge) - 1, Nt)
what_avg = zeros(length(bin_edge) - 1, Nt)

∇κ∇B_avg = zeros(length(bin_edge) - 1, Nt)
div_uB_avg = zeros(length(bin_edge) - 1, Nt)
u_bar_∇B_bar_avg = zeros(length(bin_edge) - 1, Nt)
u_prime∇B_prime_avg = zeros(length(bin_edge) - 1, Nt)
dBdt_avg = zeros(length(bin_edge) - 1, Nt)

# Load two consecutive half tidal periods for full cycle averaging from snapshots
# because the time average outputs are not calculated correctly, so we decided to just use 3D snapshots
global total_steps
tᶠ_0 = tᶠ - 1
tᶠ_first = tᶠ - 0.5
tᶠ_second = tᶠ

filename_3D_0 = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_0, "_analysis_round=all_threeD.nc")
filename_3D_first = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_first, "_analysis_round=all_threeD.nc")
filename_3D_second = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_second, "_analysis_round=all_threeD.nc")
filename_verification = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", 10, "_threeD_timeavg.nc")
ds_3D_0 = Dataset(filename_3D_0, "r")
ds_3D_first = Dataset(filename_3D_first, "r")
ds_3D_second = Dataset(filename_3D_second, "r")
ds_verification = Dataset(filename_verification, "r")

# Get dimensions
zC = ds_3D_first["z_aac"][:]
zF = ds_3D_first["z_aaf"][:]
xC = ds_3D_first["x_caa"][:]
xF = ds_3D_first["x_faa"][:]
yC = ds_3D_first["y_aca"][:]
yF = ds_3D_first["y_afa"][:]
Nz = length(zC[:])
Nx = length(xC[:])
dx = xF[end] - xF[end-1]
Ny = length(yC[:])
dy = yF[end] - yF[end-1]
z_face = zF

# Initialize variables for running sums
dB̄dx = zeros(Nx, Ny, Nz, 1)
dB̄dy = zeros(Nx, Ny, Nz, 1)
dB̄dz = zeros(Nx, Ny, Nz, 1)
b = ds_verification["b"][:, :, :, 1:1]

# Combined running sums
# first define the variables using the final timestep of tf-0.5 chunk 
# (for example, the first timestep of 452.5 chunk is 452.0833, but we want 452.0 and it is at the final timestep  in tf-0.5 chunk)
B_sum = ds_3D_0["B"][:, :, :, end:end]
uhat_sum = ds_3D_0["uhat"][:, :, :, end:end]
what_sum = ds_3D_0["what"][:, :, :, end:end]
v_sum = ds_3D_0["v"][:, :, :, end:end]
∇κ∇B_sum = ds_3D_0["∇κ∇B"][:, :, :, end:end]
div_uB_sum = ds_3D_0["div_uB"][:, :, :, end:end]
ε_sum = ds_3D_0["ε"][:, :, :, end:end]
χ_sum = ds_3D_0["χ"][:, :, :, end:end]
# Count total timesteps processed
total_steps = 1   # already have one timestep from the first dataset

# Process first dataset
Nt_first = length(ds_3D_first["time"][:])
for n in 1:Nt_first
    # Read data for this timestep
    B_sum .+= ds_3D_first["B"][:, :, :, n:n]
    uhat_sum .+= ds_3D_first["uhat"][:, :, :, n:n]
    what_sum .+= ds_3D_first["what"][:, :, :, n:n]
    v_sum .+= ds_3D_first["v"][:, :, :, n:n]
    ∇κ∇B_sum .+= ds_3D_first["∇κ∇B"][:, :, :, n:n]
    div_uB_sum .+= ds_3D_first["div_uB"][:, :, :, n:n]
    ε_sum .+= ds_3D_first["ε"][:, :, :, n:n]
    χ_sum .+= ds_3D_first["χ"][:, :, :, n:n]
    # Free memory by explicitly clearing variables if needed
    GC.gc()
    global total_steps += 1
    println(total_steps)
end

# Process second dataset
Nt_second = length(ds_3D_second["time"][:])
for n in 1:Nt_second
    # Read data for this timestep
    B_sum .+= ds_3D_second["B"][:, :, :, n:n]
    uhat_sum .+= ds_3D_second["uhat"][:, :, :, n:n]
    what_sum .+= ds_3D_second["what"][:, :, :, n:n]
    v_sum .+= ds_3D_second["v"][:, :, :, n:n]
    ∇κ∇B_sum .+= ds_3D_second["∇κ∇B"][:, :, :, n:n]
    div_uB_sum .+= ds_3D_second["div_uB"][:, :, :, n:n]
    ε_sum .+= ds_3D_second["ε"][:, :, :, n:n]
    χ_sum .+= ds_3D_second["χ"][:, :, :, n:n]

    global total_steps += 1

    # Free memory by explicitly clearing variables if needed
    GC.gc()
end

# Calculate averages
B = B_sum ./ total_steps
uhat = uhat_sum ./ total_steps
what = what_sum ./ total_steps
v = v_sum ./ total_steps
∇κ∇B = ∇κ∇B_sum ./ total_steps
div_uB = div_uB_sum ./ total_steps
ε = ε_sum ./ total_steps
χ = χ_sum ./ total_steps


# Clear the intermediate sums to free memory
B_sum = nothing
uhat_sum = nothing
what_sum = nothing
v_sum = nothing
∇κ∇B_sum = nothing
div_uB_sum = nothing
GC.gc()

what_cen = (what[:, :, 1:end-1, 1] .+ what[:, :, 2:end, 1]) ./ 2 # what at center
# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end, :, :], what_cen[1:end-1, :, :]) .+ what_cen[:, :, :]) ./ 2
u = uhat[:, :, :, 1] * cos(θ) .+ wtemp * sin(θ) # cross-slope velocity
w = -uhat[:, :, :, 1] * sin(θ) .+ wtemp * cos(θ)# slope-normal velocity


N = 1.e-3
Bz = deriv(zC, B)
Bz[b[:, :, 1:end-1, :].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
dx =  30
dy = 30
dB̄dx = diff(vcat(B[end:end,:,:,1], B[:,:,:,1]), dims=1) ./ dx
dB̄dy = diff(hcat(B[:, end:end, :, 1], B[:, :, :, 1]), dims=2) ./ dy # dB̄dy is the derivative of B with respect to y at the center cell

# interpolate Bz from faces to center cell
using Interpolations
# Interpolate each row
Bz_center = zeros(size(Bz, 1), size(Bz, 2), length(zC), 1)
for i in 1:size(Bz, 1)
    for j in 1:size(Bz, 2)
        itp = linear_interpolation(zF[2:end-1], Bz[i, j, :, 1], extrapolation_bc=Line())
        Bz_center[i, j, :, :] = itp(zC)
    end
end
Bx_center = zeros(size(B))
By_center = zeros(size(B))
# Interpolate dB̄dx and dB̄dy from faces to center cell
for i in 1:size(dB̄dx, 2)
    for j in 1:size(dB̄dx, 3)
        itp = linear_interpolation(xF[1:end], dB̄dx[:, i, j, 1], extrapolation_bc=Line())
        Bx_center[:, i, j, :] = itp(xC)
    end
end
for i in 1:size(dB̄dy, 1)
    for j in 1:size(dB̄dy, 3)
        itp = linear_interpolation(yF[1:end], dB̄dy[i, :, j, 1], extrapolation_bc=Line())
        By_center[i, :, j, :] = itp(yC)
    end
end
Bx_center[abs.(Bx_center).>0.5*nanmaximum(abs.(Bx_center))] .= 0  # set the points inside and right above the immersed boundary to NaN
By_center[abs.(By_center).>0.5*nanmaximum(abs.(By_center))] .= 0 

u_bar_∇B_bar = u .* Bx_center .+ v .* By_center .+ w .* Bz_center
u_bar_∇B_bar[b.==0] .= NaN  # set the points inside and right above the immersed boundary to NaN
div_uB[b.==0] .= NaN
u_prime∇B_prime = div_uB .- u_bar_∇B_bar

### define masks including flanks and sills using vertical velocity and bathymetry
mask_sill, mask_flanks = create_masks(; sill_height_threshold=300, high_sill_threshold=830, w=w)
# Convert the masks to Float64 before assigning NaN values
mask_rest = (1 .- mask_sill) .* (1 .- mask_flanks) 

mask_sill = Float64.(mask_sill)
mask_sill[mask_sill.==0] .= NaN

mask_flanks = Float64.(mask_flanks)
mask_flanks[mask_flanks.==0] .= NaN

mask_rest = Float64.(mask_rest)
mask_rest[mask_rest.==0] .= NaN

# terrain following quantities:
#1) full domain
bin_var=hab

@time B_avg_rest = bin_stat_over_xy(B.*mask_rest, bin_edge, bin_var; stat="mean")
@time Bz_avg_rest = bin_stat_over_xy(Bz_center.*mask_rest, bin_edge, bin_var; stat="mean")
@time what_avg_rest = bin_stat_over_xy(what_cen.*mask_rest, bin_edge, bin_var; stat="mean")
@time u_avg_rest = bin_stat_over_xy(u.*mask_rest, bin_edge, bin_var; stat="mean")
@time ∇κ∇B_avg_rest = bin_stat_over_xy(∇κ∇B.*mask_rest, bin_edge, bin_var; stat="mean")
@time div_uB_avg_rest = bin_stat_over_xy(div_uB.*mask_rest, bin_edge, bin_var; stat="mean")
@time u_bar_∇B_bar_avg_rest = bin_stat_over_xy(u_bar_∇B_bar.*mask_rest, bin_edge, bin_var; stat="mean")
@time u_prime∇B_prime_avg_rest = bin_stat_over_xy(u_prime∇B_prime.*mask_rest, bin_edge, bin_var; stat="mean")
@time uBx_center_rest = bin_stat_over_xy(u .* Bx_center.*mask_rest, bin_edge, bin_var; stat="mean")
@time vBy_center_rest = bin_stat_over_xy(v .* By_center.*mask_rest, bin_edge, bin_var; stat="mean")
@time wBz_center_rest = bin_stat_over_xy(w .* Bz_center.*mask_rest, bin_edge, bin_var; stat="mean")
@time ε_avg_rest = bin_stat_over_xy(ε.*mask_rest, bin_edge, bin_var; stat="mean")
@time χ_avg_rest = bin_stat_over_xy(χ.*mask_rest, bin_edge, bin_var; stat="mean")


#2) flanks
@time B_avg_flanks = bin_stat_over_xy(B.*mask_flanks, bin_edge, bin_var; stat="mean")
@time Bz_avg_flanks = bin_stat_over_xy(Bz_center.*mask_flanks, bin_edge, bin_var; stat="mean")
@time what_avg_flanks = bin_stat_over_xy(what_cen.*mask_flanks, bin_edge, bin_var; stat="mean")
@time u_avg_flanks = bin_stat_over_xy(u.*mask_flanks, bin_edge, bin_var; stat="mean")
@time ∇κ∇B_avg_flanks = bin_stat_over_xy(∇κ∇B.*mask_flanks, bin_edge, bin_var; stat="mean")
@time div_uB_avg_flanks = bin_stat_over_xy(div_uB.*mask_flanks, bin_edge, bin_var; stat="mean")
@time u_bar_∇B_bar_avg_flanks = bin_stat_over_xy(u_bar_∇B_bar.*mask_flanks, bin_edge, bin_var; stat="mean")
@time u_prime∇B_prime_avg_flanks = bin_stat_over_xy(u_prime∇B_prime.*mask_flanks, bin_edge, bin_var; stat="mean")
@time uBx_center_flanks = bin_stat_over_xy(u .* Bx_center.*mask_flanks, bin_edge, bin_var; stat="mean")
@time vBy_center_flanks = bin_stat_over_xy(v .* By_center.*mask_flanks, bin_edge, bin_var; stat="mean") 
@time wBz_center_flanks = bin_stat_over_xy(w .* Bz_center.*mask_flanks, bin_edge, bin_var; stat="mean")
@time ε_avg_flanks = bin_stat_over_xy(ε.*mask_flanks, bin_edge, bin_var; stat="mean")
@time χ_avg_flanks = bin_stat_over_xy(χ.*mask_flanks, bin_edge, bin_var; stat="mean")


#3) sills
@time B_avg_sill = bin_stat_over_xy(B.*mask_sill, bin_edge, bin_var; stat="mean")
@time Bz_avg_sill = bin_stat_over_xy(Bz_center.*mask_sill, bin_edge, bin_var; stat="mean")
@time what_avg_sill = bin_stat_over_xy(what_cen.*mask_sill, bin_edge, bin_var; stat="mean")
@time u_avg_sill = bin_stat_over_xy(u.*mask_sill, bin_edge, bin_var; stat="mean")
@time ∇κ∇B_avg_sill = bin_stat_over_xy(∇κ∇B.*mask_sill, bin_edge, bin_var; stat="mean")
@time div_uB_avg_sill = bin_stat_over_xy(div_uB.*mask_sill, bin_edge, bin_var; stat="mean")
@time u_bar_∇B_bar_avg_sill = bin_stat_over_xy(u_bar_∇B_bar.*mask_sill, bin_edge, bin_var; stat="mean")
@time u_prime∇B_prime_avg_sill = bin_stat_over_xy(u_prime∇B_prime.*mask_sill, bin_edge, bin_var; stat="mean")
@time uBx_center_sill = bin_stat_over_xy(u .* Bx_center.*mask_sill, bin_edge, bin_var; stat="mean")
@time vBy_center_sill = bin_stat_over_xy(v .* By_center.*mask_sill, bin_edge, bin_var; stat="mean")
@time wBz_center_sill = bin_stat_over_xy(w .* Bz_center.*mask_sill, bin_edge, bin_var; stat="mean")
@time ε_avg_sill = bin_stat_over_xy(ε.*mask_sill, bin_edge, bin_var; stat="mean")
@time χ_avg_sill = bin_stat_over_xy(χ.*mask_sill, bin_edge, bin_var; stat="mean")

dBdt = (ds_3D_second["B"][:, :, :, end] .- ds_3D_0["B"][:, :, :, end]) ./ (ds_3D_second["time"][end] .- ds_3D_0["time"][end])
@time dBdt_avg_rest = bin_stat_over_xy(dBdt.*mask_rest, bin_edge, bin_var; stat="mean")
@time dBdt_avg_flanks = bin_stat_over_xy(dBdt.*mask_flanks, bin_edge, bin_var; stat="mean")
@time dBdt_avg_sill = bin_stat_over_xy(dBdt.*mask_sill, bin_edge, bin_var; stat="mean")


close(ds_3D_0)
close(ds_3D_first)
close(ds_3D_second)
close(ds_verification)
close(ds_hab)

# This creates a new NetCDF file 
# The mode "c" stands for creating a new file
ds_create = Dataset(string("output/", simname, "/TF_avg_tᶠ=", tᶠ, "_analysis.nc"), "c")
# Define the dimension
defDim(ds_create, "z_TF", length(bin_center))
defDim(ds_create, "t", Nt)
defDim(ds_create, "xC", Nx)
defDim(ds_create, "yC", Ny)

# Define a global attribute
ds_create.attrib["title"] = "Terrain-following averages"
# Define variables for the rest of the domain
v1 = defVar(ds_create, "B_avg_rest", Float64, ("z_TF", "t"))
v1[:, :] = B_avg_rest
v2 = defVar(ds_create, "Bz_avg_rest", Float64, ("z_TF", "t"))
v2[:, :] = Bz_avg_rest
v3 = defVar(ds_create, "what_avg_rest", Float64, ("z_TF", "t"))
v3[:, :] = what_avg_rest
v4 = defVar(ds_create, "u_avg_rest", Float64, ("z_TF", "t"))
v4[:, :] = u_avg_rest
v5 = defVar(ds_create, "dBdt_avg_rest", Float64, ("z_TF", "t"))
v5[:, :] = dBdt_avg_rest
v6 = defVar(ds_create, "∇κ∇B_avg_rest", Float64, ("z_TF", "t"))
v6[:, :] = ∇κ∇B_avg_rest
v7 = defVar(ds_create, "div_uB_avg_rest", Float64, ("z_TF", "t"))
v7[:, :] = div_uB_avg_rest
v8 = defVar(ds_create, "u_bar_∇B_bar_avg_rest", Float64, ("z_TF", "t"))
v8[:, :] = u_bar_∇B_bar_avg_rest
v9 = defVar(ds_create, "u_prime∇B_prime_avg_rest", Float64, ("z_TF", "t"))
v9[:, :] = u_prime∇B_prime_avg_rest
v9a = defVar(ds_create, "uBx_center_rest", Float64, ("z_TF", "t"))
v9a[:, :] = uBx_center_rest
v9b = defVar(ds_create, "vBy_center_rest", Float64, ("z_TF", "t"))
v9b[:, :] = vBy_center_rest
v9c = defVar(ds_create, "wBz_center_rest", Float64, ("z_TF", "t"))
v9c[:, :] = wBz_center_rest
v9d = defVar(ds_create, "ε_avg_rest", Float64, ("z_TF", "t"))
v9d[:, :] = ε_avg_rest
v9e = defVar(ds_create, "χ_avg_rest", Float64, ("z_TF", "t"))
v9e[:, :] = χ_avg_rest

# Define variables for the flanks
v10 = defVar(ds_create, "B_avg_flanks", Float64, ("z_TF", "t"))
v10[:, :] = B_avg_flanks
v11 = defVar(ds_create, "Bz_avg_flanks", Float64, ("z_TF", "t"))
v11[:, :] = Bz_avg_flanks
v12 = defVar(ds_create, "what_avg_flanks", Float64, ("z_TF", "t"))
v12[:, :] = what_avg_flanks
v13 = defVar(ds_create, "u_avg_flanks", Float64, ("z_TF", "t"))
v13[:, :] = u_avg_flanks
v14 = defVar(ds_create, "dBdt_avg_flanks", Float64, ("z_TF", "t"))
v14[:, :] = dBdt_avg_flanks
v15 = defVar(ds_create, "∇κ∇B_avg_flanks", Float64, ("z_TF", "t"))
v15[:, :] = ∇κ∇B_avg_flanks
v16 = defVar(ds_create, "div_uB_avg_flanks", Float64, ("z_TF", "t"))
v16[:, :] = div_uB_avg_flanks
v17 = defVar(ds_create, "u_bar_∇B_bar_avg_flanks", Float64, ("z_TF", "t"))
v17[:, :] = u_bar_∇B_bar_avg_flanks
v18 = defVar(ds_create, "u_prime∇B_prime_avg_flanks", Float64, ("z_TF", "t"))
v18[:, :] = u_prime∇B_prime_avg_flanks
v18a = defVar(ds_create, "uBx_center_flanks", Float64, ("z_TF", "t"))
v18a[:, :] = uBx_center_flanks
v18b = defVar(ds_create, "vBy_center_flanks", Float64, ("z_TF", "t"))
v18b[:, :] = vBy_center_flanks
v18c = defVar(ds_create, "wBz_center_flanks", Float64, ("z_TF", "t"))
v18c[:, :] = wBz_center_flanks
v18d = defVar(ds_create, "ε_avg_flanks", Float64, ("z_TF", "t"))
v18d[:, :] = ε_avg_flanks
v18e = defVar(ds_create, "χ_avg_flanks", Float64, ("z_TF", "t"))
v18e[:, :] = χ_avg_flanks

# Define variables for the sills
v19 = defVar(ds_create, "B_avg_sill", Float64, ("z_TF", "t"))
v19[:, :] = B_avg_sill
v20 = defVar(ds_create, "Bz_avg_sill", Float64, ("z_TF", "t"))
v20[:, :] = Bz_avg_sill
v21 = defVar(ds_create, "what_avg_sill", Float64, ("z_TF", "t"))
v21[:, :] = what_avg_sill
v22 = defVar(ds_create, "u_avg_sill", Float64, ("z_TF", "t"))
v22[:, :] = u_avg_sill
v23 = defVar(ds_create, "dBdt_avg_sill", Float64, ("z_TF", "t"))
v23[:, :] = dBdt_avg_sill
v24 = defVar(ds_create, "∇κ∇B_avg_sill", Float64, ("z_TF", "t"))
v24[:, :] = ∇κ∇B_avg_sill
v25 = defVar(ds_create, "div_uB_avg_sill", Float64, ("z_TF", "t"))
v25[:, :] = div_uB_avg_sill
v26 = defVar(ds_create, "u_bar_∇B_bar_avg_sill", Float64, ("z_TF", "t"))
v26[:, :] = u_bar_∇B_bar_avg_sill
v27 = defVar(ds_create, "u_prime∇B_prime_avg_sill", Float64, ("z_TF", "t"))
v27[:, :] = u_prime∇B_prime_avg_sill
v27a = defVar(ds_create, "uBx_center_sill", Float64, ("z_TF", "t"))
v27a[:, :] = uBx_center_sill
v27b = defVar(ds_create, "vBy_center_sill", Float64, ("z_TF", "t"))
v27b[:, :] = vBy_center_sill
v27c = defVar(ds_create, "wBz_center_sill", Float64, ("z_TF", "t"))
v27c[:, :] = wBz_center_sill
v27d = defVar(ds_create, "ε_avg_sill", Float64, ("z_TF", "t"))
v27d[:, :] = ε_avg_sill
v27e = defVar(ds_create, "χ_avg_sill", Float64, ("z_TF", "t"))
v27e[:, :] = χ_avg_sill

# Define dimension variables
v28 = defVar(ds_create, "bin_center", Float64, ("z_TF",))
v28[:] = bin_center
v29 = defVar(ds_create, "t", Float64, ("t",))
v29[:] = tᶠ
v30 = defVar(ds_create, "mask_sill", Float64, ("xC", "yC"))
v30[:, :] = mask_sill
v31 = defVar(ds_create, "mask_flanks", Float64, ("xC", "yC"))
v31[:, :] = mask_flanks
v32 = defVar(ds_create, "mask_rest", Float64, ("xC", "yC"))
v32[:, :] = mask_rest
close(ds_create)



#### plot the buoyancy budget terms at different regions 
using PyPlot
using NCDatasets
simname = "tilt"  # or "flat" depending on the simulation
tᶠ = 454.0
# Load the saved data
ds_budget = Dataset(string("output/", simname, "/TF_avg_tᶠ=", tᶠ, "_analysis.nc"), "r")
bin_center_saved = ds_budget["bin_center"][:]

# Load budget terms for each region
dBdt_rest = ds_budget["dBdt_avg_rest"][:, 1]
∇κ∇B_rest = ds_budget["∇κ∇B_avg_rest"][:, 1]
div_uB_rest = ds_budget["div_uB_avg_rest"][:, 1]
u_bar_∇B_bar_rest = ds_budget["u_bar_∇B_bar_avg_rest"][:, 1]
u_prime∇B_prime_rest = ds_budget["u_prime∇B_prime_avg_rest"][:, 1]

dBdt_flanks = ds_budget["dBdt_avg_flanks"][:, 1]
∇κ∇B_flanks = ds_budget["∇κ∇B_avg_flanks"][:, 1]
div_uB_flanks = ds_budget["div_uB_avg_flanks"][:, 1]
u_bar_∇B_bar_flanks = ds_budget["u_bar_∇B_bar_avg_flanks"][:, 1]
u_prime∇B_prime_flanks = ds_budget["u_prime∇B_prime_avg_flanks"][:, 1]

dBdt_sill = ds_budget["dBdt_avg_sill"][:, 1]
∇κ∇B_sill = ds_budget["∇κ∇B_avg_sill"][:, 1]
div_uB_sill = ds_budget["div_uB_avg_sill"][:, 1]
u_bar_∇B_bar_sill = ds_budget["u_bar_∇B_bar_avg_sill"][:, 1]
u_prime∇B_prime_sill = ds_budget["u_prime∇B_prime_avg_sill"][:, 1]

mask_rest = ds_budget["mask_rest"][:, :]
mask_flanks = ds_budget["mask_flanks"][:, :]
mask_sill = ds_budget["mask_sill"][:, :]
mask_rest[isnan.(mask_rest)] .= 0
mask_flanks[isnan.(mask_flanks)] .= 0
mask_sill[isnan.(mask_sill)] .= 0

close(ds_budget)

using MAT
using Interpolations
import PyCall

file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic")
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo, 1))
y_topo_lin = range(y_topo[1], y_topo[end], size(z_topo, 2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1], x_topo[end], length=Nx)
y_interp = range(y_topo[1], y_topo[end], length=Ny)


# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp .- minimum(z_interp)
# plot

# Define region colors
region_colors = [
    (0.6, 0.4, 0.2, 0.8),  # Brown for sill region
    (0.7, 0.9, 1.0, 0.8),  # Light blue for flanks
    (0.2, 0.4, 0.8, 0.8)   # Darker blue for rest of domain
]

# Create figure with custom subplot layout
close("all")
# Set font size globally
plt.rcParams["font.size"] = 15

fig = figure(figsize=(12, 8))

# Left panel spanning all 3 rows
ax_map = subplot2grid((3, 2), (0, 0), rowspan=3, colspan=1)

# Right panels - one in each row
ax_sill = subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
ax_flanks = subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1)
ax_rest = subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)

# Create a combined mask for the topographic map
simname = "tilt"  # or "flat" depending on the simulation
θ = 3.6e-3
Nx = 500
Ny = 1000

filename_3D = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_analysis_round=all_threeD.nc")
ds_3D = Dataset(filename_3D, "r")
# Get dimensions
xC = ds_3D["x_caa"][:]
yC = ds_3D["y_aca"][:]

combined_mask = zeros(Float64, Nx, Ny)
mask_sill_2d = mask_sill[:, :] 
combined_mask[mask_sill_2d .== 1] .= 1
combined_mask[(mask_sill_2d .== 0) .& (mask_flanks .> 0)] .= 2
combined_mask[(mask_sill_2d .== 0) .& (mask_flanks .== 0)] .= 3
combined_mask[combined_mask .== 0] .= NaN

# Create a custom colormap with the same colors as the line plot
region_cmap = matplotlib.colors.ListedColormap([region_colors[1], region_colors[2], region_colors[3]])
bounds = [0.5, 1.5, 2.5, 3.5]
norm = matplotlib.colors.BoundaryNorm(bounds, region_cmap.N)

# Plot the topographic map with correct aspect ratio
im = ax_map.imshow(
    combined_mask', 
    origin="lower", 
    extent=[xC[1], xC[end], yC[1], yC[end]] ./ 1e3,
    aspect="equal",
    cmap=region_cmap, 
    norm=norm
)
ax_map.contour(
    xC / 1e3, yC / 1e3, combined_mask', 
    levels=[1.5, 2.5], colors="white", linewidths=1.0
)
# Add bathymetry contours
contours = ax_map.contour(
    xC / 1e3, yC / 1e3, z_interp', 
    0:200:2200, colors="black", alpha=0.5, linewidths=0.5
)
patches = matplotlib.patches
legend_elements = [
    patches.Patch(facecolor=region_colors[1], label="Sill"),
    patches.Patch(facecolor=region_colors[2], label="Flanks"),
    patches.Patch(facecolor=region_colors[3], label="Rest")
]
ax_map.legend(handles=legend_elements, loc="upper right", 
              frameon=true, framealpha=0.9, fontsize=12)

# Add labels for topography
ax_map.set_xlabel("x [km]", fontsize=14)
ax_map.set_ylabel("y [km]", fontsize=14)
# ax_map.set_title("Domain Regions", fontsize=14)
ax_map.spines["top"].set_linewidth(2)
ax_map.spines["bottom"].set_linewidth(2)
ax_map.spines["left"].set_linewidth(2)
ax_map.spines["right"].set_linewidth(2)

ax_map.tick_params(axis="both", which="major", labelsize=15)
ax_map.set_xticks([0, 5, 10, 15])
ax_map.set_xticklabels([0, 5, 10, 15])
ax_map.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax_map.set_yticklabels([0, 5, 10, 15, 20, 25, 30])
# --- Budget plots for each region ---
# Define budget term colors to match your reference
budget_colors = [[150, 148, 255] / 255, [136, 194, 115] / 255, "orange", "red", "black","gray"]
budget_labels = [L"\nabla\cdot(\overline{\mathbf{u}}\overline{B})", L" \nabla\cdot\overline{\mathbf{u'}B'}", 
                L"\overline{\nabla\cdot(\mathbf{u}B)}", L"\nabla\cdot(\overline{\mathcal{B}})",
                L"d\overline{B}/dt","RHS"]

# Function to plot budget terms for each region
function plot_budget_region(ax, dBdt, ∇κ∇B, div_uB, u_bar_∇B_bar, u_prime∇B_prime, bin_center, title, region_color)
    # Plot budget terms
    ax.plot(u_bar_∇B_bar*1e9, bin_center, color=budget_colors[1], linewidth=2.5, label=budget_labels[1])
    ax.plot(u_prime∇B_prime*1e9, bin_center, color=budget_colors[2], linewidth=2.5, label=budget_labels[2])
    ax.plot(div_uB*1e9, bin_center, color=budget_colors[3], linewidth=2.5, label=budget_labels[3])
    ax.plot(∇κ∇B*1e9, bin_center, color=budget_colors[4], linewidth=3, linestyle=":", label=budget_labels[4])
    ax.plot(dBdt*1e9, bin_center, color=budget_colors[5], linewidth=2.5, linestyle="-", label=budget_labels[5])
    ax.plot((∇κ∇B.-div_uB)*1e9, bin_center, color=budget_colors[6], linewidth=2.5, linestyle="--", label=budget_labels[6])
    
    # Add zero line
    ax.plot([0, 0], [0, 1000], color="black", linewidth=1.5)
    
    # Set symlog scale and ticks
    # ax.set_xscale("symlog", linthresh=10^(-9.1))
    # ax.set_xticks([-1e-9, -1e-10, 0, 1e-10, 1e-9])
    # ax.set_xticklabels([L"-10^{-9}", L"-10^{-10}", "0", L"10^{-10}", L"10^{-9}"])
    
    # Set y-limits and styling
    # ax.set_title(title, fontsize=14, color=region_color)
    ax.grid(true, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_yscale("log")
    ax.set_ylim(0, 500)
    # Configure minor ticks
    # matplotlib_ticker = PyCall.pyimport("matplotlib.ticker")
    # FixedLocator = matplotlib_ticker.FixedLocator
    
    # minor_ticks_negative = [-2e-10, -3e-10, -4e-10, -5e-10, -6e-10, -7e-10, -8e-10, -9e-10]
    # minor_ticks_positive = [2e-10, 3e-10, 4e-10, 5e-10, 6e-10, 7e-10, 8e-10, 9e-10]
    # all_minor_ticks = vcat(minor_ticks_negative, minor_ticks_positive)
    
    # ax.xaxis.set_minor_locator(FixedLocator(all_minor_ticks))
    # ax.tick_params(axis="x", which="minor", length=4, width=0.8)
    # ax.tick_params(axis="x", which="major", length=7, width=1.0)
    # ax.tick_params(axis="y", which="minor", length=4, width=0.8)
    # ax.tick_params(axis="y", which="major", length=7, width=1.0)
end

# Plot each region
plot_budget_region(ax_sill, dBdt_sill, ∇κ∇B_sill, div_uB_sill, u_bar_∇B_bar_sill, 
                  u_prime∇B_prime_sill, bin_center_saved, "Sill", region_colors[1])
ax_sill.set_xlim(-3, 3)

plot_budget_region(ax_flanks, dBdt_flanks, ∇κ∇B_flanks, div_uB_flanks, u_bar_∇B_bar_flanks, 
                  u_prime∇B_prime_flanks, bin_center_saved, "Flanks", region_colors[2])
ax_flanks.set_ylabel("Height above bottom [m]", fontsize=14)
ax_flanks.set_xlim(-3, 3)
plot_budget_region(ax_rest, dBdt_rest, ∇κ∇B_rest, div_uB_rest, u_bar_∇B_bar_rest, 
                  u_prime∇B_prime_rest, bin_center_saved, "Rest", region_colors[3])
ax_rest.set_xlabel(L"[10^{-9} \, \text{m s}^{-3}]", fontsize=14)
# ax_rest.set_xlabel(L"[\text{s}^{-2}]", fontsize=14)
ax_rest.set_xlim(-3, 3)
ax_rest.legend(loc="upper right", fontsize=13, frameon=true, framealpha=0.93)
# Enable minor ticks
# minorticks_on()

# Adjust layout
tight_layout()

# Save figure
savefig(string("output/", simname, "/buoyancy_budget_regions_tᶠ=", tᶠ, ".png"), 
        dpi=300, bbox_inches="tight")
display(gcf())


### plot the buoyancy advection terms
using PyPlot
using NCDatasets
simname = "tilt"  # or "flat" depending on the simulation
tᶠ = 454.0
