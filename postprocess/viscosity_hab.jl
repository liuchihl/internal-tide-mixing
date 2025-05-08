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
    dydz = diff(y[:, :, :, :], dims=3) ./ reshape(diff(z), 1, 1, length(z) - 1)
    return dydz
end

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")

tᶠ = 460
θ = 3.6e-3
if θ == 3.6e-3
    simname = "tilt"
else
    simname = "flat"
end
tᶠ = 460

bin_edge = 0:7:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
bin_mask = hab

# preallocate
nu_avg = zeros(1, length(bin_edge))

fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=460_threeD_timeavg_const_dt-b-B-nu.nc"
ds_field = Dataset(fname, "r")
t = ds_field["time"][:];
ν = mean(ds_field["νₑ"][:, :, :, :], dims=4); # viscosity
# grids
zC = ds_field["zC"][:];
zF = ds_field["zF"][:];
Nz = length(zC[:]);

xC = ds_field["xC"][:];
xF = ds_field["xF"][:];
Nx = length(xC[:]);
dx = xF[end] - xF[end-1];

yC = ds_field["yC"][:];
yF = ds_field["yF"][:];
Ny = length(yC[:]);
dy = yF[end] - yF[end-1];
z_face = zF
@time nu_avg, _ = bins(ν, bin_edge, bin_mask, dx=dx, dy=dy, z_face=z_face, normalize=true)

# analytical viscosity
nu_BOT = 1.8e-3
nu_BG = 5.3e-5
h = 230 
d = 0
z = bin_center
nu_analytical = nu_BG .+ nu_BOT.*exp.(-(z.+d)./h)

using PyPlot

# Set up the figure
close("all")
plt.figure(figsize=(8, 6))
plt.plot(nu_avg[:], bin_center, "-", linewidth=2)
plt.plot(nu_analytical, bin_center, "--", linewidth=2)
# Add labels and title
xlabel("Average Viscosity (ν)", fontsize=12)
ylabel("Height Above Bottom (m)", fontsize=12)
# plt.title("Average Viscosity vs Height Above Bottom", fontsize=14)
legend(["Numerical", "Analytical"], loc="upper right", fontsize=12)
# Add grid
grid(true, alpha=0.3)

# Improve appearance
tight_layout()

# Save the figure
savefig("output/tilt/viscosity_vs_hab.png", dpi=300)
println("Figure saved as viscosity_vs_hab.png")
