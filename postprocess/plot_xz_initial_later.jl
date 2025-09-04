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


function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end
 
θ=0.0036
# include("functions/mmderiv.jl")
simname = "tilt"
## load data
fname_early = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
early_slice = Dataset(fname_early,"r")
fname_late = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", 460.0, "_analysis_round=all_slices_xz.nc")
late_slice = Dataset(fname_late, "r")


# grids
zC = early_slice["zC"]; Nz=length(zC)
zF = early_slice["zF"]; 
xC = early_slice["xC"]; Nx=length(xC)
yC = early_slice["yC"]; Ny=length(yC)

# load all data
ind = 51
B_early = early_slice["B"][:,1,:,ind];       
b_early = early_slice["b"][:,1,:,ind]; # for mask    
ε_early = early_slice["ε"][:,1,:,ind];
uhat_early = early_slice["uhat"][:,1,:,ind];

ind = 12
B_late = late_slice["B"][:,1,:,ind];       
ε_late = late_slice["ε"][:,1,:,ind];
uhat_late = late_slice["uhat"][:,1,:,ind];


b_early[b_early.==0] .= NaN
ε_early[ε_early.==0] .= NaN
ε_late[ε_late.==0] .= NaN
uhat_early[uhat_early.==0] .= NaN
B_early[isnan.(b_early)] .= NaN
B_late[isnan.(b_early)] .= NaN
uhat_late[uhat_late.==0] .= NaN

ω₀ = 1.4e-4
M₂_period = 2π/ω₀
ω₀ = 1.4e-4
N = 1e-3
f₀ = -0.53e-4
s_ω₀ = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))
s_2ω₀ = sqrt(((2ω₀)^2-f₀^2)/(N^2-(2ω₀)^2))
s_3ω₀ = sqrt(((3ω₀)^2-f₀^2)/(N^2-(3ω₀)^2))
s_4ω₀ = sqrt(((4ω₀)^2-f₀^2)/(N^2-(4ω₀)^2))

xir = [7.5, 15].*1e3
zi = s_ω₀ * xir
xdiff = minimum(zi[:]) - 1000;
zir = s_ω₀ * xir .- xdiff

xird = [8, 11.5].*1e3
zi = -s_ω₀ * xird
xdiff = minimum(zi[:]) - 400;
zird = -s_ω₀ * xird .- xdiff

xil = [0, 7.5].*1e3
zi = -s_ω₀ * xil
xdiff = minimum(zi[:]) - 1000;
zil = -s_ω₀ * xil .- xdiff

xild = [3, 7].*1e3
zi = s_ω₀ * xild
xdiff = minimum(zi[:]) - 410;
zild = s_ω₀ * xild .- xdiff

xi2ω = [7.5, 15].*1e3
zi = 2s_ω₀ * xi2ω
xdiff = minimum(zi[:]) - 1000;
zi2ω = 2s_ω₀ * xi2ω .- xdiff

xi2ωl = [0, 7].*1e3
zi = -2s_ω₀ * xi2ωl
xdiff = minimum(zi[:]) - 1000;
zi2ωl = -2s_ω₀ * xi2ωl .- xdiff

xi3ω = [7.9, 15].*1e3
zi = 3s_ω₀ * xi3ω
xdiff = minimum(zi[:]) - 1000;
zi3ω = 3s_ω₀ * xi3ω .- xdiff

xi3ωl = [0, 7].*1e3
zi = -3s_ω₀ * xi3ωl
xdiff = minimum(zi[:]) - 1000;
zi3ωl = -3s_ω₀ * xi3ωl .- xdiff

xi4ω = [7.6, 15].*1e3
zi = 4s_ω₀ * xi4ω
xdiff = minimum(zi[:]) - 1000;
zi4ω = 4s_ω₀ * xi4ω .- xdiff

xi4ωl = [0, 7.6].*1e3
zi = -4s_ω₀ * xi4ωl
xdiff = minimum(zi[:]) - 1000;
zi4ωl = -4s_ω₀ * xi4ωl .- xdiff



# Prepare data for early and late stages
# Early: use B_early, ε_early, B_early for contours, xC, zC
# Late: use B_late, ε_late, B_late for contours, xC, zC

# Set up figure
fig = CairoMakie.Figure(resolution = (1250, 700), size=(1250, 700), fontsize=22)
axis_kwargs = (
    xlabel = "x [km]",
    ylabel = "z [m]",
    limits = ((0, 15), (0, zC[end])),
    xticklabelsize=18, yticklabelsize=18, xlabelsize=22, ylabelsize=22
    )

# First row: early stage
ax_u_early = Axis(fig[1, 1];  axis_kwargs..., titlesize=24)
ax_ε_early = Axis(fig[2, 1];  axis_kwargs..., titlesize=24)

# Second row: late stage
ax_u_late = Axis(fig[1, 2]; axis_kwargs..., titlesize=24)
ax_ε_late = Axis(fig[2, 2]; axis_kwargs..., titlesize=24)



# Early U
hm_u_early = heatmap!(ax_u_early, 1e-3*xC[:], zC[:], uhat_early;
    colorrange = (-0.08, 0.08), colormap = :diverging_bwr_40_95_c42_n256,
    nan_color = :gray)
contour!(ax_u_early, 1e-3*xC, zC, B_early, levels=1e-4:.5e-4:3.e-3, linewidth=1.2, color=:black, alpha=0.7)

# Early ε
hm_ε_early = heatmap!(ax_ε_early, 1e-3*xC[:], zC[:], log10.(ε_early);
    colorrange = (-10, -6), colormap = :matter,
    nan_color = :gray)
contour!(ax_ε_early, 1e-3*xC, zC, B_early, levels=1e-4:.5e-4:3.e-3, linewidth=1.2, color=:black, alpha=0.7)

# Late U
hm_u_late = heatmap!(ax_u_late, 1e-3*xC[:], zC[:], uhat_late;
    colorrange = (-0.08, 0.08), colormap = :diverging_bwr_40_95_c42_n256,
    nan_color = :gray)
contour!(ax_u_late, 1e-3*xC, zC, B_late, levels=10.5e-4:.5e-4:3.e-3, linewidth=1.2, color=:black, alpha=0.7)

# Late ε
hm_ε_late = heatmap!(ax_ε_late, 1e-3*xC[:], zC[:], log10.(ε_late);
    colorrange = (-10, -6), colormap = :matter,
    nan_color = :gray)
contour!(ax_ε_late, 1e-3*xC, zC, B_late, levels=10.5e-4:.5e-4:3.e-3, linewidth=1.2, color=:black, alpha=0.7)

lines!(ax_u_early, xir/1e3,zir, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xil/1e3,zil, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xild/1e3,zild, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xird/1e3,zird, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xi2ω/1e3,zi2ω, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xi2ωl/1e3,zi2ωl, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xi3ω/1e3,zi3ω, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xi3ωl/1e3,zi3ωl, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xi4ω/1e3,zi4ω, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_u_early, xi4ωl/1e3,zi4ωl, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xir/1e3,zir, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xil/1e3,zil, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xild/1e3,zild, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xi2ω/1e3,zi2ω, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xi2ωl/1e3,zi2ωl, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xi3ω/1e3,zi3ω, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xi3ωl/1e3,zi3ωl, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xi4ω/1e3,zi4ω, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xi4ωl/1e3,zi4ωl, linewidth=2.2, color=:black, linestyle=:dash)
lines!(ax_ε_early, xird/1e3,zird, linewidth=2.2, color=:black, linestyle=:dash)



# Colorbars: one for each row, placed at the right
Colorbar(fig[1, 3], hm_u_late, label="û [m s⁻¹]", width=20, height=220, ticklabelsize=18, labelsize=22)
Colorbar(fig[2, 3], hm_ε_late, label="log₁₀(ε) [W kg⁻¹]", width=20, height=220, ticklabelsize=18, labelsize=22)

fig
save("output/tilt/xz_uhat_epsilon_early_late.png", fig)