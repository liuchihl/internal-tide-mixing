using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using PyPlot
using PyCall
using NCDatasets
using Statistics
using NaNStatistics
using BSplineKit
using LinearAlgebra
using Interpolations

# tide = 0
# fn1 = "output/2D_idealized_tilt_notide_0/internal_tide_theta=0_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
# fn2 = "output/2D_idealized_tilt_notide_0.002/internal_tide_theta=0.002_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
# fn3 = "output/2D_idealized_tilt_notide_0.004/internal_tide_theta=0.004_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
# fn4 = "output/2D_idealized_tilt_notide_0.008/internal_tide_theta=0.008_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"

tide = 1
fn1 = "output/2D_idealized_tilt_0/internal_tide_theta=0_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
fn2 = "output/2D_idealized_tilt_0.002/internal_tide_theta=0.002_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
fn3 = "output/2D_idealized_tilt_0.004/internal_tide_theta=0.004_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
fn4 = "output/2D_idealized_tilt_0.008/internal_tide_theta=0.008_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"

ds1 = Dataset(fn1,"r")
ds2 = Dataset(fn2,"r")  
ds3 = Dataset(fn3,"r")  
ds4 = Dataset(fn4,"r")
zC = ds1["z_aac"][:]; Nz=length(zC)
xC = ds1["x_caa"][:]; Nx=length(xC)
t = ds1["time"][:];
time_range = 481:length(t)
# time_range = 721:length(t)
b = ds1["b"][:,:,1];
u1 = nanmean(ds1["uhat"][:,:,time_range], dim=3)
u2 = nanmean(ds2["uhat"][:,:,time_range], dim=3)
u3 = nanmean(ds3["uhat"][:,:,time_range], dim=3)
u4 = nanmean(ds4["uhat"][:,:,time_range], dim=3)

w1 = nanmean(ds1["what"][:,:,time_range], dim=3)
w2 = nanmean(ds2["what"][:,:,time_range], dim=3)
w3 = nanmean(ds3["what"][:,:,time_range], dim=3)
w4 = nanmean(ds4["what"][:,:,time_range], dim=3)

Bz1 = nanmean(ds1["Bz"][:,:,time_range], dim=3)
Bz2 = nanmean(ds2["Bz"][:,:,time_range], dim=3)
Bz3 = nanmean(ds3["Bz"][:,:,time_range], dim=3)
Bz4 = nanmean(ds4["Bz"][:,:,time_range], dim=3)

∇κ∇B1 = nanmean(ds1["∇κ∇B"][:,:,time_range], dim=3)
∇κ∇B2 = nanmean(ds2["∇κ∇B"][:,:,time_range], dim=3)
∇κ∇B3 = nanmean(ds3["∇κ∇B"][:,:,time_range], dim=3)
∇κ∇B4 = nanmean(ds4["∇κ∇B"][:,:,time_range], dim=3)

# center w
w1_cen = (w1[:,1:end-1] .+ w1[:,2:end]) ./ 2
w2_cen = (w2[:,1:end-1] .+ w2[:,2:end]) ./ 2
w3_cen = (w3[:,1:end-1] .+ w3[:,2:end]) ./ 2
w4_cen = (w4[:,1:end-1] .+ w4[:,2:end]) ./ 2

u1[u1.==0] .= NaN
u2[u2.==0] .= NaN
u3[u3.==0] .= NaN
u4[u4.==0] .= NaN
w1_cen[w1_cen.==0] .= NaN
w2_cen[w2_cen.==0] .= NaN
w3_cen[w3_cen.==0] .= NaN
w4_cen[w4_cen.==0] .= NaN
Bz1[Bz1.==0] .= NaN
Bz2[Bz2.==0] .= NaN
Bz3[Bz3.==0] .= NaN
Bz4[Bz4.==0] .= NaN
∇κ∇B1[b.==0] .= NaN
∇κ∇B2[b.==0] .= NaN
∇κ∇B3[b.==0] .= NaN
∇κ∇B4[b.==0] .= NaN

# Collect for limit calculations
U_all = [u1, u2, u3, u4]
W_all = [w1_cen, w2_cen, w3_cen, w4_cen]
Bz_all = [Bz1, Bz2, Bz3, Bz4]
∇κ∇B_all = [∇κ∇B1, ∇κ∇B2, ∇κ∇B3, ∇κ∇B4]

sym_abs_max(arrs) = maximum(abs, reduce(vcat, [vec(a[.!isnan.(a)]) for a in arrs]))
u_lim = sym_abs_max(U_all)
w_lim = sym_abs_max(W_all)
Bz_lim = sym_abs_max(Bz_all)
∇κ∇B_lim = sym_abs_max(∇κ∇B_all)
u_clim = (-u_lim, u_lim).*0.1
w_clim = (-w_lim, w_lim).*0.1
Bz_clim = (0, 2e-6)
∇κ∇B_clim = (-∇κ∇B_lim, ∇κ∇B_lim)  # Set symmetric logarithmic scale limits

# Compute 2D vorticity (∂w/∂x - ∂u/∂z) for each case
function compute_vorticity_2d(u, w, xC, zC)
    # u is on x-centers, z-centers
    # w is on x-centers, z-faces (already centered in the code above)
    Nx, Nz = size(u)
    dx = xC[2] - xC[1]
    dz = zC[2] - zC[1]
    
    vorticity = zeros(Nx, Nz) .* NaN
    
    # Compute ∂w/∂x (centered differences in x)
    dwdx = zeros(Nx, Nz) .* NaN
    for i in 2:Nx-1
        for k in 1:Nz
            if !isnan(w[i+1, k]) && !isnan(w[i-1, k])
                dwdx[i, k] = (w[i+1, k] - w[i-1, k]) / (2*dx)
            end
        end
    end
    
    # Compute ∂u/∂z (centered differences in z)
    dudz = zeros(Nx, Nz) .* NaN
    for i in 1:Nx
        for k in 2:Nz-1
            if !isnan(u[i, k+1]) && !isnan(u[i, k-1])
                dudz[i, k] = (u[i, k+1] - u[i, k-1]) / (2*dz)
            end
        end
    end
    
    # Vorticity = ∂w/∂x - ∂u/∂z
    vorticity = dwdx .- dudz
    
    return vorticity
end

# Compute vorticity for all cases
vort1 = compute_vorticity_2d(u1, w1_cen, xC, zC)
vort2 = compute_vorticity_2d(u2, w2_cen, xC, zC)
vort3 = compute_vorticity_2d(u3, w3_cen, xC, zC)
vort4 = compute_vorticity_2d(u4, w4_cen, xC, zC)

# Find symmetric vorticity limits
vort_all = [vort1, vort2, vort3, vort4]
vort_lim = sym_abs_max(vort_all)
vort_clim = (-vort_lim, vort_lim)

tilts = ["θ=0", "θ=0.002", "θ=0.004", "θ=0.008"]
data_vorticity = [vort1, vort2, vort3, vort4]
data_u = [u1, u2, u3, u4]
data_w = [w1_cen, w2_cen, w3_cen, w4_cen]
# Create figure with 1x4 layout (one per tilt case)
fig, axs = subplots(1, 4, figsize=(20, 4), constrained_layout=true)

# Quiver parameters - subsample for clarity
skip1 = 4  # Plot every Nth arrow
skip2 = 2  # Plot every Nth arrow
X_sub = xC[1:skip1:end]
Z_sub = zC[1:skip2:end]

for i in 1:4
    ax = axs[i]
    
    # Plot vorticity as background
    vort = data_vorticity[i]
    p = ax.pcolormesh(xC, zC, vort', shading="auto", cmap="RdBu_r", 
                      vmin=vort_clim[1].*0.1, vmax=vort_clim[2]*0.1, alpha=0.8)
    # Subsample velocity fields for quiver plot
    u_sub = data_u[i][1:skip1:end, 1:skip2:end]
    w_sub = data_w[i][1:skip1:end, 1:skip2:end]

    # Create meshgrid for quiver
    X_mesh = repeat(X_sub, 1, length(Z_sub))
    Z_mesh = repeat(Z_sub', length(X_sub), 1)
    
    # Plot velocity vectors
    ax.quiver(X_mesh, Z_mesh, u_sub, w_sub, 
              color="black", alpha=0.7, scale_units="xy")
    
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("z (m)", fontsize=11)
    ax.set_title("$(tilts[i])", fontsize=12)
    
    # Add colorbar for vorticity
    if i == 4  # Only add colorbar to the last subplot for cleanliness
    cbar = fig.colorbar(p, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel("Vorticity (1/s)", fontsize=10)
    end
    ax.set_xlim(5000,10000)
    ax.set_ylim(400,1000)
end
# Add overall title
# fig.suptitle("Mean Flow Vorticity and Velocity Vectors", fontsize=14, y=0.995)

# Ensure output directory exists
outdir = "output/Plots_2D_idealized_tilt_mean_flow_tide_$(tide)"
mkpath(outdir)
outfile = joinpath(outdir, "vorticity_velocity_vectors_1x4_zoom.png")
println("Saving figure to $outfile")
savefig(outfile, dpi=150)
close(fig)
