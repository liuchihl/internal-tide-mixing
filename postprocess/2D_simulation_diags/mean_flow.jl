using Printf
using PyPlot
using PyCall
using NCDatasets
using Statistics
using NaNStatistics
using Interpolations

# tide = 0
# fn1 = "output/2D_idealized_tilt_notide_0/internal_tide_theta=0_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
# fn2 = "output/2D_idealized_tilt_notide_0.002/internal_tide_theta=0.002_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
# fn3 = "output/2D_idealized_tilt_notide_0.004/internal_tide_theta=0.004_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
# fn4 = "output/2D_idealized_tilt_notide_0.008/internal_tide_theta=0.008_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"

fn1 = "output/2D_idealized_tilt_0/internal_tide_theta=0_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
fn2 = "output/2D_idealized_tilt_0.008/internal_tide_theta=0.008_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
fn3 = "output/2D_idealized_tilt_0.02/internal_tide_theta=0.02_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
# fn4 = "output/2D_idealized_tilt_0.05/internal_tide_theta=0.05_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"

ds1 = Dataset(fn1, "r")
ds2 = Dataset(fn2, "r")
ds3 = Dataset(fn3, "r")
# ds4 = Dataset(fn4, "r")
zC = ds1["z_aac"][:];
Nz = length(zC);
xC = ds1["x_caa"][:];
Nx = length(xC);
t = ds1["time"][:];
ind_begin = findfirst(t / (2 * pi / 1.4e-4) .>= 100)
# time_range = length(t)-1:length(t)
time_range = ind_begin:length(t)
# time_range = 721:length(t)
b = ds1["b"][:, :, 1];
u1 = nanmean(ds1["uhat"][:, :, time_range], dim=3)
u2 = nanmean(ds2["uhat"][:, :, time_range], dim=3)
u3 = nanmean(ds3["uhat"][:, :, time_range], dim=3)
# u4 = nanmean(ds4["uhat"][:, :, time_range], dim=3)

w1 = nanmean(ds1["what"][:, :, time_range], dim=3)
w2 = nanmean(ds2["what"][:, :, time_range], dim=3)
w3 = nanmean(ds3["what"][:, :, time_range], dim=3)
# w4 = nanmean(ds4["what"][:, :, time_range], dim=3)

B1 = nanmean(ds1["B"][:, :, time_range], dim=3)
B2 = nanmean(ds2["B"][:, :, time_range], dim=3)
B3 = nanmean(ds3["B"][:, :, time_range], dim=3)

Bz1 = nanmean(ds1["Bz"][:, :, time_range], dim=3)
Bz2 = nanmean(ds2["Bz"][:, :, time_range], dim=3)
Bz3 = nanmean(ds3["Bz"][:, :, time_range], dim=3)
# Bz4 = nanmean(ds4["Bz"][:, :, time_range], dim=3)

∇κ∇B1 = nanmean(ds1["∇κ∇B"][:, :, time_range], dim=3)
∇κ∇B2 = nanmean(ds2["∇κ∇B"][:, :, time_range], dim=3)
∇κ∇B3 = nanmean(ds3["∇κ∇B"][:, :, time_range], dim=3)

# center w
w1_cen = (w1[:, 1:end-1] .+ w1[:, 2:end]) ./ 2
w2_cen = (w2[:, 1:end-1] .+ w2[:, 2:end]) ./ 2
w3_cen = (w3[:, 1:end-1] .+ w3[:, 2:end]) ./ 2
# w4_cen = (w4[:, 1:end-1] .+ w4[:, 2:end]) ./ 2

u1[u1.==0] .= NaN
u2[u2.==0] .= NaN
u3[u3.==0] .= NaN
# u4[u4.==0] .= NaN
w1_cen[w1_cen.==0] .= NaN
w2_cen[w2_cen.==0] .= NaN
w3_cen[w3_cen.==0] .= NaN
# w4_cen[w4_cen.==0] .= NaN
Bz1[Bz1.==0] .= NaN
Bz2[Bz2.==0] .= NaN
Bz3[Bz3.==0] .= NaN
# Bz4[Bz4.==0] .= NaN
∇κ∇B1[b.==0] .= NaN
∇κ∇B2[b.==0] .= NaN
∇κ∇B3[b.==0] .= NaN
# ∇κ∇B4[b.==0] .= NaN
B1[b.==0] .= NaN
B2[b.==0] .= NaN
B3[b.==0] .= NaN
# B4[b.==0] .= NaN

# convert u into cross-slope flow
θ = 0
u1_cross = (u1[:, :]) * cos(θ) .+ w1_cen * sin(θ)
θ = 0.008
u2_cross = (u2[:, :]) * cos(θ) .+ w2_cen * sin(θ)
θ = 0.02
u3_cross = (u3[:, :]) * cos(θ) .+ w3_cen * sin(θ)


# Collect for limit calculations
U_all = [u1_cross, u2_cross, u3_cross]
W_all = [w1_cen, w2_cen, w3_cen]
Bz_all = [Bz1, Bz2, Bz3]
∇κ∇B_all = [∇κ∇B1, ∇κ∇B2, ∇κ∇B3]

sym_abs_max(arrs) = maximum(abs, reduce(vcat, [vec(a[.!isnan.(a)]) for a in arrs]))
u_lim = sym_abs_max(U_all)
w_lim = sym_abs_max(W_all)
Bz_lim = sym_abs_max(Bz_all)
# ∇κ∇B_lim = sym_abs_max(∇κ∇B_all)
u_clim = (-u_lim, u_lim) .* 0.5
w_clim = (-w_lim, w_lim) .* 0.5
Bz_clim = (0, 2e-6)
# ∇κ∇B_clim = (-∇κ∇B_lim, ∇κ∇B_lim).*0.1  # Set symmetric logarithmic scale limits

tilts = ["θ=0", "θ=0.008", "θ=0.02"]
titles = ["u", "ŵ", L"B_z", "∇κ∇B"]
data_rows = [(u1_cross, w1_cen, Bz1), (u2_cross, w2_cen, Bz2), (u3_cross, w3_cen, Bz3)]
clims = [u_clim, w_clim, Bz_clim]
cmaps = ["RdBu_r", "RdBu_r", "RdBu_r"]  # Use "coolwarm" for symmetric logarithmic scale

# Create figure with 3x3 layout
fig, axs = subplots(3, 3, figsize=(18, 10), constrained_layout=true)

for i in 1:length(data_rows)
    for j in 1:3
        ax = axs[i, j]
        dat = data_rows[i][j]
        # Create pcolormesh with appropriate colormap and limits
        p = ax.pcolormesh(xC, zC, dat', shading="auto", cmap=cmaps[j], vmin=clims[j][1], vmax=clims[j][2])

        # Axis labels only on left and bottom for cleanliness
        if j == 1
            ax.set_ylabel("z")
        end
        if i == 3
            ax.set_xlabel("x")
        end
        ax.set_title("$(titles[j]) ($(tilts[i]))", fontsize=10)
        if i == 1
            # Superimpose buoyancy contours
            b_contours = ax.contour(xC, zC, B1', levels=10, colors="k", linewidths=0.5)
            ax.clabel(b_contours, inline=1, fontsize=8, fmt="%.1e")
        elseif i == 2
            b_contours = ax.contour(xC, zC, B2', levels=10, colors="k", linewidths=0.5)
            ax.clabel(b_contours, inline=1, fontsize=8, fmt="%.1e")
        elseif i == 3
            b_contours = ax.contour(xC, zC, B3', levels=10, colors="k", linewidths=0.5)
            ax.clabel(b_contours, inline=1, fontsize=8, fmt="%.1e")
        end
    end

end
# Shared colorbars per row
for i in 1:3
    # Collect mappables from the first plot in each row
    mappable = axs[1, i].collections[1]  # pcolormesh is first collection
    cbar = fig.colorbar(mappable, ax=axs[:, i], shrink=0.3, pad=0.02, aspect=20)
    cbar.ax.set_ylabel(titles[i])
    if i == 1 || i == 2
        cbar.ax.set_title("(m/s)")
    elseif i == 3
        cbar.ax.set_title("(1/s)")
    end
end

# Add a super title summarizing limits
fig.suptitle("Mean flow comparison across bottom tilt cases", fontsize=14)

# Ensure output directory exists
outdir = "output/2D_idealized_tilt_mean_flow_tide_$(tide)"
mkpath(outdir)
outfile = joinpath(outdir, "mean_flow_comparison_3x3_150TP.png")
println("Saving figure to $outfile")
savefig(outfile)
close(fig)



ε1 = nanmean(ds1["ε"][:, :, time_range], dim=3)
ε2 = nanmean(ds2["ε"][:, :, time_range], dim=3)
ε3 = nanmean(ds3["ε"][:, :, time_range], dim=3)

χ1 = nanmean(ds1["χ"][:, :, time_range], dim=3)
χ2 = nanmean(ds2["χ"][:, :, time_range], dim=3)
χ3 = nanmean(ds3["χ"][:, :, time_range], dim=3)

# mask
ε1[ε1.==0] .= NaN
ε2[ε2.==0] .= NaN
ε3[ε3.==0] .= NaN
χ1[χ1.==0] .= NaN
χ2[χ2.==0] .= NaN
χ3[χ3.==0] .= NaN
## plot epsilon and chi
# Create a 2x3 plot for ε and χ
fig, axs = subplots(2, 3, figsize=(15, 8), constrained_layout=true)

# Define the data and titles for the plots
data = [(ε1, χ1), (ε2, χ2), (ε3, χ3)]
titles = ["ε (θ=0)", "χ (θ=0)", "ε (θ=0.008)", "χ (θ=0.008)", "ε (θ=0.02)", "χ (θ=0.02)"]

# Set the color limits and colormap for all plots
ε_clim = (1e-10,1e-8)  # Adjust limits based on your data
χ_clim = (1e-16,1e-14)  # Adjust limits based on your data
cmaps = ["rainbow", "rainbow"]

# Plot the data
for i in 1:3
    for j in 1:2
        ax = axs[j, i]
        dat = data[i][j]
        clim = j == 1 ? ε_clim : χ_clim  # Use ε_clim for ε and χ_clim for χ
        p = ax.pcolormesh(xC, zC, dat', shading="auto", cmap=cmaps[j], norm=matplotlib.colors.LogNorm(vmin=clim[1], vmax=clim[2]))
        ax.set_title(titles[(i-1)*2+j], fontsize=10)
        if i == 1
            ax.set_ylabel("z (m)", fontsize=10)
        end
        if j == 2
            ax.set_xlabel("x (m)", fontsize=10)
        end
        # Add colorbars for each plot
        cbar = fig.colorbar(p, ax=ax, shrink=0.8, pad=0.02)
        # cbar.ax.set_ylabel("Value", fontsize=10)
    end
end

# Save the figure
outdir = "output/2D_idealized_tilt_mean_flow_tide_$(tide)"
outfile = joinpath(outdir, "epsilon_chi_comparison_logscale.png")
println("Saving figure to $outfile")
savefig(outfile)
close(fig)










## plot u and w (difference from θ=0 case)
# Collect u and w differences from θ=0 case
# Create a 3x2 plot for u_cross and w anomalies
fig, axs = subplots(3, 2, figsize=(12, 10), constrained_layout=true)

# Define the data and titles for the plots
data = [u2_cross .- u1_cross, w2_cen .- w1_cen,
    u3_cross .- u1_cross, w3_cen .- w1_cen,
    u4_cross .- u1_cross, w4_cen .- w1_cen]
titles = ["Δu_cross (θ=0.008 - θ=0)", "Δŵ (θ=0.008 - θ=0)",
    "Δu_cross (θ=0.02 - θ=0)", "Δŵ (θ=0.02 - θ=0)",
    "Δu_cross (θ=0.05 - θ=0)", "Δŵ (θ=0.05 - θ=0)"]

# Set the color limits and colormap for all plots
uclim = (-u_lim * 0.3, u_lim * 0.3)  # Symmetric limits for u_cross anomalies
wclim = (-w_lim * 0.3, w_lim * 0.3)  # Symmetric limits for w anomalies
cmap = "RdBu_r"

# Plot the data
for i in 1:3
    for j in 1:2
        ax = axs[i, j]
        dat = data[(i-1)*2+j]
        clim = j == 1 ? uclim : wclim  # Use uclim for u_cross and wclim for w
        p = ax.pcolormesh(xC, zC, dat', shading="auto", cmap=cmap, vmin=clim[1], vmax=clim[2])
        ax.set_title(titles[(i-1)*2+j], fontsize=10)
        if j == 1
            ax.set_ylabel("z (m)", fontsize=10)
        end
        if i == 3
            ax.set_xlabel("x (m)", fontsize=10)
        end
        # Add colorbars for the last row
        if i == 3
            cbar = fig.colorbar(p, ax=ax, location="right", shrink=0.8, pad=0.02)
            cbar.ax.set_ylabel("Anomaly (m/s)", fontsize=12)
            if j == 1
                # cbar.ax.set_title("(u)", fontsize=10)
            else
                # cbar.ax.set_title("(w)", fontsize=10)
            end
        end
        # Set NaN values to gray
        p.cmap.set_bad(color="gray")
    end
end

# Add a single shared colorbar

# Save the figure
outdir = "output/2D_idealized_tilt_mean_flow_tide_$(tide)"
outfile = joinpath(outdir, "u_cross_w_anomalies_comparison.png")
println("Saving figure to $outfile")
savefig(outfile)
close(fig)











# Collect u profiles for all tilt cases
u_profiles = [nanmean(u, dims=1) for u in (u1, u2, u3, u4)]

# Create a figure for the vertical profiles
close("all")
figure(figsize=(6, 4))
for (i, u_profile) in enumerate(u_profiles)
    plot(vec(u_profile), zC, label="$(tilts[i])")
end

xlabel("Mean û (m/s)", fontsize=12)
ylabel("z (m)", fontsize=12)
title("Vertical profiles of mean û for all tilt cases", fontsize=12)
legend(fontsize=10)
grid(true)

# Save the figure
savefig(joinpath(outdir, "_mean_u_profiles_comparison.png"))
println("Saving figure to $(joinpath(outdir, "_mean_u_profiles_comparison.png"))")

