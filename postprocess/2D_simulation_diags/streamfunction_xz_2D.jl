using NCDatasets
using NaNStatistics
using MAT
using Interpolations

fn1 = "output/2D_idealized_tilt_0/internal_tide_theta=0_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
fn2 = "output/2D_idealized_tilt_0.008/internal_tide_theta=0.008_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
fn3 = "output/2D_idealized_tilt_0.02/internal_tide_theta=0.02_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
# fn4 = "output/2D_idealized_tilt_0.05/internal_tide_theta=0.05_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"

ds1 = Dataset(fn1, "r")
ds2 = Dataset(fn2, "r")
ds3 = Dataset(fn3, "r")

t = ds1["time"][:];
ind_begin = findfirst(t / (2 * pi / 1.4e-4) .>= 100)
# time_range = length(t)-1:length(t)
time_range = ind_begin:length(t)
# time_range = 721:length(t)
b1 = ds1["b"][:, :, 1];
b2 = ds2["b"][:, :, 1];
b3 = ds3["b"][:, :, 1];
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

zC = ds1["z_aac"][:];
Nz = length(zC);
zF = ds1["z_aaf"][:];
xC = ds1["x_caa"][:];
xF = ds1["x_faa"][:];
Nx = length(xC);
dz = abs.(zF[1:end-1] - zF[2:end]);
dx = xF[end] - xF[end-1];
what_cen1 = (w1[:, 1:end-1] .+ w1[:, 2:end]) ./ 2 # what at center
what_cen2 = (w2[:, 1:end-1] .+ w2[:, 2:end]) ./ 2
what_cen3 = (w3[:, 1:end-1] .+ w3[:, 2:end]) ./ 2

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp1 = (vcat(what_cen1[end:end, :], what_cen1[1:end-1, :]) .+ what_cen1[:, :]) ./ 2
wtemp2 = (vcat(what_cen2[end:end, :], what_cen2[1:end-1, :]) .+ what_cen2[:, :]) ./ 2
wtemp3 = (vcat(what_cen3[end:end, :], what_cen3[1:end-1, :]) .+ what_cen3[:, :]) ./ 2
θ = 0
U1 = (u1[:, :] * cos(θ) .+ wtemp1 * sin(θ)) # cross-slope velocity transport
W1 = (-u1[:, :] * sin(θ) .+ wtemp1 * cos(θ))# slope-normal velocity transport
θ = 0.008
U2 = (u2[:, :] * cos(θ) .+ wtemp2 * sin(θ)) # cross-slope velocity transport
W2 = (-u2[:, :] * sin(θ) .+ wtemp2 * cos(θ))# slope-normal velocity transport
θ = 0.02
U3 = (u3[:, :] * cos(θ) .+ wtemp3 * sin(θ)) # cross-slope velocity transport
W3 = (-u3[:, :] * sin(θ) .+ wtemp3 * cos(θ))# slope-normal velocity transport

ψ1 = zeros(Nx, Nz)
ψ2 = zeros(Nx, Nz)
ψ3 = zeros(Nx, Nz)
for i in 1:Nx
    for j in 2:Nz
        ψ1[i, j] = ψ1[i, j-1] .- U1[i, j-1] * dz[j]
        ψ2[i, j] = ψ2[i, j-1] .- U2[i, j-1] * dz[j]
        ψ3[i, j] = ψ3[i, j-1] .- U3[i, j-1] * dz[j]
    end
end
ψ1[U1.==0] .= NaN
U1[U1.==0] .= NaN
W1[W1.==0] .= NaN
ψ2[U2.==0] .= NaN
U2[U2.==0] .= NaN
W2[W2.==0] .= NaN
ψ3[U3.==0] .= NaN
U3[U3.==0] .= NaN
W3[W3.==0] .= NaN
B1[b1.==0] .= NaN
B2[b2.==0] .= NaN
B3[b3.==0] .= NaN
## plot
using PyPlot

close("all")
PyPlot.rc("font", size=18)  # Set default font size for all text elements

fig, axs = subplots(1, 3, figsize=(20, 6), sharex=true, constrained_layout=true)

ψ_list = [ψ1, ψ2, ψ3]
U_list = [U1, U2, U3]
W_list = [W1, W2, W3]
B_list = [B1, B2, B3]
titles = ["θ = 0", "θ = 0.008", "θ = 0.02"]

PyPlot.rc("font", size=18)  # Set default font size for all text elements

# Use GridSpec for equal-width panels and a dedicated colorbar axis
using PyCall
gridspec = pyimport("matplotlib.gridspec")
fig = figure(figsize=(20, 6))
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)
axs = [fig.add_subplot(gs[i]) for i in 1:3]
pcms = Vector{Any}(undef, 3)
for i in 1:3
    ax = axs[i]
    ψ = ψ_list[i]
    U = U_list[i]
    W = W_list[i]
    B = B_list[i]

    pcm = ax.pcolor(xC[:], zC[:], ψ', cmap="coolwarm")
    pcm.set_clim(-4, 4)
    pcms[i] = pcm
    ax.contour(xC[:], zC[:], ψ', levels=20, colors=[63 43 43] ./ 255, linewidths=0.8, linestyles="-")
    ax.contour(xC[:], zC[:], B', levels=30, colors="green", linewidths=0.8, linestyles="-")
    ax.quiver(xC[1:20:end], zC[1:8:end], U[1:20:end, 1:8:end]', W[1:20:end, 1:8:end]',
        angles="xy", scale_units="xy", scale=3e-5)
    ax.set_facecolor("gray")
    ax.set_xlabel("x [km]")
    if i == 1
        ax.set_ylabel("z [m]")
    else 
        ax.set_ylabel("")
        ax.set_yticklabels([])
    end
    
    ax.set_title(titles[i])
    ax.set_xticks(0:2000:14000)
    ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "12", "14"], fontsize=16)
end

# Add colorbar to the last axis (gs[3])
cax = fig.add_subplot(gs[4])
cb = fig.colorbar(pcms[1], cax=cax)
cb.set_label("ψ [m³ s⁻¹]")

savefig("output/Plots_2D_idealized_tilt_mean_flow_tide_1/streamfunction_comparison_with_colorbar.png", dpi=100)