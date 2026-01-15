using Printf
using PyPlot
using PyCall
using NCDatasets
using Statistics
using NaNStatistics


include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/mmderiv.jl")
# tide = 0
# fn1 = "output/2D_idealized_tilt_notide_0/internal_tide_theta=0_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
# fn2 = "output/2D_idealized_tilt_notide_0.002/internal_tide_theta=0.002_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
# fn3 = "output/2D_idealized_tilt_notide_0.004/internal_tide_theta=0.004_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"
# fn4 = "output/2D_idealized_tilt_notide_0.008/internal_tide_theta=0.008_Nx=1000_Nz=300_tᶠ=50.0_snapshot.nc"

θ = 0.02
fn1 = "output/2D_idealized_tilt_$(θ)/internal_tide_theta=$(θ)_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
ds1 = Dataset(fn1,"r")  
zC = ds1["z_aac"][:];
Nz = length(zC);
xC = ds1["x_caa"][:];
Nx = length(xC);
t = ds1["time"][:];
# phase average within 100TP-200TP
time_range = 2401:length(t)
b = ds1["b"][:, :, 1];
U₀ = 0.025  # Tidal velocity amplitude (m/s)
ω₀ = 1.4e-4  # Tidal frequency (rad/s)



# u_tidal_forcing(t) = U₀ * sin(ω₀ * t)
# u1 = mean(ds1["uhat"][:,:,:],dims=(1,2))
# u1[u1.==0] .= NaN

# using PyPlot
# close("all");
# figure(figsize=(16,4))
# # plot(t/(2*pi/1.4e-4), u_tidal_forcing.(t))
# plot(t/(2*pi/1.4e-4), nanmean(u1, dims=(1,2))[:], "-")
# xlim(100,120)
# xticks(100:1:120)
# savefig("output/Plots_2D_idealized_tilt_mean_flow_tide_$(tide)/tidal_forcing.png", dpi=150)
# println("output/Plots_2D_idealized_tilt_mean_flow_tide_$(tide)/tidal_forcing.png")

# compute phase-averaged fields
u_phase_avg = zeros(Nx, Nz, 8)
w_phase_avg = zeros(Nx, Nz + 1, 8)
B_phase_avg = zeros(Nx, Nz, 8)
Bz_phase_avg = zeros(Nx, Nz, 8)
eps_phase_avg = zeros(Nx, Nz, 8)
∇κ∇B_phase_avg = zeros(Nx, Nz, 8)
div_uB_phase_avg = zeros(Nx, Nz, 8)
for m in 0:7
    println("Phase average at phase = $(m)π/4:")
    for n in 1:100
        u_phase_avg[:, :, m+1] .+= ds1["uhat"][:, :, time_range[1]+3m+24(n-1)]
        w_phase_avg[:, :, m+1] .+= ds1["what"][:, :, time_range[1]+3m+24(n-1)]
        B_phase_avg[:, :, m+1] .+= ds1["B"][:, :, time_range[1]+3m+24(n-1)]
        eps_phase_avg[:, :, m+1] .+= ds1["ε"][:, :, time_range[1]+3m+24(n-1)]
        Bz_phase_avg[:, :, m+1] .+= ds1["Bz"][:, :, time_range[1]+3m+24(n-1)]
        ∇κ∇B_phase_avg[:, :, m+1] .+= ds1["∇κ∇B"][:, :, time_range[1]+3m+24(n-1)]
        div_uB_phase_avg[:, :, m+1] .+= ds1["div_uB"][:, :, time_range[1]+3m+24(n-1)]
        println("  time step: $(time_range[1]+3m+24(n-1)), n=$(n)")
    end
    u_phase_avg[:, :, m+1] ./= 100
    w_phase_avg[:, :, m+1] ./= 100
    B_phase_avg[:, :, m+1] ./= 100
    eps_phase_avg[:, :, m+1] ./= 100
    Bz_phase_avg[:, :, m+1] ./= 100
    ∇κ∇B_phase_avg[:, :, m+1] ./= 100
    div_uB_phase_avg[:, :, m+1] ./= 100
end
# release memory
GC.gc()
# compute buoyancy bugdet terms
B_phase_avg[B_phase_avg.==0] .= NaN
dB̄dx = zeros(Nx, Nz, 8)
for k in 1:size(B_phase_avg, 2)
    dB̄dx[:, k, :] = mmderiv(xC[:], B_phase_avg[:, k, :])
end
what_cen = (w_phase_avg[:, 1:end-1, :] .+ w_phase_avg[:, 2:end, :]) ./ 2 # what at center
# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end, :, :], what_cen[1:end-1, :, :]) .+ what_cen[:, :, :]) ./ 2

u_cross_phase_avg = u_phase_avg[:, :, :] * cos(θ) .+ wtemp * sin(θ) # cross-slope velocity
w_cross_phase_avg = -u_phase_avg[:, :, :] * sin(θ) .+ wtemp * cos(θ)# slope-normal velocity
u_bar_∇B_bar = u_cross_phase_avg .* dB̄dx .+ w_cross_phase_avg .* Bz_phase_avg
u_prime∇B_prime_res = div_uB_phase_avg .- u_bar_∇B_bar

u_cross_phase_avg = nothing
w_cross_phase_avg = nothing
# dB̄dx = nothing
GC.gc()
# calculate u_prime∇B_prime directly
u_prime∇B_prime_avg = zeros(Nx, Nz, 8)
u_prime∇B_bar_avg = zeros(Nx, Nz, 8)
u_bar∇B_prime_avg = zeros(Nx, Nz, 8)

Bx_prime = zeros(Nx, Nz)
for m in 0:7
    println("Phase average at phase = $(m)π/4:")
    for n in 1:100
        u_prime = ds1["uhat"][:, :, time_range[1]+3m+24(n-1)] .- u_phase_avg[:, :, m+1]
        w_prime = ds1["what"][:, :, time_range[1]+3m+24(n-1)] .- w_phase_avg[:, :, m+1]
        B_prime = ds1["B"][:, :, time_range[1]+3m+24(n-1)] .- B_phase_avg[:, :, m+1]
        Bz_prime = ds1["Bz"][:, :, time_range[1]+3m+24(n-1)] .- Bz_phase_avg[:, :, m+1]
        Bx_prime[:, :] = mmderiv(xC[:], B_prime[:, :])

        w_prime_cen = (w_prime[:, 1:end-1] .+ w_prime[:, 2:end]) ./ 2 # what at center
        # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
        wtemp = (vcat(w_prime_cen[end:end, :], w_prime_cen[1:end-1, :]) .+ w_prime_cen[:, :]) ./ 2
        u_cross_prime = u_prime * cos(θ) .+ wtemp * sin(θ) # cross-slope velocity
        w_cross_prime = -u_prime * sin(θ) .+ wtemp * cos(θ)# slope-normal velocity
        u_prime∇B_prime_avg[:, :, m+1] .+= u_cross_prime .* Bx_prime .+ w_cross_prime .* Bz_prime
        u_prime∇B_bar_avg[:, :, m+1] .+= u_cross_prime .* dB̄dx[:,:,m+1] .+ w_cross_prime .* Bz_phase_avg[:,:,m+1]
        u_bar∇B_prime_avg[:, :, m+1] .+= u_phase_avg[:, :, m+1] .* Bx_prime .+ what_cen[:, :, m+1] .* Bz_prime

        println("  time step: $(time_range[1]+3m+24(n-1))")
    end
    u_prime∇B_prime_avg[:, :, m+1] ./= 100
    u_prime∇B_bar_avg[:, :, m+1] ./= 100
    u_bar∇B_prime_avg[:, :, m+1] ./= 100
end

# (u_prime∇B_prime_avg.+u_prime∇B_bar_avg.+u_bar∇B_prime_avg.+u_bar_∇B_bar .- div_uB_phase_avg)./div_uB_phase_avg


# save phase-averaged data
output_fn = "output/2D_idealized_tilt_$(θ)/phase_avg_fields.nc"
ds_out = Dataset(output_fn, "c")
defDim(ds_out, "x", Nx)
defDim(ds_out, "z", Nz)
defDim(ds_out, "phase", 8)
defVar(ds_out, "x", xC, ("x",))
defVar(ds_out, "z", zC, ("z",))
defVar(ds_out, "phase", collect(0:7) ./ 8, ("phase",))
defVar(ds_out, "u_phase_avg", u_phase_avg, ("x", "z", "phase"))
defVar(ds_out, "w_cen_phase_avg", what_cen, ("x", "z", "phase"))
defVar(ds_out, "B_phase_avg", B_phase_avg, ("x", "z", "phase"))
defVar(ds_out, "Bz_phase_avg", Bz_phase_avg, ("x", "z", "phase"))
defVar(ds_out, "eps_phase_avg", eps_phase_avg, ("x", "z", "phase"))
defVar(ds_out, "∇κ∇B_phase_avg", ∇κ∇B_phase_avg, ("x", "z", "phase"))
defVar(ds_out, "div_uB_phase_avg", div_uB_phase_avg, ("x", "z", "phase"))
defVar(ds_out, "u_bar_∇B_bar", u_bar_∇B_bar, ("x", "z", "phase"))
defVar(ds_out, "u_prime∇B_prime_res", u_prime∇B_prime_res, ("x", "z", "phase"))
defVar(ds_out, "u_prime∇B_prime_avg", u_prime∇B_prime_avg, ("x", "z", "phase"))
close(ds_out)
println("Saved phase-averaged fields to $(output_fn)")


# load the data
θ = 0.02
output_fn = "output/2D_idealized_tilt_$(θ)/phase_avg_fields.nc"
ds_in = Dataset(output_fn, "r")
xC = ds_in["x"][:];
zC = ds_in["z"][:];
u_phase_avg = ds_in["u_phase_avg"][:, :, :];
w_cen_phase_avg = ds_in["w_cen_phase_avg"][:, :, :];
B_phase_avg = ds_in["B_phase_avg"][:, :, :];
Bz_phase_avg = ds_in["Bz_phase_avg"][:, :, :];
eps_phase_avg = ds_in["eps_phase_avg"][:, :, :];
∇κ∇B_phase_avg = ds_in["∇κ∇B_phase_avg"][:, :, :];
u_bar_∇B_bar = ds_in["u_bar_∇B_bar"][:, :, :];
u_prime∇B_prime_avg = ds_in["u_prime∇B_prime_avg"][:, :, :];
u_prime∇B_prime_res = ds_in["u_prime∇B_prime_res"][:, :, :];

u_phase_avg[u_phase_avg.==0] .= NaN
w_cen_phase_avg[w_cen_phase_avg.==0] .= NaN
B_phase_avg[repeat(b .== 0, 1, 1, size(B_phase_avg, 3))] .= NaN
Bz_phase_avg[Bz_phase_avg.==0] .= NaN
eps_phase_avg[eps_phase_avg.==0] .= NaN
∇κ∇B_phase_avg[repeat(b .== 0, 1, 1, size(∇κ∇B_phase_avg, 3))] .= NaN
u_bar_∇B_bar[repeat(b .== 0, 1, 1, size(∇κ∇B_phase_avg, 3))] .= NaN
u_prime∇B_prime_avg[repeat(b .== 0, 1, 1, size(∇κ∇B_phase_avg, 3))] .= NaN
u_prime∇B_prime_res[repeat(b .== 0, 1, 1, size(∇κ∇B_phase_avg, 3))] .= NaN

close(ds_in)
# plot the phases
# 3x8 panel: row1=u_phase_avg, row2=w_cen_phase_avg, row3=Bz_phase_avg
close("all")
fig, axs = subplots(4, 8, figsize=(24, 12), sharex=true, sharey=true)

# Phase labels
phase_labels = ["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4"]

# Helper for pcolor with vectors (xC, zC) and data (Nx x Nz)
function plot_panel(ax, x, z, field; cmap="viridis", clim=nothing)
    # PyPlot expects (Ny,Nx); Julia arrays are (Nx,Nz), so transpose
    C = transpose(field)
    pm = ax.pcolormesh(x, z, C, shading="auto", cmap=cmap)
    if clim !== nothing
        pm.set_clim(clim...)
    end
    return pm
end

# Determine color limits per row for consistent scaling
u_clim = (-0.075, 0.075)
w_clim = (-0.01, 0.01)
Bz_clim = (0, 2e-6)
eps_clim = (1e-10, 1e-8)
# Optional masking: comment out unless you specifically want to hide zeros
# u_phase_avg[u_phase_avg.==0] .= NaN
# what_cen[what_cen.==0] .= NaN
# Bz_phase_avg[Bz_phase_avg.==0] .= NaN

# Store pcolormesh objects for colorbars
pm1_list = []
pm2_list = []
pm3_list = []
pm4_list = []

for p in 1:8
    # Row 1: u_phase_avg
    pm1 = plot_panel(axs[1, p], xC, zC, u_phase_avg[:, :, p]; cmap="RdBu_r", clim=u_clim)
    axs[1, p].set_title("phase $(phase_labels[p])")
    axs[1, p].contour(xC, zC, transpose(B_phase_avg[:, :, p]), levels=30, colors="black", linewidths=0.5, alpha=0.3)
    push!(pm1_list, pm1)

    # Row 2: w_cen_phase_avg
    pm2 = plot_panel(axs[2, p], xC, zC, w_cen_phase_avg[:, :, p]; cmap="RdBu_r", clim=w_clim)
    axs[2, p].contour(xC, zC, transpose(B_phase_avg[:, :, p]), levels=30, colors="black", linewidths=0.5, alpha=0.3)
    push!(pm2_list, pm2)

    # Row 3: Bz_phase_avg
    pm3 = plot_panel(axs[3, p], xC, zC, Bz_phase_avg[:, :, p]; cmap="viridis", clim=Bz_clim)
    axs[3, p].contour(xC, zC, transpose(B_phase_avg[:, :, p]), levels=30, colors="black", linewidths=0.5, alpha=0.3)
    push!(pm3_list, pm3)

    # Row 4: eps_phase_avg
    pm4 = plot_panel(axs[4, p], xC, zC, log10.(eps_phase_avg[:, :, p]); cmap="turbo", clim=log10.(eps_clim))
    axs[4, p].contour(xC, zC, transpose(B_phase_avg[:, :, p]), levels=30, colors="black", linewidths=0.5, alpha=0.3)
    push!(pm4_list, pm4)

    # Set NaN (topography) to grey color
    for pm in [pm1, pm2, pm3, pm4]
        pm.cmap.set_bad("grey")
    end
end

# Axis labels
for r in 1:3
    axs[r, 1].set_ylabel("z (m)")
end
for p in 1:8
    axs[3, p].set_xlabel("x (m)")
end

# Shared colorbars per row using stored pcolormesh objects
cbar1 = fig.colorbar(pm1_list[1], ax=axs[1, :], location="right", fraction=0.025, pad=0.02)
cbar1.set_label("u (m s⁻¹)")
cbar2 = fig.colorbar(pm2_list[1], ax=axs[2, :], location="right", fraction=0.025, pad=0.02)
cbar2.set_label("w (m s⁻¹)")
cbar3 = fig.colorbar(pm3_list[1], ax=axs[3, :], location="right", fraction=0.025, pad=0.02)
cbar3.set_label("∂B/∂z (s⁻² m⁻¹)")
cbar4 = fig.colorbar(pm4_list[1], ax=axs[4, :], location="right", fraction=0.025, pad=0.02)
cbar4.set_label("log(ε) (m² s⁻³)")

fig.tight_layout()

plot_fn = "output/2D_idealized_tilt_$(θ)/phase_panels_$(θ).png"
savefig(plot_fn, dpi=150)
println("Saved phase panels plot to $(plot_fn)")




## plot another figure with the buoyancy budget terms
close("all")
fig, axs = subplots(4, 8, figsize=(24, 12), sharex=true, sharey=true)

# Phase labels
phase_labels = ["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4"]

# Determine symmetric color limits per row for budget terms
∇κ∇B_max = nanmaximum(abs.(∇κ∇B_phase_avg))
∇κ∇B_clim = (-1e-11, 1e-11)

u_bar_∇B_bar_max = nanmaximum(abs.(u_bar_∇B_bar))
u_bar_∇B_bar_clim = (-2e-8, 2e-8)

u_prime∇B_prime_avg_max = nanmaximum(abs.(u_prime∇B_prime_avg))
u_prime∇B_prime_avg_clim = (-4e-9, 4e-9)

u_prime∇B_prime_res_max = nanmaximum(abs.(u_prime∇B_prime_res))
u_prime∇B_prime_res_clim = (-1e-8, 1e-8)

# Store pcolormesh objects for colorbars
pmb1_list = []
pmb2_list = []
pmb3_list = []
pmb4_list = []

for p in 1:8
    # Row 1: ∇κ∇B_phase_avg (diffusion term)
    pmb1 = plot_panel(axs[1, p], xC, zC, ∇κ∇B_phase_avg[:, :, p]; cmap="RdBu_r", clim=∇κ∇B_clim)
    axs[1, p].set_title("phase $(phase_labels[p])")
    axs[1, p].contour(xC, zC, transpose(B_phase_avg[:, :, p]), levels=30, colors="black", linewidths=0.5, alpha=0.3)
    push!(pmb1_list, pmb1)

    # Row 2: u_bar_∇B_bar (mean advection)
    pmb2 = plot_panel(axs[2, p], xC, zC, -u_bar_∇B_bar[:, :, p]; cmap="RdBu_r", clim=u_bar_∇B_bar_clim)
    axs[2, p].contour(xC, zC, transpose(B_phase_avg[:, :, p]), levels=30, colors="black", linewidths=0.5, alpha=0.3)
    push!(pmb2_list, pmb2)

    # Row 3: u_prime∇B_prime_avg (turbulent flux, direct)
    pmb3 = plot_panel(axs[3, p], xC, zC, -u_prime∇B_prime_avg[:, :, p]; cmap="RdBu_r", clim=u_prime∇B_prime_avg_clim)
    axs[3, p].contour(xC, zC, transpose(B_phase_avg[:, :, p]), levels=30, colors="black", linewidths=0.5, alpha=0.3)
    push!(pmb3_list, pmb3)

    # Row 4: u_prime∇B_prime_res (turbulent flux, residual)
    pmb4 = plot_panel(axs[4, p], xC, zC, -u_prime∇B_prime_res[:, :, p]; cmap="RdBu_r", clim=u_prime∇B_prime_res_clim)
    axs[4, p].contour(xC, zC, transpose(B_phase_avg[:, :, p]), levels=30, colors="black", linewidths=0.5, alpha=0.3)
    push!(pmb4_list, pmb4)
    
    # Set NaN (topography) to grey color
    for pm in [pmb1, pmb2, pmb3, pmb4]
        pm.cmap.set_bad("grey")
    end
end

# Axis labels
for r in 1:4
    axs[r, 1].set_ylabel("z (m)")
end
for p in 1:8
    axs[4, p].set_xlabel("x (m)")
end

# Shared colorbars per row using stored pcolormesh objects
cbar1 = fig.colorbar(pmb1_list[1], ax=axs[1, :], location="right", fraction=0.025, pad=0.02)
cbar1.set_label("∇·(κ∇B) (s⁻³)")
cbar2 = fig.colorbar(pmb2_list[1], ax=axs[2, :], location="right", fraction=0.025, pad=0.02)
cbar2.set_label("-ū·∇B̄ (s⁻³)")
cbar3 = fig.colorbar(pmb3_list[1], ax=axs[3, :], location="right", fraction=0.025, pad=0.02)
cbar3.set_label("-u′·∇B′ (direct) (s⁻³)")
cbar4 = fig.colorbar(pmb4_list[1], ax=axs[4, :], location="right", fraction=0.025, pad=0.02)
cbar4.set_label("-u′·∇B′ (residual) (s⁻³)")

fig.tight_layout()

plot_fn = "output/2D_idealized_tilt_$(θ)/phase_budget_panels_$(θ).png"
savefig(plot_fn, dpi=150)
println("Saved phase budget panels plot to $(plot_fn)")