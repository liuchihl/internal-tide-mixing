# this script computes the buoyancy budget equation: 
# ∂B/∂t = -ũ⋅∇B - ∇⋅(-κ∇B)

using NCDatasets
using CairoMakie
using Statistics
using NaNStatistics
slope = "tilt"
tᶠ = 453.0
θ = 0.0036

## Decompose advection term
# <∇⋅(uB)> = <u⋅∇B> = <u>⋅∇<B> + <u'⋅∇B'>
file = string("output/", slope, "/TF_avg_tᶠ=", tᶠ, "_analysis.nc")
ds = Dataset(file, "r")
filename_3D = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_analysis_round=all_threeD.nc")
ds_3D = Dataset(filename_3D, "r")
xC = ds_3D["x_caa"][:]
zC = ds_3D["z_aac"][:]
# b = ds["b_avg"][:, :]
# what_avg = ds["what_avg"][:,:]

t = ds["t"][:]
z = ds["bin_center"][:]
ind1 = findfirst(x -> isapprox(x, 50; atol=5), z)
ind2 = findfirst(x -> isapprox(x, 100; atol=5), z)
div_uB_avg = ds["div_uB_avg"][:, :]
u_bar_∇B_bar_avg = ds["u_bar_∇B_bar_avg"][:, :]
u_prime∇B_prime_avg = ds["u_prime∇B_prime_avg"][:, :]
dBdt_avg = ds["dBdt_avg"][:, :]
∇κ∇B_avg = ds["∇κ∇B_avg"][:, :]


##
div_uB_avg_z = mean(div_uB_avg[ind1:ind2, :], dims=1)
u_bar_∇B_bar_avg_z = mean(u_bar_∇B_bar_avg[ind1:ind2, :], dims=1)
u_prime∇B_prime_avg_z = mean(u_prime∇B_prime_avg[ind1:ind2, :], dims=1)
dBdt_avg_z = mean(dBdt_avg[ind1:ind2, :], dims=1)
∇κ∇B_avg_z = mean(∇κ∇B_avg[ind1:ind2, :], dims=1)
# residual = dBdt_avg[:] - (-div_uB_avg[1,2:end] .+ ∇κ∇B_avg[1,2:end])

## Create a new figure
fig = CairoMakie.Figure(resolution=(1000, 500), fontsize=20, size=(1000, 500))

# Create an axis
ax = Axis(fig[1, 1], xlabel="Time (tidal cycle)", title="Buoyancy Budget", limits=((40, 80), nothing))

# Plot each time series
lines!(ax, t[2:end], dBdt_avg_z[:], label="<dB/dt>", color=:red, linewidth=2.5)
lines!(ax, t[2:end], u_bar_∇B_bar_avg_z[2:end], label="<ṵ>⋅<∇B>", linestyle=:dash, color=:dodgerblue, linewidth=2.5)
lines!(ax, t[2:end], div_uB_avg_z[2:end], label="<ṵ⋅∇B>", color=:dodgerblue, linewidth=2.5)
lines!(ax, t[2:end], u_prime∇B_prime_avg_z[2:end], label="<ṵ'⋅∇B'>", linestyle=:dot, color=:dodgerblue, linewidth=2.5)
lines!(ax, t[2:end], ∇κ∇B_avg_z[2:end], label="<∇⋅(κ∇B)>", color=:green, linewidth=2.5)
lines!(ax, t[2:end], -div_uB_avg_z[2:end] .+ ∇κ∇B_avg_z[2:end], label="-<ṵ⋅∇B> + <∇⋅(κ∇B)>", color=:black, linewidth=2.5)
# lines!(ax, t[2:end], residual, label = "Residual", color = :black, linewidth=2.0, linestyle=:dash)
lines!(ax, [40, 80], [0, 0], color=:black)
# Add a legend
# axislegend(ax, position = :rc, framevisible = false, ncolumns=2)
legend = Legend(fig, ax, framevisible=false, nbanks=1)
# fig[1, 1] = ax  # Ensure the axis fills the grid position
fig[1, 2] = legend  # Place the legend in the same grid position as the axis

# Display the figure
display(fig)
save("output/tilt/Buoyancy_budget_50-100m_decompose.png", fig)


## plot quantities vs hab
fig = CairoMakie.Figure(resolution=(500, 500), fontsize=20, size=(500, 500))

# Create an axis
ax = Axis(fig[1, 1], ylabel="HAB (m)", title="Buoyancy Budget", limits=((-5e-10, 5e-10), (0, 200)))
log_symm(v) = sign.(v) .* real(log10.(Complex.(v)))
# Plot each time series
lines!(ax, dropdims(mean(dBdt_avg[:, :], dims=2), dims=2), z, label="<dB/dt>", color=:red, linewidth=2.5)
lines!(ax, dropdims(mean(u_bar_∇B_bar_avg[:, :], dims=2), dims=2), z, label="<ṵ>⋅<∇B>", linestyle=:dash, color=:dodgerblue, linewidth=2.5)
lines!(ax, dropdims(mean(div_uB_avg[:, :], dims=2), dims=2), z, label="<ṵ⋅∇B>", color=:dodgerblue, linewidth=2.5)
lines!(ax, dropdims(mean(u_prime∇B_prime_avg[:, :], dims=2), dims=2), z, label="<ṵ'⋅∇B'>", linestyle=:dot, color=:dodgerblue, linewidth=2.5)
lines!(ax, dropdims(mean(∇κ∇B_avg[:, :], dims=2), dims=2), z, label="<∇⋅(κ∇B)>", color=:orange, linestyle=:dash, linewidth=2.5)
lines!(ax, dropdims(mean(-div_uB_avg[:, :] .+ ∇κ∇B_avg[:, :], dims=2), dims=2), z, label="-<ṵ⋅∇B> + <∇⋅(κ∇B)>", color=:black, linewidth=2.5)
# scatter!(ax, log_symm(dBdt_avg[:,1]), z, label="<dB/dt>", color=:red, markersize=5)
# scatter!(ax, log_symm(u_bar_∇B_bar_avg[:,1]), z, label="<ṵ>⋅<∇B>", color=:dodgerblue4, markersize=5)
# scatter!(ax, log_symm(div_uB_avg[:,1]),z, label="<ṵ⋅∇B>", color=:dodgerblue, markersize=5)
# scatter!(ax, log_symm(u_prime∇B_prime_avg[:,1]),z, label="<ṵ'⋅∇B'>", color=:blue, markersize=5)
# scatter!(ax, log_symm(∇κ∇B_avg[:,1]),z, label="<∇⋅(κ∇B)>", color=:green, markersize=5)
# scatter!(ax, log_symm(-div_uB_avg[:,1] .+ ∇κ∇B_avg_z[:,1]), z,label="-<ṵ⋅∇B> + <∇⋅(κ∇B)>", color=:black, markersize=5)# lines!(ax, t[2:end], residual, label = "Residual", color = :black, linewidth=2.0, linestyle=:dash)
# lines!(ax, t[2:end], residual, label = "Residual", color = :black, linewidth=2.0, linestyle=:dash)
# lines!(ax, [40,80], [0,0],color=:black)
# Add a legend
axislegend(ax, position=:rt, framevisible=false, nbanks=1)
# legend = Legend(fig, ax, framevisible=false, nbanks=1)
# fig[1, 1] = ax  # Ensure the axis fills the grid position
# fig[1, 2] = legend  # Place the legend in the same grid position as the axis

# Display the figure
display(fig)
save(string("output/tilt/Buoyancy_budget_hab_tᶠ=",tᶠ,"_decompose.png"),fig)



## test a few time steps and recalculate the terrain following averages
# load data

filename_field_budget = "output/tilt/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_40-80_threeD_timeavg_Bbudget.nc"
ds_budget = Dataset(filename_field_budget, "r")
filename_3D = "output/tilt/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_40-80_threeD.nc"
ds_3D = Dataset(filename_3D, "r")
θ = 0.0036
zC = ds_budget["zC"][:];
zF = ds_budget["zF"][:];
Nz = length(zC[:]);

xC = ds_budget["xC"][:];
xF = ds_budget["xF"][:];
Nx = length(xC[:]);
dx = xF[end] - xF[end-1];

yC = ds_budget["yC"][:];
yF = ds_budget["yF"][:];
Ny = length(yC[:]);
dy = yF[end] - yF[end-1];
z_face = xF * sin(θ) .+ zF' * cos(θ)     # Cartesian coordinate
t = ds_budget["time"][:];

bin_edge = 0:5:1500#0
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2

filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
bin_mask = hab

∇κ∇B_avg = zeros(length(bin_edge) - 1, length(t))
div_uB_avg = zeros(length(bin_edge) - 1, length(t))
dBdt_avg = zeros(length(bin_edge) - 1, length(t) - 1)
t_diff = diff(t)  # Precompute time differences
for n in 1:2
        # b = ds_budget["b"][:,:,:,n:n];          # buoyancy perturbation
        ∇κ∇B = ds_budget["∇κ∇B"][:, :, :, n:n]    # ∇⋅κ∇B: buoyancy flux divergence
        div_uB = ds_budget["div_uB"][:, :, :, n:n]# ∇⋅uB: buoyancy flux divergence        
        # terrain following quantities:
        @time ∇κ∇B_avg[:, n], _ = bins(∇κ∇B, bin_edge, bin_mask, dx=dx, dy=dy, z_face=z_face, normalize=true)
        @time div_uB_avg[:, n], _ = bins(div_uB, bin_edge, bin_mask, dx=dx, dy=dy, z_face=z_face, normalize=true)

        if n < length(t)
                B2 = ds_3D["B"][:, :, :, n:n+1]          # total buoyancy with two timesteps
                dBdt = (B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ t_diff[n]
                @time dBdt_avg[:, n], _ = bins(dBdt, bin_edge, bin_mask, dx=dx, dy=dy, z_face=z_face, normalize=true)
        end
end



# plot the buoyancy budget terms
# ∂B/∂t = -ũ⋅∇B - ∇⋅(-κ∇B)

using NCDatasets
using CairoMakie
using Statistics
using NaNStatistics
include("functions/bins.jl")
include("functions/mmderiv.jl")
simname = "tilt"
tᶠ = 453.0
θ = 0.0036

## Decompose advection term
# <∇⋅(uB)> = <u⋅∇B> = <u>⋅∇<B> + <u'⋅∇B'>
file = string("output/", simname, "/TF_avg_tᶠ=", tᶠ, "_analysis.nc")
ds = Dataset(file, "r")

t = ds["t"][:] / (2 * pi / 1.4e-4)
z = ds["bin_center"][:]
ind1 = findfirst(x -> isapprox(x, 50; atol=5), z)
ind2 = findfirst(x -> isapprox(x, 100; atol=5), z)
div_uB_avg = ds["div_uB_avg"][:, :]
u_bar_∇B_bar_avg = ds["u_bar_∇B_bar_avg"][:, :]
u_prime∇B_prime_avg = ds["u_prime∇B_prime_avg"][:, :]
dBdt_avg = ds["dBdt_avg"][:, :]
∇κ∇B_avg = ds["∇κ∇B_avg"][:, :]

using PyPlot
# colors = [150 148 255;136 194 115;255 41 41]./255
# Create a new figure
fig, ax = plt.subplots(figsize=(6, 4))

# Set font size
plt.rcParams["font.size"] = 25
end_time = 1
# Plot each time series
ax.plot(dropdims(mean(u_bar_∇B_bar_avg[:, 1:end_time], dims=2), dims=2), z,
        label=L"\nabla\cdot(\overline{\mathbf{u}}\,\overline{B})", linestyle="-", color=[150, 148, 255] / 255, linewidth=2.5)
ax.plot(dropdims(mean(u_prime∇B_prime_avg[:, 1:end_time], dims=2), dims=2), z,
        label=L"\nabla\cdot\overline{\mathbf{u'} B'}", linestyle="-", color=[136, 194, 115] / 255, linewidth=2.5)
ax.plot(dropdims(mean(div_uB_avg[:, 1:end_time], dims=2), dims=2), z,
        label=L"\overline{\nabla\cdot(\mathbf{u}B)}", color="orange", linewidth=2.5)
ax.plot(dropdims(mean(∇κ∇B_avg[:, 1:end_time], dims=2), dims=2), z,
        label=L"\nabla\cdot(\overline{\mathcal{B}})", color="red", linestyle="--", linewidth=3)
ax.plot(dropdims(mean(-div_uB_avg[:, 1:end_time] .+ ∇κ∇B_avg[:, 1:end_time], dims=2), dims=2), z,
        label="RHS", color="black", linewidth=2.5)
ax.plot([0, 0], [0, 500], color="black", linewidth=1.5)
ax.plot(dropdims(mean(dBdt_avg[:, 1:end_time], dims=2), dims=2), z,
        label=L"d\overline{B}/dt", color="gray", linewidth=2.5, linestyle="--")

# Set the y-limits and labels
ax.set_ylim(0, 500)
ax.set_ylabel("HAB [m]")

# Set x-axis to symlog scale
ax.set_xscale("symlog", linthresh=1 * 10^(-9.8))
ax.set_xticks([-1e-9, -1e-10,0, 1e-10, 1e-9], 
           labels=[L"-10^{-9}", L"-10^{-10}","0",L"10^{-10}", L"10^{-9}"])
        # Configure symlog minor ticks using matplotlib settings
import PyCall
matplotlib_ticker = PyCall.pyimport("matplotlib.ticker")
LogLocator = matplotlib_ticker.LogLocator
FixedLocator = matplotlib_ticker.FixedLocator

# Make sure minor ticks are visible on both sides of zero
ax.set_xlim(-2e-9, 2e-9)  # Set limits to ensure minor ticks appear


# Add legend
ax.legend(loc="right", bbox_to_anchor=(1.01, 0.86), frameon=true, ncol=2, fontsize=12,
        handlelength=1.4, columnspacing=0.9, framealpha=0.93)

# Display the figure
plt.tight_layout()
plt.show()
minorticks_on()
# Configure y-axis minor ticks
ax.tick_params(axis="y", which="minor", length=4, width=0.8)
ax.tick_params(axis="y", which="major", length=7, width=1.0)

# Create custom minor tick locations only in the logarithmic regions
minor_ticks_negative = [-2e-10, -3e-10, -4e-10, -5e-10, -6e-10, -7e-10, -8e-10, -9e-10]
minor_ticks_positive = [2e-10, 3e-10, 4e-10, 5e-10, 6e-10, 7e-10, 8e-10, 9e-10]
all_minor_ticks = vcat(minor_ticks_negative, minor_ticks_positive)

# Set custom minor tick locator
ax.xaxis.set_minor_locator(FixedLocator(all_minor_ticks))

# Configure tick appearance
ax.tick_params(axis="x", which="minor", length=4, width=0.8)
ax.tick_params(axis="x", which="major", length=7, width=1.0)

save(string("output/tilt/Buoyancy_budget_hab_tᶠ=", tᶠ, "_decompose.png"), fig)




### plot the 453-462 average profile of the buoyancy budget terms

# plot the buoyancy budget terms
# ∂B/∂t = -ũ⋅∇B - ∇⋅(-κ∇B)
using NCDatasets
using PyPlot
using Statistics
using NaNStatistics

simname = "tilt"
t_start = 453.0
t_end = 462.0
t_range = t_start:1.0:t_end

# Initialize accumulators for each term and mask count
div_uB_accum = nothing
u_bar_∇B_bar_accum = nothing
u_prime∇B_prime_accum = nothing
dBdt_accum = nothing
∇κ∇B_accum = nothing
mask_count = nothing
nfiles = 0

for tᶠ in t_range
    file = string("output/", simname, "/TF_avg_tᶠ=", tᶠ, "_analysis.nc")
    ds = Dataset(file, "r")
    # Load each region separately
    div_uB_avg = hcat(ds["div_uB_avg_sill"][:, 1], ds["div_uB_avg_rest"][:, 1], ds["div_uB_avg_flanks"][:, 1])
    u_bar_∇B_bar_avg = hcat(ds["u_bar_∇B_bar_avg_sill"][:, 1], ds["u_bar_∇B_bar_avg_rest"][:, 1], ds["u_bar_∇B_bar_avg_flanks"][:, 1])
    u_prime∇B_prime_avg = hcat(ds["u_prime∇B_prime_avg_sill"][:, 1], ds["u_prime∇B_prime_avg_rest"][:, 1], ds["u_prime∇B_prime_avg_flanks"][:, 1])
    dBdt_avg = hcat(ds["dBdt_avg_sill"][:, 1], ds["dBdt_avg_rest"][:, 1], ds["dBdt_avg_flanks"][:, 1])
    ∇κ∇B_avg = hcat(ds["∇κ∇B_avg_sill"][:, 1], ds["∇κ∇B_avg_rest"][:, 1], ds["∇κ∇B_avg_flanks"][:, 1])
    mask_sill = ds["mask_sill"][:,:]
    mask_rest = ds["mask_rest"][:,:]
    mask_flanks = ds["mask_flanks"][:,:]
    # Count valid points in each region
    n_sill = nansum(mask_sill)
    n_rest = nansum(mask_rest)
    n_flanks = nansum(mask_flanks)
    mask_total = [n_sill, n_rest, n_flanks]
    # Accumulate weighted sum and mask count for each depth
    if div_uB_accum === nothing
        ndepth = size(div_uB_avg, 1)
        div_uB_accum = zeros(ndepth)
        u_bar_∇B_bar_accum = zeros(ndepth)
        u_prime∇B_prime_accum = zeros(ndepth)
        dBdt_accum = zeros(ndepth)
        ∇κ∇B_accum = zeros(ndepth)
        mask_count = zeros(ndepth)
    end
    for r in 1:3
        div_uB_accum .+= div_uB_avg[:, r] .* mask_total[r]
        u_bar_∇B_bar_accum .+= u_bar_∇B_bar_avg[:, r] .* mask_total[r]
        u_prime∇B_prime_accum .+= u_prime∇B_prime_avg[:, r] .* mask_total[r]
        dBdt_accum .+= dBdt_avg[:, r] .* mask_total[r]
        ∇κ∇B_accum .+= ∇κ∇B_avg[:, r] .* mask_total[r]
        mask_count .+= mask_total[r]
    end
    nfiles += 1
    close(ds)
end

# Average over all files and all valid points
div_uB_mean = div_uB_accum ./ mask_count ./ nfiles
u_bar_∇B_bar_mean = u_bar_∇B_bar_accum ./ mask_count ./ nfiles
u_prime∇B_prime_mean = u_prime∇B_prime_accum ./ mask_count ./ nfiles
dBdt_mean = dBdt_accum ./ mask_count ./ nfiles
∇κ∇B_mean = ∇κ∇B_accum ./ mask_count ./ nfiles

# Load z for plotting (from last file)
ds = Dataset(string("output/", simname, "/TF_avg_tᶠ=", t_end, "_analysis.nc"), "r")
z = ds["bin_center"][:]
close(ds)

fig, ax = plt.subplots(figsize=(6, 4))
plt.rcParams["font.size"] = 25

ax.plot(-u_bar_∇B_bar_mean, z, label=L"-\nabla\cdot(\overline{\mathbf{u}}\,\overline{B})", linestyle="-", color=[150, 148, 255] / 255, linewidth=2.5)
ax.plot(-u_prime∇B_prime_mean, z, label=L"-\nabla\cdot\overline{\mathbf{u'} B'}", linestyle="-", color=[136, 194, 115] / 255, linewidth=2.5)
ax.plot(-div_uB_mean, z, label=L"-\overline{\nabla\cdot(\mathbf{u}B)}", color="orange", linewidth=2.5)
ax.plot(∇κ∇B_mean, z, label=L"-\nabla\cdot(\overline{\mathcal{B}})", color="red", linestyle="--", linewidth=3)
ax.plot(-div_uB_mean .+ ∇κ∇B_mean, z, label="RHS", color="black", linewidth=2.5)
ax.plot([0, 0], [0, 500], color="black", linewidth=1.5)
ax.plot(dBdt_mean, z, label=L"\partial\overline{B}/\partial t", color="gray", linewidth=2.5, linestyle="--")

ax.set_ylim(0, 500)
ax.set_ylabel("HAB [m]")
# ax.set_xscale("symlog", linthresh=1 * 10^(-9.8))
# ax.set_xticks([-1e-9, -1e-10,0, 1e-10, 1e-9], 
#            labels=[L"-10^{-9}", L"-10^{-10}","0",L"10^{-10}", L"10^{-9}"])
# ax.set_xlim(-2e-9, 2e-9)
ax.set_xlim(-.15e-9, .15e-9)
ax.legend(loc="right", bbox_to_anchor=(1.01, 0.86), frameon=true, ncol=2, fontsize=12,
        handlelength=1.4, columnspacing=0.9, framealpha=0.93)
plt.tight_layout()
# plt.show()

PyPlot.savefig(string("output/tilt/Buoyancy_budget_hab_tidalavg_", t_start, "-", t_end, "_decompose.png"))
println(string("Saved to output/tilt/Buoyancy_budget_hab_tidalavg_", t_start, "-", t_end, "_decompose.png"))