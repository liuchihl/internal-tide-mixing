# this script computes the height above bottom averages for each cases
# Load two consecutive half tidal periods for full cycle averaging from snapshots
# because the time average outputs are not calculated correctly, so we decided to just use 3D snapshots
using NCDatasets
using Statistics
using NaNStatistics
using RollingFunctions

# Helper function for rolling mean that preserves array size
function rollmean_samesize(x, window)
    n = length(x)
    result = similar(x)
    half = window ÷ 2
    for i in 1:n
        i_start = max(1, i - half)
        i_end = min(n, i + half)
        result[i] = mean(x[i_start:i_end])
    end
    return result
end

global total_steps
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bin_stat_over_xy.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/mmderiv.jl")
θ = 0.02
simname = "2D_idealized_tilt_$(θ)"
fn1 = "output/2D_idealized_tilt_$(θ)/internal_tide_theta=$(θ)_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
ds = Dataset(fn1, "r")


# ds4 = Dataset(fn4, "r")
zC = ds["z_aac"][:];
Nz = length(zC);
xC = ds["x_caa"][:];
Nx = length(xC);
t = ds["time"][:];
ind_begin = findfirst(t / (2 * pi / 1.4e-4) .>= 50)
# time_range = length(t)-1:length(t)
time_range = ind_begin:length(t)

div_uB = nanmean(ds["div_uB"][:, :, time_range], dim=3);
div_uB[div_uB .== 0] .= NaN
using PyPlot
# Plot a pcolor of div_uB
# figure(figsize=(10, 6))
# # pcolormesh(xC, zC, ds["div_uB"][:,:,4000]', shading="auto", cmap="RdBu_r")
# # clim(-nanmaximum(ds["div_uB"][:,:,4000])*0.1, nanmaximum(ds["div_uB"][:,:,4000])*0.1)
# pcolormesh(xC, zC, div_uB', shading="auto", cmap="RdBu_r")
# clim(-nanmaximum(div_uB)*0.1, nanmaximum(div_uB)*0.1)

# colorbar(label="div_uB (1/s)")
# xlabel("x (m)")
# ylabel("z (m)")
# # title("Snapshot at t=$(round(t[4000]/(2*pi/1.4e-4)))  of div_uB")
# title("Time-averaged div_uB from t=$(round(t[ind_begin]/(2*pi/1.4e-4))) to t=$(round(t[end]/(2*pi/1.4e-4)))")
# grid(true)
# show()
# savefig(string("output/", simname, "/div_uB_pcolor.png"), dpi=150)
# println("Saved figure to ", string("output/", simname, "/div_uB_pcolor.png"))
# time_range = 721:length(t)
b = ds["b"][:, :, 1];

# compute height above bottom grids
mask = [findfirst(b[i,:] .> 0) for i in 1:Nx]
hab = repeat(zC, 1, Nx)' .- zC[mask] 
# Initialize variables for running sums
dB̄dx = zeros(Nx, Nz, 1)
dB̄dz = zeros(Nx, Nz, 1)

B_sum = zeros(size(ds["B"][:, :, 1]))
Bz_sum = zeros(size(ds["Bz"][:, :, 1]))
uhat_sum = zeros(size(B_sum))
what_sum = zeros(size(ds["what"][:, :, 1]))
∇κ∇B_sum = zeros(size(B_sum))
div_uB_sum = zeros(size(B_sum))
ε_sum = zeros(size(B_sum))
χ_sum = zeros(size(B_sum))
# # Count total timesteps processed
B_avg = nanmean(ds["B"][:, :, time_range], dim=3)
Bz_avg = nanmean(ds["Bz"][:, :, time_range], dim=3)
uhat_avg = nanmean(ds["uhat"][:, :, time_range], dim=3)
what_avg = nanmean(ds["what"][:, :, time_range], dim=3)
what_avg_cen = (what_avg[:, 1:end-1] .+ what_avg[:, 2:end]) ./ 2
∇κ∇B_avg = nanmean(ds["∇κ∇B"][:, :, time_range], dim=3)
div_uB_avg = nanmean(ds["div_uB"][:, :, time_range], dim=3)
ε_avg = nanmean(ds["ε"][:, :, time_range], dim=3)
χ_avg = nanmean(ds["χ"][:, :, time_range], dim=3)

u_bar_∇B_bar = zeros(Nx, Nz)
u_prime∇B_prime = zeros(Nx, Nz)

# take the tidal average first, then take the long time average (this only applies to u_bar∇B_bar and u'_∇B')
global total_steps = 0
for n in 1:100
    # Read data for this timestep
    start = time_range[1] + 24(n - 1)
    final = start + 23
    B_temp = nanmean(ds["B"][:, :, start:final], dim=3)
    Bz_temp = nanmean(ds["Bz"][:, :, start:final], dim=3)
    uhat_temp = nanmean(ds["uhat"][:, :, start:final], dim=3)
    what_temp = nanmean(ds["what"][:, :, start:final], dim=3)
    div_uB_temp = nanmean(ds["div_uB"][:, :, start:final], dim=3)
    what_cen = (what_temp[:, 1:end-1] .+ what_temp[:, 2:end]) ./ 2 # what at center
    uhat_cen = (uhat_temp[:,:].+ vcat(uhat_temp[2:end, :], uhat_temp[end:end, :])) ./ 2
    u = uhat_cen * cos(θ) .+ what_cen * sin(θ) # cross-slope velocity
    w = -uhat_cen * sin(θ) .+ what_cen * cos(θ)# slope-normal velocity
    # mask B to avoid erroneous derivatives
    B_temp[b.==0] .= NaN
    dB̄dx = mmderiv(xC[:], B_temp)
    u_bar_∇B_bar_temp = u .* dB̄dx .+ w .* Bz_temp
    u_prime∇B_prime_temp = div_uB_temp .- u_bar_∇B_bar_temp

    u_bar_∇B_bar .+= u_bar_∇B_bar_temp
    u_prime∇B_prime .+= u_prime∇B_prime_temp
    global total_steps += 1
    println(total_steps)
end
u_bar_∇B_bar_avg = u_bar_∇B_bar./total_steps
u_prime∇B_prime_avg = u_prime∇B_prime./total_steps
# terrain following quantities:
bin_start = 0
bin_stop = 1200
bin_width = 20     # total window width
bin_stride = 5    # spacing between centers
bin_center = bin_start:bin_stride:bin_stop  # centers: 0, 4, 8, 12, ..., 1200

B_avg_hab = zeros(length(bin_center))
Bz_avg_hab = zeros(length(bin_center))
what_avg_hab = zeros(length(bin_center))
uhat_avg_hab = zeros(length(bin_center))
∇κ∇B_avg_hab = zeros(length(bin_center))
div_uB_avg_hab = zeros(length(bin_center))
u_bar_∇B_bar_avg_hab = zeros(length(bin_center))
u_prime∇B_prime_avg_hab = zeros(length(bin_center))
ε_avg_hab = zeros(length(bin_center))
χ_avg_hab = zeros(length(bin_center))

@time B_avg_hab = bin_stat_over_xy_overlapping(B_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time Bz_avg_hab = bin_stat_over_xy_overlapping(Bz_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time what_avg_hab = bin_stat_over_xy_overlapping(what_avg_cen, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time uhat_avg_hab = bin_stat_over_xy_overlapping(uhat_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time ∇κ∇B_avg_hab = bin_stat_over_xy_overlapping(∇κ∇B_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time div_uB_avg_hab = bin_stat_over_xy_overlapping(div_uB_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time u_bar_∇B_bar_avg_hab = bin_stat_over_xy_overlapping(u_bar_∇B_bar_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time u_prime∇B_prime_avg_hab = bin_stat_over_xy_overlapping(u_prime∇B_prime_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time ε_avg_hab = bin_stat_over_xy_overlapping(ε_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
@time χ_avg_hab = bin_stat_over_xy_overlapping(χ_avg, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")
ds = Dataset(fn1, "r")
dBdt = (ds["B"][:, :, time_range[end]] .- ds["B"][:, :, time_range[1]]) ./ (t[time_range[end]] .- t[time_range[1]])
@time dBdt_avg_hab = bin_stat_over_xy_overlapping(dBdt, bin_start, bin_stop, bin_width, bin_stride, hab; stat="mean")

# This creates a new NetCDF file 
# The mode "c" stands for creating a new file
rm(string("output/", simname, "/TF_avg_analysis_$(θ).nc"))  # remove existing file if any
ds_create = Dataset(string("output/", simname, "/TF_avg_analysis_$(θ).nc"), "c")
# Define the dimension
Nt = length(time_range)
defDim(ds_create, "z_TF", length(bin_center))
defDim(ds_create, "t", Nt)

# Define a global attribute
ds_create.attrib["title"] = "Terrain-following averages"
# Define the variables
v1 = defVar(ds_create, "B_avg", Float64, ("z_TF",))
v1[:] = B_avg_hab
v2 = defVar(ds_create, "Bz_avg", Float64, ("z_TF",))
v2[:] = Bz_avg_hab
v3 = defVar(ds_create, "uhat_avg", Float64, ("z_TF",))
v3[:] = uhat_avg_hab
v4 = defVar(ds_create, "what_avg", Float64, ("z_TF",))
v4[:] = what_avg_hab
v6 = defVar(ds_create, "dBdt_avg", Float64, ("z_TF",))
v6[:] = dBdt_avg_hab
v7 = defVar(ds_create, "∇κ∇B_avg", Float64, ("z_TF",))
v7[:] = ∇κ∇B_avg_hab
v8 = defVar(ds_create, "div_uB_avg", Float64, ("z_TF",))
v8[:] = div_uB_avg_hab
v9 = defVar(ds_create, "u_bar_∇B_bar_avg", Float64, ("z_TF",))
v9[:] = u_bar_∇B_bar_avg_hab
v10 = defVar(ds_create, "u_prime∇B_prime_avg", Float64, ("z_TF",))
v10[:] = u_prime∇B_prime_avg_hab
v11 = defVar(ds_create, "ε_avg", Float64, ("z_TF",))
v11[:] = ε_avg_hab
v12 = defVar(ds_create, "χ_avg", Float64, ("z_TF",))
v12[:] = χ_avg_hab
v13 = defVar(ds_create, "bin_center", Float64, ("z_TF",))
v13[:] = bin_center

# write attributes
v1.attrib["units"] = "m/s²"
v2.attrib["units"] = "1/s²"
v3.attrib["units"] = "m/s"
v4.attrib["units"] = "m/s"
v6.attrib["units"] = "m/s³"
v6.attrib["units"] = "WKg"
v7.attrib["units"] = "m/s³"
v8.attrib["units"] = "m/s³"
v9.attrib["units"] = "m/s³"
v10.attrib["units"] = "m/s³"
v11.attrib["units"] = "m²/s³"
v12.attrib["units"] = "m²/s³"
v13.attrib["units"] = "m"

close(ds_create)

# Plot terrain-following diagnostics below save
using PyPlot
using NCDatasets
θ = 0.02
simname = "2D_idealized_tilt_$(θ)"
ds = Dataset(string("output/", simname, "/TF_avg_analysis_$(θ).nc"), "r")
# load data
B_avg_hab = ds["B_avg"][:];
Bz_avg_hab = ds["Bz_avg"][:];
uhat_avg_hab = ds["uhat_avg"][:];
what_avg_hab = ds["what_avg"][:];
dBdt_avg_hab = ds["dBdt_avg"][:];
∇κ∇B_avg_hab = ds["∇κ∇B_avg"][:];
div_uB_avg_hab = ds["div_uB_avg"][:];
u_bar_∇B_bar_avg_hab = ds["u_bar_∇B_bar_avg"][:];
u_prime∇B_prime_avg_hab = ds["u_prime∇B_prime_avg"][:];
ε_avg_hab = ds["ε_avg"][:];
χ_avg_hab = ds["χ_avg"][:];
bin_center = ds["bin_center"][:];
# Ensure data vectors and x-axis

# Create a 2-row layout:
# Row 1: 4 columns (B_avg, Bz_avg, uhat_avg, what_avg)
# Row 2: 2 small columns (epsilon, chi) + 1 wide column with buoyancy budget terms
fig = figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

# Row 1 plots
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])
ax3 = fig.add_subplot(gs[1, 3])
ax4 = fig.add_subplot(gs[1, 4])

ax1.plot(B_avg_hab, bin_center, label="B_avg")
ax1.set_ylabel("HAB (m)")
ax1.set_xlabel("B (m/s²)")
ax1.set_title("B_avg")
ax1.grid(true)

ax2.plot(Bz_avg_hab, bin_center, label="Bz_avg", color="tab:orange")
ax2.set_ylabel("HAB (m)")
ax2.set_xlabel("Bz (1/s²)")
ax2.set_title("Bz_avg")
ax2.grid(true)

ax3.plot(uhat_avg_hab, bin_center, label="uhat_avg", color="tab:green")
ax3.set_ylabel("HAB (m)")
ax3.set_xlabel("u (m/s)")
ax3.set_title("uhat_avg")
ax3.set_xlim(-0.002, 0.002)
ax3.grid(true)

ax4.plot(what_avg_hab, bin_center, label="what_avg", color="tab:red")
ax4.set_xlim(-0.002, 0.002)
ax4.set_ylabel("HAB (m)")
ax4.set_xlabel("w (m/s)")
ax4.set_title("what_avg")
ax4.grid(true)

# Row 2: epsilon and chi on first two columns
ax5 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[2, 2])

ax5.plot(ε_avg_hab, bin_center, label="ε_avg", color="tab:purple")
ax5.set_ylabel("HAB (m)")
ax5.set_xlabel("ε (m²/s³)")
ax5.set_title("ε_avg")
ax5.set_xscale("log")
ax5.grid(true)

ax6.plot(χ_avg_hab, bin_center, label="χ_avg", color="tab:brown")
ax6.set_ylabel("HAB (m)")
ax6.set_xlabel("χ (m²/s³)")
ax6.set_title("χ_avg")
ax6.set_xscale("log")
ax6.grid(true)

# Row 2: wide budget column spanning last two columns
ax7 = fig.add_subplot(gs[2, 3])
ax7.plot(dBdt_avg_hab, bin_center, label="dBdt_avg", linewidth=2)
ax7.plot(∇κ∇B_avg_hab, bin_center, label="∇κ∇B_avg")
ax7.plot(-div_uB_avg_hab, bin_center, label="-div_uB_avg", alpha=0.4)
div_uB_smooth = rollmean_samesize(div_uB_avg_hab, 30)
ax7.plot(-div_uB_smooth, bin_center, label="-div_uB_avg (smoothed)", linewidth=2)
# ax7.plot(-u_bar_∇B_bar_avg_hab, bin_center, label="-u_bar_∇B_bar_avg")
# ax7.plot(-u_prime∇B_prime_avg_hab, bin_center, label="-u_prime∇B_prime_avg")
ax7.set_ylabel("HAB (m)")
ax7.set_xlabel("Buoyancy budget (m/s³)")
ax7.set_title("Buoyancy budget terms")
ax7.grid(true)
ax7.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

fig.tight_layout()
# Optionally save a PNG next to the NetCDF output
savefig(string("output/", simname, "/TF_avg_analysis_", θ, "_plots.png"), dpi=150)
println("Saved figure to ", string("output/", simname, "/TF_avg_analysis_", θ, "_plots.png"))





## plot all variables except the budget terms 
using PyPlot
using NCDatasets
θ = 0
simname = "2D_idealized_tilt_$(θ)"
ds = Dataset(string("output/", simname, "/TF_avg_analysis_$(θ).nc"), "r")
# load data
B_avg_hab1 = ds["B_avg"][:];
Bz_avg_hab1  = ds["Bz_avg"][:];
uhat_avg_hab1 = ds["uhat_avg"][:];
what_avg_hab1 = ds["what_avg"][:];
dBdt_avg_hab1 = ds["dBdt_avg"][:];
∇κ∇B_avg_hab1 = ds["∇κ∇B_avg"][:];
div_uB_avg_hab1 = ds["div_uB_avg"][:];
u_bar_∇B_bar_avg_hab1 = ds["u_bar_∇B_bar_avg"][:];
u_prime∇B_prime_avg_hab1 = ds["u_prime∇B_prime_avg"][:];
ε_avg_hab1 = ds["ε_avg"][:];
χ_avg_hab1 = ds["χ_avg"][:];
bin_center = ds["bin_center"][:];

θ = 0.008
simname = "2D_idealized_tilt_$(θ)"
ds = Dataset(string("output/", simname, "/TF_avg_analysis_$(θ).nc"), "r")
# load data
B_avg_hab2 = ds["B_avg"][:];
Bz_avg_hab2  = ds["Bz_avg"][:];
uhat_avg_hab2 = ds["uhat_avg"][:];
what_avg_hab2 = ds["what_avg"][:];
dBdt_avg_hab2 = ds["dBdt_avg"][:];
∇κ∇B_avg_hab2 = ds["∇κ∇B_avg"][:];
div_uB_avg_hab2 = ds["div_uB_avg"][:];
u_bar_∇B_bar_avg_hab2 = ds["u_bar_∇B_bar_avg"][:];
u_prime∇B_prime_avg_hab2 = ds["u_prime∇B_prime_avg"][:];
ε_avg_hab2 = ds["ε_avg"][:];
χ_avg_hab2 = ds["χ_avg"][:];
bin_center = ds["bin_center"][:];

θ = 0.02
simname = "2D_idealized_tilt_$(θ)"
ds = Dataset(string("output/", simname, "/TF_avg_analysis_$(θ).nc"), "r")
# load data
B_avg_hab3 = ds["B_avg"][:];
Bz_avg_hab3  = ds["Bz_avg"][:];
uhat_avg_hab3 = ds["uhat_avg"][:];
what_avg_hab3 = ds["what_avg"][:];
dBdt_avg_hab3 = ds["dBdt_avg"][:];
∇κ∇B_avg_hab3 = ds["∇κ∇B_avg"][:];
div_uB_avg_hab3 = ds["div_uB_avg"][:];
u_bar_∇B_bar_avg_hab3 = ds["u_bar_∇B_bar_avg"][:];
u_prime∇B_prime_avg_hab3 = ds["u_prime∇B_prime_avg"][:];
ε_avg_hab3 = ds["ε_avg"][:];
χ_avg_hab3 = ds["χ_avg"][:];
bin_center = ds["bin_center"][:];


# Ensure data vectors and x-axis

# Create a 2-row layout:
# Row 1: 4 columns (B_avg, Bz_avg, uhat_avg, what_avg)
# Row 2: 2 small columns (epsilon, chi) + 1 wide column with buoyancy budget terms
fig = figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 3)

# Row 1 plots
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])
ax3 = fig.add_subplot(gs[1, 3])

ax1.plot(B_avg_hab1, bin_center, label="θ=0")
ax1.plot(B_avg_hab2, bin_center, label="θ=0.008")
ax1.plot(B_avg_hab3, bin_center, label="θ=0.02")
ax1.set_ylabel("HAB (m)")
ax1.set_xlabel("B (m/s²)")
# ax1.set_title("B_avg")
ax1.legend()
ax1.grid(true)

ax2.plot(Bz_avg_hab1, bin_center, label="θ=0")
ax2.plot(Bz_avg_hab2, bin_center, label="θ=0.008")
ax2.plot(Bz_avg_hab3, bin_center, label="θ=0.02")
ax2.set_ylabel("HAB (m)")
ax2.set_xlabel("Bz (1/s²)")
# ax2.set_title("Bz_avg")
ax2.legend()
ax2.grid(true)

ax3.plot(uhat_avg_hab1, bin_center, label="θ=0")
ax3.plot(uhat_avg_hab2, bin_center, label="θ=0.008")
ax3.plot(uhat_avg_hab3, bin_center, label="θ=0.02")
ax3.set_ylabel("HAB (m)")
ax3.set_xlabel("û (m/s)")
# ax3.set_title("uhat_avg")
ax3.set_xlim(-0.002, 0.002)
ax3.legend()
ax3.grid(true)

# Row 2 plots
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[2, 2])
ax6 = fig.add_subplot(gs[2, 3])

ax4.plot(what_avg_hab1, bin_center, label="θ=0")
ax4.plot(what_avg_hab2, bin_center, label="θ=0.008")
ax4.plot(what_avg_hab3, bin_center, label="θ=0.02")
ax4.set_xlim(-0.002, 0.002)
ax4.set_ylabel("HAB (m)")
ax4.set_xlabel("ŵ (m/s)")
# ax4.set_title("what_avg")
ax4.legend()
ax4.grid(true)

ax5.plot(ε_avg_hab1, bin_center, label="θ=0")
ax5.plot(ε_avg_hab2, bin_center, label="θ=0.008")
ax5.plot(ε_avg_hab3, bin_center, label="θ=0.02")
ax5.set_ylabel("HAB (m)")
ax5.set_xlabel("ε (m²/s³)")
# ax5.set_title("ε_avg")
ax5.set_xscale("log")
ax5.legend()
ax5.grid(true)

ax6.plot(χ_avg_hab1, bin_center, label="θ=0")
ax6.plot(χ_avg_hab2, bin_center, label="θ=0.008")
ax6.plot(χ_avg_hab3, bin_center, label="θ=0.02")
ax6.set_ylabel("HAB (m)")
ax6.set_xlabel("χ (m²/s³)")
# ax6.set_title("χ_avg")
ax6.set_xscale("log")
ax6.legend()
ax6.grid(true)

# Adjust tick parameters to avoid overlapping
for ax in [ax3, ax4]
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelsize=10)
end

fig.tight_layout()
# Optionally save a PNG next to the NetCDF output
savefig(string("output/", simname, "/TF_avg_analysis_comparison_plots.png"), dpi=150)
println("Saved figure to ", string("output/", simname, "/TF_avg_analysis_comparison_plots.png"))





# Plot the buoyancy budget terms in 3 panels (1x3 layout)
fig = figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 3)

# Panel 1: θ = 0
ax1 = fig.add_subplot(gs[1])
ax1.plot(dBdt_avg_hab1, bin_center, label="dBdt_avg", linewidth=2, color="black")
ax1.plot(∇κ∇B_avg_hab1, bin_center, label="∇κ∇B_avg", color="red")
ax1.plot(-div_uB_avg_hab1, bin_center, label="-div_uB_avg", alpha=0.4, color="green")
div_uB_smooth1 = rollmean_samesize(div_uB_avg_hab1, 30)
ax1.plot(-div_uB_smooth1, bin_center, label="-div_uB_avg (smoothed)", linewidth=2, color="green")
ax1.set_ylabel("HAB (m)")
ax1.set_xlabel("Buoyancy budget (m/s³)")
ax1.set_title("Buoyancy budget (θ=0)")
ax1.grid(true)
ax1.legend(fontsize=8)

# Panel 2: θ = 0.008
ax2 = fig.add_subplot(gs[2])
ax2.plot(dBdt_avg_hab2, bin_center, label="dBdt_avg", linewidth=2, color="black")
ax2.plot(∇κ∇B_avg_hab2, bin_center, label="∇κ∇B_avg", color="red")
ax2.plot(-div_uB_avg_hab2, bin_center, label="-div_uB_avg", alpha=0.4, color="green")
div_uB_smooth2 = rollmean_samesize(div_uB_avg_hab2, 30)
ax2.plot(-div_uB_smooth2, bin_center, label="-div_uB_avg (smoothed)", linewidth=2, color="green")
ax2.set_ylabel("HAB (m)")
ax2.set_xlabel("Buoyancy budget (m/s³)")
ax2.set_title("Buoyancy budget (θ=0.008)")
ax2.grid(true)
ax2.legend(fontsize=8)

# Panel 3: θ = 0.02
ax3 = fig.add_subplot(gs[3])
ax3.plot(dBdt_avg_hab3, bin_center, label="dBdt_avg", linewidth=2, color="black")
ax3.plot(∇κ∇B_avg_hab3, bin_center, label="∇κ∇B_avg", color="red")
ax3.plot(-div_uB_avg_hab3, bin_center, label="-div_uB_avg", alpha=0.4, color="green")
div_uB_smooth3 = rollmean_samesize(div_uB_avg_hab3, 30)
ax3.plot(-div_uB_smooth3, bin_center, label="-div_uB_avg (smoothed)", linewidth=2, color="green")
ax3.set_ylabel("HAB (m)")
ax3.set_xlabel("Buoyancy budget (m/s³)")
ax3.set_title("Buoyancy budget (θ=0.02)")
ax3.grid(true)
ax3.legend(fontsize=8)

fig.tight_layout()
# Save the figure
savefig(string("output/", simname, "/TF_avg_analysis_buoyancy_budget_comparison.png"), dpi=150)
println("Saved figure to ", string("output/", simname, "/TF_avg_analysis_buoyancy_budget_comparison.png"))
