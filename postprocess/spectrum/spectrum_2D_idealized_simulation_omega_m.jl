using NCDatasets
using PyPlot
using Statistics
using Interpolations
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/fft_psd_2d.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/fft_psd.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/fft_cpsd_2d.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bin_z_dimension.jl")


# tf = 451.5:0.5:452.0
fname = "output/2D_idealized_tilt_0/internal_tide_theta=0_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
ds = Dataset(fname, "r")
zC = ds["z_aac"][:]
zF = ds["z_aaf"][:]
xC = ds["x_caa"][:]

left = 1
right = argmin(abs.(xC .- 5000.0))
up = argmin(abs.(zC .- 2300.0))
low = argmin(abs.(zC .- 600.0))
low_f = argmin(abs.(zF .- 600.0))
up_f = argmin(abs.(zF .- 2300.0))

t1 = argmin(abs.(ds["time"][:]/(2*pi/1.4e-4) .- 160))
t2 = argmin(abs.(ds["time"][:]/(2*pi/1.4e-4) .- 200))

B = ds["B"][left:2:right, low:up, t1:t2]
Bz = ds["Bz"][left:2:right, low:up, t1:t2]
uhat = ds["uhat"][left:2:right, low:up, t1:t2]
what = ds["what"][left:2:right, low_f-1:up_f+1, t1:t2]
# eps = ds["ε"][left:right, low:up, t1:t2]
time = ds["time"][t1:t2]


# interpolate to uniform vertical grid
Nz_new = up-low+1
zC_new = range(zC[low], stop=zC[up], length=Nz_new)
B_interp = Array{Float64}(undef, size(B, 1), Nz_new, size(B, 3))
Bz_interp = Array{Float64}(undef, size(Bz, 1), Nz_new, size(Bz, 3))
uhat_interp = Array{Float64}(undef, size(uhat, 1), Nz_new, size(uhat, 3))
what_interp = Array{Float64}(undef, size(what, 1), Nz_new, size(what, 3))
for i in axes(B, 1)
    for j in axes(B, 3)
        B_interp[i, :, j] = Interpolations.interpolate((zC[low:up],), B[i, :, j], Gridded(Linear())).(zC_new)
        Bz_interp[i, :, j] = Interpolations.interpolate((zC[low:up],), Bz[i, :, j], Gridded(Linear())).(zC_new)
        uhat_interp[i, :, j] = Interpolations.interpolate((zC[low:up],), uhat[i, :, j], Gridded(Linear())).(zC_new)
        what_interp[i, :, j] = Interpolations.interpolate((zF[low_f-1:up_f+1],), what[i, :, j], Gridded(Linear())).(zC_new)
    end
end
# free memory
B=nothing
Bz=nothing
uhat=nothing
what=nothing
GC.gc()

# time derivative of each
B_interp_dt = diff(B_interp, dims=3) ./ reshape(diff(time), 1, 1, :)
uhat_interp_dt = diff(uhat_interp, dims=3) ./ reshape(diff(time), 1, 1, :)
what_interp_dt = diff(what_interp, dims=3) ./ reshape(diff(time), 1, 1, :)

# move the time derivative quantities to the center
B_interp_dt = (cat(B_interp_dt, zeros(size(B_interp, 1), size(B_interp, 2), 1), dims=3) .+
               cat(zeros(size(B_interp, 1), size(B_interp, 2), 1), B_interp_dt, dims=3)) / 2
uhat_interp_dt = (cat(uhat_interp_dt, zeros(size(uhat_interp, 1), size(uhat_interp, 2), 1), dims=3) .+
                  cat(zeros(size(uhat_interp, 1), size(uhat_interp, 2), 1), uhat_interp_dt, dims=3)) / 2
what_interp_dt = (cat(what_interp_dt, zeros(size(what_interp, 1), size(what_interp, 2), 1), dims=3) .+
                  cat(zeros(size(what_interp, 1), size(what_interp, 2), 1), what_interp_dt, dims=3)) / 2


# only take the layers that are stably stratified
Bz_interp[Bz_interp.<0] .= NaN  # mask out the unstable regions
Bz_avg_x = nanmean(Bz_interp[:, :, :], dims=1)

N = sqrt.(Bz_avg_x)




dt = mean(diff(time))
dz = mean(diff(zC_new))

_, _, _, kj_cp, fj_cp = fft_cpsd_2d(uhat_interp[1, :, :], uhat_interp_dt[1, :, :]; dx=dz, dt=dt, win="hann")

global E_uu_t = zeros(length(kj_cp), length(fj_cp))
# E_vv_t = zeros(length(kj_cp), length(fj_cp))
global E_ww_t = zeros(length(kj_cp), length(fj_cp))
global E_BB_t = zeros(length(kj_cp), length(fj_cp))
global E_BB = zeros(length(kj_cp), length(fj_cp))
global E_uu = zeros(length(kj_cp), length(fj_cp))
# E_vv = zeros(length(kj_cp), length(fj_cp))
E_ww = zeros(length(kj_cp), length(fj_cp))
# compute ∂ₜE(m,ω) using cross-spectral density
for i in 1:size(uhat_interp, 1)
    E_BB_temp, _, E_BB_t_temp, _, _ = fft_cpsd_2d(B_interp[i, :, :], B_interp_dt[i, :, :] ./ N[1, :, :]; dx=dz, dt=dt, win="hann")
    E_uu_temp, _, E_uu_t_temp, _, _ = fft_cpsd_2d(uhat_interp[i, :, :], uhat_interp_dt[i, :, :]; dx=dz, dt=dt, win="hann")
    # E_vv_temp, _, E_vv_t_temp, _, _ = fft_cpsd_2d(v_interp[i, :, :], v_interp_dt[i, :, :]; dx=dz, dt=dt, win="hann")
    E_ww_temp, _, E_ww_t_temp, _, _ = fft_cpsd_2d(what_interp[i, :, :], what_interp_dt[i, :, :]; dx=dz, dt=dt, win="hann")

    if i == 1
        global E_uu_t = E_uu_t_temp
        # E_vv_t = E_vv_t_temp
        global E_ww_t = E_ww_t_temp
        global E_BB_t = E_BB_t_temp
        global E_BB = E_BB_temp
        global E_uu = E_uu_temp
        # E_vv = E_vv_temp
        global E_ww = E_ww_temp
    else
        global E_uu_t .+= E_uu_t_temp
        # E_vv_t .+= E_vv_t_temp
        global E_ww_t .+= E_ww_t_temp
        global E_BB_t .+= E_BB_t_temp
        global E_BB .+= E_BB_temp
        global E_uu .+= E_uu_temp
        # E_vv .+= E_vv_temp
        global E_ww .+= E_ww_temp
    end
    println("Processed slice $i / $(size(uhat_interp, 1))")
end
# horizontal average the spectra
E_uu_t ./= (size(uhat_interp, 1))
# E_vv_t ./= (size(uhat_interp, 1))
E_ww_t ./= (size(uhat_interp, 1))
E_BB_t ./= (size(uhat_interp, 1))
E_uu ./= (size(uhat_interp, 1))
# E_vv ./= (size(uhat_interp, 1))
E_ww ./= (size(uhat_interp, 1))
E_BB ./= (size(uhat_interp, 1))
KE_t = real(E_uu_t .+ E_ww_t)
PE_t = real(E_BB_t)
E_t = KE_t .+ PE_t  # this has the dimension of [ω, m]

KE = real(E_uu .+ E_ww)
PE = real(E_BB)
E = KE .+ PE  # this has the dimension of [ω, m]

# release memory
B_interp=nothing
B_interp_dt=nothing
uhat_interp=nothing
uhat_interp_dt=nothing
what_interp=nothing
what_interp_dt=nothing
GC.gc()

f = 5.3e-5
ω₀ = 1.4e-4
N = 1e-3
# convert to radian
kj = 2 * pi .* kj_cp
fj = 2 * pi .* fj_cp



# save E and dEdt and kj and fj as a netcdf file
outfname = "output/2D_idealized_tilt_0/E_m_omega.nc"
ds_out = Dataset(outfname, "c")
defDim(ds_out, "m", length(kj))
defDim(ds_out, "omega", length(fj))
defVar(ds_out, "E", Float64, ("m", "omega"))
defVar(ds_out, "E_t", Float64, ("m", "omega"))

defVar(ds_out, "m", Float64, ("m",))
defVar(ds_out, "omega", Float64, ("omega",))


# Assign data to the variables
ds_out["E"][:] = E
ds_out["E_t"][:] = E_t
ds_out["m"][:] = kj
ds_out["omega"][:] = fj

println("Saving joint spectrum to $outfname")
close(ds_out)



# load data
using NCDatasets
fname = "output/2D_idealized_tilt_0/E_m_omega.nc"
ds = Dataset(fname, "r")
E = ds["E"][:, :]
# E_t = ds["E_t"][:, :]
kj = ds["m"][:]
fj = ds["omega"][:]
close(ds)
# plot both the fj and kj spectra with power-law fitting
using PyPlot
using Printf

# Generate synthetic data for demonstration
f = 5.3e-5
ω₀ = 1.4e-4
N = 1e-3

kj=kj[2:end]
fj=fj[2:end]
E_fit_m_2 = 1e-1*kj[argmin(abs.(kj .- 0.01)):argmin(abs.(kj .- 0.06))] .^ -2
E_fit_m_35 = 1e-3*kj[argmin(abs.(kj .- 0.05)):argmin(abs.(kj .- 0.2))] .^ -3.5
E_fit_f = 10^(-9.5)*fj[argmin(abs.(fj .- 4f)):argmin(abs.(fj .- 13f))] .^ -2

# Create figure with two subplots
fig, axs = subplots(2, 1, figsize=(10, 8), constrained_layout=true)

# Top plot: Energy spectrum vs vertical wavenumber
ax1 = axs[1]
ax1.loglog(kj, sum(E[2:end,:], dims=2) * diff(kj)[1], label="original", color="black")
ax1.loglog(kj[argmin(abs.(kj .- 0.01)):argmin(abs.(kj .- 0.06))], E_fit_m_2, linestyle="--", color="magenta", label="power-law fitting")
ax1.loglog(kj[argmin(abs.(kj .- 0.05)):argmin(abs.(kj .- 0.2))], E_fit_m_35, linestyle="--", color="blue")
ax1.set_xlabel("Vertical wavenumber m [m⁻¹]")
ax1.set_ylabel("E(m) [m² s⁻²]")
ax1.set_title("Total energy spectra of BBTRE")
ax1.legend()
ax1.grid(true, which="both", linestyle="--", alpha=0.5)
ax1.text(0.35, 0.8, "E(m) ∝ m⁻²", transform=ax1.transAxes, fontsize=14, color="magenta")
ax1.text(0.7, 0.7, L"E(m) ∝ $m^{-3.5}$", transform=ax1.transAxes, fontsize=14, color="blue")

# Bottom plot: Energy spectrum vs frequency
ax2 = axs[2]
ax2.loglog(fj/f, sum(E[2:end,2:end], dims=1)' * diff(fj)[1], label="original", color="black")
ax2.loglog(fj[argmin(abs.(fj .- 4f)):argmin(abs.(fj .- 13f))]./f, E_fit_f, linestyle="--", color="magenta", label="power-law fitting")
ax2.set_xlabel("ω / f")
ax2.set_ylabel("E(ω) [m² s⁻²]")
ax2.legend()
ax2.grid(true, which="both", linestyle="--", alpha=0.5)
ax2.text(0.85, 0.4, "E(ω) ∝ ω⁻²", transform=ax2.transAxes, fontsize=14, color="magenta")

# Add vertical lines for f, M2, and N
ax2.axvline(f/f, color="blue", linestyle="-", label="f")
ax2.axvline(ω₀/f, color="blue", linestyle="--", label="M2")
ax2.axvline(N/f, color="blue", linestyle=":", label="N")
ax2.legend()

# Save the figure
PyPlot.savefig("output/2D_idealized_tilt_0/spectrum_plot.png", dpi=300)
println("Spectrum plot saved as spectrum_plot.png")



# plot joint spectrum of E * kj * fj and dE/dt * kj * fj

using NCDatasets
fname = "output/2D_idealized_tilt_0/E_m_omega.nc"
ds = Dataset(fname, "r")
E = ds["E"][:, :]
E_t = ds["E_t"][:, :]
kj = ds["m"][:]
fj = ds["omega"][:]

using PyCall
using NaNStatistics
mpl = pyimport("matplotlib")

# Define levels clipped to your desired color range
data_min = nanminimum(kj' .* fj .* E_t' * 1e9)
data_max = nanmaximum(kj' .* fj .* E_t' * 1e9)
vmin = -1
vmax = 1
levels = range(nanmax(data_min, vmin), nanmin(data_max, vmax), length=51)

# plot this joint spectrum close("all")
fig, axs = subplots(1, 2, figsize=(16, 5), constrained_layout=true)

# Panel 1: E * kj * fj
data1 = kj[2:end]' .* fj[2:end] .* E[2:end,2:end]'  # [m²/s³]
vmin1 = -5
vmax1 = -2
levels1 = range(-9.5, -2, length=51)
c1 = axs[1].contourf(kj[2:end], fj[2:end] / f, log10.(data1), cmap="RdBu_r", levels=levels1, vmin=vmin1, vmax=vmax1, extend="both")
cbar1 = fig.colorbar(c1, ax=axs[1], label=L"$log mωE$ [m²/s³]")
cbar1.ax.tick_params(labelsize=14)
cbar1.ax.yaxis.label.set_size(16)
axs[1].set_xlabel(L"$m$ [1/m]", fontsize=16)
axs[1].set_ylabel(L"$ω/f$", fontsize=16)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].tick_params(axis="x", labelsize=14)
axs[1].tick_params(axis="y", labelsize=14)
axs[1].set_xlim(kj[2], maximum(kj))
axs[1].set_ylim(fj[2] / f, maximum(fj) / f)
axs[1].grid(true, which="both", linestyle="--", alpha=0.5)
# axs[1].set_title("E ⋅ mω")

# Panel 2: dE/dt * kj * fj (your existing code)
data2 = kj' .* fj .* E_t' * 1e9
vmin2 = -1.5
vmax2 = 1.5
levels2 = range(vmin2, vmax2, length=51)
c2 = axs[2].contourf(kj, fj / f, data2, cmap="RdBu", levels=levels2, vmin=vmin2, vmax=vmax2, extend="max")
cbar2 = fig.colorbar(c2, ax=axs[2], label=L"$mω(\partial E/\partial t)$ [m²/s³] ⋅ 10⁹")
cbar2.ax.tick_params(labelsize=14)
cbar2.ax.yaxis.label.set_size(16)
axs[2].set_xlabel(L"$m$ [1/m]", fontsize=16)
axs[2].set_ylabel(L"$ω/f$", fontsize=16)
axs[2].set_xscale("log")
axs[2].set_yscale("log")
axs[2].tick_params(axis="x", labelsize=14)
axs[2].tick_params(axis="y", labelsize=14)
axs[2].set_xlim(kj[2], maximum(kj))
axs[2].set_ylim(fj[2] / f, maximum(fj) / f)
axs[2].grid(true, which="both", linestyle="--", alpha=0.5)
# axs[2].set_title("∂ₜE ⋅ mω")

savefig("output/2D_idealized_tilt_0/joint_spectrum_E_and_dEt_m_f.png", dpi=300)




## check the distribution of epsilon
# left = 1
# right = argmin(abs.(xC .- 5000.0))
# low = argmin(abs.(zC .- 600.0))
# up = argmin(abs.(zC .- 1750.0))
# # eps_range = eps_all[:, :, :]
# eps_range = eps_all[left:right, low:up, :]
# eps_range = eps_range[eps_range.>1e-10]  # only keep positive values
# # Flatten the eps_range array and remove NaN values
# eps_values = filter(!isnan, vec(eps_range))
# bins = 10 .^ range(-10, log10(maximum(eps_values)), length=100)

# # Plot the PDF in log scale
# close("all")
# fig, ax = subplots(figsize=(8, 5))
# ax.hist(eps_values, bins=bins, density=true, color="blue", alpha=0.7, edgecolor="black")
# ax.set_xscale("log")
# # ax.set_yscale("log")
# ax.set_xlabel(L"$\epsilon$ [m²/s³]", fontsize=16)
# ax.set_ylabel("Probability Density", fontsize=16)
# ax.tick_params(axis="x", labelsize=14)
# ax.tick_params(axis="y", labelsize=14)
# ax.grid(true, which="both", linestyle="--", alpha=0.5)
# # ax.set_title(L"PDF of $\epsilon$", fontsize=18)
# savefig("output/tilt/pdf_eps_log.png", dpi=300)



# using PyCall
# mpl = pyimport("matplotlib")
# np = pyimport("numpy")

# data = kj' .* fj .* E_t'

# absmax = maximum(abs.(data))
# vmin = -absmax
# vmax = absmax
# linthresh = absmax / 1000  # adjust as needed

# # Number of levels in each region
# n_log = 20
# n_lin = 11

# # Log-spaced levels for positive and negative
# log_pos = exp.(range(log(linthresh), log(vmax), length=n_log))
# log_neg = -reverse(log_pos)

# # Linear levels around zero
# lin_levels = range(-linthresh, linthresh, length=n_lin)

# # Combine all levels and ensure uniqueness/sorting
# levels = vcat(log_neg, lin_levels, log_pos)
# levels = unique(sort(levels))

# norm = mpl.colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

# close("all")
# fig, ax = subplots(1, 1, figsize=(8, 5))
# c1 = ax.contourf(kj, fj / f, data, cmap="RdBu", levels=levels, norm=norm)
# cbar = fig.colorbar(c1, ax=ax, label=L"$mω(\partial E/\partial t)$ [m²/s³]")
# cbar.ax.tick_params(labelsize=14)
# cbar.ax.yaxis.label.set_size(16)
# ax.axhline(1, color="black", linestyle="--")
# ax.axhline(0.5ω₀ / f, color="black", linestyle="--")
# ax.text(kj[45], 0.5ω₀ / f, "0.5ω₀", color="black", va="bottom", ha="left")
# ax.axhline(ω₀ / f, color="black", linestyle="--")
# ax.text(kj[45], ω₀ / f, "ω₀", color="black", va="bottom", ha="left")
# ax.axhline(2ω₀ / f, color="black", linestyle="--")
# ax.text(kj[45], 2ω₀ / f, "2ω₀", color="black", va="bottom", ha="left")
# ax.axhline(3ω₀ / f, color="black", linestyle="--")
# ax.text(kj[45], 3ω₀ / f, "3ω₀", color="black", va="bottom", ha="left")
# ax.axhline(N / f, color="black", linestyle="--")
# ax.text(kj[45], N / f, "N", color="black", va="bottom", ha="left")
# ax.set_xlabel(L"$m$ [1/m]", fontsize=16)
# ax.set_ylabel(L"$ω/f$", fontsize=16)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.tick_params(axis="x", labelsize=14)
# ax.tick_params(axis="y", labelsize=14)
# ax.set_xlim(kj[2], maximum(kj))
# ax.set_ylim(fj[2] / f, maximum(fj) / f)
# ax.grid(true, which="both", linestyle="--", alpha=0.5)
# savefig("output/tilt/joint_spectrum_Et_m_f_log.png", dpi=300)