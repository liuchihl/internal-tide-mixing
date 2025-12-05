using NCDatasets
using PyPlot
using Statistics
using Interpolations
include("functions/fft_psd_2d.jl")
include("functions/fft_psd.jl")
include("functions/fft_cpsd_2d.jl")
include("functions/bin_z_dimension.jl")


# first analyze the 1D profile
simname = "tilt"    # this is the only thing that needs to be changed
θ = simname == "tilt" ? 0.0036 : 0
tf = 451.5:0.5:462.0
# tf = 451.5:0.5:452.0
fname = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tf[1], "_analysis_round=all_slices_xz.nc")
ds = Dataset(fname, "r")
zC = ds["z_aac"][:]
zF = ds["z_aaf"][:]
xC = ds["x_caa"][:]

B_all = Float64[]
uhat_all = Float64[]
v_all = Float64[]
what_all = Float64[]
eps_all = Float64[]
time_all = Float64[]
for i in eachindex(tf)
    fname = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tf[i], "_analysis_round=all_slices_xz.nc")
    ds = Dataset(fname, "r")
    B = ds["B"][:, 1, :, :]
    uhat = ds["uhat"][:, 1, :, :]
    vhat = ds["v"][:, 1, :, :]
    what = ds["what"][:, 1, :, :]
    eps = ds["ε"][:, 1, :, :]
    time = ds["time"][:]
    close(ds)
    if i == 1
        B_all = B
        uhat_all = uhat
        v_all = vhat
        what_all = what
        eps_all = eps
        time_all = time
    else
        B_all = cat(B_all, B, dims=3)
        uhat_all = cat(uhat_all, uhat, dims=3)
        v_all = cat(v_all, vhat, dims=3)
        what_all = cat(what_all, what, dims=3)
        eps_all = cat(eps_all, eps, dims=3)
        time_all = vcat(time_all, time)
    end
    println("Processed tᶠ=$(tf[i])")
end


# interpolate to uniform vertical grid
Nz_new = 250
zC_new = range(minimum(zC), stop=maximum(zC), length=Nz_new)
zF_new = range(minimum(zF), stop=maximum(zF), length=Nz_new + 1)
B_interp = Array{Float64}(undef, size(B_all, 1), Nz_new, size(B_all, 3))
uhat_interp = Array{Float64}(undef, size(uhat_all, 1), Nz_new, size(uhat_all, 3))
v_interp = Array{Float64}(undef, size(v_all, 1), Nz_new, size(v_all, 3))
what_interp = Array{Float64}(undef, size(what_all, 1), Nz_new, size(what_all, 3))
for i in axes(B_all, 1)
    for j in axes(B_all, 3)
        B_interp[i, :, j] = Interpolations.interpolate((zC,), B_all[i, :, j], Gridded(Linear())).(zC_new)
        uhat_interp[i, :, j] = Interpolations.interpolate((zC,), uhat_all[i, :, j], Gridded(Linear())).(zC_new)
        v_interp[i, :, j] = Interpolations.interpolate((zC,), v_all[i, :, j], Gridded(Linear())).(zC_new)
        what_interp[i, :, j] = Interpolations.interpolate((zF,), what_all[i, :, j], Gridded(Linear())).(zC_new)
    end
end
# time derivative of each
B_interp_dt = diff(B_interp, dims=3) ./ reshape(diff(time_all), 1, 1, :)
uhat_interp_dt = diff(uhat_interp, dims=3) ./ reshape(diff(time_all), 1, 1, :)
v_interp_dt = diff(v_interp, dims=3) ./ reshape(diff(time_all), 1, 1, :)
what_interp_dt = diff(what_interp, dims=3) ./ reshape(diff(time_all), 1, 1, :)

# move the time derivative quantities to the center
B_interp_dt = (cat(B_interp_dt, zeros(size(B_interp, 1), size(B_interp, 2), 1), dims=3) .+ 
               cat(zeros(size(B_interp, 1), size(B_interp, 2), 1), B_interp_dt, dims=3)) / 2
uhat_interp_dt = (cat(uhat_interp_dt, zeros(size(uhat_interp, 1), size(uhat_interp, 2), 1), dims=3) .+ 
                  cat(zeros(size(uhat_interp, 1), size(uhat_interp, 2), 1), uhat_interp_dt, dims=3)) / 2
v_interp_dt = (cat(v_interp_dt, zeros(size(v_interp, 1), size(v_interp, 2), 1), dims=3) .+ 
               cat(zeros(size(v_interp, 1), size(v_interp, 2), 1), v_interp_dt, dims=3)) / 2
what_interp_dt = (cat(what_interp_dt, zeros(size(what_interp, 1), size(what_interp, 2), 1), dims=3) .+ 
                  cat(zeros(size(what_interp, 1), size(what_interp, 2), 1), what_interp_dt, dims=3)) / 2

# compute buoyancy frequency squared profile N(z)
# Calculate Bz
Bz = diff(B_all, dims=2) ./ reshape(diff(zC[:]), 1, length(zC) - 1)
# Center Bz at cell centers
Nx = size(B_all, 1)
Nt = size(B_all, 3)
Bz_center = (cat(zeros(Nx, 1, Nt), Bz, dims=2) .+ cat(Bz, zeros(Nx, 1, Nt), dims=2)) / 2
Bz_center[Bz_center.<0] .= NaN  # mask out the unstable regions

# exclude the center sill for a deeper water column
left = 1
right = argmin(abs.(xC .- 5000.0))

# only take the layers that are stably stratified
Bz_avg_x = nanmean(Bz_center[left:right, :, :], dims=1)
N = sqrt.(Bz_avg_x)

# free memory
what_all = nothing
B_all = nothing
uhat_all = nothing
v_all = nothing
GC.gc()


dt = mean(diff(time_all))
dz = mean(diff(zC_new))
low = argmin(abs.(zC_new .- 600.0))
up = argmin(abs.(zC_new .- 1750.0))

_, _, _, kj_cp, fj_cp = fft_cpsd_2d(uhat_interp[1, low:up, :], uhat_interp_dt[1, low:up, :]; dx=dz, dt=dt, win="hann")

E_uu_t = zeros(length(kj_cp), length(fj_cp))
E_vv_t = zeros(length(kj_cp), length(fj_cp))
E_ww_t = zeros(length(kj_cp), length(fj_cp))
E_BB_t = zeros(length(kj_cp), length(fj_cp))
E_BB = zeros(length(kj_cp), length(fj_cp))
E_uu = zeros(length(kj_cp), length(fj_cp))
E_vv = zeros(length(kj_cp), length(fj_cp))
E_ww = zeros(length(kj_cp), length(fj_cp))
# compute ∂ₜE(m,ω) using cross-spectral density
for i in left:right
    E_BB_temp, _, E_BB_t_temp, _, _ = fft_cpsd_2d(B_interp[i, low:up, :], B_interp_dt[i, low:up, :] ./ N[1, low:up, :]; dx=dz, dt=dt, win="hann")
    E_uu_temp, _, E_uu_t_temp, _, _ = fft_cpsd_2d(uhat_interp[i, low:up, :], uhat_interp_dt[i, low:up, :]; dx=dz, dt=dt, win="hann")
    E_vv_temp, _, E_vv_t_temp, _, _ = fft_cpsd_2d(v_interp[i, low:up, :], v_interp_dt[i, low:up, :]; dx=dz, dt=dt, win="hann")
    E_ww_temp, _, E_ww_t_temp, _, _ = fft_cpsd_2d(what_interp[i, low:up, :], what_interp_dt[i, low:up, :]; dx=dz, dt=dt, win="hann")

    if i == left
        E_uu_t = E_uu_t_temp
        E_vv_t = E_vv_t_temp
        E_ww_t = E_ww_t_temp
        E_BB_t = E_BB_t_temp
        E_BB = E_BB_temp
        E_uu = E_uu_temp
        E_vv = E_vv_temp
        E_ww = E_ww_temp
    else
        E_uu_t .+= E_uu_t_temp
        E_vv_t .+= E_vv_t_temp
        E_ww_t .+= E_ww_t_temp
        E_BB_t .+= E_BB_t_temp
        E_BB .+= E_BB_temp
        E_uu .+= E_uu_temp
        E_vv .+= E_vv_temp
        E_ww .+= E_ww_temp
    end
    println("Processed slice $i / $(right-left+1)")
end
# horizontal average the spectra
E_uu_t ./= (right - left + 1)
E_vv_t ./= (right - left + 1)
E_ww_t ./= (right - left + 1)
E_BB_t ./= (right - left + 1)
E_uu ./= (right - left + 1)
E_vv ./= (right - left + 1)
E_ww ./= (right - left + 1)
E_BB ./= (right - left + 1)
KE_t = real(E_uu_t .+ E_vv_t .+ E_ww_t)
PE_t = real(E_BB_t)
E_t = KE_t .+ PE_t  # this has the dimension of [ω, m]

KE = real(E_uu .+ E_vv .+ E_ww)
PE = real(E_BB)
E = KE .+ PE  # this has the dimension of [ω, m]


f = 5.3e-5
ω₀ = 1.4e-4
N = 1e-3
# convert to radian
kj = 2*pi .* kj_cp
fj = 2*pi .* fj_cp

using PyCall
mpl = pyimport("matplotlib")

# Define levels clipped to your desired color range
data_min = nanminimum(kj'.*fj.*E_t'*1e9)
data_max = nanmaximum(kj'.*fj.*E_t'*1e9)
vmin = -1
vmax = 1
levels = range(nanmax(data_min, vmin), nanmin(data_max, vmax), length=51)

# plot this joint spectrum close("all")
fig, axs = subplots(1, 2, figsize=(16, 5), constrained_layout=true)

# Panel 1: E * kj * fj
data1 = kj' .* fj .* E'  # [m²/s³]
vmin1 = minimum(data1)
vmax1 = maximum(data1)
levels1 = range(vmin1, vmax1, length=51)
c1 = axs[1].contourf(kj, fj / f, data1, cmap="RdBu_r", levels=levels1)
cbar1 = fig.colorbar(c1, ax=axs[1], label=L"$mωE$ [m²/s³]")
cbar1.ax.tick_params(labelsize=14)
cbar1.ax.yaxis.label.set_size(16)
axs[1].set_xlabel(L"$m$ [1/m]", fontsize=16)
axs[1].set_ylabel(L"$ω/f$", fontsize=16)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].tick_params(axis="x", labelsize=14)
axs[1].tick_params(axis="y", labelsize=14)
axs[1].set_xlim(kj[2], maximum(kj))
axs[1].set_ylim(fj[2]/f, maximum(fj)/f)
axs[1].grid(true, which="both", linestyle="--", alpha=0.5)
# axs[1].set_title("E ⋅ mω")

# Panel 2: dE/dt * kj * fj (your existing code)
data2 = kj' .* fj .* E_t' * 1e9
vmin2 = -1
vmax2 = 1
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
axs[2].set_ylim(fj[2]/f, maximum(fj)/f)
axs[2].grid(true, which="both", linestyle="--", alpha=0.5)
# axs[2].set_title("∂ₜE ⋅ mω")

savefig("output/tilt/joint_spectrum_E_and_dEt_m_f.png", dpi=300)

# plot E vs fj
fig, axs = subplots(1, 2, figsize=(16, 5), constrained_layout=true)

# Panel 1: E * kj * fj
data1 =  E'  # [m²/s³]
axs[1].loglog(fj,data1[:,:])
axs[1].axvline(f, color="black", linestyle="--", label="f")
axs[1].axvline(ω₀, color="red", linestyle="--", label="ω₀")
axs[1].axvline(N, color="green", linestyle="--", label="N")
axs[1].set_xlabel(L"$ω$ [1/s]", fontsize=16)
axs[1].set_ylabel(L"$E$ [m²/s³]", fontsize=16)
axs[1].tick_params(axis="x", labelsize=14)
axs[1].tick_params(axis="y", labelsize=14)

data2 =  E'[2:end,:]./fj[2:end]  # [m²/s³]
axs[2].loglog(fj[2:end],data2[:,:])
axs[2].axvline(f, color="black", linestyle="--", label="f")
axs[2].axvline(ω₀, color="red", linestyle="--", label="ω₀")
axs[2].axvline(N, color="green", linestyle="--", label="N")
axs[2].set_xlabel(L"$ω$ [1/s]", fontsize=16)
axs[2].set_ylabel(L"$E/ω$ [m²/s³]", fontsize=16)
axs[2].tick_params(axis="x", labelsize=14)
axs[2].tick_params(axis="y", labelsize=14)


savefig("output/tilt/spectrum_E_f_multiple_m.png", dpi=300)


# help me save E and kj and fj as a netcdf file
outfname = "output/tilt/E_m_omega.nc"
ds_out = Dataset(outfname, "c")
defDim(ds_out, "m", length(kj))
defDim(ds_out, "omega", length(fj))
defVar(ds_out, "E", Float64, ("m", "omega"))
defVar(ds_out, "m", Float64, ("m",))
defVar(ds_out, "omega", Float64, ("omega",))

# Assign data to the variables
ds_out["E"][:] = E
ds_out["m"][:] = kj
ds_out["omega"][:] = fj

println("Saving joint spectrum to $outfname")
close(ds_out)




## check the distribution of epsilon
left = 1
right = argmin(abs.(xC .- 5000.0))
low = argmin(abs.(zC .- 600.0))
up = argmin(abs.(zC .- 1750.0))
# eps_range = eps_all[:, :, :]
eps_range = eps_all[left:right, low:up, :]
eps_range = eps_range[eps_range .> 1e-10]  # only keep positive values
# Flatten the eps_range array and remove NaN values
eps_values = filter(!isnan, vec(eps_range))
bins = 10 .^ range(-10, log10(maximum(eps_values)), length=100)

# Plot the PDF in log scale
close("all")
fig, ax = subplots(figsize=(8, 5))
ax.hist(eps_values, bins=bins, density=true, color="blue", alpha=0.7, edgecolor="black")
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(L"$\epsilon$ [m²/s³]", fontsize=16)
ax.set_ylabel("Probability Density", fontsize=16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.grid(true, which="both", linestyle="--", alpha=0.5)
# ax.set_title(L"PDF of $\epsilon$", fontsize=18)
savefig("output/tilt/pdf_eps_log.png", dpi=300)



using PyCall
mpl = pyimport("matplotlib")
np = pyimport("numpy")

data = kj'.*fj.*E_t'

absmax = maximum(abs.(data))
vmin = -absmax
vmax = absmax
linthresh = absmax / 1000  # adjust as needed

# Number of levels in each region
n_log = 20
n_lin = 11

# Log-spaced levels for positive and negative
log_pos = exp.(range(log(linthresh), log(vmax), length=n_log))
log_neg = -reverse(log_pos)

# Linear levels around zero
lin_levels = range(-linthresh, linthresh, length=n_lin)

# Combine all levels and ensure uniqueness/sorting
levels = vcat(log_neg, lin_levels, log_pos)
levels = unique(sort(levels))

norm = mpl.colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

close("all")
fig, ax = subplots(1, 1, figsize=(8, 5))
c1 = ax.contourf(kj, fj / f, data, cmap="RdBu", levels=levels, norm=norm)
cbar = fig.colorbar(c1, ax=ax, label=L"$mω(\partial E/\partial t)$ [m²/s³]")
cbar.ax.tick_params(labelsize=14)
cbar.ax.yaxis.label.set_size(16)
ax.axhline(1, color="black", linestyle="--")
ax.axhline(.5ω₀ / f, color="black", linestyle="--")
ax.text(kj[45], .5ω₀ / f, "0.5ω₀", color="black", va="bottom", ha="left")
ax.axhline(ω₀ / f, color="black", linestyle="--")
ax.text(kj[45], ω₀ / f, "ω₀", color="black", va="bottom", ha="left")
ax.axhline(2ω₀ / f, color="black", linestyle="--")
ax.text(kj[45], 2ω₀ / f, "2ω₀", color="black", va="bottom", ha="left")
ax.axhline(3ω₀ / f, color="black", linestyle="--")
ax.text(kj[45], 3ω₀ / f, "3ω₀", color="black", va="bottom", ha="left")
ax.axhline(N / f, color="black", linestyle="--")
ax.text(kj[45], N / f, "N", color="black", va="bottom", ha="left")
ax.set_xlabel(L"$m$ [1/m]",fontsize=16)
ax.set_ylabel(L"$ω/f$",fontsize=16)
ax.set_xscale("log")
ax.set_yscale("log")
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_xlim(kj[2], maximum(kj))
ax.set_ylim(fj[2]/f, maximum(fj)/f)
ax.grid(true, which="both", linestyle="--", alpha=0.5)
savefig("output/tilt/joint_spectrum_Et_m_f_log.png", dpi=300)