using NCDatasets
using PyPlot
using Statistics
include("functions/fft_psd_2d.jl")
include("functions/fft_cpsd_2d.jl")
include("functions/bin_z_dimension.jl")


simname = "tilt"    # this is the only thing that needs to be changed
θ = simname == "tilt" ? 0.0036 : 0
tᶠ = 10
dx = 30
y_ind = 100   # center or south flank of the domain
# load the data and convert the slice into hab format
bin_edges = 0:10:200
bin_center = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, y_ind, :];
nx, nz = size(hab)
what_hab_cat = Array{Float64}(undef, size(hab, 1), length(bin_center), 0)
uhat_hab_cat = Array{Float64}(undef, size(hab, 1), length(bin_center), 0)
B_hab_cat = Array{Float64}(undef, size(hab, 1), length(bin_center), 0)
t = Float64[]
file_type = "final"    # initial or final
if file_type == "initial"
    filename = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_slices_xz.nc"
    ds = Dataset(filename, "r")
    uhat = ds["uhat"][:, 1:1, :, :]
    uhat_cen = (cat(uhat[end:end,:,:,:],uhat[1:end-1, :, :, :],dims=1) .+ uhat[:,:,:,:]) ./ 2
    B = ds["B"][:, y_ind:y_ind, :, :]
    time = ds["time"][:]
    # Convert to hab format
    what_hab, _ = bin_z_dimension(what_cen, hab, bin_edges)
    B_hab, _ = bin_z_dimension(B, hab, bin_edges)
    what_hab_cat = cat(what_hab_cat, what_hab; dims=3)
    B_hab_cat = cat(B_hab_cat, what_hab; dims=3)
    t = vcat(t, time)
    @info "Processed tᶠ=$(tᶠ)"
elseif file_type == "final"
    tᶠ = 453.5:0.5:462.0
    for i in eachindex(tᶠ)
        # Load data
        filename = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ[i], "_analysis_round=all_threeD.nc")
        ds = Dataset(filename, "r")
        uhat = ds["uhat"][:, y_ind:y_ind, :, :]
        what = ds["what"][:, y_ind:y_ind, :, :]
        what_cen = (what[:, :, 1:end-1, :] .+ what[:, :, 2:end, :]) ./ 2 # what at center
        # Compute uhat at cell centers, accounting for periodicity in the x-direction
        uhat_cen = (cat(uhat[end:end,:,:,:],uhat[1:end-1, :, :, :],dims=1) .+ uhat[:,:,:,:]) ./ 2
        B = ds["B"][:, y_ind:y_ind, :, :]
        time = ds["time"][:]
        # Convert to hab format
        what_hab, _ = bin_z_dimension(what_cen, hab, bin_edges)
        uhat_hab, _ = bin_z_dimension(uhat_cen, hab, bin_edges)
        B_hab, _ = bin_z_dimension(B, hab, bin_edges)
        what_hab_cat = cat(what_hab_cat, what_hab; dims=3)
        uhat_hab_cat = cat(uhat_hab_cat, uhat_hab; dims=3)
        B_hab_cat = cat(B_hab_cat, what_hab; dims=3)
        
        t = vcat(t, time)
        @info "Processed tᶠ=$(tᶠ[i])"
    end
end
close(ds)
# Save what_hab_cat, B_hab_cat, and t as a NetCDF file
using NCDatasets
simname = "tilt"

output_nc = string("output/", simname, "/what_B_hab_cat_center_final.nc")

ds_out = Dataset(output_nc, "c")

# Dimensions
ds_out.dim["x"] = size(what_hab_cat, 1)
ds_out.dim["zbin"] = size(what_hab_cat, 2)
ds_out.dim["time"] = size(what_hab_cat, 3)

# Define variables before writing data
defVar(ds_out, "what_hab_cat", Float64, ("x", "zbin", "time"))
defVar(ds_out, "uhat_hab_cat", Float64, ("x", "zbin", "time"))
defVar(ds_out, "B_hab_cat", Float64, ("x", "zbin", "time"))
defVar(ds_out, "t", Float64, ("time",))

# Now you can write data
ds_out["what_hab_cat"][:, :, :] = what_hab_cat
ds_out["uhat_hab_cat"][:, :, :] = uhat_hab_cat
ds_out["B_hab_cat"][:, :, :] = B_hab_cat
ds_out["t"][:] = t

close(ds_out)
@info "Saved uhat_hab_cat, what_hab_cat, B_hab_cat, and t to $output_nc"

# Load data
using NCDatasets
using Statistics
using PyPlot
file_type = "final"    # initial or final
if file_type == "initial"
    simname = "tilt"
    filename = string("output/", simname, "/what_B_hab_cat_initial.nc")
    ds = Dataset(filename, "r")
elseif file_type == "final"
    simname = "tilt"
    filename = string("output/", simname, "/what_B_hab_cat_center_final.nc")
    ds = Dataset(filename, "r")
else
    error("Unknown file type: $file_type")
end
what_hab_cat = ds["what_hab_cat"][:, :, :];
what_hab_cat_pert = what_hab_cat .- mean(what_hab_cat, dims=3)  # remove mean
B_hab_cat = ds["B_hab_cat"][:, :, :];
B_hab_cat_pert = B_hab_cat .- mean(B_hab_cat, dims=3)  # remove mean
t = ds["t"][:]

dx = 30
dt = diff(t)[1]

#####1) calculate w'B' PSD (might not be ideal)
# wB = what_hab_cat_pert .* B_hab_cat_pert
# # Call fft_psd on a single slice to get wavenumbers
# E_avg = nothing
# kj = nothing
# fj = nothing
# for i in 1:size(wB, 2)    # loop over hab slices
#     # Calculate the PSD for the first time slice
#     E, kj_tmp, fj_tmp = fft_psd_2d(wB[:, i, :]; dx=dx, dt=dt, win="rec")
#     @info "Wavenumbers calculated for time slice $i"
#     if E_avg === nothing
#         E_avg = zeros(size(E))  # Use the shape of E
#         kj = kj_tmp
#         fj = fj_tmp
#     end
#     E_avg .+= E
# end
# E_avg ./= size(wB, 2)  # Average over all hab slices

######2) calculate w' and B' cross PSD
E_avg = nothing
kj = nothing
fj = nothing
for i in 1:size(what_hab_cat_pert, 2)    # loop over hab slices
    # Calculate the PSD for the first time slice
    E_ww, E_BB, E_wB, kj_tmp, fj_tmp = fft_cpsd_2d(what_hab_cat_pert[:, i, :], B_hab_cat_pert[:, i, :]; dx=dx, dt=dt, win="rec")
    @info "Wavenumbers calculated for time slice $i"
    if E_avg === nothing
        E_avg = zeros(size(E_wB))  # Use the shape of E
        kj = kj_tmp
        fj = fj_tmp
    end
    E_avg .+= E_wB
end
E_avg ./= size(what_hab_cat_pert, 2)  # Average over all hab slices

Tt = 2 * pi / 1.4e-4 / 3600
Tf = 2 * pi / (5.3e-5) / 3600
TN = 2 * pi / 1e-3 / 3600
vmin = 4.5e-5
vmax = 0.005
using PyPlot
close("all")
fig, ax = PyPlot.subplots(figsize=(10, 6))

# Prepare axes and data (variance-preserving spectrum)
kj_plot = kj[2:end]
fj_plot = fj[2:end]
E_plot = 3600 * E_avg[2:end, 2:end] .* kj_plot .* fj_plot'

# Use pcolormesh for plotting
pcm = ax.pcolormesh(
    kj_plot, 3600 * fj_plot, E_plot',
    shading="auto",
    cmap="turbo",
    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
)
ax.set_xlabel("Wavenumber [cpm]", fontsize=18)
ax.set_ylabel("Frequency [cphr]", fontsize=18)
ax.set_xscale("log")
ax.set_yscale("log")

# Optionally, add lines for specific frequencies
ax.axhline(1 / Tt, color="black", linestyle="--")
ax.axhline(2 / Tt, color="black", linestyle="--")
ax.axhline(1 / TN, color="black", linestyle="--")
ax.axhline(1 / Tf, color="black", linestyle="--")
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(L"f \cdot k \cdot \hat{w'}^* \hat{B'}~[\mathrm{m}^2~\mathrm{s}^{-3}]", fontsize=20)
cbar.ax.tick_params(labelsize=18)
ax.tick_params(axis="both", which="major", labelsize=18)
tight_layout()
savefig("output/tilt/cpsd_2d_wb_pert_kf.png")



using PyPlot
close("all")
fig, ax = PyPlot.subplots(figsize=(10, 6))

# Prepare axes and data (variance-preserving spectrum)
kj_plot = kj[2:end]
fj_plot = fj[2:end]
E_plot = 3600 * E_avg[2:end, 2:end]

# Use pcolormesh for plotting
pcm = ax.pcolormesh(
    kj_plot, 3600 * fj_plot, E_plot',
    shading="auto",
    cmap="turbo",
    norm=matplotlib.colors.LogNorm()
)
ax.set_xlabel("Wavenumber [cpm]", fontsize=18)
ax.set_ylabel("Frequency [cphr]", fontsize=18)
ax.set_xscale("log")
ax.set_yscale("log")

# Optionally, add lines for specific frequencies
ax.axhline(1 / Tt, color="black", linestyle="--")
ax.axhline(2 / Tt, color="black", linestyle="--")
ax.axhline(1 / TN, color="black", linestyle="--")
ax.axhline(1 / Tf, color="black", linestyle="--")
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(L"\hat{w'}^* \hat{B'}~[\mathrm{m}^2~\mathrm{s}^{-3}/(cpm*cphr)]", fontsize=20)
cbar.ax.tick_params(labelsize=18)
ax.tick_params(axis="both", which="major", labelsize=18)
tight_layout()
savefig("output/tilt/cpsd_2d_wb_pert.png")


## integral of the spectrum over frequency and wavenumber
#separate the integral at the tidal frequency
E_avg_norm = E_avg[2:end, 2:end]# .* (kj[2:end] .* fj[2:end]')
E_avg_low = E_avg_norm[:, fj[2:end].<1/Tt/3600]
E_avg_high = E_avg_norm[:, fj[2:end].>1/Tt/3600]
df = mean(diff(fj))
dk = mean(diff(kj))
E_avg_low_int = sum(E_avg_low) * df * dk  # integrate over frequency
E_avg_high_int = sum(E_avg_high) * df * dk  # integrate over frequency
using PyPlot
fig, ax = PyPlot.subplots(figsize=(6, 4))
freq_labels = ["Low freq (< tidal)", "High freq (> tidal)"]
integrated_values = [E_avg_low_int, E_avg_high_int]
bar_width = 0.4  # Make bars thinner (default is 0.8)
bars = ax.bar(freq_labels, integrated_values, color=["skyblue", "orange"], width=bar_width)
ax.set_ylabel("Integrated Spectral Power\n" * L"\int\int E_{w'B'}(k, f)\,dk\,df", fontsize=16)
ax.set_xlabel("Frequency Range", fontsize=16)
tight_layout()
savefig("output/tilt/integrated_spectrum_bar.png")

## pick out some wavenumbers to plot the spectrum 
ind1 = 3

using PyPlot
fig, ax = PyPlot.subplots(figsize=(10, 6))
ax.plot(1 ./ (fj[2:end]) ./ 3600, 3600 * E_avg[ind1, 2:end], label="wavenumber $(kj[ind1]) m⁻¹", color="blue")
ax.set_xlabel("Time [hr]", fontsize=18)
ax.set_ylabel("m²s⁻³/(m⁻¹ hr⁻¹)", fontsize=18)
ax.set_xscale("log")
ax.set_yscale("log")

savefig("output/tilt/spectrum_wavenumber_$(ind1).png")
