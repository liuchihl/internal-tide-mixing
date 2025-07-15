# this script plots the cross sections of terrain following heatmaps
using Interpolations
using NCDatasets

function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end

include("functions/interpolation_z_dimension.jl")
simname = "tilt"
# timerange = "80-120"

tᶠ = 454.0
θ = 3.6e-3
if θ == 3.6e-3
    simname = "tilt"
else
    simname = "flat"
end

if tᶠ ≤ 10
    output_mode = "verification"
    Nt = 11      # number of time averages
    endtime = tᶠ
elseif tᶠ ≤ 450
    output_mode = "spinup"
    Nt = 4
    if tᶠ == 450
        endtime = ["$i" for i in 50:40:450]
        # endtime = tᶠ   # if you want to run only one case
    end
else
    output_mode = "analysis"
    Nt = 1    # only 1 tidal average
    endtime = tᶠ
end


bin_edge = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
bin_mask = hab

# preallocate
B_avg = zeros(length(bin_edge) - 1, Nt)
Bz_avg = zeros(length(bin_edge) - 1, Nt)
uhat_avg = zeros(length(bin_edge) - 1, Nt)
u_avg = zeros(length(bin_edge) - 1, Nt)
what_avg = zeros(length(bin_edge) - 1, Nt)

∇κ∇B_avg = zeros(length(bin_edge) - 1, Nt)
div_uB_avg = zeros(length(bin_edge) - 1, Nt)
u_bar_∇B_bar_avg = zeros(length(bin_edge) - 1, Nt)
u_prime∇B_prime_avg = zeros(length(bin_edge) - 1, Nt)
dBdt_avg = zeros(length(bin_edge) - 1, Nt)

# Load two consecutive half tidal periods for full cycle averaging from snapshots
# because the time average outputs are not calculated correctly, so we decided to just use 3D snapshots
global total_steps
tᶠ_0 = tᶠ - 1
tᶠ_first = tᶠ - 0.5
tᶠ_second = tᶠ

filename_3D_0 = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_0, "_analysis_round=all_threeD.nc")
filename_3D_first = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_first, "_analysis_round=all_threeD.nc")
filename_3D_second = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_second, "_analysis_round=all_threeD.nc")
filename_verification = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", 10, "_threeD_timeavg.nc")
ds_3D_0 = Dataset(filename_3D_0, "r")
ds_3D_first = Dataset(filename_3D_first, "r")
ds_3D_second = Dataset(filename_3D_second, "r")
ds_verification = Dataset(filename_verification, "r")

# Get dimensions
zC = ds_3D_first["z_aac"][:]
zF = ds_3D_first["z_aaf"][:]
xC = ds_3D_first["x_caa"][:]
xF = ds_3D_first["x_faa"][:]
yC = ds_3D_first["y_aca"][:]
yF = ds_3D_first["y_afa"][:]
Nz = length(zC[:])
Nx = length(xC[:])
dx = xF[end] - xF[end-1]
Ny = length(yC[:])
dy = yF[end] - yF[end-1]
z_face = zF

uhat_sum = ds_3D_0["uhat"][:, :, :, end:end]
what_sum = ds_3D_0["what"][:, :, :, end:end]
v_sum = ds_3D_0["v"][:, :, :, end:end]
# Count total timesteps processed
total_steps = 1   # already have one timestep from the first dataset

# Process first dataset
Nt_first = length(ds_3D_first["time"][:])
for n in 1:Nt_first
    # Read data for this timestep
    uhat_sum .+= ds_3D_first["uhat"][:, :, :, n:n]
    what_sum .+= ds_3D_first["what"][:, :, :, n:n]
    v_sum .+= ds_3D_first["v"][:, :, :, n:n]
    # Free memory by explicitly clearing variables if needed
    GC.gc()
    global total_steps += 1
    println(total_steps)
end

# Process second dataset
Nt_second = length(ds_3D_second["time"][:])
for n in 1:Nt_second
    # Read data for this timestep
    uhat_sum .+= ds_3D_second["uhat"][:, :, :, n:n]
    what_sum .+= ds_3D_second["what"][:, :, :, n:n]
    v_sum .+= ds_3D_second["v"][:, :, :, n:n]
    global total_steps += 1

    # Free memory by explicitly clearing variables if needed
    GC.gc()
end

# Calculate averages
uhat = uhat_sum ./ total_steps
what = what_sum ./ total_steps
v = v_sum ./ total_steps

# Clear the intermediate sums to free memory
uhat_sum = nothing
what_sum = nothing
v_sum = nothing
GC.gc()

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
uhat_cen = (uhat[1:end,:,:] .+ vcat(uhat[2:end,:,:],uhat[1:1,:,:]))./2 # u at center in the x dimension

what_cen = (what[:, :, 1:end-1, 1] .+ what[:, :, 2:end, 1]) ./ 2 # what at center
# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end, :, :], what_cen[1:end-1, :, :]) .+ what_cen[:, :, :]) ./ 2
u = uhat[:, :, :, 1] * cos(θ) .+ wtemp * sin(θ) # cross-slope velocity
w = -uhat[:, :, :, 1] * sin(θ) .+ wtemp * cos(θ)# slope-normal velocity
v_cen = (v[:,1:end,:,1] .+ cat(v[:,2:end,:,1], v[:,1:1,:,1], dims=2))./2 # v at center in the x dimension
# topostrophy
uhat_cen[uhat_cen.==0] .= NaN
what_cen[what_cen.==0] .= NaN
v_cen[v_cen.==0] .= NaN

# calculate the topostrophy
include("functions/mmderiv.jl")
# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];
bin_mask = hab
dHdy = zeros(size(hab))
dHdx = zeros(size(hab))
for k in 1:size(hab,3)
dHdy[:,:,k] = mmderiv(yC,hab[:,:,k]')'
dHdx[:,:,k] = mmderiv(xC,hab[:,:,k])
end

τ = @. -(uhat_cen*dHdy - v_cen*dHdx)

file_TF = Dataset(string("output/", simname, "/TF_avg_tᶠ=", tᶠ, "_analysis.nc"), "r")
mask_flanks = file_TF["mask_flanks"][:,:]
τ_flanks = τ .* mask_flanks
τ_flanks[τ_flanks .== 0] .= NaN  # Set zero values to NaN for better visualization
what_cen_flanks = what_cen .* mask_flanks
what_cen_flanks[what_cen_flanks .== 0] .= NaN

# interpolation of τ and what_cen to hab

new_bin_edges = 0:8:1500
bin_centers = @. (new_bin_edges[1:end-1] + new_bin_edges[2:end]) / 2
interpolated_what = zeros(Nx,Ny,size(bin_centers,1))
interpolated_τ = zeros(Nx,Ny,size(bin_centers,1))
interpolated_what_flanks = zeros(Nx,Ny,size(bin_centers,1))
interpolated_τ_flanks = zeros(Nx,Ny,size(bin_centers,1))
# interpolation
@time interpolated_what[:,:,:],_ =  interpolate_z_dimension(what_cen, hab, new_bin_edges)
@time interpolated_what_flanks[:,:,:],_ =  interpolate_z_dimension(what_cen_flanks, hab, new_bin_edges)
@time interpolated_τ[:,:,:],_ =  interpolate_z_dimension(τ, hab, new_bin_edges)
@time interpolated_τ_flanks[:,:,:],_ =  interpolate_z_dimension(τ_flanks, hab, new_bin_edges)

# hab = 204 m 
zind = argmin(abs.(bin_centers .- 75)) # find the index of the bin closest to 204 m
τ_valid = filter(!isnan, interpolated_τ[:,:, 1:zind][:])
what_cen_valid = filter(!isnan, interpolated_what[:,:, 1:zind][:])
τ_flanks_valid = filter(!isnan, interpolated_τ_flanks[:,:, 1:zind][:])
what_cen_flanks_valid = filter(!isnan, interpolated_what_flanks[:,:, 1:zind][:])




# Compute PDFs of τ and what_cen
using StatsBase

# Filter out NaNs for valid calculations
# τ_valid = filter(!isnan, τ[:,:, 1:197][:])
# what_cen_valid = filter(!isnan, what_cen[:,:, 1:197][:])
# τ_flanks_valid = filter(!isnan, τ_flanks[:,:, 1:197][:])
# what_cen_flanks_valid = filter(!isnan, what_cen_flanks[:,:, 1:197][:])


# Set up histogram bins
lg = 400
τ_edges = range(minimum(τ_valid), maximum(τ_valid), length=lg)
what_edges = range(minimum(what_cen_valid), maximum(what_cen_valid), length=lg)
τ_flanks_edges = range(minimum(τ_flanks_valid), maximum(τ_flanks_valid), length=lg)
what_flanks_edges = range(minimum(what_cen_flanks_valid), maximum(what_cen_flanks_valid), length=lg)

# Compute histograms
τ_hist = StatsBase.fit(Histogram, τ_valid, τ_edges)
τ_hist = normalize(τ_hist, mode=:pdf)
what_hist = StatsBase.fit(Histogram, what_cen_valid, what_edges)
what_hist = normalize(what_hist, mode=:pdf)
τ_flanks_hist = StatsBase.fit(Histogram, τ_flanks_valid, τ_flanks_edges)
τ_flanks_hist = normalize(τ_flanks_hist, mode=:pdf)
what_flanks_hist = StatsBase.fit(Histogram, what_cen_flanks_valid, what_flanks_edges)
what_flanks_hist = normalize(what_flanks_hist, mode=:pdf)


# Calculate bin centers for plotting
τ_centers = (τ_edges[1:end-1] .+ τ_edges[2:end]) ./ 2
what_centers = (what_edges[1:end-1] .+ what_edges[2:end]) ./ 2
τ_flanks_centers = (τ_flanks_edges[1:end-1] .+ τ_flanks_edges[2:end]) ./ 2
what_flanks_centers = (what_flanks_edges[1:end-1] .+ what_flanks_edges[2:end]) ./ 2

# Visualize the PDFs
using PyPlot

# Create a new figure with two subplots side by side
close("all")
fig, (ax1, ax2) = PyPlot.subplots(1, 2, figsize=(10, 4))

# Plot PDF of topostrophy (τ)
ax1.plot(τ_centers, τ_hist.weights, linewidth=2)
ax1.plot(τ_flanks_centers, τ_flanks_hist.weights, linewidth=2, linestyle="-", color="orange", label="Flanks")
ax1.set_xlabel("Topostrophy (τ)")
ax1.set_ylabel("PDF")
ax1.set_xlim(-0.1, 0.1)
# ax1.set_title("PDF of Topostrophy")
ax1.grid(true, alpha=0.3)
ax1.axvline(x=0, color="black", linestyle="-", linewidth=2)  # Add vertical line at zero
ax1.set_yscale("log")  # Set y-axis to logarithmic scale for better visibility
# Plot PDF of vertical velocity component (what_cen)
ax2.plot(what_centers, what_hist.weights, linewidth=2)
ax2.plot(what_flanks_centers, what_flanks_hist.weights, linewidth=2, linestyle="-", color="orange", label="Flanks")
ax2.set_xlabel("ŵ [m s⁻¹]")
ax2.set_ylabel("PDF")
ax2.set_xlim(-0.07, 0.04)
# ax2.set_title("PDF of Vertical Velocity")
ax2.grid(true, alpha=0.3)
ax2.axvline(0, color="black", linestyle="-", linewidth=2)  # Add vertical line at zero
ax2.set_yscale("log")  # Set y-axis to logarithmic scale for better visibility
# Adjust layout
tight_layout()
# Add legends to both plots
# ax1.legend(["Full Domain", "Flanks"], loc="best")
ax2.legend(["Full Domain", "Flanks"], loc="best")
# Save the figure
savefig(string("output/tilt/pdf_topostrophy_w_",bin_center[zind],".png"), dpi=300)

# Calculate skewness for each distribution
using Statistics

# Function to calculate skewness from raw data
function calculate_skewness(data)
    μ = mean(data)
    σ = std(data)
    n = length(data)
    skew = sum(((data .- μ) ./ σ).^3) / n
    return skew
end

# Calculate skewness for each distribution
τ_skew = calculate_skewness(τ_valid)
τ_flanks_skew = calculate_skewness(τ_flanks_valid)
w_skew = calculate_skewness(what_cen_valid)
w_flanks_skew = calculate_skewness(what_cen_flanks_valid)

# Print the results
println("Skewness statistics:")
println("Topostrophy (full domain): $(round(τ_skew, digits=4))")
println("Topostrophy (flanks): $(round(τ_flanks_skew, digits=4))")
println("Vertical velocity (full domain): $(round(w_skew, digits=4))")
println("Vertical velocity (flanks): $(round(w_flanks_skew, digits=4))")

# Add skewness values as text annotations to the plots
ax1.text(0.05, 0.95, "Skew = $(round(τ_skew, digits=2))", transform=ax1.transAxes, 
         va="top", ha="left", bbox=Dict("facecolor"=>"white", "alpha"=>0.7))
ax1.text(0.05, 0.85, "Skew (flanks) = $(round(τ_flanks_skew, digits=2))", transform=ax1.transAxes, 
         va="top", ha="left", bbox=Dict("facecolor"=>"white", "alpha"=>0.7))

ax2.text(0.05, 0.95, "Skew = $(round(w_skew, digits=2))", transform=ax2.transAxes, 
         va="top", ha="left", bbox=Dict("facecolor"=>"white", "alpha"=>0.7))
ax2.text(0.05, 0.85, "Skew (flanks) = $(round(w_flanks_skew, digits=2))", transform=ax2.transAxes, 
         va="top", ha="left", bbox=Dict("facecolor"=>"white", "alpha"=>0.7))