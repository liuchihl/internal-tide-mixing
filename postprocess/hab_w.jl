using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics

simname = "tilt"
t_start = 458.0
t_end = 462.0
t_range = t_start:1.0:t_end

what_accum = nothing
count_accum = nothing
nfiles = 0

for tᶠ in t_range
    file = string("output/", simname, "/TF_avg_tᶠ=", tᶠ, "_analysis.nc")
    ds = Dataset(file, "r")
    what_avg_sill = ds["what_avg_sill"][:,1]      # 187
    what_avg_rest = ds["what_avg_rest"][:,1]      # 187
    what_avg_flanks = ds["what_avg_flanks"][:,1]  # 187
    mask_sill = ds["mask_sill"][:,:]              # 500x1000
    mask_rest = ds["mask_rest"][:,:]              # 500x1000
    mask_flanks = ds["mask_flanks"][:,:]          # 500x1000

    # Count valid points in each region
    n_sill = nansum(mask_sill)
    n_rest = nansum(mask_rest)
    n_flanks = nansum(mask_flanks)

    # For each depth, accumulate weighted sum and total count
    if what_accum === nothing
        what_accum = zeros(length(what_avg_sill))
        count_accum = zeros(length(what_avg_sill))
    end
    what_accum .+= what_avg_sill .* n_sill
    count_accum .+= n_sill
    what_accum .+= what_avg_rest .* n_rest
    count_accum .+= n_rest
    what_accum .+= what_avg_flanks .* n_flanks
    count_accum .+= n_flanks

    nfiles += 1
    close(ds)
end

# Average over all files and all valid points
what_avg_mean = what_accum ./ count_accum ./ nfiles

# Load z for plotting (from last file)
ds = Dataset(string("output/", simname, "/TF_avg_tᶠ=", t_end, "_analysis.nc"), "r")
z = ds["bin_center"][:]
close(ds)

fig = Figure()
ax = Axis(fig[1,1], xlabel="Averaged true vertical velocity ŵ [m/s]", ylabel="Height above bottom [m]", title="5 tidal period average (total region)",
    )
lines!(ax, what_avg_mean, z, color=:blue, linewidth=2.8, label="Total mean profile")
vlines!(ax, 0, color=:black, linestyle=:dash)
save(string("output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=", t_start, "-", t_end, "_what_avg_total_mean_profile.png"),fig)
println(string("Saved to output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=", t_start, "-", t_end, "_what_avg_total_mean_profile.png"))  