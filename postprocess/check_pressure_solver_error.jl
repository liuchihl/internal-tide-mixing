# Plot time-averaged profiles of |div(u)| vs hab before and after t=450 tidal periods.
# Compares both the xy-bin mean and xy-bin maximum of |div(u)|.

using Printf
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using LinearAlgebra
using Interpolations
using JLD2
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bin_stat_over_xy.jl")

###### Load topography
Nx = 500
Ny = 1000
Nz = 250
basefile = "/scratch/bcpi/cliu28/internal-tide-mixing"
cd(basefile)
using MAT
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic")
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo, 1))
y_topo_lin = range(y_topo[1], y_topo[end], size(z_topo, 2))
close(file)
x_interp = range(x_topo[1], x_topo[end], length=Nx)
y_interp = range(y_topo[1], y_topo[end], length=Ny)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp .- minimum(z_interp)

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/bins.jl")
include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/mmderiv.jl")

###### Grid and hab
depth = 1:50
bin_edge   = 0:8:1500
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2
nbins      = length(bin_center)

filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, depth]

###### File paths
spinup_dir1   = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=410_threeD_timeavg.nc"
spinup_dir2   = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=450_threeD_timeavg.nc"
analysis_dirs = ["output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=$(tᶠ)_analysis_round=all_threeD.nc"
                 for tᶠ in 451.5:0.5:461.0]

ds_spinup1 = Dataset(spinup_dir1, "r")
ds_spinup2 = Dataset(spinup_dir2, "r")

###### Grid spacing
zC = ds_spinup1["zC"][:]
zF = ds_spinup1["zF"][:]
xC = ds_spinup1["xC"][:]
xF = ds_spinup1["xF"][:]
yC = ds_spinup1["yC"][:]
yF = ds_spinup1["yF"][:]
dx = xF[2] - xF[1]
dy = yF[2] - yF[1]
dz = diff(zF)

###### Core function: return (mean, max) of |div(u)| binned by hab
function divu_hab_stats(uhat, v, what, hab, bin_edge, dx, dy, dz)
    du_dx  = (circshift(uhat, (-1, 0, 0)) .- uhat) ./ dx
    dv_dy  = (circshift(v,    (0, -1, 0)) .- v)    ./ dy
    dw_dz  = (what[:, :, 2:end] .- what[:, :, 1:end-1]) ./ reshape(dz, 1, 1, :)
    div_u  = du_dx .+ dv_dy .+ dw_dz
    div_u[hab .<= 0] .= NaN
    abs_divu = abs.(div_u)
    mean_col = bin_stat_over_xy(abs_divu, bin_edge, hab; stat="mean")
    max_col  = bin_stat_over_xy(abs_divu, bin_edge, hab; stat="maximum")
    return mean_col, max_col
end

###### Compute or load profiles
savefile = "output/divu_hab_profiles.jld2"

if isfile(savefile)
    @info "Loading saved profiles from $savefile"
    @load savefile before_40_mean before_40_max before_80_mean before_80_max after_10_mean after_10_max bin_center
else
    @info "Computing profiles..."

    # ── before_40: tᶠ=450 spinup file only (~40 tidal periods) ────────────────
    before_40_mean = zeros(nbins)
    before_40_max  = zeros(nbins)
    count_40 = 0
    times = Float64.(ds_spinup2["time"][:])
    for n in 1:length(times)
        @info "before_40: timestep $n / $(length(times))"
        m, mx = divu_hab_stats(ds_spinup2["uhat"][:,:,depth,n], ds_spinup2["v"][:,:,depth,n],
                               ds_spinup2["what"][:,:,1:depth[end]+1,n], hab, bin_edge, dx, dy, dz[depth])
        before_40_mean .+= m
        before_40_max  .+= mx
        count_40 += 1
        GC.gc()
    end
    before_40_mean ./= count_40
    before_40_max  ./= count_40

    # ── before_80: both tᶠ=410 and tᶠ=450 spinup files (~80 tidal periods) ───
    before_80_mean = zeros(nbins)
    before_80_max  = zeros(nbins)
    count_80 = 0
    for (label, ds) in [("tᶠ=410", ds_spinup1), ("tᶠ=450", ds_spinup2)]
        times = Float64.(ds["time"][:])
        for n in 1:length(times)
            @info "before_80 ($label): timestep $n / $(length(times))"
            m, mx = divu_hab_stats(ds["uhat"][:,:,depth,n], ds["v"][:,:,depth,n],
                                   ds["what"][:,:,1:depth[end]+1,n], hab, bin_edge, dx, dy, dz[depth])
            before_80_mean .+= m
            before_80_max  .+= mx
            count_80 += 1
            GC.gc()
        end
    end
    before_80_mean ./= count_80
    before_80_max  ./= count_80

    # ── after_10: all 20 analysis files (~10 tidal periods) ───────────────────
    after_10_mean = zeros(nbins)
    after_10_max  = zeros(nbins)
    count_after = 0
    for path in analysis_dirs
        ds = Dataset(path, "r")
        Nt_local = length(ds["time"][:])
        @info "after_10: $path — averaging $Nt_local snapshots"

        uhat_sum = zeros(Float64, size(ds["uhat"][:,:,depth,1]))
        v_sum    = zeros(Float64, size(ds["v"][:,:,depth,1]))
        what_sum = zeros(Float64, size(ds["what"][:,:,1:depth[end]+1,1]))
        for n in 1:Nt_local
            uhat_sum .+= ds["uhat"][:,:,depth,n]
            v_sum    .+= ds["v"][:,:,depth,n]
            what_sum .+= ds["what"][:,:,1:depth[end]+1,n]
        end
        m, mx = divu_hab_stats(uhat_sum ./ Nt_local, v_sum ./ Nt_local,
                               what_sum ./ Nt_local, hab, bin_edge, dx, dy, dz[depth])
        after_10_mean .+= m
        after_10_max  .+= mx
        count_after += 1
        close(ds)
        GC.gc()
    end
    after_10_mean ./= count_after
    after_10_max  ./= count_after

    @save savefile before_40_mean before_40_max before_80_mean before_80_max after_10_mean after_10_max bin_center
    @info "Saved profiles to $savefile"
end

savefile = "output/divu_hab_profiles.jld2"
d = JLD2.load(savefile)
before_40_mean = d["before_40_mean"]
before_40_max  = d["before_40_max"]
before_80_mean = d["before_80_mean"]
before_80_max  = d["before_80_max"]
after_10_mean  = d["after_10_mean"]
after_10_max   = d["after_10_max"]
bin_center     = d["bin_center"]

###### Plot
hab_range = 1:10    # bins ~12–204 m above bottom

fig = CairoMakie.Figure(size=(500, 500))

ax = Axis(fig[1, 1],
          xlabel = "log₁₀|∇·u| (s⁻¹)",
          ylabel = "Height above bottom (m)",
          title  = "Maximum |∇·u| (s⁻¹)")

yvals  = bin_center[hab_range]
x40    = log10.(max.(before_40_max[hab_range], 1e-20))
xafter = log10.(max.(after_10_max[hab_range],  1e-20))

lines!(ax, x40,    yvals, color=:steelblue, linewidth=2, label="t=410-450 averaged (FFT pressure solver)")
lines!(ax, xafter, yvals, color=:firebrick, linewidth=2, label="t=451-461 averaged (CG pressure solver)")

axislegend(ax, position=:rt)

save("output/divu_hab_profiles_maximum.png", fig)
@info "Saved output/divu_hab_profiles_maximum.png"

# fig = CairoMakie.Figure(size=(900, 500))

# for (col, (title, before_40, before_80, after_10)) in enumerate([
#         ("Mean |∇·u| (s⁻¹)",    before_40_mean, before_80_mean, after_10_mean),
#         ("Maximum |∇·u| (s⁻¹)", before_40_max,  before_80_max,  after_10_max),
#     ])

#     ax = Axis(fig[1, col],
#               xlabel = "log₁₀|∇·u| (s⁻¹)",
#               ylabel = col == 1 ? "Height above bottom (m)" : "",
#               title  = title,
#               yticklabelsvisible = col == 1)

#     x40    = log10.(max.(before_40[hab_range], 1e-20))
#     x80    = log10.(max.(before_80[hab_range], 1e-20))
#     xafter = log10.(max.(after_10[hab_range],  1e-20))
#     yvals  = bin_center[hab_range]

#     lines!(ax, x40,    yvals, color=:steelblue,   linewidth=2, label="Before t=450 (40 TP)")
#     lines!(ax, x80,    yvals, color=:dodgerblue,  linewidth=2, label="Before t=450 (80 TP)")
#     lines!(ax, xafter, yvals, color=:firebrick,   linewidth=2, label="After t=450 (10 TP)")

#     if col == 2
#         Legend(fig[1, 3], ax)
#     end
# end

# save("output/divu_hab_profiles.png", fig)
# @info "Saved output/divu_hab_profiles.png"
