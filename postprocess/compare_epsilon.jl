# Equilibrium test: plot Bz (vertical buoyancy gradient) as a hovmoller diagram.
# Handles files with shifted/incorrect time windows by sorting all timesteps globally
# and removing duplicates before assembling the continuous 0–180 tidal-period record.

using CairoMakie
using NCDatasets
using NaNStatistics

using Interpolations
using MAT
Nx = 1000
Ny=500
Nz=250
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

T_tide = 2π / 1.4e-4          # tidal period [s]
dir    = "output/tilt"


fn = "$dir/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=456.0_analysis_round=all_threeD.nc"

ds = Dataset(fn, "r")

epsilon = ds["ε"][:,:,:,end]
epsilon[epsilon .== 0] .= NaN
epsilon_avg_tilt = dropdims(nanmean(epsilon, dims=(1,2))[:,:,:], dims=(1,2))

#flat
file = "output/flat/epsilon_avg_snapshot.nc"
ds = Dataset(file, "r")

epsilon_avg_flat = ds["ε_avg"][:]


# Save epsilon_avg to NetCDF
z = ds["zC"][:]

#plotting
fig = Figure(size = (500, 600))
ax = Axis(fig[1, 1],
    xlabel = L"\langle ε \rangle_{xy} [m^2 s^{-3}]",
    ylabel = "Height above mean slope (m)",
    xscale = log10,
    # title  = "Horizontally averaged dissipation rate"
)

lines!(ax, epsilon_avg_tilt, z, label = "Tilt")
lines!(ax, epsilon_avg_flat, z, label = "Flat")
hlines!(ax, mean(z_interp), color=:black, linestyle=:dash, label="Mean topo height")
axislegend(ax, position = :rt)

save("output/compare_epsilon.png", fig)
println("Saved figure to output/compare_epsilon.png")
