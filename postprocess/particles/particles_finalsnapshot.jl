using MAT
using NCDatasets
using NaNStatistics
using Interpolations
using CairoMakie
using ColorSchemes
using Statistics
using Printf

# --- Load topography and transform it into Cartesian coordinates ---
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic")
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo, 1))
y_topo_lin = range(y_topo[1], y_topo[end], size(z_topo, 2))
close(file)
Nx = 500
Ny = 1000
θ = 0.0036
x_interp = range(x_topo[1], x_topo[end], length=Nx)
y_interp = range(y_topo[1], y_topo[end], length=Ny)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp .- minimum(z_interp)
dx_interp = x_interp[2] - x_interp[1]
dy_interp = y_interp[2] - y_interp[1]
Lx_interp = x_interp[end]
Ly_interp = y_interp[end]
x_interp_ext = vcat(x_interp[3Nx÷4:Nx] .- Lx_interp .- dx_interp, x_interp, x_interp .+ Lx_interp .+ dx_interp, x_interp .+ 2Lx_interp .+ dx_interp)
y_interp_ext = vcat(y_interp, y_interp .+ Ly_interp .+ dy_interp)
z_interp_ext = vcat(z_interp[3Nx÷4:Nx, :], z_interp, z_interp, z_interp)
z_interp_ext = vcat(z_interp_ext', z_interp_ext')'
X_interp_ext = repeat(x_interp_ext, 1, length(y_interp_ext))
Y_interp_ext = repeat(y_interp_ext', length(x_interp_ext), 1)
Z_interp_ext = z_interp_ext
X_cart_interp_ext = X_interp_ext .* cos(θ) .- Z_interp_ext .* sin(θ)
Y_cart_interp_ext = Y_interp_ext
Z_cart_interp_ext = X_interp_ext .* sin(θ) .+ Z_interp_ext .* cos(θ)

simname = "tilt"
z_center_particle = 1000
output_file = string("output/", simname, "/concatenated_particle_data_z", z_center_particle, "_all.nc")
@info "Loading concatenated data for final snapshot..."
ds = NCDataset(output_file, "r")
x_cart_full = ds["main_x_cartesian"][:, :]
y_cart_full = ds["main_y_cartesian"][:, :]
z_cart_full = ds["main_z_cartesian"][:, :]
B_full = ds["main_buoyancy"][:, :]
B_pert = B_full .- B_full[:, 1]
time_full = ds["main_time"][:]
close(ds)
B_full = nothing
GC.gc()

# --- Plot final snapshot ---
n_particles, n_time_steps = size(x_cart_full)
final_time_idx = n_time_steps
particle_sample = 1:5:n_particles

particle_x = x_cart_full[particle_sample, final_time_idx]
particle_y = y_cart_full[particle_sample, final_time_idx]
particle_z = z_cart_full[particle_sample, final_time_idx]
particle_colors = B_pert[particle_sample, final_time_idx]

fig = CairoMakie.Figure(resolution=(1200, 800), layout=GridLayout(1, 2, width_ratios=[0.9, 0.15]))

ax = Axis3(fig[1, 1],
    xlabel="x̂ [km]",
    ylabel="y [km]",
    zlabel="\n ẑ [m]",
    aspect=(3, 4, 0.8),
    limits=(minimum(X_cart_interp_ext)*1e-3, maximum(X_cart_interp_ext)*1e-3,
        minimum(Y_cart_interp_ext)*1e-3, maximum(Y_cart_interp_ext)*1e-3,
        minimum(Z_cart_interp_ext), maximum(Z_cart_interp_ext)),
    xlabelsize=24,  # Increase label font size
    xticklabelsize=24,  # Increase tick label font size
    ylabelsize=24,  # Increase label font size
    yticklabelsize=24,  # Increase tick label font size
    zlabelsize=24,  # Increase label font size
    zticklabelsize=24  # Increase tick label font size

)

full_cmap = ColorSchemes.terrain.colors
custom_cmap = full_cmap[1:floor(Int, 1 * length(full_cmap))]

surface!(ax, X_cart_interp_ext*1e-3, Y_cart_interp_ext*1e-3, Z_cart_interp_ext,
    colormap=:terrain,
    shading=NoShading,
    transparency=false)

scatter!(ax, particle_x*1e-3, particle_y*1e-3, particle_z,
    color=particle_colors,
    markersize=4,
    colormap=reverse(cgrad(:RdBu)),
    colorrange=(-2e-5, 2e-5),
    alpha=0.4)
Lx=x_interp[end]
Ly=y_interp[end]
scatter!(ax, 0.25*Lx*1e-3, 0.5*Ly*1e-3, 1000,
    color=:green,
    strokecolor=:black,
    markersize=25,
    alpha=0.8)

cbar_layout = GridLayout(2, 1)
fig[1, 2] = cbar_layout
# terrain_cbar = Colorbar(cbar_layout[1, 1], label="Terrain Elevation (m)", colormap=custom_cmap,
#     limits=(minimum(Z_cart_interp_ext), maximum(Z_cart_interp_ext)))
particle_cbar = Colorbar(cbar_layout[2, 1], label="ΔBᵖ [m s⁻²]", colormap=reverse(cgrad(:RdBu)),
    limits=(-2e-5, 2e-5),
    ticklabelsize=24,
    labelsize=24)
ax.azimuth = 1.55π
ax.elevation = 0.12π
save("output/tilt/particle_positions_final_snapshot.png", fig, dpi=350)