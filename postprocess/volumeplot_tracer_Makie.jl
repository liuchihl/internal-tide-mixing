using NCDatasets
using WGLMakie
using WGLMakie: Vec3f0, FRect3D
using FileIO
using MAT
using Statistics
using Interpolations

# --- Parameters ---
output_dir = "output/tilt/"
all_times = [461.5, 462.0]  # Example times, update as needed
isos = Float32[1e-1, 1e-2, 1e-3]  # Isosurface values
framerate = 5

# --- Load grid and topography ---
first_file = string(output_dir, "internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=", all_times[1], "_analysis_round=all_threeD.nc")
ds_temp = Dataset(first_file, "r")
xC = ds_temp["x_caa"][:]
yC = ds_temp["y_aca"][:]
zC = ds_temp["z_aac"][:]
close(ds_temp)

# Topography from .mat
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic")
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
close(file)
Nx, Ny = 500, 1000
θ = 0.0036
x_interp = range(x_topo[1], x_topo[end], length=Nx)
y_interp = range(y_topo[1], y_topo[end], length=Ny)
itp = LinearInterpolation((range(x_topo[1], x_topo[end], size(z_topo, 1)),
        range(y_topo[1], y_topo[end], size(z_topo, 2))), z_topo)
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp .- minimum(z_interp)
X_topo = repeat(x_interp, 1, length(y_interp))
Y_topo = repeat(y_interp', length(x_interp), 1)
Z_topo = z_interp
X_topo_cart = X_topo .* cos(θ) .- Z_topo .* sin(θ)
Y_topo_cart = Y_topo
Z_topo_cart = X_topo .* sin(θ) .+ Z_topo .* cos(θ)

# --- Set up Makie scene ---
fig = Figure(size=(1000, 800))
ax = Axis3(fig[1, 1], perspectiveness=0.8, aspect=(3, 4, 0.6))
surface_plot_ref = Observable(Vector{Any}())
tracer_frame = Observable(zeros(Float32, length(xC), length(yC), length(zC)))

# Function to update isosurfaces
function update_volume!(ax, volume, surface_plot_ref)
    # Remove old volume plot if present
    if !isempty(surface_plot_ref[])
        Makie.delete!(ax, surface_plot_ref[][1])
        empty!(surface_plot_ref[])
    end
    # Add new volume plot
    volplot = volume!(ax, volume;
        algorithm=:absorption,   # or :mip for max intensity
        colormap=:viridis,
        transparency=true,
        alpha=1,
        colorrange=(-6, -1)
    )

    push!(surface_plot_ref[], volplot)
end

# --- Add topography surface (semi-transparent) ---
topo = surface!(ax, x_interp, y_interp, Z_topo_cart; color=:gray, shading=true, alpha=1)

# Camera and axis settings
hidedecorations!(ax)
ax.azimuth = 4π / 3
# ax.elevation = π / 6
# ax.limits = (
#     (minimum(xC), minimum(yC), minimum(zC)),
#     (maximum(xC), maximum(yC), maximum(zC))
# )

# --- Animation loop ---
for t_idx in 1:length(all_times)
    # Load tracer data for this frame
    tval = all_times[t_idx]
    data_file = string(output_dir, "internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=", tval, "_analysis_round=all_threeD.nc")
    ds = Dataset(data_file, "r")
    # Create a downsampled version for performance
    downsample = 10
    tracer = ds["c"][1:downsample:end, 1:downsample:end, 1:downsample:end, t_idx]
    tracer[tracer.<1e-6] .= 1e-10
    close(ds)
    update_volume!(ax, log10.(tracer), surface_plot_ref)
    ax.title = "Time = $tval"
    png_path = joinpath(output_dir, "frames", "frame_$(lpad(t_idx, 4, '0')).png")
    mkpath(dirname(png_path))
    save(png_path, fig)
    println("Saved $png_path")
    println("Tracer min/max: ", minimum(tracer), " / ", maximum(tracer))
    println("Tracer log10 min/max: ", minimum(log10.(tracer)), " / ", maximum(log10.(tracer)))
    using StatsBase
    println("Tracer value histogram: ", countmap(round.(log10.(tracer), digits=1)))
    println("Tracer size: ", size(tracer))
end
tracer = tracer_frame[]
fig = Figure()
ax = Axis3(fig[1, 1])
volume!(ax, log10.(tracer); algorithm=:absorption, colormap=:viridis, transparency=true, alpha=1, colorrange=(-6, -1))
save("output/tilt/final_volume.png", fig)
display(fig)