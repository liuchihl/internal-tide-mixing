cd("/scratch/bcpi/cliu28/internal-tide-mixing/")
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using LinearAlgebra
using MAT
using Interpolations
using ColorSchemes

# --- Utility Functions ---

function load_particle_data(; simname, z_center_particle=1000, tᶠ="457.0")
    θ = simname == "tilt" ? 0.0036 : 0
    filename_particles = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_analysis_round=all_particles_z=", z_center_particle, ".nc")
    ds_particles = Dataset(filename_particles, "r")
    return Dict(
        "x" => ds_particles["x"][:, :],
        "y" => ds_particles["y"][:, :],
        "z" => ds_particles["z"][:, :],
        "time" => ds_particles["time"][:],
        "b" => ds_particles["b"][:, :]
    )
end

function calculate_background_buoyancy(θ)
    θ = θ
    ĝ = (sin(θ), 0, cos(θ))
    N = 1e-3
    @inline ẑ(x, z, ĝ) = x * ĝ[1] + z * ĝ[3]
    @inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
    B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N²=N^2))
    H = 2.25kilometers
    Lx = 15kilometers
    Ly = 30kilometers
    Nx = 500
    Ny = 1000
    Nz = 250
    z_faces(k) = -H * ((1 + ((Nz + 1 - k) / Nz - 1) / 1.2) * (1 - exp(-15 * (Nz + 1 - k) / Nz)) / (1 - exp(-15)) - 1)
    grid = RectilinearGrid(size=(Nx, Ny, Nz),
        x=(0, Lx),
        y=(0, Ly),
        z=z_faces,
        halo=(4, 4, 4),
        topology=(Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded))
    model = NonhydrostaticModel(
        grid=grid,
        background_fields=(; b=B̄_field),
        tracers=:b
    )
    return interior(compute!(Field(model.background_fields.tracers.b)))[:, 1, :]
end

# --- Main Processing ---

simname = "flat"
z_center_particle = 1000
θ = 0
Lx = 15kilometers
Nx = 500
Ny = 1000
Ly = 30kilometers
N = 1e-3
Lz = 2.25kilometers

# File time lists
tᶠ_all = ["451.5", "452.0", "452.5", "453.0", "453.5", "454.0", "454.5", "455.0", "455.5", "456.0"]
# Particle counts
n_main = 499574

# Load grid data for buoyancy calculation
@info "Loading grid data for buoyancy calculation..."
filename = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ_all[1], "_analysis_round=all_threeD.nc")
ds = Dataset(filename, "r")
zC = ds["z_aac"][:];
xC = ds["x_caa"][:];
close(ds)

# Background buoyancy
B̄ = calculate_background_buoyancy(θ)
ΔB = N^2 * Lx * sin(θ)

# --- First, figure out total number of timesteps ---
@info "Calculating total timesteps..."
global total_timesteps = 0
timesteps_per_file = []

for tᶠ in tᶠ_all
    data = load_particle_data(; simname, z_center_particle=z_center_particle, tᶠ=tᶠ)
    n_steps = length(data["time"])
    push!(timesteps_per_file, n_steps)
    global total_timesteps += n_steps
    @info "File $tᶠ: $n_steps timesteps"
end

@info "Total timesteps: $total_timesteps"


# --- Preallocate arrays for all particles and timesteps ---
main_x = Array{Float64}(undef, n_main, total_timesteps)
main_y = similar(main_x)
main_z = similar(main_x)
main_b = similar(main_x)
main_time = Vector{Float64}(undef, total_timesteps)

# --- Load and concatenate data ---
@info "Loading and concatenating all particle data..."

global current_idx = 1

for (file_idx, tᶠ) in enumerate(tᶠ_all)
    @info "Loading file $file_idx/$length(tᶠ_all): $tᶠ"
    data = load_particle_data(; simname, z_center_particle=z_center_particle, tᶠ=tᶠ)

    n_steps = length(data["time"])
    idx_range = current_idx:(current_idx+n_steps-1)

    # Main particles (always load first n_main)
    main_x[:, idx_range] = data["x"][1:n_main, :]
    main_y[:, idx_range] = data["y"][1:n_main, :]
    main_z[:, idx_range] = data["z"][1:n_main, :]
    main_b[:, idx_range] = data["b"][1:n_main, :]
    main_time[idx_range] = data["time"][:]

    global current_idx += n_steps
end

# --- Unwrap and adjust buoyancy for each set independently ---

function unwrap_and_adjust!(x, y, z, b, B̄, ΔB, xC, zC, Lx, Ly, θ)
    n_particles, n_times = size(x)
    unwrapped_x = copy(x)
    unwrapped_y = copy(y)
    unwrapped_z = copy(z)
    B = zeros(n_particles, n_times)

    @info "Processing $n_particles particles across $n_times timesteps..."

    for i in 1:n_particles
        if i % 50000 == 0
            @info "Processing particle $i/$n_particles"
        end

        # Unwrap x
        for j in 2:n_times
            dx = x[i, j] - x[i, j-1]
            if dx > 0.5 * Lx
                dx -= Lx
            elseif dx < -0.5 * Lx
                dx += Lx
            end
            unwrapped_x[i, j] = unwrapped_x[i, j-1] + dx
        end

        # Unwrap y
        for j in 2:n_times
            dy = y[i, j] - y[i, j-1]
            if dy > 0.5 * Ly
                dy -= Ly
            elseif dy < -0.5 * Ly
                dy += Ly
            end
            unwrapped_y[i, j] = unwrapped_y[i, j-1] + dy
        end

        # Tilt correction for z
        for j in 1:n_times
            x_domains_moved = (unwrapped_x[i, j] - unwrapped_x[i, 1]) / Lx
            unwrapped_z[i, j] = z[i, j] + x_domains_moved * Lx * sin(θ)
        end

        # Buoyancy
        for j in 1:n_times
            ind_x = argmin(abs.(xC[:] .- x[i, j]))
            ind_z = argmin(abs.(zC[:] .- z[i, j]))
            n_crossings_x = round(Int, (unwrapped_x[i, j] - x[i, j]) / Lx)
            B[i, j] = B̄[ind_x, ind_z] + b[i, j] + n_crossings_x * ΔB
        end
    end

    # Cartesian transform
    x_cart = unwrapped_x .* cos(θ) .- unwrapped_z .* sin(θ)
    y_cart = unwrapped_y
    z_cart = unwrapped_x .* sin(θ) .+ unwrapped_z .* cos(θ)
    return unwrapped_x, unwrapped_y, unwrapped_z, x_cart, y_cart, z_cart, B
end

@info "Unwrapping and adjusting main particles..."
main_unwrap_x, main_unwrap_y, main_unwrap_z, main_x_cart, main_y_cart, main_z_cart, main_B = unwrap_and_adjust!(main_x, main_y, main_z, main_b, B̄, ΔB, xC, zC, Lx, Ly, θ)

# --- Save to NetCDF ---

output_file = string("output/", simname, "/concatenated_particle_data_z_flat_", z_center_particle, "_all.nc")
@info "Saving all particles to $output_file"
ds_output = NCDataset(output_file, "c")

# Dimensions
ds_output.dim["main_particle"] = n_main
ds_output.dim["time"] = total_timesteps

# Main particles (1:499774, all times)
defVar(ds_output, "main_x_unwrapped", Float64, ("main_particle", "time"))[:] = main_unwrap_x
defVar(ds_output, "main_y_unwrapped", Float64, ("main_particle", "time"))[:] = main_unwrap_y
defVar(ds_output, "main_z_unwrapped", Float64, ("main_particle", "time"))[:] = main_unwrap_z
defVar(ds_output, "main_x_cartesian", Float64, ("main_particle", "time"))[:] = main_x_cart
defVar(ds_output, "main_y_cartesian", Float64, ("main_particle", "time"))[:] = main_y_cart
defVar(ds_output, "main_z_cartesian", Float64, ("main_particle", "time"))[:] = main_z_cart
defVar(ds_output, "main_buoyancy", Float64, ("main_particle", "time"))[:] = main_B
defVar(ds_output, "main_time", Float64, ("time",))[:] = main_time

# Attributes
ds_output.attrib["title"] = "Concatenated Particle Data (main: 1-499774, new: 499775-524774)"
ds_output.attrib["simulation"] = simname
ds_output.attrib["z_center_particle"] = z_center_particle
ds_output.attrib["theta"] = θ
ds_output.attrib["Lx"] = Lx
ds_output.attrib["Ly"] = Ly
ds_output.attrib["N"] = N
ds_output.attrib["time_range"] = "$(tᶠ_all[1]) to $(tᶠ_all[end])"
ds_output.attrib["main_particles"] = n_main
ds_output.attrib["main_time_steps"] = total_timesteps

close(ds_output)
@info "✓ All particles saved successfully to $output_file"