#this is wrong for some reason
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
n_main = 499574
Lx = 15kilometers
Nx = 500
Ny = 1000
Ly = 30kilometers

# File time lists
tᶠ_all = ["451.5", "452.0", "452.5", "453.0", "453.5", "454.0", "454.5", "455.0", "455.5", "456.0"]

# --- First, figure out total number of timesteps ---
@info "Calculating total timesteps..."
global total_timesteps = 0
timesteps_per_file = []

for tᶠ in tᶠ_all
    data = load_particle_data(; simname, z_center_particle=1000, tᶠ=tᶠ)
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
main_buoyancy = similar(main_x)
main_time = Vector{Float64}(undef, total_timesteps)

# --- Load and concatenate data ---
@info "Loading and concatenating all particle data..."

global current_idx = 1

for tᶠ in tᶠ_all
    data = load_particle_data(; simname, z_center_particle=1000, tᶠ=tᶠ)
    n_steps = length(data["time"])
    idx_range = current_idx:(current_idx+n_steps-1)

    main_x[:, idx_range] = data["x"][1:n_main, :]
    main_y[:, idx_range] = data["y"][1:n_main, :]
    main_z[:, idx_range] = data["z"][1:n_main, :]
    main_buoyancy[:, idx_range] = data["b"][1:n_main, :]
    main_time[idx_range] = data["time"][:]

    global current_idx += n_steps
end

# --- Unwrap x and y only ---
function unwrap_xy!(x, y, Lx, Ly)
    n_particles, n_times = size(x)
    unwrapped_x = copy(x)
    unwrapped_y = copy(y)
    for i in 1:n_particles
        for j in 2:n_times
            dx = x[i, j] - x[i, j-1]
            if dx > 0.5 * Lx
                dx -= Lx
            elseif dx < -0.5 * Lx
                dx += Lx
            end
            unwrapped_x[i, j] = unwrapped_x[i, j-1] + dx

            dy = y[i, j] - y[i, j-1]
            if dy > 0.5 * Ly
                dy -= Ly
            elseif dy < -0.5 * Ly
                dy += Ly
            end
            unwrapped_y[i, j] = unwrapped_y[i, j-1] + dy
        end
    end
    return unwrapped_x, unwrapped_y
end
@info "Unwrapping x and y positions for flat bottom..."
main_x_unwrapped, main_y_unwrapped = unwrap_xy!(main_x, main_y, Lx, Ly)

# --- Get grid coordinates ---
first_file = string("output/", simname, "/internal_tide_theta=0.0_Nx=500_Nz=250_tᶠ=", tᶠ_all[1], "_analysis_round=all_particles_z=1000.nc")
ds_grid = Dataset(first_file, "r")
xC = ds_grid["x_caa"][:]
zC = ds_grid["z_aac"][:]
close(ds_grid)

# --- Calculate background buoyancy field ---
@info "Calculating background buoyancy field for flat case..."
B̄ = calculate_background_buoyancy(0) # size (Nx, Nz)

# --- Add background buoyancy to each particle ---
@info "Adding background buoyancy to each particle..."
main_buoyancy_total = similar(main_buoyancy)
for i in 1:n_main
    for j in 1:total_timesteps
        ind_x = argmin(abs.(xC .- main_x[i, j]))
        ind_z = argmin(abs.(zC .- main_z[i, j]))
        main_buoyancy_total[i, j] = B̄[ind_x, ind_z] + main_buoyancy[i, j]
    end
end

# --- Save to NetCDF ---
z_center_particle = 1000
output_file = string("output/", simname, "/concatenated_particle_data_z", z_center_particle, "_all.nc")
@info "Saving all particles to $output_file"
ds_output = NCDataset(output_file, "c")

ds_output.dim["main_particle"] = n_main
ds_output.dim["time"] = total_timesteps

defVar(ds_output, "main_x_unwrapped", Float64, ("main_particle", "time"))[:] = main_x_unwrapped
defVar(ds_output, "main_y_unwrapped", Float64, ("main_particle", "time"))[:] = main_y_unwrapped
defVar(ds_output, "main_z", Float64, ("main_particle", "time"))[:] = main_z
defVar(ds_output, "main_buoyancy", Float64, ("main_particle", "time"))[:] = main_buoyancy
defVar(ds_output, "main_buoyancy_total", Float64, ("main_particle", "time"))[:] = main_buoyancy_total
defVar(ds_output, "main_time", Float64, ("time",))[:] = main_time

close(ds_output)
@info "✓ All particles saved successfully to $output_file"