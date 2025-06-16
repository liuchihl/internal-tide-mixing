# this script is to postprocess the particle data: including 1) the particle trajectories, 2) the comparison of buoyancy on each particle between two cases
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
# define a function that loads particle data
function load_particle_data(; simname, z_center_particle=500)
    θ = simname == "tilt" ? 0.0036 : 0
    tᶠ = "460"
    
    filename_particles = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_particles_z=",z_center_particle,".nc")
    ds_particles = Dataset(filename_particles,"r")
    
    return Dict(
        "x" => ds_particles["x"][:,:],
        "y" => ds_particles["y"][:,:],
        "z" => ds_particles["z"][:,:],
        "time" => ds_particles["time"][:],
        "b" => ds_particles["b"][:,:]
    )
end

function calculate_background_buoyancy(θ)
    θ = θ
    ĝ = (sin(θ), 0, cos(θ)) # the vertical unit vector in rotated coordinates
    N = 1e-3
    @inline ẑ(x, z, ĝ) = x*ĝ[1] + z*ĝ[3]
    @inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)

    # Create a background field
    B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

    # Setup grid
    H = 2.25kilometers # vertical extent
    Lx = 15kilometers # along-canyon extent
    Ly = 30kilometers # cross-canyon extent
    Nx = 500
    Ny = 1000
    Nz = 250
    
    # Bottom-intensified stretching for vertical grid
    z_faces(k) = - H * ((1 + ((Nz + 1 - k) / Nz - 1) / 1.2) * 
                        (1 - exp(-15 * (Nz + 1 - k) / Nz)) / (1 - exp(-15)) - 1)

    grid = RectilinearGrid(size=(Nx, Ny, Nz), 
            x = (0, Lx),
            y = (0, Ly), 
            z = z_faces,
            halo = (4, 4, 4),
            topology = (Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded))
            
    model = NonhydrostaticModel(
        grid = grid,
        background_fields = (; b=B̄_field),
        tracers = :b
    )
    
    return interior(compute!(Field(model.background_fields.tracers.b)))[:,1,:]
end
# Load data for both cases
simname = "tilt"    # this is the only thing that needs to be changed
z_center_particle = 500
data = load_particle_data(; simname, z_center_particle=z_center_particle)
θ = simname == "tilt" ? 0.0036 : 0
tᶠ = "460"
Lx = 15kilometers  # Domain length in x
Nx = 500
Ny = 1000
Ly = 30kilometers  # Domain length in y
N = 1e-3           # Buoyancy frequency
Lz = 2.25kilometers  # Vertical extent (assuming Lz = H from your grid)
# load topography 
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
y_interp = range(y_topo[1],y_topo[end], length=Ny)


# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp.-minimum(z_interp)

# plot the particle trajectories

# create an extended topography
dx_interp = x_interp[2]-x_interp[1]
dy_interp = y_interp[2]-y_interp[1]
Lx_interp = x_interp[end]
Ly_interp = y_interp[end]

x_interp_ext = vcat(x_interp[3Nx÷4:Nx].-Lx_interp.-dx_interp, x_interp, x_interp.+ Lx_interp.+dx_interp, x_interp.+2Lx_interp.+dx_interp)
y_interp_ext = vcat(y_interp, y_interp.+ Ly_interp.+dy_interp)
z_interp_ext = vcat(z_interp[3Nx÷4:Nx,:], z_interp, z_interp, z_interp )
z_interp_ext = vcat(z_interp_ext', z_interp_ext')'

## transform the topography into Cartesian coordinates
# create a meshgrid
X_interp_ext = repeat(x_interp_ext, 1, length(y_interp_ext)) 
Y_interp_ext = repeat(y_interp_ext', length(x_interp_ext), 1)
Z_interp_ext = z_interp_ext
# Transform topography to Cartesian coordinates
X_cart_interp_ext = X_interp_ext .* cos(θ) .- Z_interp_ext .* sin(θ)
Y_cart_interp_ext = Y_interp_ext  # y-coordinate is unchanged
Z_cart_interp_ext = X_interp_ext .* sin(θ) .+ Z_interp_ext .* cos(θ)



unwrapped_x = copy(data["x"])  # Unwrapped x-position
unwrapped_y = copy(data["y"])  # Unwrapped y-position
unwrapped_z = copy(data["z"])  # Unwrapped z-position
B = zeros(size(data["x"]))  # Total buoyancy

n_crossings = zeros(Int, size(data["x"]))  # Track number of crossings in x-direction

# background buoyancy
B̄ = calculate_background_buoyancy(θ)
ΔB = N^2 * Lx*sin(θ)      # Buoyancy increment per domain crossing

# define the grids
filename = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_B-c.nc")
ds = Dataset(filename,"r")
zC = ds["zC"][:];
xC = ds["xC"][:]; 


# Adjust initial positions for particles starting on the right side
for i in 1:size(data["x"], 1)
    if data["x"][i, 1] > 3Lx/4
        unwrapped_x[i, 1] = data["x"][i, 1] - Lx
        unwrapped_z[i, 1] = data["z"][i, 1]
        n_crossings[i, 1] = -1  # Particle starts in the "previous" domain
    else
        unwrapped_x[i, 1] = data["x"][i, 1]
        unwrapped_z[i, 1] = data["z"][i, 1]
        n_crossings[i, 1] = 0
    end
end
particle_time = data["time"][:]

# Compute unwrapped positions
for i in 1:size(data["x"], 1)  # Particle ID
    for j in 1:size(data["x"], 2)-1  # Time steps
        # Calculate displacement between consecutive time steps
        dx = data["x"][i, j+1] - data["x"][i, j]
        dy = data["y"][i, j+1] - data["y"][i, j]
        dz = data["z"][i, j+1] - data["z"][i, j]
        
        # Handle periodic crossing in x direction
        if dx > 0.5 * Lx       # Moved left-to-right across boundary
            n_crossings[i, j+1] = n_crossings[i, j] - 1
        elseif dx < -0.5 * Lx  # Moved right-to-left across boundary
            n_crossings[i, j+1] = n_crossings[i, j] + 1
        else
            n_crossings[i, j+1] = n_crossings[i, j]  # No crossing
        end

        # Compute unwrapped x position
        unwrapped_x[i, j+1] = data["x"][i, j+1] + n_crossings[i, j+1] * Lx
        unwrapped_z[i, j+1] = data["z"][i, j+1]
        
        # Handle periodic crossing in y direction
        if dy > 0.5 * Ly  # Moved south-to-north across boundary
            dy -= Ly
        elseif dy < -0.5 * Ly  # Moved north-to-south across boundary
            dy += Ly
        end
        
        # Update unwrapped y position
        unwrapped_y[i, j+1] = unwrapped_y[i, j] + dy

        # Grid indices for background buoyancy
        ind_x = argmin(abs.(xC[:] .- data["x"][i, j]))
        ind_z = argmin(abs.(zC[:] .- data["z"][i, j]))
        
        # Compute total buoyancy: background + perturbation + domain crossing increment
        B[i, j] = (B̄[ind_x, ind_z] + data["b"][i, j] + n_crossings[i, j] * ΔB)
    end
    
    # Handle the last time step
    ind_x = argmin(abs.(xC[:] .- data["x"][i, end]))
    ind_z = argmin(abs.(zC[:] .- data["z"][i, end]))
    B[i, end] = (B̄[ind_x, ind_z] + data["b"][i, end] + n_crossings[i, end] * ΔB)
end
# transform the particle positions to the Cartesian coordinates
# Initialize Cartesian coordinates
x_cart = similar(unwrapped_x)
y_cart = similar(unwrapped_y)
z_cart = similar(unwrapped_z)

# Compute sin and cos of θ for the transformation
θ = 0.0036

# Transform particle positions to Cartesian coordinates
x_cart = unwrapped_x .* cos(θ) .- unwrapped_z .* sin(θ)
y_cart = unwrapped_y
z_cart = unwrapped_x .* sin(θ) .+ unwrapped_z .* cos(θ)


# Create a figure to plot the particle trajectories
using CairoMakie

# Create a 3D figure for topography and particle trajectories with a specific layout
fig = CairoMakie.Figure(resolution = (1200, 800), 
                       layout = GridLayout(1, 2, width_ratios = [0.85, 0.15]))

ax = Axis3(fig[1, 1], 
           xlabel = "X (m)", 
           ylabel = "Y (m)", 
           zlabel = "Z (m)",
           title = "3D Particle Trajectories",
           aspect = (3, 4, 0.6))

# Create a nested layout for the colorbars
cbar_layout = GridLayout(2, 1)
fig[1, 2] = cbar_layout

# Create all static elements (terrain + walls) ONCE outside the animation loop
# Main topography
# Get the full colormap
full_cmap = ColorSchemes.deepsea.colors
# Take only the first 80% of colors
custom_cmap = full_cmap[1:floor(Int, 1*length(full_cmap))]

surface!(ax, X_cart_interp_ext, Y_cart_interp_ext, Z_cart_interp_ext, 
         colormap = custom_cmap,
         shading = NoShading,
         transparency = false)

# Define number of particles to plot
n_particles, n_time_steps = size(x_cart) 

# Animation parameters
time_steps = 1:n_time_steps 
n_frames = 1:length(time_steps)

# Create particle objects list to keep track of what to delete each frame
particle_objects = []

record(fig, string("output/", simname, "/3D_particle_trajectories_animated_extended_buoyancycolor_code_z_center=",z_center_particle,".mp4"), n_frames; framerate = 30) do frame
    # Delete only the previous particles, not the terrain
    for obj in particle_objects
        try
            delete!(ax.scene, obj)
        catch
            # Object may already be gone
        end
    end
    empty!(particle_objects)
    
    # Get current time step
    current_time_step = time_steps[frame]
    # Get z values for current time step to use as color data
    z_colors = B[:, current_time_step]
    
    # Plot current particle positions with depth-based coloring
    particles = scatter!(ax, 
            x_cart[1:n_particles, current_time_step], 
            y_cart[1:n_particles, current_time_step], 
            z_cart[1:n_particles, current_time_step],
            color = z_colors,
            markersize = 5,
            colormap = :YlOrRd_9,
            colorrange = (0.001, 0.0013)
          )
    
    # Add colorbars only once
    if frame == 1
        Colorbar(cbar_layout[1, 1], label="Terrain Elevation", colormap=custom_cmap, 
                 limits=(minimum(Z_cart_interp_ext), maximum(Z_cart_interp_ext)))
        Colorbar(cbar_layout[2, 1], label="Particle Buoyancy", colormap=:YlOrRd_9, 
                 limits=(0.001, 0.0013))
    end
    push!(particle_objects, particles)
    
    # Add trails (previous positions)
    trail_length = 8

    for i in 1:n_particles
            start_idx = max(1, current_time_step - trail_length)
            if start_idx < current_time_step
                # Get the buoyancy value for this particle at the current time step
                particle_color = B[i, current_time_step]
                
                trail = lines!(ax, 
                    x_cart[i, start_idx:current_time_step], 
                    y_cart[i, start_idx:current_time_step], 
                    z_cart[i, start_idx:current_time_step],
                    color = particle_color,  # Use the particle's current buoyancy as color
                    colormap = :YlOrRd_9,    # Same colormap as the particles
                    # color = :gray,
                    colorrange = (0.001, 0.0013),
                    linewidth = 1.5,
                    alpha = 0.6)
                push!(particle_objects, trail)
            end
    end

    # Control camera rotation
    azimuth = range(1.3π, 1.6π, length=length(n_frames))[frame]
    ax.azimuth = azimuth
    
    # Update title to show time
    ax.title = @sprintf("Particle Movement at t = %.2f hours", particle_time[current_time_step]/3600)
    @info current_time_step
end


## tracer weighted buoyancy: comparison with the particle buoyancy





################# compare buoyancy on each particles
# Function to load particle data for a given simulation
# using Oceananigans
# using Oceananigans.Units
# using CairoMakie
# using NCDatasets
# using Statistics
# using NaNStatistics


# simname = "tilt" 
# θ = θ = simname == "tilt" ? 0.0036 : 0
# tᶠ = "460"

# # Function to calculate background buoyancy field for a given slope
# function calculate_background_buoyancy(θ)
#     θ = θ
#     ĝ = (sin(θ), 0, cos(θ)) # the vertical unit vector in rotated coordinates
#     N = 1e-3
#     @inline ẑ(x, z, ĝ) = x*ĝ[1] + z*ĝ[3]
#     @inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)

#     # Create a background field
#     B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

#     # Setup grid
#     H = 2.25kilometers # vertical extent
#     Lx = 15kilometers # along-canyon extent
#     Ly = 30kilometers # cross-canyon extent
#     Nx = 500
#     Ny = 1000
#     Nz = 250
    
#     # Bottom-intensified stretching for vertical grid
#     z_faces(k) = - H * ((1 + ((Nz + 1 - k) / Nz - 1) / 1.2) * 
#                         (1 - exp(-15 * (Nz + 1 - k) / Nz)) / (1 - exp(-15)) - 1)

#     grid = RectilinearGrid(size=(Nx, Ny, Nz), 
#            x = (0, Lx),
#            y = (0, Ly), 
#            z = z_faces,
#            halo = (4, 4, 4),
#            topology = (Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded))
           
#     model = NonhydrostaticModel(
#         grid = grid,
#         background_fields = (; b=B̄_field),
#         tracers = :b
#     )
    
#     return interior(compute!(Field(model.background_fields.tracers.b)))[:,1,:]
# end

# # define a function that loads particle data
# function load_particle_data(; simname,depth=depth)
#     θ = simname == "tilt" ? 0.0036 : 0
#     tᶠ = "460"
    
#     filename_particles = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_particles_z=",Int(depth),".nc")
#     ds_particles = Dataset(filename_particles,"r")
    
#     return Dict(
#         "x" => ds_particles["x"][:,:],
#         "y" => ds_particles["y"][:,:],
#         "z" => ds_particles["z"][:,:],
#         "time" => ds_particles["time"][:],
#         "b" => ds_particles["b"][:,:]
#     )
# end

# # load background buoyancy fields for both cases
# B̄_tilt = calculate_background_buoyancy(0.0036)
# B̄_flat = calculate_background_buoyancy(0.0)

# # Load data for both cases
# depth = 1000;
# tilt_data = load_particle_data(; simname="tilt", depth=depth)
# flat_data = load_particle_data(; simname="flat", depth=depth)


# filename = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_B-c.nc")
# ds = Dataset(filename,"r")
# zC = ds["zC"][:]; Nz=length(zC);
# xC = ds["xC"][:]; Nx=length(xC)
# # find the index of each particles in the grid and compute total buoyancy
# # Constants from your grid setup
# Lx = 15kilometers  # Domain length in x
# Ly = 30kilometers  # Domain length in y
# N = 1e-3           # Buoyancy frequency
# Lz = 2.25kilometers  # Vertical extent (assuming Lz = H from your grid)
# θ = 0.0036
# ΔB = N^2 * Lx*sin(θ)      # Buoyancy increment per domain crossing

# # Initialize arrays
# B_tilt = zeros(size(tilt_data["x"]))  # Total buoyancy
# unwrapped_x = zeros(size(tilt_data["x"]))  # Unwrapped x-position
# unwrapped_y = zeros(size(tilt_data["y"]))  # Unwrapped y-position
# unwrapped_z = zeros(size(tilt_data["z"]))  # Unwrapped y-position
# n_crossings = zeros(Int, size(tilt_data["x"]))  # Number of domain crossings

# # Set initial unwrapped positions (at t=1)
# unwrapped_x[:, 1] = tilt_data["x"][:, 1]
# unwrapped_y[:, 1] = tilt_data["y"][:, 1]
# unwrapped_z[:, 1] = tilt_data["z"][:, 1]
# # Compute unwrapped positions and buoyancy over time
# for i in 1:size(tilt_data["x"], 1)  # Particle ID
#     for j in 1:size(tilt_data["x"], 2)-1  # Time steps
#         # Calculate displacement between consecutive time steps
#         dx = tilt_data["x"][i, j+1] - tilt_data["x"][i, j]
#         dy = tilt_data["y"][i, j+1] - tilt_data["y"][i, j]
#         dz = tilt_data["z"][i, j+1] - tilt_data["z"][i, j]
        
#         # Handle periodic crossing in x direction
#         if dx > 0.5 * Lx  # Moved left-to-right across boundary
#             dx -= Lx
#             dz -= Lx*sin(θ)
#             n_crossings[i, j+1] = n_crossings[i, j] - 1
#         elseif dx < -0.5 * Lx  # Moved right-to-left across boundary
#             dx += Lx
#             dz += Lx*sin(θ)
#             n_crossings[i, j+1] = n_crossings[i, j] + 1
#         else
#             n_crossings[i, j+1] = n_crossings[i, j]  # No crossing
#         end
        
#         # Handle periodic crossing in y direction
#         if dy > 0.5 * Ly  # Moved across boundary in y-direction
#             dy -= Ly
#         elseif dy < -0.5 * Ly
#             dy += Ly
#         end
        
#         # Update unwrapped positions
#         unwrapped_x[i, j+1] = unwrapped_x[i, j] + dx
#         unwrapped_y[i, j+1] = unwrapped_y[i, j] + dy
#         unwrapped_z[i, j+1] = unwrapped_z[i, j] + dz
        
#         # Grid indices for background buoyancy
#         ind_tilt_x = argmin(abs.(xC[:] .- tilt_data["x"][i, j]))
#         ind_tilt_z = argmin(abs.(zC[:] .- tilt_data["z"][i, j]))
        
#         # Compute total buoyancy: background + perturbation + domain crossing increment
#         B_tilt[i, j] = (B̄_tilt[ind_tilt_x, ind_tilt_z] + 
#                         tilt_data["b"][i, j] + 
#                         n_crossings[i, j] * ΔB)
#     end
    
#     # Handle the last time step
#     ind_tilt_x = argmin(abs.(xC[:] .- tilt_data["x"][i, end]))
#     ind_tilt_z = argmin(abs.(zC[:] .- tilt_data["z"][i, end]))
#     B_tilt[i, end] = (B̄_tilt[ind_tilt_x, ind_tilt_z] + 
#                       tilt_data["b"][i, end] + 
#                       n_crossings[i, end] * ΔB)
# end

# # For flat case (no slope, so no ΔB increment needed)
# B_flat = zeros(size(flat_data["x"]))
# for i in 1:size(flat_data["x"], 1)
#     for j in 1:size(flat_data["x"], 2)
#         ind_flat_x = argmin(abs.(xC[:] .- flat_data["x"][i, j]))
#         ind_flat_z = argmin(abs.(zC[:] .- flat_data["z"][i, j]))
#         B_flat[i, j] = B̄_flat[ind_flat_x, ind_flat_z] + flat_data["b"][i, j]
#     end
# end

# ## filter out the particles 
# # 1) not moving
# # 2) moves away into the northern channel
# # Find particles that get stuck (don't move) at any point during the simulation
# stuck_particles = findall(vec(any(abs.(diff(unwrapped_y, dims=2)) .< 1e-6, dims=2)))
# # Find particles that move north beyond the domain
# north_moving_particles = findall(vec(unwrapped_y[:,end] .> Ly))
# # Combine both filters to get all particles to exclude
# exclude_particles = union(stuck_particles, north_moving_particles)

# # Create a mask for valid particles (those we want to keep)
# valid_particles = setdiff(1:size(B_tilt, 1), exclude_particles)

# # Use only valid particles for buoyancy calculations
# B_tilt_filtered = B_tilt[valid_particles, :]
# # B_flat_filtered = B_flat[valid_particles, :]

# # Calculate mean total buoyancy for each case
# tilt_mean_total_B = nanmean(B_tilt_filtered, dims=1)[1, :]
# flat_mean_total_B = nanmean(B_flat, dims=1)[1, :]
# # Plot time in hours
# tilt_time = tilt_data["time"] / 3600
# flat_time = flat_data["time"] / 3600
# # Create figure with two subplots to compare total buoyancy evolution
# # empty!(fig)
# fig = Figure(resolution=(1200, 600), fontsize=20)
# using Printf

# ax1 = Axis(fig[1, 1], 
#           xlabel="Time (hours)", 
#           ylabel="Particle-averaged total buoyancy", 
#           title="Tilt Case (θ=0.0036)", 
#           limits = ((tilt_time[1], tilt_time[end]), nothing),
#           xminorticksvisible = true,
#           yminorticksvisible = true,
#           xminorticks = IntervalsBetween(5),
#           yminorticks = IntervalsBetween(4))

# ax2 = Axis(fig[1, 2], 
#           xlabel="Time (hours)", 
#           title="Flat Case (θ=0)", 
#           limits = ((tilt_time[1],tilt_time[end]),nothing),
#           xminorticksvisible = true,
#           yminorticksvisible = true,
#           xminorticks = IntervalsBetween(5),
#           yminorticks = IntervalsBetween(4))

# # Set consistent scientific notation format for y-axis ticks on both plots
# for ax in [ax1, ax2]
#     ax.ytickformat = x -> [@sprintf("%.5f", v) for v in x]  # Increase precision to 5 decimal places
# end

# # Calculate standard deviations for each case
# tilt_std_B = nanstd(B_tilt_filtered, dims=1)[1, :]
# flat_std_B = nanstd(B_flat, dims=1)[1, :]

# # Tilt case - first subplot
# band!(ax1, tilt_time, tilt_mean_total_B .- tilt_std_B, tilt_mean_total_B .+ tilt_std_B, 
#     color=(Makie.RGB(1,0,0), 0.3), label="Standard deviation")
# lines!(ax1, tilt_time, tilt_mean_total_B, 
#      linewidth=3, color=:red, label="Mean buoyancy")

# # Flat case - second subplot
# band!(ax2, flat_time, flat_mean_total_B .- flat_std_B, flat_mean_total_B .+ flat_std_B, 
#     color=(Makie.RGB(0,0,1), 0.3), label="Standard deviation")
# lines!(ax2, flat_time, flat_mean_total_B, 
#      linewidth=3, color=:blue, label="Mean buoyancy")

# # Add legend to both subplots
# # axislegend(ax1, position=:lt)
# # axislegend(ax2, position=:lt)

# # Link y-axes for better comparison
# # linkaxes!(ax1, ax2)

# # Save the figure
# save(string("output/comparison_mean_particle_total_buoyancy_subplots_z=",depth,".png"), fig)



# Plot the topography




# Create a histogram of tracer concentration values
# function plot_concentration_histogram(c_data)
#     # Flatten the array and remove NaN values
#     c_flat = filter(!isnan, c_data[:])
    
#     # Create histogram figure
#     hist_fig = Figure(resolution = (800, 600))
#     hist_ax = Axis(hist_fig[1, 1], 
#                   xlabel = "Concentration",
#                   ylabel = "Frequency",
#                   title = "Distribution of Tracer Concentration")
    
#     # Calculate appropriate bin range based on data
#     # Define the bin range
#     min_val = -0.05
#     max_val = 0.05
    
#     # Create bins with explicit range
#     bin_edges = range(min_val, max_val, length=41)  # 41 edges for 40 bins
    
#     # Plot histogram
#     hist!(hist_ax, c_flat, bins = bin_edges)
    
#     # Add vertical lines at important values
#     vlines!(hist_ax, 0, color = :red, linewidth = 2, label = "c = 0")
#     # vlines!(hist_ax, 1, color = :blue, linewidth = 2, label = "c = 1")
    
#     # Add legend
#     axislegend(hist_ax)
    
#     return hist_fig
# end

# # Create and save histogram
# hist_figure = plot_concentration_histogram(c[:, 1, :, :])
# save(string("output/",simname,"/tracer_concentration_histogram.png"), hist_figure)



# c[c.<=0] .= 1e-10
# fig = Figure(resolution = (1200, 800))
# ax = Axis(fig[1, 1], 
#     xlabel = "Distance (m)",
#     ylabel = "Depth (m)",
#     title = "Tracer Concentration and particles")

# # Create a observable for the time index and data
# time_idx = Observable(1)
# c_data = @lift(log10.(c[:, 1, 1:160, $time_idx]))
# u_data = @lift(u_cen[1:10:end, 1, 1:6:160, $time_idx])
# w_data = @lift(1*w[1:10:end, 1, 1:6:160, $time_idx])

# # Create the heatmap plot
# hm = heatmap!(ax, xC, zC[1:160], c_data, 
#     colormap = :OrRd_9,
#     colorrange = (-3,0),
#     nan_color = :grey)

# # Add colorbar
# Colorbar(fig[1, 2], hm, label = "Concentration")
# # Extract particles at specific time
# # x_particles and z_particles have shape (particle_id, time)
# particle_x = @lift begin
#     time_index = min($time_idx, size(x_particles, 2))
#     x_particles[1:5000, time_index]
# end

# particle_z = @lift begin
#     time_index = min($time_idx, size(z_particles, 2))
#     z_particles[1:5000, time_index]
# end

# # Plot particles as scatter points
# particles = scatter!(ax, particle_x, particle_z, 
#     color = :green, 
#     markersize = 4.0,
#     marker = :circle,
#     alpha = 0.9)  # Adjust alpha between 0 (fully transparent) and 1 (fully opaque)

#     # Add quiver plot (flow arrows)
# arrows!(ax, 
#     xC[1:10:end], zC[1:6:160],
#     u_data, w_data,
#     arrowsize = 7,
#     lengthscale = 2000,
#     linewidth = 1.0,
#     color = :black)
# # Create and save the animation
# # Animation parameters
# fps = 10
# times = 1:2:length(t)
# record(fig, string("output/",simname,"/tracer_particle_animation_withoutnegativevalues.mp4"), times; framerate = fps) do i
#     time_idx[] = i
#     ax.title = @sprintf("Tracer Concentration and Particles at t = %.2f hr", t[i]/3600)
#     @info "Frame $i"
# end