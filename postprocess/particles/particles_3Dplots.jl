# cd("scratch/bcpi/cliu28/internal-tide-mixing/")
using MAT
using NCDatasets
using NaNStatistics
using Interpolations
using CairoMakie
using ColorSchemes
using Printf
# load topography and transform it into Cartesian coordinates
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
Nx = 500
Ny = 1000
θ=0.0036
x_interp = range(x_topo[1],x_topo[end], length=Nx)
y_interp = range(y_topo[1],y_topo[end], length=Ny)

# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp.-minimum(z_interp)

# create an extended topography
dx_interp = x_interp[2]-x_interp[1]
dy_interp = y_interp[2]-y_interp[1]
Lx_interp = x_interp[end]
Ly_interp = y_interp[end]

x_interp_ext = vcat(x_interp[3Nx÷4:Nx].-Lx_interp.-dx_interp, x_interp, x_interp.+ Lx_interp.+dx_interp)
y_interp_ext = vcat(y_interp, y_interp.+ Ly_interp.+dy_interp)
z_interp_ext = vcat(z_interp[3Nx÷4:Nx,:], z_interp, z_interp )
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


# Now plot the animation using the saved data
simname = "tilt"  
z_center_particle = 1000  # Center height of particles
output_file = string("output/", simname, "/concatenated_particle_data_z", z_center_particle, ".nc")
@info "Loading concatenated data for animation..."
ds = NCDataset(output_file, "r")
x_cart_full = ds["x_cartesian"][:,:]
y_cart_full = ds["y_cartesian"][:,:]
z_cart_full = ds["z_cartesian"][:,:]
B_full = ds["buoyancy"][:,:]
time_full = ds["time"][:]
close(ds)

# Create a 4-panel figure for topography and particle trajectories
fig = CairoMakie.Figure(resolution = (1000, 700))

# Define the four evenly distributed time steps
n_particles, n_time_steps = size(x_cart_full)
time_indices = [1, div(n_time_steps, 3), div(2*n_time_steps, 3), n_time_steps]

# Calculate tidal period (assuming 12.42 hours for M2 tidal period)
T_tidal = 2*pi/1.4e-4  # Tidal period in seconds

# Set consistent camera angle for all subplots
camera_azimuth = 1.6π
camera_elevation = 0.15π

# Create the four subplots
axes = []
for i in 1:4
    ax = Axis3(fig[div(i-1, 2)+1, mod(i-1, 2)+1], 
               xlabel = "x̂ [km]", 
               ylabel = "y [km]", 
               zlabel = "ẑ [m]",
               aspect = (3.5, 4, 2),
               limits = ((0, 30e3), (0, 40e3), (0, 1500))
               )
    
    # Set consistent camera angle
    ax.azimuth = camera_azimuth
    ax.elevation = camera_elevation
    
    push!(axes, ax)
end

# Create terrain colormap
full_cmap = ColorSchemes.terrain.colors
custom_cmap = full_cmap[1:floor(Int, 1*length(full_cmap))]

# MEMORY OPTIMIZATION: Subsample particles and trails
particle_subsample = 1:1:n_particles  # Use every 5th particle
trail_subsample = 1:10:n_particles    # Use every 10th particle for trails
trail_length = 8

# Plot terrain and particles for each time step
for (subplot_idx, time_idx) in enumerate(time_indices)
    ax = axes[subplot_idx]
    
    # Plot the terrain surface (use alpha to make it less memory intensive)
    # surface!(ax, X_cart_interp_ext, Y_cart_interp_ext, Z_cart_interp_ext, 
    #          colormap = :terrain,
    #          shading = NoShading,
    #          transparency = false,
    #          alpha = 0.8,
    #          colorrange = (0, 1500)) 
    
    # OPTIMIZED: Plot only subsampled particles
    particles = scatter!(ax, 
            x_cart_full[particle_subsample, time_idx], 
            y_cart_full[particle_subsample, time_idx], 
            z_cart_full[particle_subsample, time_idx],
            color = B_full[particle_subsample, time_idx],
            markersize = 5,
            colormap = reverse(cgrad(:RdBu)),
            colorrange = (0.001, 0.0013)
          )
    
    # OPTIMIZED: Create trails using vectorized approach
    # start_idx = max(1, time_idx - trail_length)
    # if start_idx < time_idx
    #     # Pre-allocate trail arrays
    #     trail_x = Vector{Float64}()
    #     trail_y = Vector{Float64}()
    #     trail_z = Vector{Float64}()
    #     trail_colors = Vector{Float64}()
        
    #     # Build trail data efficiently
    #     for i in trail_subsample
    #         for j in start_idx:(time_idx-1)
    #             # Add line segment points
    #             push!(trail_x, x_cart_full[i, j])
    #             push!(trail_y, y_cart_full[i, j])
    #             push!(trail_z, z_cart_full[i, j])
    #             push!(trail_colors, B_full[i, time_idx])
                
    #             push!(trail_x, x_cart_full[i, j+1])
    #             push!(trail_y, y_cart_full[i, j+1])
    #             push!(trail_z, z_cart_full[i, j+1])
    #             push!(trail_colors, B_full[i, time_idx])
    #         end
    #     end
        
    #     # Create single trail object instead of many
    #     if !isempty(trail_x)
    #         linesegments!(ax, 
    #             Point3f.(zip(trail_x, trail_y, trail_z)),
    #             color = trail_colors,
    #             colormap = :YlOrRd_9,
    #             colorrange = (0.001, 0.0013),
    #             linewidth = 1.5,
    #             alpha = 0.6)
    #     end
    # end
    
    # Calculate time in tidal periods
    current_time_seconds = time_full[time_idx]
    tidal_periods = current_time_seconds / T_tidal
    
    # Set title with time in tidal periods
    ax.title = @sprintf("t = %.2f tidal period", tidal_periods)
    ax.xticks = ([0, 10, 20, 30].*1e3, ["0", "10", "20", "30"])
    ax.yticks = ([0, 10, 20, 30, 40, 50, 60].*1e3, ["0", "10", "20", "30", "40", "50", "60"])
    @info "Plotted subplot $subplot_idx at time step $time_idx (t = %.2f T)" tidal_periods
    
    # MEMORY CLEANUP: Force garbage collection after each subplot
    GC.gc()
end

# Add thin colorbars to the right of the fourth subplot
# Create a layout for colorbars
cbar_layout = GridLayout(2, 1, height = Relative(0.6))
fig[1:2, 3] = cbar_layout

# Terrain colorbar
Colorbar(cbar_layout[1, 1], 
         label="Terrain Elevation [m]", 
         colormap=custom_cmap, 
         limits=(0, 1500),
         width = 15)

# Particle buoyancy colorbar
Colorbar(cbar_layout[2, 1], 
         label="Particle Buoyancy [m s⁻²]", 
         colormap=reverse(cgrad(:RdBu)), 
         limits=(0.001, 0.0013),
         width = 15)

# Adjust layout to give more space to the plots
colgap!(fig.layout, 1, 10)
colgap!(fig.layout, 2, 20)
rowgap!(fig.layout, 1, 5)

# Add overall title
# Label(fig[0, 1:2], "3D Particle Trajectories at Different Tidal Phases", 
#       fontsize = 18, tellwidth = false)

# Save the figure
save(string("output/", simname, "/3D_particle_trajectories_4panels_z_center=", z_center_particle, ".png"), fig)

@info "4-panel 3D particle trajectory plot saved"

# Final cleanup
GC.gc()