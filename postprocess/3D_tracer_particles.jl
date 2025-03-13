cd("/scratch/bcpi/cliu28/internal-tide-mixing/")
using Printf
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using LinearAlgebra
using MAT
using Interpolations


simname = "flat" 
θ = θ = simname == "tilt" ? 0.0036 : 0
tᶠ = "460"

## load data
filename = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_B-c.nc")
ds = Dataset(filename,"r")
c = ds["c"][:,500,:,1]
c[c.==0] .= NaN
filename_particles = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_particles.nc")
ds_particles = Dataset(filename_particles,"r")
x_particles = ds_particles["x"][:,:]
y_particles = ds_particles["y"][:,:]
z_particles = ds_particles["z"][:,:]
particle_time = ds_particles["time"][:]
particle_b = ds_particles["b"][:,:]


# grids
zC = ds["zC"][:]; Nz=length(zC); 
zF = ds["zF"][:]; 
xF = ds["xF"][:];
xC = ds["xC"][:]; Nx=length(xC)
yC = ds["yC"][:]; Ny=length(yC)
t = ds["time"][:];



####### see if b is increasing
# Create a figure to plot all particle b values over time
# using PyPlot
# close("all")
# figure(figsize=(12, 8))
# # Plot for each particle
# for i in 1:size(particle_b, 1)
#     PyPlot.plot(particle_time / 3600, particle_b[i, :], 
#          linewidth=0.5, alpha=0.3, color="blue")
# end

# # Add mean buoyancy line for reference
# mean_b = nanmean(particle_b, dims=1)[1, :]
# PyPlot.plot(particle_time / 3600, mean_b, 
#      linewidth=3, color="red", 
#      label="Mean buoyancy")

# xlabel("Time (hours)")
# ylabel("Buoyancy")
# title("Buoyancy of particles over time")
# legend()
# tight_layout()
# savefig(string("output/", simname, "/b_vs_time.png"), dpi=300)
#######




####### plot the particle trajectories

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

# Create a figure to plot the particle trajectories
using CairoMakie

# Create a 3D figure for topography and particle trajectories
fig = CairoMakie.Figure(resolution = (1200, 800))
ax = Axis3(fig[1, 1], 
           xlabel = "X (m)", 
           ylabel = "Y (m)", 
           zlabel = "Z (m)",
           title = "3D Particle Trajectories",
           aspect = (1, 2, 0.25))

# Create all static elements (terrain + walls) ONCE outside the animation loop
# Main topography
surface!(ax, x_interp, y_interp, z_interp, 
         colormap = :terrain,
         shading = NoShading,
         transparency = false)

# Add legend
# fig[1, 2] = Legend(fig, 
#                   [MarkerElement(color = :blue, marker = :circle)],
#                   ["Particles"],
#                   )

# Define number of particles to plot
n_particles, n_time_steps = size(x_particles) 

# Animation parameters
n_frames = 1:n_time_steps
time_steps = round.(Int, range(1, n_time_steps, length=length(n_frames)))

# Create particle objects list to keep track of what to delete each frame
particle_objects = []

record(fig, string("output/", simname, "/3D_particle_trajectories_animated.mp4"), n_frames; framerate = 30) do frame
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
    
    # Plot current particle positions
    particles = scatter!(ax, 
             x_particles[1:n_particles, current_time_step], 
             y_particles[1:n_particles, current_time_step], 
             z_particles[1:n_particles, current_time_step],
             color = :blue,
             markersize = 5)
    push!(particle_objects, particles)
    
    # Add trails (previous positions)
    trail_length = 8

    for i in 1:n_particles
        if xC[2]+500 < x_particles[i,current_time_step] < xF[end]-500
            start_idx = max(1, current_time_step - trail_length)
            if start_idx < current_time_step
                trail = lines!(ax, 
                    x_particles[i, start_idx:current_time_step], 
                    y_particles[i, start_idx:current_time_step], 
                    z_particles[i, start_idx:current_time_step],
                    color = :cyan,
                    linewidth = 1.0,
                    alpha = 0.5)
                push!(particle_objects, trail)
            end
        end
    end
    # Control the camera rotation within a specific range center around 0.5π

    azimuth = range(1.3π,1.6π, length=length(n_frames))[frame]
    ax.azimuth = azimuth
    # ax.elevation = elevation
    # Update title to show time
    ax.title = @sprintf("Particle Movement at t = %.2f hours", particle_time[current_time_step]/3600)
    @info current_time_step
end

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