using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using BSplineKit
using LinearAlgebra
using Interpolations
using MAT

simname = "2D_notilt"
tᶠ = "146"
θ = 0
## load data
filename_slice = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_tᶠ=",tᶠ, "_snapshot.nc")
# ds = Dataset(filename_slice,"r")
# filename_slice = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_tᶠ=",tᶠ, "_1TP_timeavg.nc")
ds = Dataset(filename_slice,"r")


filename_particles = string("output/",simname,"/internal_tide_theta=",θ,"_realtopo2D_Nx=500_Nz=250_tᶠ=",tᶠ, "_particles.nc")
ds_particles = Dataset(filename_particles,"r")
x_particles = ds_particles["x"][:,:]
z_particles = ds_particles["z"][:,:]
particle_time = ds_particles["time"][:]

# grids
zC = ds["zC"][:]; Nz=length(zC); 
zF = ds["zF"][:]; 
xF = ds["xF"][:];
xC = ds["xC"][:]; Nx=length(xC)
yC = ds["yC"][:]; Ny=length(yC)
t = ds["time"][:];
c = ds["c"][:,:,:,:];
u = ds["uhat"][:,:,:,:];
u_cen = (u[1:end,:,:,:] .+ vcat(u[2:end,:,:,:],u[1:1,:,:,:]))./2 # u at center in the x dimension
w = ds["what"][:,:,:,:];


u_cen[u_cen.==0].=NaN
w[w.==0].=NaN
c[c.==0] .= NaN


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



c[c.<=0] .= 1e-10
fig = Figure(resolution = (1200, 800))
ax = Axis(fig[1, 1], 
    xlabel = "Distance (m)",
    ylabel = "Depth (m)",
    title = "Tracer Concentration and particles")

# Create a observable for the time index and data
time_idx = Observable(1)
c_data = @lift(log10.(c[:, 1, 1:160, $time_idx]))
u_data = @lift(u_cen[1:10:end, 1, 1:6:160, $time_idx])
w_data = @lift(1*w[1:10:end, 1, 1:6:160, $time_idx])

# Create the heatmap plot
hm = heatmap!(ax, xC, zC[1:160], c_data, 
    colormap = :OrRd_9,
    colorrange = (-3,0),
    nan_color = :grey)

# Add colorbar
Colorbar(fig[1, 2], hm, label = "Concentration")
# Extract particles at specific time
# x_particles and z_particles have shape (particle_id, time)
particle_x = @lift begin
    time_index = min($time_idx, size(x_particles, 2))
    x_particles[1:5000, time_index]
end

particle_z = @lift begin
    time_index = min($time_idx, size(z_particles, 2))
    z_particles[1:5000, time_index]
end

# Plot particles as scatter points
particles = scatter!(ax, particle_x, particle_z, 
    color = :green, 
    markersize = 4.0,
    marker = :circle,
    alpha = 0.9)  # Adjust alpha between 0 (fully transparent) and 1 (fully opaque)

    # Add quiver plot (flow arrows)
arrows!(ax, 
    xC[1:10:end], zC[1:6:160],
    u_data, w_data,
    arrowsize = 7,
    lengthscale = 2000,
    linewidth = 1.0,
    color = :black)
# Create and save the animation
# Animation parameters
fps = 10
times = 1:2:length(t)
record(fig, string("output/",simname,"/tracer_particle_animation_withoutnegativevalues.mp4"), times; framerate = fps) do i
    time_idx[] = i
    ax.title = @sprintf("Tracer Concentration and Particles at t = %.2f hr", t[i]/3600)
    @info "Frame $i"
end