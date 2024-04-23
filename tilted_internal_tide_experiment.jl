using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra
using Adapt
using MAT
using Statistics
using Oceanostics
using Oceanostics.TKEBudgetTerms: BuoyancyProductionTerm


using CUDA

function log_gpu_memory_usage()
# Capture the output of CUDA.memory_status()
    output = IOBuffer()
    CUDA.memory_status(output)
# Convert the captured output to a string
    mem_info_str = String(take!(output))
    return mem_info_str
end

suffix = "90s"

## Simulation parameters
const Nx = 150#1000 #150 500 750 1000
const Ny = 300#2000 #300 1000 1500 2000
const Nz = 100 # 250

const tᶠ = 90#120 # simulation run time
const Δtᵒ = 30 # interval for saving output

const H = 3.5kilometers # 4.926e3, 6.e3 # vertical extent
const Lx = 15kilometers
const Ly = 30kilometers

print("dx=",Lx/Nx,", dy=",Ly/Ny)

## Create grid
# Creates a vertical grid with near-constant spacing `refinement * Lz / Nz` near the bottom:
# "Warped" coordinate
kwarp(k, N) = (N + 1 - k) / N
# Linear near-surface generator
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
# Bottom-intensified stretching function
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))

# Generating function
z_faces(k) = - H * (ζ(k, Nz, 1.8) * Σ(k, Nz, 10) - 1)

#using CairoMakie

#lines(zspacings(grid, Center()), znodes(grid, Center()),
#      axis = (ylabel = "Depth (m)",
#              xlabel = "Vertical spacing (m)"))

#scatter!(zspacings(grid, Center()), znodes(grid, Center()))


grid = RectilinearGrid(GPU(),size=(Nx, Ny, Nz), 
        x = (0, Lx),
        y = (0, Ly), 
        z = z_faces,
        halo = (4,4,4),
        topology = (Periodic, Periodic, Bounded)
)
# yᶜ = ynodes(grid, Center())
# Δyᶜ = yspacings(grid, Center())

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
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=Ny)

using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)
# heatmap(x_interp, y_interp, z_interp'; color = :balance, xlabel = "x", ylabel = "z", aspect_ratio = :equal)



# # grids
# zC = adapt(Array,znodes(grid, Center()))
# zF = adapt(Array,znodes(grid, Face()))
# xC = xnodes(grid, Center())
# yC = ynodes(grid, Center())
# # find the grid that is above z_interp at x-y plane
# inx = zeros(Nx,Ny)  # Preallocate inx array to store the indices
# # create an array of indices that captures the frist element above the topography
# for i in 1:Nx
#    for j in 1:Ny
#     inx[i,j] = adapt(Array,findfirst(x -> x > z_interp[i,j], zC))
#    end
# end

# using Plots, StatsBase

# function terrain_follow_average(ϕ)

# ## creating terrain-aligned horizontal average

# # Find the tallest point and use histogram to bin the vertical grids first
# binsize = ceil(maximum(diff(zF)));  # smallest binsize has to be larger than the maximun Δz
# row, col = findmax(z_interp)[2][1],findmax(z_interp)[2][2]
# h = fit(Histogram, zC[Int(inx[row,col]):end],[zC[Int(inx[row,col])]:binsize:maximum(zC);])
# bins = diff(h.edges[1])/2 .+ h.edges[1][1:end-1]   # central bin
# # preallocation
# temp = adapt(Array,zeros(Nx,Ny,length(h.weights)));
# CUDA.allowscalar(false) 

# # loop over the entire x-y plane to get the depth histogram from the topography to the surface
# for k in 1:length(h.weights)
#     for i in 1:Nx
#         for j in 1:Ny
#             hl = fit(Histogram, zC[Int(inx[i,j]):end],[zC[Int(inx[i,j])]:binsize:maximum(zC);]) # histogram in the loop
#             window = 0; 
#             # lg and ug are the lowest and upmost grids within the bin.
#             # For example, if zF = 0, 0.5, 1, 1.5, 2, 2.5, 3, and bins = 0.1, 2.1, 4.1. Within the first bin, lg=0 and ug=2
#             lg = Int(inx[i,j])+window # lowest grid in the bin
#             ug = Int(inx[i,j])+window+hl.weights[k]-1 #upmost grid in the bin
#             CUDA.@allowscalar temp[i,j,k] = sum(adapt(Array,ϕ[i,j,lg:ug]).*diff(zF)[lg:ug]) ./ (zF[lg].-zF[ug]) 
#             # temp[i,j,k] = sum(ϕ[i,j,lg:ug].*diff(zF)[lg:ug]) ./ (zF[lg].-zF[ug])
#             window = window + hl.weights[k]
#         end
#     end
# end
# ϕ̄ = vec(mean(temp,dims=(1,2))) 
# # uu = vec(mean(temp,dims=(1,2))) 
# # shift the bins to 0
# return ϕ̄, bins.-minimum(bins)    
# end


# Create immersed boundary grid
	# GridFittedBottom: real topography. GridFittedBoundary: need a mask (logical)
grid_real = ImmersedBoundaryGrid(grid, GridFittedBottom(z_interp)) 
velocity_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0));


# Environmental parameters
const N = 1.e-3 # Brunt-Väisälä buoyancy frequency
const f₀ = 0.53e-4 # Coriolis frequency
const θ = 3.6e-3 # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates

# Tidal forcing
const U₀ = 0.025
const ω₀ = 1.4e-4
u_tidal_forcing(x, y, z, t) = U₀*ω₀*sin(ω₀*t)

# IC such that flow is in phase with predicted linear response, but otherwise quiescent
Uᵣ = U₀ * ω₀^2/(ω₀^2 - f₀^2 - (N*sin(θ))^2) # quasi-resonant linear barotropic response
uᵢ(x, y, z) = -Uᵣ
vᵢ(x, y, z) = 0.
bᵢ(x, y, z) = 1e-9*rand() # seed infinitesimal perturbations in buoyancy field

s = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))
#γ = h*π/(s*6kilometers)
#print("Steepness parameter of γ=",round(γ, digits=3))
-
# Rotate gravity vector
buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -[ĝ...])
coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)

# Linear background stratification (in ẑ)
@inline ẑ(x, z, ĝ) = x*ĝ[1] .+ z*ĝ[3]
@inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))


model = NonhydrostaticModel(
    grid = grid_real,
    advection = WENO(),
    buoyancy = buoyancy,
    coriolis = coriolis,
    boundary_conditions=(u=velocity_bcs, v=velocity_bcs, w=velocity_bcs),
    forcing = (u = u_tidal_forcing,),
    closure = ScalarDiffusivity(; ν=1e-4, κ=1e-4),
    tracers = :b,
    timestepper = :RungeKutta3,
    background_fields = (; b=B̄_field),
)

set!(model, b=bᵢ, u=uᵢ, v=vᵢ)

## Configure simulation
const Δt = (1/N)*0.03
# Δt = 0.5 * minimum_zspacing(grid) / Uᵣ
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

# # The `TimeStepWizard` manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy
# # (CFL) number close to `0.5` while ensuring the time-step does not increase beyond the
# # maximum allowable value for numerical stability.

#wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.2)
#simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


## Diagnostics
b = model.tracers.b
B̄ = model.background_fields.tracers.b
B = B̄ + b # total buoyancy field

u, v, w = model.velocities
û = @at (Face, Center, Center) u*ĝ[3] - w*ĝ[1] # true zonal velocity
ŵ = @at (Center, Center, Face) w*ĝ[3] + u*ĝ[1] # true vertical velocity
ζᶻ = ∂x(v) - ∂y(û)

ν = model.closure.ν
κ = model.closure.κ

# Oceanostics
KE = KineticEnergy(model)
ε = KineticEnergyDissipationRate(model)
χ = TracerVarianceDissipationRate(model, :b)
wb = BuoyancyProductionTerm(model)


# # Terrain-following horizontal averages
# ε_avg, bin = terrain_follow_average(ε) # includes the surface grid to the grid right above the topography
# χ_avg = terrain_follow_average(χ)[1]
# KE_avg = terrain_follow_average(KE)[1]
# wb_avg = terrain_follow_average(wb)[1]

# # # Write custom vectors and arrays to disk
# ε_avg_disk(model) = ε_avg
# χ_avg_disk(model) = χ_avg
# KE_avg_disk(model) = KE_avg
# wb_avg_disk(model) = wb_avg
# bin_disk(model) = bin
# # bathy(model) = z_interp


state_diags = merge(model.velocities, model.tracers)
Oceanostics_diags = (; KE, ε, wb, χ)
custom_diags = (; uhat=û, what=ŵ,B=B,vorticity=ζᶻ)
all_diags = merge(state_diags,Oceanostics_diags,custom_diags) 


fname = string("internal_tide_", suffix,"-theta=",string(θ),"_realtopo3D_Nx",Nx,"_Nz",Nz)

# output 3D field data
simulation.output_writers[:nc_fields] = NetCDFOutputWriter(model, state_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        verbose=true,
					filename = string("output/", fname, "_fields.nc"),
                                        overwrite_existing = true)
# output 2D slices
#simulation.output_writers[:nc_slice] = NetCDFOutputWriter(model, custom_diags,
#                                       schedule = TimeInterval(Δtᵒ),
#                                       indices = (:,Ny÷2,:), # center of the domain (on the canyon)
#                                       #max_filesize = 500MiB, #needs to be uncommented when running large simulation
#                                       verbose=true,
#                                       filename = string("output/", fname, "_slices.nc"),
#			               overwrite_existing = true)

# # output terrain-following horizontal averages
# outputs = Dict("ε_avg" => ε_avg_disk, "χ_avg" => χ_avg_disk, 
#         "KE_avg" => KE_avg_disk, "wb_avg" => wb_avg_disk, "bin" => bin_disk)

# dims = Dict("ε_avg" => ("zC",), "χ_avg" => ("zC",), "KE_avg" => ("zC",), 
#         "wb_avg" => ("zC",), "bin" => ("bin",))
# output_attributes = Dict(
#         "ε_avg"  => Dict("long_name" => "Terrain-following horizontal average KE dissipation rate", "units" => "m²/s³"),
#         "χ_avg" => Dict("long_name" => "Terrain-following horizontal average buoyancy variance dissipation rate", "units" => "m²/s³"),
#         "wb_avg"   => Dict("long_name" => "Terrain-following horizontal average buoyancy flux", "units" => "m²/s³"),
#         "KE_avg"   => Dict("long_name" => "Terrain-following horizontal average KE", "units" => "m²/s²"),
#         "bin"   => Dict("long_name" => "Vertical bins", "units" => "m")

#         )

# simulation.output_writers[:TFH] = NetCDFOutputWriter(model, outputs,
#                                        schedule = TimeInterval(Δtᵒ), dimensions=dims,
#                                        filename = string("output/", fname, "_TF_horizontal_average.nc"),
#                                        output_attributes=output_attributes,
#                                        overwrite_existing = true)

# bathymetry = Dict("bathy" => bathy)
# dims = Dict("bathy" => ("xC","yC")) 
# output_attributes = Dict(
#         "bathy"   => Dict("long_name" => "Bathymetry", "units" => "m"))
# simulation.output_writers[:bathy] = NetCDFOutputWriter(model, bathymetry,
#                                        schedule = TimeInterval(0), dimensions=dims,
#                                        filename = "output/bathymetry.nc",
#                                        verbose=true,
#                                        output_attributes=output_attributes,
#                                        overwrite_existing = true)      
          

## Progress messages
progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, Δt: %.3f,
                            advective CFL: %.2e, diffusive CFL: %.2e, gpu_memory_usage:%s\n",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w), Δt,
                            AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model),log_gpu_memory_usage())
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))

## Running the simulation!
run!(simulation)

@info """
    Simulation complete.
"""
##@info """
##    Simulation complete.
##    Output: $(abspath(simulation.output_writers[:fields].filepath))
##"""







# JLD2OutputWriter  
#simulation.output_writers[:checkpointer] = Checkpointer(
#                                        model,
#                                        schedule=TimeInterval(tᶠ),
#                                        dir="output",
#                                        prefix=string(fname, "_checkpoint"),
#                                        cleanup=true)

#simulation.output_writers[:fields] = JLD2OutputWriter(model, custom_diags,
#                                        schedule = TimeInterval(30),
#                                        filename = string("output/", fname, "_fields.jld2"),
#					                    max_filesize = 500MiB, 
#					                    verbose=true,
#                                        overwrite_existing = true)

#simulation.output_writers[:slice] = JLD2OutputWriter(model, custom_diags,
#                                        schedule = TimeInterval(Δtᵒ),
#                                        indices = (:,Ny÷2,:), # center of the domain (on the canyon)
#					#max_filesize = 500MiB, #needs to be uncommented when running large simulation
#                                        verbose=true,
#                                        filename = string("output/", fname, "_slices.jld2"),
#                                        overwrite_existing = true)
##### output netcdf
#simulation.output_writers[:checkpointer] = Checkpointer(
#                                        model,
#                                        schedule=TimeInterval(tᶠ),
#                                        dir="output",
#                                        prefix=string(fname, "_checkpoint"),
#					verbose=true,
#                                        cleanup=true)
