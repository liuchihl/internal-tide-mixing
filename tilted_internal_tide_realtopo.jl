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

suffix = "3days"

## Simulation parameters
const Nx = 150 #250 500 1000
const Ny = 300 #500 1000 2000
const Nz = 100

const tᶠ = 3days # simulation run time
const Δtᵒ = 30minutes # interval for saving output

const H = 4.926kilometers # 6.e3 # vertical extent
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
file = matopen("/pub/chihlul1/work/PROJECT/topo.mat")
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


# Create immersed boundary grid
	# GridFittedBottom: real topography. GridFittedBoundary: need a mask (logical)
grid_real = ImmersedBoundaryGrid(grid, GridFittedBottom(z_interp)) 
velocity_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0));

## creating terrain-aligned horizontal average
# center z grid
zc = znodes(grid, Center())
# find the grid that is above z_interp at x-y plane
inx = zeros(Nx,Ny)  # Preallocate inx array to store the indices
# create an array of indices that captures the frist element above the topography
#for i in 1:Nx
#    for j in 1:Ny
#inx[i,j] = findfirst(x -> x > z_interp[i,j], zc)
#    end
#end

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
#Δt = (1/N)*0.03
Δt = 0.5 * minimum_zspacing(grid) / Uᵣ
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

# # The `TimeStepWizard` manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy
# # (CFL) number close to `0.5` while ensuring the time-step does not increase beyond the
# # maximum allowable value for numerical stability.

wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.2)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


## Diagnostics
b = model.tracers.b
B̄ = model.background_fields.tracers.b
B = B̄ + b # total buoyancy field

u, v, w = model.velocities
û = @at (Face, Center, Center) u*ĝ[3] - w*ĝ[1] # true zonal velocity
ŵ = @at (Center, Center, Face) w*ĝ[3] + u*ĝ[1] # true vertical velocity

ν = model.closure.ν
κ = model.closure.κ
# only works when running on CPU, comment it out when running on GPU
#ε = Field(ν*(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2 + ∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2 + ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2))
# when run on GPU
ddx² = Field(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2)
ddy² = Field(∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2)
ddz² = Field(∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2)
ε = Field(ν * (ddx² + ddy² + ddz²))
χ = @at (Center, Center, Center) κ[1] * (∂x(b)^2 + ∂z(b)^2)

KE = KineticEnergy(model)
ε_oceanostics = KineticEnergyDissipationRate(model)
wb = BuoyancyProductionTerm(model)
#∫KE = Integral(KE)
#∫ε_oceanostics = Integral(ε_oceanostics)
#∫wb = Integral(wb)

custom_diags = (B=B, uhat=û, what=ŵ, χ=χ, ε=ε, KE=KE, ε_oceanostics=ε_oceanostics, wb=wb)

all_diags = merge(model.velocities, model.tracers, custom_diags)

fname = string("internal_tide_", suffix,"-theta=",string(θ),"_realtopo3D_Nx150")

# JLD2OutputWriter  
simulation.output_writers[:checkpointer] = Checkpointer(
                                        model,
                                        schedule=TimeInterval(Δtᵒ*10),
                                        dir="output",
                                        prefix=string(fname, "_checkpoint"),
                                        cleanup=true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, all_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        filename = string("output/", fname, "_fields.jld2"),
					max_filesize = 500MiB, 
					verbose=true,
                                        overwrite_existing = true)

simulation.output_writers[:slice] = JLD2OutputWriter(model, all_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        indices = (:,Ny÷2,:), # center of the domain (on the canyon)
					#max_filesize = 500MiB, #needs to be uncommented when running large simulation
                                        verbose=true,
                                        filename = string("output/", fname, "_slice.jld2"),
                                        overwrite_existing = true)
#simulation.output_writers[:slice] = JLD2OutputWriter(model, (; wb, ε_oceanostics, KE, χ),
#                                        schedule = TimeInterval(Δtᵒ),
#                                        indices = (:,1,:), # edge of the domain (on the ridge)
#                                        max_filesize = 500MiB,
#                                        verbose=true,
#                                        filename = string("output/", fname, "_slice.jld2"),
#                                        overwrite_existing = true)
#simulation.output_writers[:zonal_time_means] = JLD2OutputWriter(model, (; ε),
#                                        schedule = AveragedTimeInterval(Δtᵒ÷2, window=Δtᵒ÷2),
#                                        filename = string("output/", fname, "_zonal_time_means.jld2"),
#                                        overwrite_existing = true)

#simulation.output_writers[:TF_horizontal_average] = JLD2OutputWriter(model, avg_diags;
#                                        schedule = AveragedTimeInterval(Δtᵒ÷2, window=Δtᵒ÷2),
#                                        filename = string("output/", fname, "_TF_horizontal_average.jld2"),
#                                        overwrite_existing = true)



## Progress messages
progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, 
                            advective CFL: %.2e, diffusive CFL: %.2e\n",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w),
                            AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))

## Running the simulation!
run!(simulation)

@info """
    Simulation complete.
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""
