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

suffix = "5days"

## Simulation parameters
const Nx = 500 #250 500 1000
const Ny = 1000 #500 1000 2000
const Nz = 250

const tᶠ = 5days # simulation run time
const Δtᵒ = 30minutes # interval for saving output

const H = 3.5kilometers # 6.e3 # vertical extent
const Lx = 15kilometers
const Ly = 30kilometers

# print("dx=",Lx/Nx,", dy=",Ly/Ny)

## Create grid
# Creates a vertical grid with near-constant spacing `refinement * Lz / Nz` near the bottom:
# "Warped" coordinate
kwarp(k, N) = (N + 1 - k) / N
# Linear near-surface generator
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
# Bottom-intensified stretching function
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))

# Generating function
z_faces(k) = - H * (ζ(k, Nz, 1.5) * Σ(k, Nz, 15) - 1)

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
zC = adapt(Array,znodes(grid, Center()))

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


# Environmental parameters
const N = 1.e-3 # Brunt-Väisälä buoyancy frequency
const f₀ = 0.53e-4 # Coriolis frequency
const θ = 3.6e-3 # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
const ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates


# Create immersed boundary grid
	# GridFittedBottom: real topography. GridFittedBoundary: need a mask (logical)
grid_real = ImmersedBoundaryGrid(grid, GridFittedBottom(z_interp)) 
velocity_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0));


# setting quadratic drag BC at domain bottom and top of the immersed boundary
const z₀ = 0.1 # m (roughness length)
const κ_von = 0.4  # von Karman constant

CUDA.@allowscalar z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
const cᴰ = (κ_von / log(z₁ / z₀))^2 # Drag coefficient
# non-immersed and immersed boundary conditions
@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v
@inline immersed_drag_u(x, y, z, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
@inline immersed_drag_v(x, y, z, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))
immersed_drag_bc_u = FluxBoundaryCondition(immersed_drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
immersed_drag_bc_v = FluxBoundaryCondition(immersed_drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))

u_immerse = ImmersedBoundaryCondition(top=immersed_drag_bc_u)
v_immerse = ImmersedBoundaryCondition(top=immersed_drag_bc_v)

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u, immersed=u_immerse)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v, immersed=v_immerse)
# ImpenetrableBoundaryCondition is used as default at other non-immersed boundary points
w_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0))   

# no-flux boundary condition
normal = -N^2*cos(θ)    # normal slope 
cross = -N^2*sin(θ)     # cross slope
# directions are defined relative to non-immersed grids
B_immerse = ImmersedBoundaryCondition(bottom=GradientBoundaryCondition(normal),
                    west = GradientBoundaryCondition(cross), east = GradientBoundaryCondition(-cross))
B_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(normal),immersed=B_immerse);


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
    boundary_conditions=(u=u_bcs, v=v_bcs, w=w_bcs,  b = B_bcs,),
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

# Oceanostics
KE = KineticEnergy(model)
# PE = PotentialEnergy(model)
ε = KineticEnergyDissipationRate(model)
χ = TracerVarianceDissipationRate(model, :b)
wb = BuoyancyProductionTerm(model)


state_diags = merge(model.velocities, model.tracers)
Oceanostics_diags = (; KE, ε, wb, χ)
custom_diags = (; uhat=û, what=ŵ,B=B)
all_diags = merge(state_diags,Oceanostics_diags,custom_diags)

fname = string("internal_tide_", suffix,"-theta=",string(θ),"_realtopo3D_Nx",Nx,"_Nz",Nz)
# checkpoint  
simulation.output_writers[:checkpointer] = Checkpointer(
                                        model,
                                        schedule=TimeInterval(tᶠ),
                                        dir="output/supercritical_tilt",
                                        prefix=string(fname, "_checkpoint"),
                                        cleanup=true)
# output 3D field data
simulation.output_writers[:nc_fields] = NetCDFOutputWriter(model, (; uhat=û ,B=B, ε=ε, χ=χ),
                                        schedule = TimeInterval(43Δtᵒ),
                                        verbose=true,
                                        filename = string("output/supercritical_tilt/", fname, "_fields.nc"),
                                        overwrite_existing = true)
# output 2D slices
#1) xz
simulation.output_writers[:nc_slice_xz] = NetCDFOutputWriter(model, all_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        indices = (:,Ny÷2,:), # center of the domain (on the canyon)
                                        #max_filesize = 500MiB, #needs to be uncommented when running large simulation
                                        verbose=true,
                                        filename = string("output/supercritical_tilt/", fname, "_slices_xz.nc"),
                                        overwrite_existing = true)
#2) xy
ind = argmin(abs.(zC .- 1300))   # 1300 m height above bottom

simulation.output_writers[:nc_slice_xy] = NetCDFOutputWriter(model, all_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        indices = (:,:,ind), # center of the domain (on the canyon)
                                        #max_filesize = 500MiB, #needs to be uncommented when running large simulation
                                        verbose=true,
                                        filename = string("output/supercritical_tilt/", fname, "_slices_xy.nc"),
                                        overwrite_existing = true)
#3) yz
simulation.output_writers[:nc_slice_yz] = NetCDFOutputWriter(model, all_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        indices = (Nx÷2,:,:), # center of the domain (on the canyon)
                                        #max_filesize = 500MiB, #needs to be uncommented when running large simulation
                                        verbose=true,
                                        filename = string("output/supercritical_tilt/", fname, "_slices_yz.nc"),
                                        overwrite_existing = true)

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
