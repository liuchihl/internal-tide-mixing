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
ν = model.closure.ν
κ = model.closure.κ

state_diags = merge(model.velocities, model.tracers)

fname = string("internal_tide_", suffix,"-theta=",string(θ),"_realtopo3D_Nx",Nx,"_Nz",Nz)

# output 3D field data
simulation.output_writers[:nc_fields] = NetCDFOutputWriter(model, state_diags,
                                        schedule = TimeInterval(Δtᵒ),
                                        verbose=true,
					filename = string("output/", fname, "_fields.nc"),
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
                         
