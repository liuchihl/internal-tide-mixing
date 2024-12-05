
## this script is a test of the bottom boundary condition as well as the added diffusive 
# flux forcing term by comparing the numerical result with the steady state analytical solution
# from Callies 2018 in terms of stratification. The setup is a gentle slope without topographic 
# features and the diffusivity is set to vary in the vertical direction.
using MPI
MPI.Init()
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra
using MAT
using Statistics
using Oceanostics
using Oceanostics.TKEBudgetTerms: BuoyancyProductionTerm
using CUDA
using NCDatasets

function log_gpu_memory_usage()
    return @capture_out CUDA.memory_status()  # retrieve raw string status
end

suffix = "1000days"

## Simulation parameters
const tᶠ = 1000days # simulation run time
const Δtᵒ = 5days   # interval for saving output

const H = 1.2kilometers
const Lx = 500meters#15kilometers
const Ly = 500meters
rx = parse(Int, get(ENV, "RX", "4"))
ry = parse(Int, get(ENV, "RY", "1"))
ranks = (rx, ry, 1)
Nx = parse(Int, get(ENV, "NX", "4"))
Ny = parse(Int, get(ENV, "NY", "1"))
Nz = parse(Int, get(ENV, "NZ", "250")) 

## Create grid
# Creates a vertical grid with near-constant spacing `refinement * Lz / Nz` near the bottom:
# "Warped" coordinate
kwarp(k, N) = (N + 1 - k) / N
# Linear near-surface generator
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
# Bottom-intensified stretching function
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))

# Generating function
# large refinement gets better resolution at the bottom but coarser at the surface
# large stretching: resolution difference between the bottom and surface increases
# z_faces(k) = - H * (ζ(k, Nz, 1.8) * Σ(k, Nz, 10) - 1)
z_faces(k) = - H * (ζ(k, Nz, 2) * Σ(k, Nz, 10) - 1)


arch = Distributed(GPU(); partition = Partition(ranks...))
grid = RectilinearGrid(arch,size=(Nx,Nz), 
        x = (0,Lx),        
        # y = (0,Ly),        
        z = z_faces,
        halo=(2,2),
        topology = (Oceananigans.Periodic, Oceananigans.Flat, Bounded)
)
bottomimmerse = 0#grid.zᵃᵃᶜ[3]   # if immersed boundary is at z=0, no effect of gradient BC is found
grid_immerse = ImmersedBoundaryGrid(grid, GridFittedBottom(bottomimmerse)) 


# Sinusoidal topography of height h and mode number n
# topog = [0 0 0 0; 0 0 0 0; 0 0 0 0; 1 1 1 1]
# # Create immersed boundary grid
# grid_immerse = ImmersedBoundaryGrid(grid, GridFittedBoundary(topog))

# Environmental parameters
const N = 1.3e-3              # Brunt-Väisälä buoyancy frequency        
const f₀ = -5.3e-5            # Coriolis frequency
const θ = 1.26e-3  #2e-1# 2e-3                # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
const ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates
const κ₀ = 5.3e-5             # Far-Field diffusivity
const κ₁ = 1.8e-3            # Bottom enhancement of diffusivity
const h = 230meter            # decay scale of diffusivity
const σ = 1                   # Prandtl number
const ν₀ = κ₀
const ν₁ = κ₁

# diffusivity
@inline κ(x,z,t) = κ₀ + κ₁*exp(-z / h)
closure = ScalarDiffusivity(;κ=κ, ν=κ)

# u_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0))   
# v_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0))   
# w_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0))  

noslip_freeslip = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0),top = FluxBoundaryCondition(nothing))
no_penetrate = FieldBoundaryConditions(ValueBoundaryCondition(0.0))
# no-flux boundary condition
normal = -N^2*cos(θ)    # normal slope 
cross = -N^2*sin(θ)     # cross slope
B_immerse = ImmersedBoundaryCondition(bottom=GradientBoundaryCondition(normal),
                    west = GradientBoundaryCondition(cross), east = GradientBoundaryCondition(cross))
B_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(normal),
                                top = GradientBoundaryCondition(0), 
                                immersed=B_immerse);


# Rotate gravity vector
buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -[ĝ...])
coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)

# Linear background stratification (in ẑ)
@inline ẑ(x, z, ĝ) = x*ĝ[1].+z*ĝ[3]
@inline constant_stratification(x, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))


# Tidal forcing 
const U₀ = 0#0.025
const ω₀ = 1.4e-4
u_tidal_forcing(x, z, t) = U₀*ω₀*sin(ω₀*t)

model = NonhydrostaticModel(
    grid = grid_immerse,
    advection = WENO(),
    buoyancy = buoyancy,
    coriolis = coriolis,
    boundary_conditions=(u=noslip_freeslip, w=no_penetrate,  b = B_bcs,),
    forcing = (u = u_tidal_forcing,),
    closure = closure,
    tracers = :b,
    timestepper = :RungeKutta3,
    # hydrostatic_pressure_anomaly = CenterField(grid_immerse),
    background_fields = Oceananigans.BackgroundFields(; background_closure_fluxes=true, b=B̄_field),
)

# IC such that flow is in phase with predicted linear response, but otherwise quiescent
Uᵣ = U₀ * ω₀^2/(ω₀^2 - f₀^2 - (N*sin(θ))^2) # quasi-resonant linear barotropic response
uᵢ(x,z) = -Uᵣ
vᵢ(x,z) = 0.
bᵢ(x,z) = 1e-9*rand() # seed infinitesimal perturbations in buoyancy field

set!(model, b=bᵢ, u=uᵢ, v=vᵢ)

## Configure simulation
Δt = 150#(1/N)*0.03
# Δt = 0.5 * minimum_zspacing(grid) / Uᵣ
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)
# wizard = TimeStepWizard(cfl=.5, diffusive_cfl=.5)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

## Diagnostics
b = model.tracers.b
B̄ = model.background_fields.tracers.b
B = B̄ + b # total buoyancy field
κ_save = model.closure.κ.b

u, v, w = model.velocities
û = @at (Face, Center, Center) u*ĝ[3] - w*ĝ[1] # true zonal velocity
Bz = @at (Center, Center, Center) ∂z(B)            

fname = string("test_background_diffusive_flux_divergence_", suffix,"_theta=",string(θ),"_2D_higherresolution_Nz=",string(Nz),"_4GPU")
# rm(string(fname,"/test_background_diffusive_flux_divergence_50days_theta=1_z.nc"))

# simulation.output_writers[:checkpointer] = Checkpointer(
#                                         model,
#                                         schedule=TimeInterval(250days),
#                                         dir=fname,
#                                         prefix=string(fname, "_checkpoint"),
#                                         cleanup=true)
simulation.output_writers[:oneD_z_nc] = NetCDFOutputWriter(model, (;B=B, Bz=Bz, b=b, uhat=û, u=u),
                                       schedule = TimeInterval(Δtᵒ),
                                       indices = (:,:,:), # center of the domain (on the canyon)
                                       verbose=true,
                                       filename = string("test_background_diffusive_flux_divergence/",fname, "_z.nc"),
			                  		   overwrite_existing = true)

## Progress messages
progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, Δt: %.3f,
                            advective CFL: %.2e, diffusive CFL: %.2e, gpu_memory_usage:%s\n",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w), s.Δt,
                            AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model)
                            ,log_gpu_memory_usage())

simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))

## Running the simulation!
run!(simulation; pickup=false)

