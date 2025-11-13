using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver, FourierTridiagonalPoissonSolver, AsymptoticPoissonPreconditioner
using LinearAlgebra
using Adapt
using MAT
using CUDA
using Statistics
using Oceanostics
using Oceanostics.TKEBudgetTerms: BuoyancyProductionTerm
using Interpolations: LinearInterpolation
using Suppressor
using NCDatasets

include("../functions/diagnostics_budget.jl")
function log_gpu_memory_usage()
    return @capture_out CUDA.memory_status()  # retrieve raw string status
end

const Nx = 1e3
const Ny = 1
const Nz = 300
const ω₀ = 1.4e-4     # tidal freq.
const Δtᵒ = 1/24*2π / ω₀ # interval for saving output
const tᶠ = 50 * 2π / ω₀    # endtime of the simulation
const θ = 4e-3       # slope angle
const U₀ = 0.025      # tidal amplitude
const N = 1.e-3       # Buoyancy frequency
const f₀ = -0.53e-4   # Coriolis frequency
simname = "2D_idealized_tilt_$(θ)"

architecture = GPU()
closure = (SmagorinskyLilly(), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))
## Simulation parameters
H = 2.25kilometers # vertical extent
Lx = 15kilometers # along-canyon extent

## Create vertical grid
# Creates a vertical grid with near-constant spacing `refinement * Lz / Nz` near the bottom:
# "Warped" coordinate
kwarp(k, N) = (N + 1 - k) / N
# Linear near-surface generator
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
# Bottom-intensified stretching function
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))
# Generating function
z_faces(k) = -H * (ζ(k, Nz, 1.2) * Σ(k, Nz, 15) - 1)

grid = RectilinearGrid(architecture, size=(Nx, Nz),
    x=(0, Lx),
    z=z_faces,
    halo=(4, 4),
    topology=(Oceananigans.Periodic, Oceananigans.Flat, Oceananigans.Bounded)
)

# topography: similar scale to topo.mat but smoothed triangle

function smooth_triangle(x; h=1000, w=3000, edge=400, tip_width=4000, tip_smoothness=1)
    tri = h * (1 .- abs.(x) ./ w)
    tri = max.(tri, 0)
    left = exp.(-((x .+ w) ./ edge) .^ 2)
    right = exp.(-((x .- w) ./ edge) .^ 2)

    # Tanh-based tip smoothing
    # The tanh function creates a smooth transition near x=0
    # tip_smoothness controls how sharp/smooth the tip is (lower = smoother)
    tip_smooth = h * (1 .- tanh.((abs.(x) ./ tip_width) .^ tip_smoothness))

    # Blend between the triangular shape and the smooth tip
    # Use tanh to create a smooth mask
    blend_factor = 0.5 * (1 .+ tanh.((tip_width .- abs.(x)) ./ (tip_width * 0.2)))

    # Combine: use smooth tip near center, transition to triangle further out
    tri_smooth = blend_factor .* tip_smooth .+ (1 .- blend_factor) .* tri

    # Apply edge smoothing as before
    topo = tri_smooth .* (1 .- left) .* (1 .- right) .+ h .* left .* right

    return topo
end
x_symm = range(-Lx / 2, stop=Lx / 2, length=Nx)
z_triangle = smooth_triangle(x_symm, h=850, w=3600, edge=1600, tip_width=3000, tip_smoothness=1.4)

# Environmental parameters
ĝ = (sin(θ), 0, cos(θ)) # the vertical (oriented opposite gravity) unit vector in rotated coordinates

# Create immersed boundary grid
z_triangle = architecture == CPU() ? z_triangle : CuArray(z_triangle)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(z_triangle))

# setting quadratic drag BC at domain bottom and top of the immersed boundary
z₀ = 0.1 # m (roughness length)
κ_von = 0.4  # von Karman constant

CUDA.@allowscalar z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
# z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
cᴰ = (κ_von / log(z₁ / z₀))^2 # Drag coefficient
# non-immersed and immersed boundary conditions
@inline drag_u(x, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
@inline immersed_drag_u(x, z, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
@inline immersed_drag_v(x, z, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))
immersed_drag_bc_u = FluxBoundaryCondition(immersed_drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
immersed_drag_bc_v = FluxBoundaryCondition(immersed_drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))

u_immerse = ImmersedBoundaryCondition(bottom=immersed_drag_bc_u)
v_immerse = ImmersedBoundaryCondition(bottom=immersed_drag_bc_v)

u_bcs = FieldBoundaryConditions(bottom=drag_bc_u, top=FluxBoundaryCondition(nothing), immersed=u_immerse)
v_bcs = FieldBoundaryConditions(bottom=drag_bc_v, top=FluxBoundaryCondition(nothing), immersed=v_immerse)
# tracer: no-flux boundary condition
∂B̄∂z = N^2 * cos(θ)
∂B̄∂x = N^2 * sin(θ)
# In the following comments, ẑ=slope-normal unit vector and x̂=cross-slope unit vector
B_bcs_immersed = ImmersedBoundaryCondition(
    bottom=GradientBoundaryCondition(-∂B̄∂z), # ∇B⋅ẑ = 0 → ∂B∂z = 0 → ∂b∂z = -∂B̄∂z
    west=GradientBoundaryCondition(-∂B̄∂x), # ∇B⋅x̂ = 0 → ∂B∂x = 0 → ∂b∂x = -∂B̄∂x
    east=GradientBoundaryCondition(-∂B̄∂x)) # ∇B⋅x̂ = 0 → ∂B∂x = 0 → ∂b∂x = -∂B̄∂x

B_bcs = FieldBoundaryConditions(
    bottom=GradientBoundaryCondition(-∂B̄∂z), # ∇B⋅ẑ = 0 → ∂B∂z = 0 → ∂b∂z = -∂B̄∂z
    top=GradientBoundaryCondition(0), # ∇B⋅ẑ = ∂B̄∂ẑ → ∂b∂z = 0 and ∂b∂x = 0 (periodic)
    immersed=B_bcs_immersed);
## Notes:
# (1) directions are defined relative to domain coordinates.
# (2) Gradients are positive in the direction of the coordinate.

# Tidal forcing
@inline u_tidal_forcing(x, z, t) = U₀ * ω₀ * sin(ω₀ * t)

# s = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))

# Rotate gravity vector
buoyancy = BuoyancyForce(BuoyancyTracer(), gravity_unit_vector=-[ĝ...])
coriolis = ConstantCartesianCoriolis(f=f₀, rotation_axis=ĝ)

# Linear background stratification (in ẑ)
@inline ẑ(x, z, ĝ) = x * ĝ[1] .+ z * ĝ[3]
@inline constant_stratification(x, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N²=N^2))

# IC such that flow is in phase with predicted linear response, but otherwise quiescent
Uᵣ = U₀ * ω₀^2 / (ω₀^2 - f₀^2 - (N * sin(θ))^2) # quasi-resonant linear barotropic response
uᵢ(x, z) = -Uᵣ
vᵢ(x, z) = 0
# Bᵢ(x, z) = constant_stratification(x, z, 0, (; N² = N^2, ĝ=ĝ)) + 1e-9*rand()   # background + perturbation (only works in flat)
bᵢ(x, z) = 1e-9 * rand()   # background + perturbation (only works in flat)

# tol = 1e-9
model = NonhydrostaticModel(
    grid=grid,
    # pressure_solver=ConjugateGradientPoissonSolver(
    #     grid; maxiter=500, preconditioner=AsymptoticPoissonPreconditioner(),
    #     reltol=tol, abstol=tol),
    advection=WENO(),
    buoyancy=buoyancy,
    coriolis=coriolis,
    boundary_conditions=(u=u_bcs, v=v_bcs, b=B_bcs,),
    forcing=(u=u_tidal_forcing,),
    tracers=:b,
    closure=closure,
    timestepper=:RungeKutta3,
    hydrostatic_pressure_anomaly=CenterField(grid),
    background_fields=Oceananigans.BackgroundFields(; background_closure_fluxes=true, b=B̄_field),
)
set!(model, b=bᵢ, u=uᵢ, v=vᵢ)

## Configure simulation
Δt = 12#(1 / N) * 0.03
simulation = Simulation(model, Δt=Δt, stop_time=tᶠ + 20Δt, minimum_relative_step=0.01)

# # The `TimeStepWizard` manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy
# # (CFL) number close to `0.5` while ensuring the time-step does not increase beyond the
# # maximum allowable value for numerical stability.

# wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.2)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Diagnostics
b = model.tracers.b
B̄ = model.background_fields.tracers.b
B = B̄ + b # total buoyancy field
u, v, w = model.velocities
û = @at (Face, Center, Center) u * ĝ[3] - w * ĝ[1] # true zonal velocity
ŵ = @at (Center, Center, Face) w * ĝ[3] + u * ĝ[1] # true vertical velocity
νₑ = simulation.model.diffusivity_fields[1].νₑ    # eddy viscosity
Bz = @at (Center, Center, Center) ∂z(B)
# Oceanostics
ε = KineticEnergyDissipationRate(model)
χ = TracerVarianceDissipationRate(model, :b)
Rig = RichardsonNumber(model, u, v, w, B, .-model.buoyancy.gravity_unit_vector)
# Manually compute Richardson number to avoid KernelFunctionOperation on GPU
# Ri = N² / S² where N² = -g/ρ₀ * ∂B/∂z and S² = (∂u/∂z)² + (∂v/∂z)²
# N² = @at (Center, Center, Face) -∂z(B)
# S² = @at (Center, Center, Face) (∂z(u))^2 + (∂z(model.velocities.v))^2
# Rig = Field(N² / (S² + 1e-10))  # Add small number to avoid division by zero
Bbudget = get_budget_outputs_tuple(model;)
twoD_diags = merge(Bbudget, (; νₑ=νₑ, ε=ε, Rig=Rig, χ=χ, uhat=û, what=ŵ, B=B, Bz=Bz, b=b))

checkpoint_interval = tᶠ/2 * 2π / ω₀
fname = string("internal_tide_theta=", θ, "_Nx=", Nx, "_Nz=", Nz, "_tᶠ=", round(tᶠ / (2 * pi / 1.4e-4), digits=1))
dir = string("output/", simname, "/")
## checkpoint  
simulation.output_writers[:checkpointer] = Checkpointer(
    model,
    schedule=TimeInterval(checkpoint_interval),
    dir=dir,
    prefix="checkpoint",
    cleanup=true)
## output particles
# simulation.output_writers[:particles] = NetCDFOutputWriter(model, model.particles, filename=string(dir, fname, "_particles.nc"), schedule=TimeInterval(Δtᵒ))
## output field 
simulation.output_writers[:nc_snapshot] = NetCDFWriter(model, twoD_diags,
    verbose=true,
    filename=string(dir, fname, "_snapshot.nc"),
    overwrite_existing=true,
    schedule=TimeInterval(Δtᵒ)
)
## Progress messages


function progress_message(s)
    model = s.model
    w = model.velocities.w
    arch = Oceananigans.architecture(model.grid)
    progress = 100 * model.clock.time / s.stop_time
    iteration = model.clock.iteration
    current_time = model.clock.time
    current_dt = s.Δt

    # Get CG solver parameters
    cg = model.pressure_solver.conjugate_gradient_solver
    cg_iter = cg.iteration
    cg_maxiter = cg.maxiter

    if arch isa CPU
        maximum_w = maximum(abs, w)
        adv_cfl = AdvectiveCFL(s.Δt)(model)
        diff_cfl = DiffusiveCFL(s.Δt)(model)
        cg_residual = maximum(abs, cg.residual)
        memory_usage = "CPU"
    else
        CUDA.@allowscalar begin
            maximum_w = maximum(abs, w)
            cg_residual = maximum(abs, cg.residual)
            adv_cfl = AdvectiveCFL(s.Δt)(model)
            diff_cfl = DiffusiveCFL(s.Δt)(model)
        end
        memory_usage = log_gpu_memory_usage()
    end

    @info @sprintf(
        "[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, Δt: %.3f, advective CFL: %.2e, diffusive CFL: %.2e, memory_usage: %s, CG residual: %.2e, CG iteration: %d/%d\n",
        progress, iteration, current_time, maximum_w, current_dt, adv_cfl, diff_cfl, memory_usage,
        cg_residual, cg_iter, cg_maxiter
        )
end
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))    # interval is 110s
## Run the simulation
run!(simulation)