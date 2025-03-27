using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver, FourierTridiagonalPoissonSolver, AsymptoticPoissonPreconditioner
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: all_reduce
using LinearAlgebra
using Adapt
using MAT
using Statistics
using Oceanostics
using Oceanostics.TKEBudgetTerms: BuoyancyProductionTerm
using Interpolations: LinearInterpolation
using CUDA
using Suppressor


function initialize_internal_tide(
    simname,
    Nx,
    Ny,
    Nz;
    Δtᵒ=30minutes,
    tᶠ=6*2π/ω₀,
    θ=3.6e-3,
    U₀=0.025,
    N=1.e-3,
    f₀ = -0.53e-4,
    threeD_snapshot_interval=2Δtᵒ,
    architecture=GPU(),
    closure = SmagorinskyLilly(),
    solver = "FFT",
    timerange = "",
    output_mode = "test",
    output_writer = true,
    topo_file = "topo.mat",
    clean_checkpoint = "false",
    overwrite_output = "true"
)


function log_gpu_memory_usage()
    return @capture_out CUDA.memory_status()  # retrieve raw string status
end

## Simulation parameters
 H = 2.25kilometers # vertical extent
 Lx = 15kilometers # along-canyon extent
 Ly = 30kilometers # cross-canyon extent

## Create vertical grid
# Creates a vertical grid with near-constant spacing `refinement * Lz / Nz` near the bottom:
# "Warped" coordinate
kwarp(k, N) = (N + 1 - k) / N
# Linear near-surface generator
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
# Bottom-intensified stretching function
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))
# Generating function
z_faces(k) = - H * (ζ(k, Nz, 1.2) * Σ(k, Nz, 15) - 1)

grid = RectilinearGrid(architecture,size=(Nx, Ny, Nz), 
        x = (0, Lx),
        y = (0, Ly), 
        z = z_faces,
        halo = (4,4,4),
        topology = (Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded)
)
# yᶜ = ynodes(grid, Center())
# Δyᶜ = yspacings(grid, Center())
# zC = adapt(Array,znodes(grid, Center()))

# load topography 
file = matopen(topo_file)
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

# Environmental parameters
ĝ = (sin(θ), 0, cos(θ)) # the vertical (oriented opposite gravity) unit vector in rotated coordinates

# Create immersed boundary grid
z_interp_data = architecture == CPU() ? z_interp : CuArray(z_interp)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(z_interp_data))

# setting quadratic drag BC at domain bottom and top of the immersed boundary
 z₀ = 0.1 # m (roughness length)
 κ_von = 0.4  # von Karman constant
 z₁ = architecture == CPU() ? first(znodes(grid, Center())) :
     CUDA.@allowscalar(first(znodes(grid, Center())))
     


cᴰ = (κ_von / log(z₁ / z₀))^2 # Drag coefficient
# non-immersed and immersed boundary conditions
@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v
@inline immersed_drag_u(x, y, z, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
@inline immersed_drag_v(x, y, z, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))
immersed_drag_bc_u = FluxBoundaryCondition(immersed_drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
immersed_drag_bc_v = FluxBoundaryCondition(immersed_drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))

u_immerse = ImmersedBoundaryCondition(bottom=immersed_drag_bc_u)
v_immerse = ImmersedBoundaryCondition(bottom=immersed_drag_bc_v)

# u_bcs = FieldBoundaryConditions(bottom = drag_bc_u, top = FluxBoundaryCondition(nothing), immersed=u_immerse)
# v_bcs = FieldBoundaryConditions(bottom = drag_bc_v, top = FluxBoundaryCondition(nothing), immersed=v_immerse)
u_bcs = FieldBoundaryConditions(bottom = drag_bc_u, top = FluxBoundaryCondition(nothing))
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v, top = FluxBoundaryCondition(nothing))
# tracer: no-flux boundary condition
∂B̄∂z = N^2*cos(θ)
∂B̄∂x = N^2*sin(θ)
# In the following comments, ẑ=slope-normal unit vector and x̂=cross-slope unit vector
B_bcs_immersed = ImmersedBoundaryCondition(
        bottom = GradientBoundaryCondition(0), # ∇B⋅ẑ = 0 → ∂B∂z = 0 → ∂b∂z = -∂B̄∂z
          west = GradientBoundaryCondition(-∂B̄∂x), # ∇B⋅x̂ = 0 → ∂B∂x = 0 → ∂b∂x = -∂B̄∂x
          east = GradientBoundaryCondition(-∂B̄∂x)) # ∇B⋅x̂ = 0 → ∂B∂x = 0 → ∂b∂x = -∂B̄∂x

B_bcs = FieldBoundaryConditions(
          bottom = GradientBoundaryCondition(0), # ∇B⋅ẑ = 0 → ∂B∂z = 0 → ∂b∂z = -∂B̄∂z
             top = GradientBoundaryCondition(∂B̄∂z) # ∇B⋅ẑ = ∂B̄∂ẑ → ∂b∂z = 0 and ∂b∂x = 0 (periodic)
        );
# B_bcs = FieldBoundaryConditions(
#           bottom = GradientBoundaryCondition(-∂B̄∂z), # ∇B⋅ẑ = 0 → ∂B∂z = 0 → ∂b∂z = -∂B̄∂z
#              top = GradientBoundaryCondition(0.), # ∇B⋅ẑ = ∂B̄∂ẑ → ∂b∂z = 0 and ∂b∂x = 0 (periodic)
#         immersed = B_bcs_immersed);
## Notes:
# (1) directions are defined relative to domain coordinates.
# (2) Gradients are positive in the direction of the coordinate.

# Tidal forcing
 U₀ = U₀
 ω₀ = 1.4e-4
u_tidal_forcing(x, y, z, t) = U₀*ω₀*sin(ω₀*t)

# IC such that flow is in phase with predicted linear response, but otherwise quiescent
Uᵣ = U₀ * ω₀^2/(ω₀^2 - f₀^2 - (N*sin(θ))^2) # quasi-resonant linear barotropic response
uᵢ(x, y, z) = -Uᵣ
vᵢ(x, y, z) = 0.
# bᵢ(x, y, z) = 1e-9*rand() # seed infinitesimal random perturbations in the buoyancy field
s = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))

# Rotate gravity vector
buoyancy = BuoyancyForce(BuoyancyTracer(), gravity_unit_vector = -[ĝ...])
coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)

# Linear background stratification (in ẑ)
@inline ẑ(x, z, ĝ) = x*ĝ[1] .+ z*ĝ[3]
@inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
# B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

Bᵢ(x, y, z) = constant_stratification(x, y, z, 0, (; N² = N^2, ĝ=ĝ)) + 1e-9*rand()   # background + perturbation (only works in flat)

if solver == "FFT"
    model = NonhydrostaticModel(
        grid = grid,
        advection = WENO(),
        buoyancy = buoyancy,
        coriolis = coriolis,
        boundary_conditions=(u=u_bcs, v=v_bcs,  b = B_bcs,),
        forcing = (u = u_tidal_forcing,),
        closure = closure,
        tracers = :b,
        timestepper = :RungeKutta3,
        hydrostatic_pressure_anomaly = CenterField(grid),
        # background_fields = Oceananigans.BackgroundFields(; background_closure_fluxes=true, b=B̄_field),
    )
else solver == "Conjugate Gradient"
    tol = 1e-9
    model = NonhydrostaticModel(;
        grid=grid,
        pressure_solver = ConjugateGradientPoissonSolver(
                grid; maxiter=100, preconditioner=AsymptoticPoissonPreconditioner(),
                reltol=tol),
        advection = WENO(),
        buoyancy = buoyancy,
        coriolis = coriolis,
        boundary_conditions=(u=u_bcs, v=v_bcs, b = B_bcs,),
        forcing = (u = u_tidal_forcing,),
        closure = closure,
        tracers = :b,
        timestepper = :RungeKutta3,
        hydrostatic_pressure_anomaly = CenterField(grid),
        # background_fields = Oceananigans.BackgroundFields(; background_closure_fluxes=true, b=B̄_field),
    )
end    
set!(model, b=Bᵢ, u=uᵢ, v=vᵢ)

## Configure simulation
Δt = (1/N)*0.03
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

# # The `TimeStepWizard` manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy
# # (CFL) number close to `0.5` while ensuring the time-step does not increase beyond the
# # maximum allowable value for numerical stability.

wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.2)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


## Diagnostics
B = model.tracers.b
u, v, w = model.velocities
û = @at (Face, Center, Center) u*ĝ[3] - w*ĝ[1] # true zonal velocity
ŵ = @at (Center, Center, Face) w*ĝ[3] + u*ĝ[1] # true vertical velocity

Bz = @at (Center, Center, Center) ∂z(B)
# uz = Field(∂z(û))
# Rig = RichardsonNumber(model; location=(Center, Center, Face), add_background=true)

# Oceanostics
# KE = KineticEnergy(model)
# PE = PotentialEnergy(model)
# ε = KineticEnergyDissipationRate(model)
# χ = TracerVarianceDissipationRate(model, :b)

# eddy viscosity 
# νₑ = simulation.model.diffusivity_fields.νₑ
# buoyancy budget
# Bbudget=get_budget_outputs_tuple(model;)

# testing divergence 
# udiv = KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, model.grid, u, v, w)

# set the ouput mode:
if output_mode == "spinup"
        checkpoint_interval = 20*2π/ω₀
        slice_diags = (; uhat=û, B=B, b=b)
        threeD_diags = (; uhat=û, what=ŵ, B=B, b=b)
        # threeD_diags = (; B=B, b=b)
        threeD_buoyancy = (; B=B,)
elseif output_mode == "test"
        checkpoint_interval = 2*2π/ω₀
        # slice_diags = (; ε, χ, uhat=û, B=B, b=b, Bz=Bz, uhat_z=uz)
        # threeD_diags = (; ε, χ, uhat=û, what=ŵ,  B=B, b=b, Bz=Bz, uhat_z=uz)
        slice_diags = (; χ, uhat=û, B=B, b=b)
        threeD_diags = (; χ, uhat=û, what=ŵ,  B=B, b=b)
elseif output_mode == "certain-diagnostics"
        checkpoint_interval = 20*2π/ω₀
        # slice_diags = (; Bz=Bz, what=ŵ,)
        threeD_diags = (; Bz=Bz, what=ŵ,)
        
        
else output_mode == "analysis"
        checkpoint_interval = 20*2π/ω₀
        slice_diags = (; ε, χ, uhat=û, what=ŵ, v=v, B=B, b=b, Bz=Bz, uhat_z=uz, Rig=Rig)
        point_diags = (; ε, χ, uhat=û, what=ŵ, v=v, B=B, b=b, Bz=Bz, uhat_z=uz, Rig=Rig)
        threeD_diags = (; uhat=û, what=ŵ, v=v, B=B, b=b, Bz=Bz)

end
fname = string("internal_tide_theta=",string(θ),"_realtopo3D_Nx=",Nx,"_Nz=",Nz,"_",timerange)
dir = string("output/",simname, "/")
if output_writer
    ## checkpoint  
    # simulation.output_writers[:checkpointer] = Checkpointer(
    #                                     model,
    #                                     schedule=TimeInterval(checkpoint_interval),
    #                                     dir=dir,
    #                                     prefix=string(fname, "_checkpoint"),
    #                                     cleanup=clean_checkpoint)

    ## output 3D field window time average
    # tidal_period = 2π/ω₀ 
    simulation.output_writers[:nc_threeD_timeavg] = NetCDFOutputWriter(model, threeD_diags,
                                        verbose=true,
                                        filename = string(dir, fname, "_threeD_timeavg.nc"),
                                        overwrite_existing = overwrite_output,
                                        # schedule = AveragedTimeInterval(1tidal_period, window=1tidal_period, stride=1),
                                        schedule = AveragedTimeInterval(1hour, window=1hour, stride=1),






                                        # indices = (:,Ny÷2,:) # take this out when running real simulation
                                        )
    # simulation.output_writers[:nc_threeD_timeavg_Bbudget] = NetCDFOutputWriter(model, Bbudget,
    #                                     verbose=true,
    #                                     filename = string(dir, fname, "_threeD_timeavg_Bbudget.nc"),
    #                                     overwrite_existing = overwrite_output,
    #                                     schedule = AveragedTimeInterval(1tidal_period, window=1tidal_period, stride=1),
    #                                     # indices = (:,Ny÷2,:) # take this out when running real simulation
    #                                     )

    # output 3D field snapshots
    # simulation.output_writers[:nc_threeD] = NetCDFOutputWriter(model, threeD_diags,
    #                                         verbose=true,
    #                                         filename = string(dir, fname, "_threeD.nc"),
    #                                         overwrite_existing = overwrite_output,
    #                                         schedule = TimeInterval(threeD_snapshot_interval))


    ## output 2D slices
    #1) xz
    # simulation.output_writers[:nc_slice_xz] = NetCDFOutputWriter(model, slice_diags,
    #                                         schedule = TimeInterval(Δtᵒ),
    #                                         indices = (:,Ny÷2,:), # center of the domain (along thalweg)
    #                                         verbose=true,
    #                                         filename = string(dir, fname, "_slices_xz.nc"),
    #                                         overwrite_existing = overwrite_output)
    # #2) xy
    # ind = argmin(abs.(zC .- 1300))   # 1300 m height above bottom
    # simulation.output_writers[:nc_slice_xy] = NetCDFOutputWriter(model, slice_diags,
    #                                         schedule = TimeInterval(Δtᵒ),
    #                                         indices = (:,:,ind),
    #                                         verbose=true,
    #                                         filename = string(dir, fname, "_slices_xy.nc"),
    #                                         overwrite_existing = overwrite_output)
    #3) yz
    # simulation.output_writers[:nc_slice_yz] = NetCDFOutputWriter(model, slice_diags,
    #                                         schedule = TimeInterval(Δtᵒ),
    #                                         indices = (Nx÷2,:,:), # center of the domain (along the sill)
    #                                         verbose=true,
    #                                         filename = string(dir, fname, "_slices_yz.nc"),
    #                                         overwrite_existing = overwrite_output)
    # save 3D snapshots of buoyancy fields
    # simulation.output_writers[:nc_threeD] = NetCDFOutputWriter(model, merge(Bbudget,threeD_buoyancy),
    #                                         verbose=true,
    #                                         filename = string(dir, fname, "_threeD.nc"),
    #                                         overwrite_existing = overwrite_output,
    #                                         schedule = TimeInterval(tidal_period÷2) )

    ## output that is saved only when reaching quasi-equilibrium
    if output_mode=="analysis"
        # output 3D field snapshots
        # simulation.output_writers[:nc_threeD] = NetCDFOutputWriter(model, threeD_diags,
        #                                         verbose=true,
        #                                         filename = string(dir, fname, "_threeD.nc"),
        #                                         overwrite_existing = overwrite_output,
        #                                         schedule = TimeInterval(threeD_snapshot_interval))
        # 1D profile
        # simulation.output_writers[:nc_point] = NetCDFOutputWriter(model, point_diags,
        #                                         schedule = TimeInterval(Δtᵒ÷30),
        #                                         indices = (Nx÷2,Ny÷2,:), # center of the domain (at the sill)
        #                                         verbose=true,
        #                                         filename = string(dir, fname, "_point_center.nc"),
        #                                         overwrite_existing = overwrite_output)
    end
end
### Progress messages

        # progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, Δt: %.3f,
        #                     advective CFL: %.2e, diffusive CFL: %.2e, gpu_memory_usage:%s\n",
        #                     100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
        #                     s.model.clock.time, maximum(abs, model.velocities.w), s.Δt,
        #                     AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model)
        #                     ,log_multi_gpu_memory_usage())
                            # log_gpu_memory_usage()
                            # multi_gpu_memory_info
        progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, Δt: %.3f,
                            advective CFL: %.2e, diffusive CFL: %.2e, %s\n",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w), s.Δt,
                            AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model),
                            log_gpu_memory_usage())

        # progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, Δt: %.3f,
        #                     advective CFL: %.2e, diffusive CFL: %.2e \n",
        #                     100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
        #                     s.model.clock.time, maximum(abs, model.velocities.w), s.Δt,
        #                     AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model)
        #                     )


# simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))


    return simulation
end