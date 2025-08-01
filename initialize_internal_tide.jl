using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver, FourierTridiagonalPoissonSolver, FFTBasedPoissonSolver, AsymptoticPoissonPreconditioner
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
using StructArrays: StructArray
using NCDatasets

struct particles_analysis_period
    x::Float64  # x-coordinate
    y::Float64  # y-coordinate
    z::Float64  # z-coordinate
    b::Float64  # buoyancy
end

function initialize_internal_tide(
    simname,
    Nx,
    Ny,
    Nz;
    Δtᵒ=30minutes,
    tᶠ=6 * 2π / ω₀,
    θ=3.6e-3,
    U₀=0.025,
    N=1.e-3,
    f₀=-0.53e-4,
    architecture=GPU(),
    closure=SmagorinskyLilly(),
    solver="FFT",
    output_mode="test",
    output_writer=true,
    avg_interval=avg_interval,
    snapshot_interval=snapshot_interval,
    slice_interval=slice_interval,
    topo_file="topo.mat",
    clean_checkpoint="false",
    overwrite_output="true",
    analysis_round=analysis_round
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
    z_faces(k) = -H * (ζ(k, Nz, 1.2) * Σ(k, Nz, 15) - 1)

    grid = RectilinearGrid(architecture, size=(Nx, Ny, Nz),
        x=(0, Lx),
        y=(0, Ly),
        z=z_faces,
        halo=(4, 4, 4),
        topology=(Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded)
    )
    zC = adapt(Array, znodes(grid, Center()))

    # load topography 
    file = matopen(topo_file)
    z_topo = read(file, "z_noslope_periodic")
    x_topo = read(file, "x_domain")
    y_topo = read(file, "y_domain")
    # grids has to be evenly spaced
    x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo, 1))
    y_topo_lin = range(y_topo[1], y_topo[end], size(z_topo, 2))
    close(file)
    # high-resolution grids
    x_interp = range(x_topo[1], x_topo[end], length=Nx)
    y_interp = range(y_topo[1], y_topo[end], length=Ny)


    # Interpolation object (caches coefficients and such)
    itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
    # Interpolate z_topo onto a higher-resolution grid
    z_interp = [itp(x, y) for x in x_interp, y in y_interp]
    z_interp = z_interp .- minimum(z_interp)

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
    @inline drag_u(x, y, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
    @inline drag_v(x, y, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
    @inline immersed_drag_u(x, y, z, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
    @inline immersed_drag_v(x, y, z, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
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
        top=GradientBoundaryCondition(0.), # ∇B⋅ẑ = ∂B̄∂ẑ → ∂b∂z = 0 and ∂b∂x = 0 (periodic)
        immersed=B_bcs_immersed)
    ## Notes:
    # (1) directions are defined relative to domain coordinates.
    # (2) Gradients are positive in the direction of the coordinate.

    # Tidal forcing
    U₀ = U₀

    ω₀ = 1.4e-4  # tidal frequency
    u_tidal_forcing(x, y, z, t) = U₀ * ω₀ * sin(ω₀ * t)

    # IC such that flow is in phase with predicted linear response, but otherwise quiescent
    Uᵣ = U₀ * ω₀^2 / (ω₀^2 - f₀^2 - (N * sin(θ))^2) # quasi-resonant linear barotropic response
    uᵢ(x, y, z) = -Uᵣ
    vᵢ(x, y, z) = 0.
    bᵢ(x, y, z) = 1e-9 * rand() # seed infinitesimal random perturbations in the buoyancy field


    # set tracer and particles
    if output_mode == "analysis"
        # tracer initial distribution
        x_center_cart = 0.25 * Lx # in meters
        y_center_cart = 0.5 * Ly # in meters
        if analysis_round == 1
            z_center_cart = 1000
        elseif analysis_round == "all"
            z_center_cart = 1000
        end
        σ_x = 1000  # in meters
        σ_y = 1000
        σ_z = 50
        C = 1.0  # Amplitude of the tracer concentration at the center

        # Create a function that takes tilted coordinates (x', y', z') and returns concentration
        function cᵢ(x_tilted, y_tilted, z_tilted)
            # Transform tilted coordinates back to Cartesian coordinates
            x_cart = x_tilted * cos(θ) - z_tilted * sin(θ)
            y_cart = y_tilted  # y coordinate doesn't change in the transform
            z_cart = x_tilted * sin(θ) + z_tilted * cos(θ)

            # Calculate concentration using Cartesian coordinates
            return C * exp(-((x_cart - x_center_cart)^2 / (2σ_x^2) +
                             (y_cart - y_center_cart)^2 / (2σ_y^2) +
                             (z_cart - z_center_cart)^2 / (2σ_z^2)))
        end

        # cᵢ(x, y, z) = C * exp(-((x - x_center)^2 / (2σ_x^2) + (y - y_center)^2 / (2σ_y^2) + (z - z_center)^2 / (2σ_z^2)))

        # define particles
            # 451-451.5, invalid particles (e.g., below topography, outside domain) are removed,
            # after 451.5, since we are picking up from a checkpoint file, there is no need to 
            # exclude the out-of-bounds particles (may cause particles number inconsistency). 
            # Applying periodic boundary conditions to particles is necessary; when apply periodic bounds, no vertical criterion
            # is applied as well. 
            # 
        apply_periodic_bounds = tᶠ == 451.5 * 2π / ω₀ ? false : true 

        # all tracers and particles
        if tᶠ <= 451*2π/ω₀ # before 451, the model is still spinning up from the FFT solver, so we don't need c and paritcles
            tracers = (; b=CenterField(grid))
            particles = nothing

        elseif tᶠ <= 458*2π/ω₀ # beyond 451 and included, we will always need c and particles    
            Nparticles = 499829 # number of particles to generate (this is the number of particles in the checkpoint file of 452.5)
            # particles are released at 1000 m above the bottom, which is about z=967 m
            x₀, y₀, z₀ = gaussian_particle_generator(
                Nparticles, Lx, Nx, Ly, Ny, z_interp, architecture, H;
                x_center_ratio=0.25, y_center_ratio=0.5, z_center=z_center_cart,
                σ_x=σ_x, σ_y=σ_y, σ_z=σ_z, apply_periodic_bounds=apply_periodic_bounds)
            b = 1e-5 * ones(Float64, Int(length(x₀)))
            b = architecture == GPU() ? CuArray(b) : b
            tracers = (; b=CenterField(grid), c=CenterField(grid))
            # Define tracked fields as a NamedTuple
            lagrangian_particles = StructArray{particles_analysis_period}((x₀, y₀, z₀, b))
            tracked_fields = (; b=tracers.b)
            particles = LagrangianParticles(lagrangian_particles; tracked_fields=tracked_fields, restitution=1)                

        elseif tᶠ > 458*2π/ω₀ && simname=="tilt"  # after 458, we want to generate additional particles at the original points and continue tracking the previous particles
            Nparticles = 499829 # number of particles to generate (this is the number of particles in the checkpoint file of 452.5)
            x₀1, y₀1, z₀1 = gaussian_particle_generator(
                Nparticles, Lx, Nx, Ly, Ny, z_interp, architecture, H;
                x_center_ratio=0.25, y_center_ratio=0.5, z_center=z_center_cart,
                σ_x=σ_x, σ_y=σ_y, σ_z=σ_z, apply_periodic_bounds=apply_periodic_bounds)
            b1 = 1e-5 * ones(Float64, Int(length(x₀1)))
            b1 = architecture == GPU() ? CuArray(b1) : b1
            # additional to the particles that have already traveled, create additional particles at the original point
            # to compare with the new tracer release at original point
            x₀2, y₀2, z₀2 = gaussian_particle_generator(
                250000, Lx, Nx, Ly, Ny, z_interp, architecture, H;
                x_center_ratio=0.25, y_center_ratio=0.5, z_center=z_center_cart,
                σ_x=σ_x, σ_y=σ_y, σ_z=σ_z, apply_periodic_bounds=apply_periodic_bounds)
            b2 = 1e-5 * ones(Float64, Int(length(x₀2)))
            b2 = architecture == GPU() ? CuArray(b2) : b2
            # concatenate both sets of particles
            x₀ = vcat(x₀1, x₀2)
            y₀ = vcat(y₀1, y₀2)
            z₀ = vcat(z₀1, z₀2)
            b = vcat(b1, b2)
            tracers = (; b=CenterField(grid), c=CenterField(grid))
            # Define tracked fields as a NamedTuple
            lagrangian_particles = StructArray{particles_analysis_period}((x₀, y₀, z₀, b))
            tracked_fields = (; b=tracers.b)
            particles = LagrangianParticles(lagrangian_particles; tracked_fields=tracked_fields, restitution=1)                
        end
    end
    # s = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))

    # Rotate gravity vector
    buoyancy = BuoyancyForce(BuoyancyTracer(), gravity_unit_vector=-[ĝ...])
    coriolis = ConstantCartesianCoriolis(f=f₀, rotation_axis=ĝ)

    # Linear background stratification (in ẑ)
    @inline ẑ(x, z, ĝ) = x * ĝ[1] .+ z * ĝ[3]
    @inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
    B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N²=N^2))

    # set model
    if solver == "FFT"
        model = NonhydrostaticModel(;
            grid=grid,
            advection=WENO(),
            buoyancy=buoyancy,
            coriolis=coriolis,
            boundary_conditions=(u=u_bcs, v=v_bcs, b=B_bcs,),
            forcing=(u=u_tidal_forcing,),
            closure=closure,
            particles=output_mode !== "analysis" ? nothing : particles,
            tracers=output_mode !== "analysis" ? :b : tracers,
            timestepper=:RungeKutta3,
            hydrostatic_pressure_anomaly=CenterField(grid),
            background_fields=Oceananigans.BackgroundFields(; background_closure_fluxes=true, b=B̄_field),
        )
    else
        solver == "Conjugate Gradient"
        # this is for analysis period because CG solver is much slower than FFT solver but more accurate near the boundaries
        tol = 1e-9
        # add particles
        model = NonhydrostaticModel(;
            grid=grid,
            pressure_solver=ConjugateGradientPoissonSolver(
                grid; maxiter=500, preconditioner=AsymptoticPoissonPreconditioner(),
                reltol=tol, abstol=tol), 
            advection=WENO(),
            buoyancy=buoyancy,
            coriolis=coriolis,
            boundary_conditions=(u=u_bcs, v=v_bcs, b=B_bcs,),
            forcing=(u=u_tidal_forcing,),
            closure=closure,
            particles=particles,
            tracers=tracers,
            timestepper=:RungeKutta3,
            hydrostatic_pressure_anomaly=CenterField(grid),
            background_fields=Oceananigans.BackgroundFields(; background_closure_fluxes=true, b=B̄_field)
        )
    end

    # initialize the model
    if output_mode !== "analysis"    # before analysis period, set initial condition as usual because we are picking up from a checkpoint
        set!(model, b=bᵢ, u=uᵢ, v=vᵢ)
    else    # during analysis period, set initial condition from the final checkpoint of the spinup period
        function find_checkpoint_by_rank(dir, rank=1)
            # Get all jld2 files in the directory
            checkpoint_files = filter(f -> endswith(f, ".jld2"), readdir(dir))
            # Extract iteration numbers using regex
            # Filter out any files that contain "analysis"
            checkpoint_files = filter(f -> !occursin("analysis", f), checkpoint_files)
            if isempty(checkpoint_files)
                error("No valid checkpoint files found")
            end
            if length(checkpoint_files) < rank
                error("Need at least $rank checkpoint files to find rank $rank")
            end
            
            iter_numbers = map(f -> parse(Int, match(r"iteration(\d+)", f)[1]), checkpoint_files)
            
            # Sort iteration numbers in descending order and get the specified rank
            sorted_indices = sortperm(iter_numbers, rev=true)
            target_idx = sorted_indices[rank]
            
            return joinpath(dir, checkpoint_files[target_idx])
        end
        if tᶠ <= 451*2π/ω₀ # without c and particles 
                checkpoint_file = find_checkpoint_by_rank(string("output/", simname),1)
                set!(model, checkpoint_file)
        elseif tᶠ < 452*2π/ω₀ 
                checkpoint_file = find_checkpoint_by_rank(string("output/", simname),1)
                set!(model, checkpoint_file)
                set!(model, c=cᵢ)
        elseif tᶠ <= 458*2π/ω₀  # c and particles are present in the checkpoint file
                checkpoint_file = find_checkpoint_by_rank(string("output/", simname),1)
                set!(model, checkpoint_file)
        elseif tᶠ == 458.5*2π/ω₀ && simname == "tilt"
                @info "Before loading checkpoint: $(length(model.particles.properties.x)) particles"
                checkpoint_file = find_checkpoint_by_rank(string("output/", simname),1)
                set!(model, checkpoint_file) 
                @info "After loading checkpoint: $(length(model.particles.properties.x)) particles"
                # reset initial condition for c (set to the correct position)
                set!(model, c=cᵢ)
                
        elseif tᶠ > 458.5*2π/ω₀  # no need to reset c, just use the checkpoint file to set initial condition
                checkpoint_file = find_checkpoint_by_rank(string("output/", simname),1)
                set!(model, checkpoint_file)
        end
    end



    ## Configure simulation
    Δt = 12
    # Δt = (1/N)*0.03
    simulation = Simulation(model, Δt=Δt, stop_time=tᶠ + 20Δt, minimum_relative_step=0.01)
    # add 50Δt to ensure the simulation runs past the final time average window to avoid missing averaged data
    # minimum_relative_step=0.1: eliminates the possibility of the time-step being too small.
    
    # # The `TimeStepWizard` manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy
    # # (CFL) number close to `0.5` while ensuring the time-step does not increase beyond the
    # # maximum allowable value for numerical stability. However, window time average would not work properly if wizard is used.

    # wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.2)
    # simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    # set the ouput mode:
    if output_mode == "verification"
        b = model.tracers.b
        B̄ = model.background_fields.tracers.b
        B = B̄ + b # total buoyancy field
        u, v, w = model.velocities
        û = @at (Face, Center, Center) u * ĝ[3] - w * ĝ[1] # true zonal velocity
        ŵ = @at (Center, Center, Face) w * ĝ[3] + u * ĝ[1] # true vertical velocity
        ε = KineticEnergyDissipationRate(model)
        χ = TracerVarianceDissipationRate(model, :b)

        checkpoint_interval = 5 * 2π / ω₀
        slice_diags = (; uhat=û, B=B, b=b, ε=ε, χ=χ)
        threeD_diags_avg = (; uhat=û, what=ŵ, B=B, b=b)
    elseif output_mode == "spinup"
        b = model.tracers.b
        B̄ = model.background_fields.tracers.b
        B = B̄ + b # total buoyancy field
        u, v, w = model.velocities
        û = @at (Face, Center, Center) u * ĝ[3] - w * ĝ[1] # true zonal velocity
        ŵ = @at (Center, Center, Face) w * ĝ[3] + u * ĝ[1] # true vertical velocity

        checkpoint_interval = 20 * 2π / ω₀
        slice_diags = (; uhat=û, what=ŵ, B=B)
        threeD_diags_avg = (; uhat=û, v=v, what=ŵ, B=B)
    elseif output_mode == "analysis"
        b = model.tracers.b
        B̄ = model.background_fields.tracers.b
        B = B̄ + b # total buoyancy field
        if tᶠ > 451*2π/ω₀
            c = model.tracers.c
        end
        u, v, w = model.velocities
        û = @at (Face, Center, Center) u * ĝ[3] - w * ĝ[1] # true zonal velocity
        ŵ = @at (Center, Center, Face) w * ĝ[3] + u * ĝ[1] # true vertical velocity
        νₑ = simulation.model.diffusivity_fields[1].νₑ    # eddy viscosity
        Bz = @at (Center, Center, Center) ∂z(B)
        # Oceanostics
        ε = KineticEnergyDissipationRate(model)
        χ = TracerVarianceDissipationRate(model, :b)
        Rig = RichardsonNumber(model, u, v, w, B, .-model.buoyancy.gravity_unit_vector)
        Bbudget = get_budget_outputs_tuple(model;)
        # force garbage collection, free some memory
        if architecture == GPU()
            CUDA.@sync begin
                GC.gc()
                CUDA.reclaim()
            end
        end

        if analysis_round !== "all"
            #1)first round output 
            checkpoint_interval = 0.6 * 2π / ω₀
            threeD_diags_avg = (; b=b)
            slice_diags = (; B=B, uhat=û)
        elseif analysis_round == "all"
            checkpoint_interval = 0.5 * 2π / ω₀
            point_diags = (; uhat=û, what=ŵ, B=B)
            # threeD_diags_avg = merge(Bbudget, (; ε=ε, χ=χ, uhat=û, v=v, what=ŵ, B=B))
            threeD_diags = merge(Bbudget, (; ε=ε, νₑ=νₑ, Rig=Rig, χ=χ, uhat=û, v=v, what=ŵ, B=B, c=c))
            slice_diags = (; uhat=û, v=v, what=ŵ, B=B, ε=ε, χ=χ, νₑ=νₑ)
        end
    elseif output_mode == "customized"
        checkpoint_interval = 20 * 2π / ω₀
        # threeD_diags = (; Bz=Bz, what=ŵ, u=u)        
        # slice_interval = Δtᵒ
        threeD_diags_avg = Bbudget
    end
    fname = string("internal_tide_theta=", θ, "_Nx=", Nx, "_Nz=", Nz, "_tᶠ=", round(tᶠ / (2π / ω₀), digits=1),
             "_analysis_round=", analysis_round)
    dir = string("output/", simname, "/")
    # create output path if the folder does not exist
    if !isdir(dir)
        mkdir(dir)
    end
    if output_writer
        
        ## output 3D field window time average
        # simulation.output_writers[:nc_threeD_timeavg] = NetCDFWriter(model, threeD_diags_avg,
        #     filename=string(dir, fname, "_threeD_timeavg.nc"),
        #     schedule=AveragedTimeInterval(avg_interval, window=avg_interval, stride=1),
        #     verbose=true,
        #     overwrite_existing=overwrite_output
        # )

        ## output 2D slices
        # xz
        simulation.output_writers[:nc_slice_xz] = NetCDFWriter(model, slice_diags,
            schedule=TimeInterval(slice_interval),
            indices=(:, Ny ÷ 2, :), # center of the domain (along thalweg)
            verbose=true,
            filename=string(dir, fname, "_slices_xz.nc"),
            overwrite_existing=overwrite_output)

        ## output that is saved only when reaching analysis period (quasi-equilibrium in terms of bottom buoyancy)
        if output_mode == "analysis"
            if analysis_round == "all"
                # xy
                ind = argmin(abs.(zC .- 1300))   # 1300 m height above bottom
                simulation.output_writers[:nc_slice_xy] = NetCDFWriter(model, slice_diags,
                    schedule=TimeInterval(slice_interval),
                    indices=(:, :, ind),
                    verbose=true,
                    filename=string(dir, fname, "_slices_xy.nc"),
                    overwrite_existing=overwrite_output)
                # yz
                simulation.output_writers[:nc_slice_yz] = NetCDFWriter(model, slice_diags,
                    schedule=TimeInterval(slice_interval),
                    indices=(Nx ÷ 2, :, :), # center of the domain (along the sill)
                    verbose=true,
                    filename=string(dir, fname, "_slices_yz.nc"),
                    overwrite_existing=overwrite_output)
                # output 3D field snapshots
                simulation.output_writers[:nc_threeD] = NetCDFWriter(model, threeD_diags,
                    verbose=true,
                    filename=string(dir, fname, "_threeD.nc"),
                    overwrite_existing=overwrite_output,
                    schedule=TimeInterval(snapshot_interval))
                # 1D profile
                simulation.output_writers[:nc_point] = NetCDFWriter(model, point_diags,
                    schedule=TimeInterval(5Δt),
                    indices=(Nx ÷ 2, Ny ÷ 2, :), # center of the domain (at the sill)
                    verbose=true,
                    filename=string(dir, fname, "_point_center.nc"),
                    overwrite_existing=overwrite_output)
                # particles
                simulation.output_writers[:particles] = NetCDFWriter(model, (particles=model.particles,),
                    verbose=true,
                    filename=string(dir, fname, "_particles_z=", z_center_cart, ".nc"),
                    schedule=TimeInterval(Δtᵒ / 3),
                    overwrite_existing=true)
            end
            
        end

        # checkpoint  
        checkpoint_prefix = "checkpoint"
        simulation.output_writers[:checkpointer] = Checkpointer(
            model,
            schedule=TimeInterval(checkpoint_interval),
            dir=dir,
            prefix=checkpoint_prefix,
            cleanup=clean_checkpoint)

    end
    ### Progress messages

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
            CUDA.synchronize()
            # Force cleanup before checking memory usage
            GC.gc()
            CUDA.reclaim()
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
    simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(10Δt))    # interval is 110s
    return simulation
end
