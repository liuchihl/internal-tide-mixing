## This script uses initialize_internal_tide.jl file to initialize and run a simulation.
# The script should be run on a GPU node by submitting a bash run script as a slurm batch job.
include("../functions/diagnostics_budget.jl")
include("../functions/gaussian_particle_generator.jl")
include("../initialize_internal_tide.jl")

function run_internal_tide(tᶠ,θ)
    Nx = 500
    Ny = 1000
    Nz = 250        
    ω₀ = 1.4e-4     # tidal freq.
    T₀ = 2π/ω₀      # tidal period
    Δtᵒ = 1/24*T₀ # interval for saving output: 1/24 of a tidal period (~31 minutes) 
    U₀ = 0.025      # tidal amplitude
    N = 1.e-3       # Buoyancy frequency
    f₀ = -0.53e-4   # Coriolis frequency
    closure = (SmagorinskyLilly(), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))
    architecture = GPU()             # CPU() or GPU() or Distributed(GPU(); partition = Partition(ranks...))
    output_writer = true
    clean_checkpoint = false         # cleanup checkpoint
    overwrite_output = true          # overwrite existing output (if pickup=true, clean=false, and vice versa)

    if θ == 0        # slope angle
        simname = "flat"
    else
        simname = "tilt"
    end
    # 3 modes: "verification", "spinup", "analysis"
    if tᶠ ≤ 10*T₀
        output_mode = "verification"
        solver = "FFT"   
        avg_interval = 1*T₀
        snapshot_interval = 0              # no snapshot
        slice_interval = Δtᵒ
        pickup = false             
    elseif tᶠ ≤ 450*T₀
        output_mode = "spinup"
        solver = "FFT"    
        avg_interval = 10*T₀
        snapshot_interval = 0           # no snapshot
        slice_interval = 13/12*T₀       # snapshot at different point in the tidal cycle
        pickup = true  
        ###### remember to switch back to true after testing this part    
        # pickup = "output/tilt/checkpoint_iteration315391.jld2"      
    else
        output_mode = "analysis"
        solver = "Conjugate Gradient"                
        avg_interval = T₀
        snapshot_interval = 1/12*T₀
        slice_interval = Δtᵒ
        pickup = false  
        
        # set initial condition to be the final state of the spinup simulation by extracting information from the checkpoint file         
    end

    # Running the simulation!
    println("Starting simulation with tᶠ = $tᶠ, θ = $θ")
    println("Mode: $output_mode, Solver: $solver")
  
    if pickup == false && tᶠ ≤ 450*T₀
        simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                        Δtᵒ=Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, f₀=f₀,
                                        output_mode=output_mode, output_writer=output_writer,
                                        architecture=architecture,
                                        clean_checkpoint=clean_checkpoint, overwrite_output=overwrite_output, 
                                        closure=closure, solver=solver, snapshot_interval=snapshot_interval, 
                                        slice_interval=slice_interval, avg_interval=avg_interval)
        run!(simulation)
        checkpointed_wta = simulation.output_writers[:nc_threeD_timeavg].outputs["B"]
        checkpointed_actuations = checkpointed_wta.schedule.actuations
        # Saving actuation to a text file
        open(string("output/",simname,"/actuation.txt"), "w") do file write(file, string(checkpointed_actuations)) end
    else
        # since time-average window is different between modes, actuation has to be recalculated
        actuation = open(string("output/",simname,"/actuation.txt"), "r") do file parse(Float64, readline(file)) end
        if tᶠ == 50*T₀
            # actuation = the endtime of the verification period divided by the average interval in spinup period 
            checkpointed_actuations = round(10*T₀/(avg_interval))
        elseif 50*T₀ < tᶠ ≤ 450*T₀
            # the rest of the time during spinup period
            checkpointed_actuations = actuation
        else 
            # the time during analysis period: we don't pickup checkpointer, 
            # we set initial condition, set!(model,checkpoint_path.jld2), from the end of the spinup period
            pickup = false
            # checkpointed_actuations = round(450*T₀/(avg_interval))
        end
        # Reading actuation from a text file
        simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                        Δtᵒ=Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, f₀=f₀,
                                        output_mode=output_mode, output_writer=output_writer,
                                        architecture=architecture,
                                        clean_checkpoint=clean_checkpoint, overwrite_output=overwrite_output, 
                                        closure=closure, solver=solver, snapshot_interval=snapshot_interval, 
                                        slice_interval=slice_interval, avg_interval=avg_interval)
        simulation.output_writers[:nc_threeD_timeavg].outputs["B"].schedule.actuations = checkpointed_actuations
        
        run!(simulation; pickup=pickup)
        # Overwrite and save actuation to actuation.txt
        checkpointed_wta = simulation.output_writers[:nc_threeD_timeavg].outputs["B"]
        checkpointed_actuations = checkpointed_wta.schedule.actuations
        open(string("output/",simname,"/actuation.txt"), "w") do file write(file, string(checkpointed_actuations)) end
    end
    println("Simulation finished successfully!")
end
