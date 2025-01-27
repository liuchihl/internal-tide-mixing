## This script uses initialize_internal_tide.jl file to initialize and run a simulation.
# The script should be run on a GPU node by submitting a bash run script as a slurm batch job.
include("../diagnostics_budget.jl")
include("../initialize_internal_tide.jl")

function run_internal_tide(tᶠ,θ)
    Nx = 500
    Ny = 1000
    Nz = 250        
    Δtᵒ = 30minutes # interval for saving output
    ω₀ = 1.4e-4     # tidal freq.
    U₀ = 0.025      # tidal amplitude
    N = 1.e-3       # Buoyancy frequency
    f₀ = -0.53e-4   # Coriolis frequency
    closure = (SmagorinskyLilly(), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))
    architecture = GPU()             # CPU() or GPU() or Distributed(GPU(); partition = Partition(ranks...))
    output_writer = true
    clean_checkpoint = false         # cleanup checkpoint
    overwrite_output = true          # overwrite existing output (if pickup=true, clean=false, and vice versa)
    threeD_snapshot_interval = 12Δtᵒ  # effective only when output_mode="analysis"

    if θ == 0        # slope angle
        simname = "flat"
    else
        simname = "tilt"
    end
    # 3 modes: "verification", "spinup", "analysis"
    if tᶠ ≤ 10*2π/ω₀
        output_mode = "verification"
        solver = "FFT"   
        pickup = false             
    elseif tᶠ ≤ 1010*2π/ω₀
        output_mode = "spinup"
        solver = "FFT"    
        pickup = true            
    else
        output_mode = "analysis"
        solver = "Conjugate Gradient"                
        pickup = true  
        # set initial condition to be the final state of the spinup simulation by extracting information from the checkpoint file         
    end

    # Running the simulation!
    println("Starting simulation with tᶠ = $tᶠ, θ = $θ")
    println("Mode: $output_mode, Solver: $solver")
  
    if pickup == false
        simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                        Δtᵒ=Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, f₀=f₀,
                                        output_mode=output_mode, output_writer=output_writer,
                                        threeD_snapshot_interval=threeD_snapshot_interval, architecture=architecture,
                                        clean_checkpoint=clean_checkpoint, overwrite_output=overwrite_output, 
                                        closure=closure, solver=solver)
        run!(simulation)
        checkpointed_wta = simulation.output_writers[:nc_threeD_timeavg].outputs["B"]
        checkpointed_actuations = checkpointed_wta.schedule.actuations
        # Saving actuation to a text file
        open(string("output/",simname,"/actuation.txt"), "w") do file write(file, string(checkpointed_actuations)) end
    else
        # Reading actuation from a text file
        checkpointed_actuations = open(string("output/",simname,"/actuation.txt"), "r") do file parse(Float64, readline(file)) end
        simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                        Δtᵒ=Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, f₀=f₀,
                                        output_mode=output_mode, output_writer=output_writer,
                                        threeD_snapshot_interval=threeD_snapshot_interval, architecture=architecture,
                                        clean_checkpoint=clean_checkpoint, overwrite_output=overwrite_output, 
                                        closure=closure, solver=solver)
        simulation.output_writers[:nc_threeD_timeavg].outputs["B"].schedule.actuations = checkpointed_actuations
        run!(simulation; pickup=pickup)
        # Overwrite and save actuation to actuation.txt
        checkpointed_wta = simulation.output_writers[:nc_threeD_timeavg].outputs["B"]
        checkpointed_actuations = checkpointed_wta.schedule.actuations
        open(string("output/",simname,"/actuation.txt"), "w") do file write(file, string(checkpointed_actuations)) end
    end
    println("Simulation finished successfully!")
end
