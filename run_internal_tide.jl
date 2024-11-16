## This script uses initialize_internal_tide.jl file to initialize and run a simulation.
# The script should be run on a GPU node by submitting a bash run script as a slurm batch job.
include("diagnostics_budget.jl")
include("initialize_internal_tide.jl")

simname = "tilt"
const Nx = 500
const Ny = 1000
const Nz = 250        
const Δtᵒ = 30minutes # interval for saving output
const ω₀ = 1.4e-4     # tidal freq.
const tᶠ = 120*2π/ω₀    # endtime of the simulation: 120 tidal cycle
const θ = 3.6e-3      # slope angle
const U₀ = 0.025      # tidal amplitude
const N = 1.e-3       # Buoyancy frequency
const f₀ = -0.53e-4   # Coriolis frequency
threeD_snapshot_interval = 12Δtᵒ  # effective only when output_mode="analysis"
# closure = SmagorinskyLilly()
closure = (SmagorinskyLilly(), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))
solver = "FFT"
architecture = GPU()
# 3 modes to choose: "spinup", "test", "analysis"
output_mode = "certain-diagnostics"
output_writer = true
clean_checkpoint = false         # cleanup checkpoint
overwrite_output = true          # overwrite existing output (if pickup=true, clean=false, and vice versa)
timerange = "100-120"
# Running the simulation!
# Comment either
    # pickup = false
    pickup = string("output/", simname, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_80-120_checkpoint_iteration160110.jld2")

if pickup == false
    simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                    Δtᵒ=Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, f₀=f₀,
                                    output_mode=output_mode, output_writer=output_writer,
                                    threeD_snapshot_interval=threeD_snapshot_interval, architecture=architecture,
                                    clean_checkpoint=clean_checkpoint, overwrite_output=overwrite_output, 
                                    closure=closure, solver=solver, timerange=timerange)
    run!(simulation)
    checkpointed_wta = simulation.output_writers[:nc_threeD_timeavg].outputs["b"]
    checkpointed_actuations = checkpointed_wta.schedule.actuations
    # Saving actuation to a text file
    open("actuation.txt", "w") do file write(file, string(checkpointed_actuations)) end
else
    # Reading actuation from a text file
    checkpointed_actuations = open("actuation.txt", "r") do file parse(Float64, readline(file)) end
    simulation = initialize_internal_tide(simname, Nx, Ny, Nz; 
                                    Δtᵒ=Δtᵒ, tᶠ=tᶠ, θ=θ, U₀=U₀, N=N, f₀=f₀,
                                    output_mode=output_mode, output_writer=output_writer,
                                    threeD_snapshot_interval=threeD_snapshot_interval, architecture=architecture,
                                    clean_checkpoint=clean_checkpoint, overwrite_output=overwrite_output, 
                                    closure=closure, solver=solver, timerange=timerange)
    simulation.output_writers[:nc_threeD_timeavg].outputs["b"].schedule.actuations = checkpointed_actuations
    run!(simulation; pickup=pickup)
    # Overwrite and save actuation to actuation.txt
    checkpointed_wta = simulation.output_writers[:nc_threeD_timeavg].outputs["b"]
    checkpointed_actuations = checkpointed_wta.schedule.actuations
    open("actuation.txt", "w") do file write(file, string(checkpointed_actuations)) end
end

@info """
    Simulation complete.
"""