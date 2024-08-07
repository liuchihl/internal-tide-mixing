using Oceananigans
using Oceananigans.Units
using Printf

Ns = 10    # number of time saves
T = 200 # simulation stop time (s)
Δt = 1  # timestep (s)

grid = RectilinearGrid(size=(), topology=(Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Flat))
model = NonhydrostaticModel(; grid, timestepper=:RungeKutta3)
simulation = Simulation(model; Δt, stop_time = T)
progress_message(sim) = @printf("Iteration: %03d, time: %s, Δt: %s, wall time: %s\n",
	iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(sim.run_wall_time))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(10))
dir = "output/test_MWE"
## checkpoint  
simulation.output_writers[:checkpointer] = Checkpointer(
                            model,
                            schedule=TimeInterval(T),
                            dir=dir,
                            prefix=string("checkpoint"),
                            cleanup=true)
# simulation.callbacks[:progress].schedule.actuations = 1
file = string(dir,"/checkpoint_iteration100.jld2")

run!(simulation,pickup=file)
