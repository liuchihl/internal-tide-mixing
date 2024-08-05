using Oceananigans
using Oceananigans.Units
using Printf

grid = RectilinearGrid(CPU(),size=128, z=(0, 10meters), topology=(Flat, Flat, Bounded))

closure = ScalarDiffusivity(κ=1)

model = NonhydrostaticModel(; grid, closure, tracers=:T)

width = 0.1
initial_temperature(z) = exp(-z^2 / (2width^2))
set!(model, T=initial_temperature)


## Time-scale for diffusion across a grid cell
min_Δz = minimum_zspacing(model.grid)
diffusion_time_scale = min_Δz^2 / model.closure.κ.T
iteration_final = 5minutes
simulation = Simulation(model, Δt = 0.1 * diffusion_time_scale, stop_iteration = iteration_final)

dir = "test_MWE"
## checkpoint  
simulation.output_writers[:checkpointer] = Checkpointer(
    model,
    schedule=IterationInterval(iteration_final),
    dir=dir,
    prefix=string("_checkpoint"),
    cleanup=false)

# simulation.output_writers[:temperature] = NetCDFOutputWriter(model, model.tracers,
#                                         verbose=true,
#                                         filename = string(dir,"/one_dimensional_diffusion.nc"),
#                                         overwrite_existing = false,
#                                         schedule = TimeInterval(1000))

progress_message(s) = @info @sprintf("iteration: %d, time: %.3f, max|w|: %.2e, Δt: %.3f,
                    advective CFL: %.2e, diffusive CFL: %.2e\n",
                    s.model.clock.iteration,
                    s.model.clock.time, maximum(abs, model.velocities.w), s.Δt,
                    AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model))
times = [model.clock.time:0.01:0.5]                    
simulation.callbacks[:progress] = Callback(progress_message, SpecifiedTimes(0.01))
            
file = "test_MWE/_checkpoint_iteration60.jld2"
run!(simulation; pickup=file)


# read file
# file = "MWE/one_dimensional_diffusion.nc"
# ds = Dataset(file,"r")
# t = ds["time"][:]

# diff(t)
# close(ds)