# this is a MWE for reproducing discontinuity of AveragedTimeInterval
using Oceananigans
using Printf
using Plots

""" Set up a simple simulation to test picking up from a checkpoint. """

function test_simulation(stop_time, Δt, δt, overwrite)
    grid = RectilinearGrid(size=(2), z = (-1,1), topology=(Oceananigans.Flat, Oceananigans.Flat, Oceananigans.Bounded))
    uᵢ(z) = 0
    T=2
    u_forcing(z, t) = 10*sin(2*pi/T*t)

    model = NonhydrostaticModel(; grid, 
                                  forcing = (u = u_forcing,),
                                  timestepper = :RungeKutta3)

   
    set!(model, u=uᵢ)

    simulation = Simulation(model; Δt, stop_time)
    u = model.velocities.u

    progress_message(sim) = @info string("Iter: ", iteration(sim), ", time: ", prettytime(sim))
    simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(δt))

    simulation.output_writers[:timeavg] = NetCDFOutputWriter(model, (u=u,),
                                        filename = "timeavg.nc",
                                        schedule = AveragedTimeInterval(δt, window=δt),
                                        # schedule = TimeInterval(δt),
                                        overwrite_existing = overwrite)
                                 
    checkpointer = Checkpointer(model,
                                schedule = TimeInterval(stop_time),
                                prefix = "test",
                                cleanup = true)

    simulation.output_writers[:checkpointer] = checkpointer

    return simulation
end

# run(`sh -c "rm test_iteration*.jld2"`)

Δt = .01    # timestep (s)
T1 = 6      # first simulation stop time (s)
T2 = 2T1    # second simulation stop time (s)
δt = .02   # progress message interval and output saving interval

# Run a simulation that saves data to a checkpoint
simulation = test_simulation(T1, Δt, δt, true)
run!(simulation)

# Now try again, but picking up from the previous checkpoint
N = iteration(simulation)
checkpoint = "test_iteration$N.jld2"
simulation = test_simulation(T2, Δt, δt, false)
run!(simulation, pickup=checkpoint)

using NCDatasets
file = "timeavg.nc"
ds = Dataset(file)
t = ds["time"][:]
z = ds["zC"][:]
u = dropdims(ds["u"][:,:,:,:],dims = (1,2))
ln = Plots.plot(t[:],u[end,:],xlabel="time", ylabel="u",legend=nothing)
# Plots.savefig("ln.png")
# close(ds)

# using IJulia
# jupyterlab(dir=pwd())