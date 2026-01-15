using JLD2
using FileIO

filepath = "output/3D_idealized_tilt_0.008/checkpoint_iteration74976.jld2"
# filepath = "output/3D_idealized_tilt_0/checkpoint_iteration74880.jld2"
data = jldopen(filepath, "r")
# Access the NonhydrostaticModel and then the clock
model = data["NonhydrostaticModel"]
time_value = model["clock"].time

println("The value of 'time' is: ", time_value/(2*pi/1.4e-4))