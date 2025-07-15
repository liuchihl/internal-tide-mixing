using JLD2
using FileIO

filepath = "output/flat/checkpoint_iteration775116.jld2"
data = jldopen(filepath, "r")
# Access the NonhydrostaticModel and then the clock
model = data["NonhydrostaticModel"]
time_value = model["clock"].time

println("The value of 'time' is: ", time_value/(2*pi/1.4e-4))