# this script analyzes the diapycanl velocity
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
using Interpolations
using Printf
include("../functions/bins_2d.jl")
include("../functions/mmderiv.jl")

    # load data
tᶠ = 460
θ = 3.6e-3
if θ==3.6e-3
    simname = "tilt"
else 
    simname = "flat"
end

base_folder = "/scratch/bcpi/cliu28/internal-tide-mixing/"
filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_Bbudget.nc")
ds_field = Dataset(filename_field,"r")
filename_field_B = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_B-c.nc")
ds_B = Dataset(filename_field_B,"r")
filename_b_verification = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_threeD_timeavg.nc"
ds_b = Dataset(filename_b_verification,"r")

# grids
zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]); 
xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];
yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];


b = ds_b["b"][:,:,:,1]
B_time = ds_B["time"][:]
Nt = length(B_time)

Bx = zeros(Nx,Ny,Nz);
By = zeros(Nx,Ny,Nz);
Bz = zeros(Nx,Ny,Nz);
ẽ = zeros(Nx, Ny, Nz)

for tt in 1:Nt
    ∇κ∇B = ds_field["∇κ∇B"][:,:,:,tt]
    ∇κ∇B[b .== 0] .= NaN
    B = ds_B["B"][:,:,:,tt]
    
    # Initialize temporary arrays for derivatives
    Bx = zeros(size(B))
    By = zeros(size(B))
    Bz = zeros(size(B))
    
    for j in 1:Ny
        Bx[:,j,:] = mmderiv(xC, B[:,j,:])
        Bz[:,j,:] = mmderiv(zC, B[:,j,:]')'
    end
    
    for i in 1:Nx
        By[i,:,:] = mmderiv(yC, B[i,:,:])
    end
    
    # Add to ∇B_mag using the global scope
    ∇B_mag = sqrt.(Bx.^2 + By.^2 + Bz.^2) 
    global ẽ += ∇κ∇B ./ ∇B_mag
    @info @sprintf("%.2f%%", 100*tt/Nt)
end
ẽ_avg = ẽ ./ Nt
# Save ẽ to a NetCDF file
output_dir = string("output/", simname, "/")

output_filename = string(output_dir, "diapycnal_velocity_tᶠ=", tᶠ, "_theta=", θ, ".nc")
ds_save = Dataset(output_filename, "c")

# Define dimensions
defDim(ds_save, "xC", length(xC))
defDim(ds_save, "yC", length(yC))
defDim(ds_save, "zC", length(zC))

# Define variables
v1 = defVar(ds_save, "ẽ_avg", Float64, ("xC", "yC", "zC"))
v1.attrib["long_name"] = "10 tidal period average diapycnal velocity"
v1.attrib["units"] = "m/s"
v1[:,:,:] = ẽ_avg

v2 = defVar(ds_save, "xC", Float64, ("xC",))
v2[:] = xC[:]

v3 = defVar(ds_save, "yC", Float64, ("yC",))
v3[:] = yC[:]

v4 = defVar(ds_save, "zC", Float64, ("zC",))
v4[:] = zC[:]

close(ds_save)
@info "Saved diapycnal velocity to $output_filename"



## read the data
# Read the saved diapycnal velocity data

# load data
# Function to load data for a given case
# function load_diapycnal_velocity(tᶠ, θ)
#     if θ==3.6e-3
#         simname = "tilt"
#     else 
#         simname = "flat"
#     end
#     output_filename = string("output/", simname, "/diapycnal_velocity_tᶠ=", tᶠ, "_theta=", θ, ".nc")
#     ds_e = Dataset(output_filename, "r")
#     ẽ = ds_e["ẽ"][:,:,:]
#     xC = ds_e["xC"][:]
#     yC = ds_e["yC"][:]
#     zC = ds_e["zC"][:]
#     close(ds_e)
#     return ẽ, xC, yC, zC, simname
# end

# # Load flat case (θ = 0)
# tᶠ = 460
# ẽ_flat, xC_flat, yC_flat, zC_flat, simname_flat = load_diapycnal_velocity(tᶠ, 0)

# # Load tilt case (θ = 3.6e-3)
# ẽ_tilt, xC_tilt, yC_tilt, zC_tilt, simname_tilt = load_diapycnal_velocity(tᶠ, 3.6e-3)

# # Find common color range for proper comparison
# max_val = max(
#     nanmaximum(abs.(filter(isfinite, ẽ_flat))),
#     nanmaximum(abs.(filter(isfinite, ẽ_tilt)))
# ) * 0.01

# # Create a figure with two subplots side by side
# fig = Figure(size=(1200, 500))

# # Flat case subplot
# ax1 = Axis(fig[1, 1], 
#     xlabel="x (m)", 
#     ylabel="z (m)",
#     title="Flat Case (θ = 0)")

# # Tilt case subplot
# ax2 = Axis(fig[1, 2], 
#     xlabel="x (m)", 
#     ylabel="z (m)",
#     title="Tilt Case (θ = 3.6e-3)")

# # Plot the heatmaps with the same colorrange
# hm1 = heatmap!(ax1, xC_flat, zC_flat, ẽ_flat[:,length(yC_flat)÷2,:],
#     colormap=:balance,
#     colorrange=(-max_val, max_val),
#     nan_color=:gray)

# hm2 = heatmap!(ax2, xC_tilt, zC_tilt, ẽ_tilt[:,length(yC_tilt)÷2,:],
#     colormap=:balance,
#     colorrange=(-max_val, max_val),
#     nan_color=:gray)

# # Add colorbar
# Colorbar(fig[1, 3], hm2, label="ẽ (m/s)")

# # Save the figure
# save("output/diapycnal_velocity_comparison.png", fig)

# # Display the figure
# fig
