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
filename_field = string(base_folder, "output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg_Bbudget-wb.nc")
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

∇κ∇B = dropdims(mean(ds_field["∇κ∇B"][:,:,:,:],dims=4),dims=4)
b = ds_b["b"][:,:,:,1]
∇κ∇B[b .== 0] .= NaN
B_time = ds_B["time"][:]
Nt = length(B_time)

Bx = zeros(Nx,Ny,Nz);
By = zeros(Nx,Ny,Nz);
Bz = zeros(Nx,Ny,Nz);
∇B_mag = zeros(Nx,Ny,Nz);

for tt in 1:Nt
    B = ds_B["B"][:,:,:,tt]
    for j in 1:Ny
        Bx[:,j,:] = mmderiv(xC, B[:,j,:])
        Bz[:,j,:] = mmderiv(zC, B[:,j,:]')'
    end
    for i in 1:Nx
        By[i,:,:] = mmderiv(yC, B[i,:,:])
    end
    ∇B_mag += sqrt.(Bx.^2 + By.^2 + Bz.^2) 
    @info @sprintf("%.2f%%", 100*tt/Nt)
end
∇B_mag_avg = ∇B_mag ./ Nt
# 10 TP average diapycnal velocity
ẽ = ∇κ∇B ./ ∇B_mag[:,:,:]  
# Save ẽ to a NetCDF file
output_dir = string("output/", simname, "/")

output_filename = string(output_dir, "diapycnal_velocity_tᶠ=", tᶠ, "_theta=", θ, ".nc")
ds_out = NCDataset(output_filename, "c")

# Define dimensions
ds_out.dim["x"] = Nx
ds_out.dim["y"] = Ny
ds_out.dim["z"] = Nz

# Define coordinates
ds_out.defVar("xC", xC, ("x",))
ds_out.defVar("yC", yC, ("y",))
ds_out.defVar("zC", zC, ("z",))

# Define and write the diapycnal velocity
diapycnal_vel = ds_out.defVar("e_tilde", ẽ, ("x", "y", "z"), fillvalue=NaN)
diapycnal_vel.attrib["long_name"] = "Diapycnal velocity"
diapycnal_vel.attrib["units"] = "m/s"

# Close the NetCDF file
close(ds_out)
@info "Saved diapycnal velocity to $output_filename"



# ẽ[b .== 0] .= NaN
# ẽ_p = ẽ[ẽ .> 0]
# ẽ_n = ẽ[ẽ .< 0]



# # create a heatmap of the diapycnal velocity ẽ
# fig = Figure(size=(800, 400))
# ax = Axis(fig[1, 1], 
#     xlabel="x (m)", 
#     ylabel="z (m)",
#     title="Diapycnal velocity ẽ")

# # Plot the heatmap
# hm = heatmap!(ax, xC, zC, ẽ,
#     colormap=:balance,
#     colorrange=(-nanmaximum(abs.(filter(isfinite, ẽ)))*0.1, nanmaximum(abs.(filter(isfinite, ẽ)))*0.1),
#     nan_color = :gray)

# # Add colorbar
# Colorbar(fig[1, 2], hm, label="ẽ (m/s)")

# # Save the figure
# save(string("output/",simname,"/diapycnal_velocity_xz.png"), fig)

# # Display the figure
# fig

