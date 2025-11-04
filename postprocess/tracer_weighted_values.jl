# Computing tracer-weighted buoyancy B_c as volume integral of tracer*B divided by volume integral of tracer

# Loading required packages
using Oceananigans
using Oceananigans.Units
using CairoMakie
using NCDatasets: Dataset, defVar
using Statistics
using NaNStatistics
using Printf
using Dates
# General function to calculate tracer-weighted values for any field
# function tracer_weighted_values(c_field, target_field, cell_volumes; 
#                                chunk_size=50, tracer_threshold=1e-12)
#     """
#     Calculate tracer-weighted values for any field.
    
#     Parameters:
#     - c_field: 3D tracer concentration field [Nx, Ny, Nz]
#     - target_field: 3D field to be weighted [Nx, Ny, Nz]
#     - cell_volumes: 1D array of cell volumes [Nz]
#     - chunk_size: number of vertical levels to process at once
#     - tracer_threshold: minimum tracer value to consider
    
#     Returns:
#     - tracer-weighted value (scalar)
#     """
    
#     Nx, Ny, Nz = size(c_field)
    
#     # Initialize accumulators
#     field_weighted_sum = 0.0
#     total_tracer_volume = 0.0
    
#     # Process in chunks to save memory
#     for k_start in 1:chunk_size:Nz
#         k_end = min(k_start + chunk_size - 1, Nz)
        
#         # Process this chunk
#         for k in k_start:k_end
#             cell_vol = cell_volumes[k]
            
#             # Get slices for this level
#             c_slice = c_field[:,:,k]
#             target_slice = target_field[:,:,k]
            
#             # Create mask for cells with tracer
#             tracer_mask = c_slice .> tracer_threshold
            
#             if any(tracer_mask)
#                 # Calculate weighted sums
#                 tracer_weights = c_slice[tracer_mask] * cell_vol
#                 field_weighted_sum += sum(tracer_weights .* target_slice[tracer_mask])
#                 total_tracer_volume += sum(tracer_weights)
#             end
#         end
        
#         # Force garbage collection for chunks
#         GC.gc()
#     end
    
#     # Calculate tracer-weighted average
#     if total_tracer_volume > tracer_threshold
#         return field_weighted_sum / total_tracer_volume
#     else
#         return NaN
#     end
# end

# # Rest as normal code for flexibility:

# # Define time range to process
# tᶠ_start = 451.5#458.5
# tᶠ_end = 456.0#462.0
# tᶠ_step = 0.5
# tᶠ_list = collect(tᶠ_start:tᶠ_step:tᶠ_end)

# # Initialize arrays to store results
# all_times_tilt = Float64[]
# all_times_flat = Float64[]
# all_w_tracer_tilt = Float64[]
# all_w_tracer_flat = Float64[]
# all_B_weighted_tilt = Float64[]
# all_gradB_weighted_tilt = Float64[]
# all_gradB_weighted_flat = Float64[]
# all_B_weighted_flat = Float64[]

# println("Processing $(length(tᶠ_list)) files from tᶠ=$tᶠ_start to tᶠ=$tᶠ_end")

# # Initialize grid variables
# zC = nothing
# zF = nothing
# Nx = Ny = Nz = 0
# dx = dy = 0.0
# cell_volumes = nothing

# # Process each file
# for (file_idx, tᶠ) in enumerate(tᶠ_list)
#     println("Processing file $file_idx/$(length(tᶠ_list)): tᶠ=$tᶠ")
    
#     # Process both cases
#     for simname in ["flat"]
#         θ = simname == "tilt" ? 0.0036 : 0
#         filename = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_analysis_round=all_threeD.nc")
        
#         try
#             ds = Dataset(filename, "r")
            
#             # Get grid information (only once)
#             if file_idx == 1
#                 global zC = ds["z_aac"][:]
#                 global  zF = ds["z_aaf"][:]
#                 global Nx, Ny, Nz, _ = size(ds["c"])
                
#                 # Grid parameters
#                 H = 2.25kilometers
#                 Lx = 15kilometers
#                 Ly = 30kilometers
#                 global dx = Lx / Nx
#                 global dy = Ly / Ny
#                 global dz = diff(zF)
                
#                 # Pre-calculate cell volumes
#                 global cell_volumes = zeros(Nz)
#                 for k in 1:Nz
#                 cell_volumes[k] = dx * dy * dz[k]
#                 end
#             end
            
#             time_file = ds["time"][:]
#             Nt = length(time_file)
#             w_tracer_file = zeros(Nt)
#             B_weighted_file = zeros(Nt)
#             grad_B_weighted_file = zeros(Nt)
#             # Process each time step
#             for t in 1:Nt
#                 # Load current time step data
#                 c_field = ds["c"][:,:,:,t]
#                 B_field = ds["B"][:,:,:,t]
#                 omega_field = ds["∇κ∇B"][:,:,:,t]
                
#                 # Compute gradient magnitude using your exact method
#                 grad_B_magnitude = zeros(Nx, Ny, Nz)
#                 for k in 1:Nz
#                     B_slice = B_field[:,:,k]
                    
#                     # X-direction gradient using periodic boundaries
#                     B_face_x = diff(vcat(B_slice[end:end,:], B_slice), dims=1) ./ dx
#                     grad_B_x = (vcat(B_face_x[2:end, :], B_face_x[1:1, :]) .+ B_face_x) ./ 2
                    
#                     # Y-direction gradient using periodic boundaries
#                     B_face_y = diff(hcat(B_slice[:,end:end], B_slice), dims=2) ./ dy
#                     grad_B_y = (hcat(B_face_y[:,2:end], B_face_y[:,1:1]) .+ B_face_y) ./ 2
                    
#                     # Z-direction gradient
#                     if k > 1 && k < Nz
#                         grad_B_z_above = (B_field[:,:,k+1] - B_slice) / (zC[k+1] - zC[k])
#                         grad_B_z_below = (B_slice - B_field[:,:,k-1]) / (zC[k] - zC[k-1])
#                         grad_B_z = (grad_B_z_above .+ grad_B_z_below) ./ 2
#                     elseif k == 1
#                         grad_B_z = zeros(Nx, Ny)
#                     elseif k == Nz
#                         grad_B_z = fill(1e-6, Nx, Ny)
#                     end
                    
#                     grad_B_magnitude[:,:,k] = sqrt.(grad_B_x.^2 + grad_B_y.^2 + grad_B_z.^2)
#                 end
                
#                 # Compute tracer-weighted values
#                 omega_weighted = tracer_weighted_values(c_field, omega_field, cell_volumes)
#                 grad_B_weighted = tracer_weighted_values(c_field, grad_B_magnitude, cell_volumes)
#                 B_weighted = tracer_weighted_values(c_field, B_field, cell_volumes)
                
#                 # Compute diapycnal velocity
#                 if grad_B_weighted > 1e-12
#                     w_tracer_file[t] = 2 * omega_weighted / grad_B_weighted
#                 else
#                     w_tracer_file[t] = NaN
#                 end
                
#                 # Store B_weighted and gradB in an array over time
#                 B_weighted_file[t] = B_weighted     
#                 grad_B_weighted_file[t] = grad_B_weighted     
                
#                 # Clear memory
#                 c_field = nothing
#                 B_field = nothing
#                 omega_field = nothing
#                 grad_B_magnitude = nothing
#                 B_magnitude = nothing
#                 GC.gc()
#                 println("Processed timestep $t of $Nt for $simname")
#             end
            
#             # Store results
#             if simname == "tilt"
#                 append!(all_times_tilt, time_file)
#                 append!(all_w_tracer_tilt, w_tracer_file)
#                 append!(all_B_weighted_tilt, B_weighted_file)
#                 append!(all_gradB_weighted_tilt, grad_B_weighted_file)
#             else
#                 append!(all_times_flat, time_file)
#                 append!(all_w_tracer_flat, w_tracer_file)
#                 append!(all_B_weighted_flat, B_weighted_file)
#                 append!(all_gradB_weighted_flat, grad_B_weighted_file)

#             end
            
#             close(ds)
            
#         catch e
#             println("Error processing $simname tᶠ=$tᶠ: $e")
#             continue
#         end
#     end
    
#     GC.gc()
# end

# # Sort results by time
# sort_indices_tilt = sortperm(all_times_tilt)
# all_times_tilt = all_times_tilt[sort_indices_tilt]
# all_w_tracer_tilt = all_w_tracer_tilt[sort_indices_tilt]
# all_B_weighted_tilt = all_B_weighted_tilt[sort_indices_tilt]
# all_gradB_weighted_tilt = all_gradB_weighted_tilt[sort_indices_tilt]

# sort_indices_flat = sortperm(all_times_flat)
# all_times_flat = all_times_flat[sort_indices_flat]
# all_w_tracer_flat = all_w_tracer_flat[sort_indices_flat]
# all_B_weighted_flat = all_B_weighted_flat[sort_indices_flat]
# all_gradB_weighted_flat = all_gradB_weighted_flat[sort_indices_flat]

# # Save as NetCDF files
# println("Saving results...")

# # Save tilt case
# output_filename_tilt = string("output/tilt/tracer_weighted_vertical_velocity_tilt_",tᶠ_start,"-",tᶠ_end,".nc")
# Dataset(output_filename_tilt, "c") do ds
#     # Define dimensions
#     ds.dim["time"] = length(all_times_tilt)
    
#     # Define variables
#     time_var = defVar(ds, "time", Float64, ("time",))
#     w_tracer_var = defVar(ds, "w_tracer", Float64, ("time",))
    
#     # Add attributes
#     time_var.attrib["long_name"] = "Time"
#     time_var.attrib["units"] = "seconds"
#     time_var.attrib["description"] = "Time since simulation start"
    
#     w_tracer_var.attrib["long_name"] = "Tracer-weighted vertical velocity"
#     w_tracer_var.attrib["units"] = "m/s"
#     w_tracer_var.attrib["description"] = "Vertical velocity computed as 2*omega/|∇B| where omega is ∇κ∇B"
#     w_tracer_var.attrib["formula"] = "w = 2 * <c*∇κ∇B> / <c*|∇B|>"

#     # Add B-weighted variable
#     B_weighted_var = defVar(ds, "B_weighted", Float64, ("time",))
#     B_weighted_var.attrib["long_name"] = "Tracer-weighted buoyancy"
#     B_weighted_var.attrib["units"] = "m/s²"
#     B_weighted_var.attrib["description"] = "Buoyancy weighted by tracer concentration"  
#     B_weighted_var.attrib["formula"] = "B_c = <c*B> / <c>"
    
#     # Add gradB-weighted variable
#     gradB_weighted_var = defVar(ds, "gradB_weighted", Float64, ("time",))
#     gradB_weighted_var.attrib["long_name"] = "Tracer-weighted buoyancy gradient magnitude"
#     gradB_weighted_var.attrib["units"] = "1/s²"
#     gradB_weighted_var.attrib["description"] = "Buoyancy gradient magnitude weighted by tracer concentration"  
#     gradB_weighted_var.attrib["formula"] = "|∇B_c| = <c*|∇B|> / <c>"
    
#     # Global attributes
#     ds.attrib["title"] = "Tracer-weighted vertical velocity for tilt case"
#     ds.attrib["created"] = string(now())
#     ds.attrib["case"] = "tilt"
#     ds.attrib["time_range"] = "tᶠ = $tᶠ_start to $tᶠ_end with step $tᶠ_step"
#     ds.attrib["total_time_points"] = length(all_times_tilt)
    
#     # Write data
#     time_var[:] = all_times_tilt
#     w_tracer_var[:] = all_w_tracer_tilt
#     B_weighted_var[:] = all_B_weighted_tilt
#     gradB_weighted_var[:] = all_gradB_weighted_tilt
# end

# println("Tilt results saved to: $output_filename_tilt")

# # Save flat case
# output_filename_flat = string("output/flat/tracer_weighted_vertical_velocity_flat_",tᶠ_start,"-",tᶠ_end,".nc")
# Dataset(output_filename_flat, "c") do ds
#     # Define dimensions
#     ds.dim["time"] = length(all_times_flat)
    
#     # Define variables
#     time_var = defVar(ds, "time", Float64, ("time",))
#     w_tracer_var = defVar(ds, "w_tracer", Float64, ("time",))
    
#     # Add attributes
#     time_var.attrib["long_name"] = "Time"
#     time_var.attrib["units"] = "seconds"
#     time_var.attrib["description"] = "Time since simulation start"
    
#     w_tracer_var.attrib["long_name"] = "Tracer-weighted vertical velocity"
#     w_tracer_var.attrib["units"] = "m/s"
#     w_tracer_var.attrib["description"] = "Vertical velocity computed as 2*omega/|∇B| where omega is ∇κ∇B"
#     w_tracer_var.attrib["formula"] = "w = 2 * <c*∇κ∇B> / <c*|∇B|>"
    
#     # Add B-weighted variable
#     B_weighted_var = defVar(ds, "B_weighted", Float64, ("time",))
#     B_weighted_var.attrib["long_name"] = "Tracer-weighted buoyancy"
#     B_weighted_var.attrib["units"] = "m/s²"
#     B_weighted_var.attrib["description"] = "Buoyancy weighted by tracer concentration"  
#     B_weighted_var.attrib["formula"] = "B_c = <c*B> / <c>"
    

#     # Add gradB-weighted variable
#     gradB_weighted_var = defVar(ds, "gradB_weighted", Float64, ("time",))
#     gradB_weighted_var.attrib["long_name"] = "Tracer-weighted buoyancy gradient magnitude"
#     gradB_weighted_var.attrib["units"] = "1/s²"
#     gradB_weighted_var.attrib["description"] = "Buoyancy gradient magnitude weighted by tracer concentration"  
#     gradB_weighted_var.attrib["formula"] = "|∇B_c| = <c*|∇B|> / <c>"
    
#     # Global attributes
#     ds.attrib["title"] = "Tracer-weighted vertical velocity for flat case"
#     ds.attrib["created"] = string(now())
#     ds.attrib["case"] = "flat"
#     ds.attrib["time_range"] = "tᶠ = $tᶠ_start to $tᶠ_end with step $tᶠ_step"
#     ds.attrib["total_time_points"] = length(all_times_flat)
    
#     # Write data
#     time_var[:] = all_times_flat
#     w_tracer_var[:] = all_w_tracer_flat
#     B_weighted_var[:] = all_B_weighted_flat
#     gradB_weighted_var[:] = all_gradB_weighted_flat
    
# end

# println("Flat results saved to: $output_filename_flat")

# Create a combined NetCDF file with both cases
# combined_filename = "output/tracer_weighted_vertical_velocity_combined.nc"
# Dataset(combined_filename, "c") do ds
#     # Define dimensions
#     defDim(ds, "time_tilt", length(all_times_tilt))
#     defDim(ds, "time_flat", length(all_times_flat))
    
#     # Define variables for tilt case
#     time_tilt_var = defVar(ds, "time_tilt", Float64, ("time_tilt",))
#     w_tracer_tilt_var = defVar(ds, "w_tracer_tilt", Float64, ("time_tilt",))
    
#     # Define variables for flat case
#     time_flat_var = defVar(ds, "time_flat", Float64, ("time_flat",))
#     w_tracer_flat_var = defVar(ds, "w_tracer_flat", Float64, ("time_flat",))
    
#     # Add attributes
#     time_tilt_var.attrib["long_name"] = "Time - Tilt Case"
#     time_tilt_var.attrib["units"] = "seconds"
#     w_tracer_tilt_var.attrib["long_name"] = "Tracer-weighted vertical velocity - Tilt Case"
#     w_tracer_tilt_var.attrib["units"] = "m/s"
    
#     time_flat_var.attrib["long_name"] = "Time - Flat Case"
#     time_flat_var.attrib["units"] = "seconds"
#     w_tracer_flat_var.attrib["long_name"] = "Tracer-weighted vertical velocity - Flat Case"
#     w_tracer_flat_var.attrib["units"] = "m/s"
    
#     # Global attributes
#     ds.attrib["title"] = "Tracer-weighted vertical velocity comparison"
#     ds.attrib["created"] = string(now())
#     ds.attrib["time_range"] = "tᶠ = $tᶠ_start to $tᶠ_end with step $tᶠ_step"
    
#     # Write data
#     time_tilt_var[:] = all_times_tilt
#     w_tracer_tilt_var[:] = all_w_tracer_tilt
#     time_flat_var[:] = all_times_flat
#     w_tracer_flat_var[:] = all_w_tracer_flat
# end

# println("Combined results saved to: $combined_filename")

# plot tracer weighted buoyancy equation: < bₜ > = 2 < ∇⋅(κ∇B) >, where < > is the tracer weighted average
# using NCDatasets
# using CairoMakie
# file = "output/tilt/tracer_weighted_vertical_velocity_tilt.nc"
# ds = Dataset(file, "r")
# w_tracer = ds["w_tracer"][:]
# B_weighted = ds["B_weighted"][:]
# time = ds["time"][:]
# time_edge = time[1:end-1] .+ diff(time) ./ 2
# Bt_weighted = diff(B_weighted) ./ diff(time)
# two∇κ∇B_weighted = w_tracer.*ds["gradB_weighted"][:]
# fig_w = CairoMakie.Figure(resolution=(1000, 600), fontsize=20)

# # First panel: dB̄/dt and -2∇⋅𝔅
# ax1_w = Axis(fig_w[1, 1], 
#     xlabel="Tidal cycles", 
#     ylabel="", 
#     # title="Tracer-weighted Vertical Velocity - Tilt Case",
#     xminorticksvisible = true,
#     yminorticksvisible = true,
#     xminorticks = IntervalsBetween(5),
#     yminorticks = IntervalsBetween(4),
#     limits= ((0, maximum(time.-time[1]) ./ (2*pi/1.4e-4)), nothing)
# )

# lines!(ax1_w, (time_edge.-time_edge[1]) ./ (2*pi/1.4e-4), Bt_weighted, 
#     linewidth=3, color=:red, label=L"d\bar{B}/dt" )
# lines!(ax1_w, (time.-time[1]) ./ (2*pi/1.4e-4), two∇κ∇B_weighted, 
#     linewidth=3, color=:blue, label=L"2\bar{ω}")

# axislegend(ax1_w, position=:rt)

# # Second panel: w_tracer
# ax2_w = Axis(fig_w[2, 1],
#     xlabel="Tidal cycles",
#     ylabel=L"10^{-4}~ w_{tracer} ~\mathrm{[m ~s^{-1}]}",
#     xminorticksvisible = true,
#     yminorticksvisible = true,
#     xminorticks = IntervalsBetween(5),
#     yminorticks = IntervalsBetween(4),
#     limits = ((0, maximum(time.-time[1]) ./ (2*pi/1.4e-4)), (0, 3.6)))

# lines!(ax2_w, (time.-time[1]) ./ (2*pi/1.4e-4), w_tracer*1e4, 
#     linewidth=3, color=:green, label=L"w_{tracer}")

# # hlines!(ax2_w, [0.0], color=:black, linewidth=2, linestyle=:dash)
# # axislegend(ax2_w, position=:rt)

# # fig_w[1, 1].title = "Tracer-weighted Buoyancy Budget"
# # fig_w[2, 1].title = "Tracer-weighted Vertical Velocity"

# save("output/tilt/tracer_weighted_Bbudget&w_southbottomflank.png", fig_w)

# # Print summary statistics
# println("\nSummary Statistics:")
# println("Tilt case ($(length(all_times_tilt)) time points):")
# println("  Time range: $(minimum(all_times_tilt)/3600) to $(maximum(all_times_tilt)/3600) hours")
# println("  Mean w_tracer: $(mean(skipmissing(all_w_tracer_tilt))) m/s")
# println("  Std w_tracer: $(std(skipmissing(all_w_tracer_tilt))) m/s")

# println("Flat case ($(length(all_times_flat)) time points):")
# println("  Time range: $(minimum(all_times_flat)/3600) to $(maximum(all_times_flat)/3600) hours")
# println("  Mean w_tracer: $(mean(skipmissing(all_w_tracer_flat))) m/s")
# println("  Std w_tracer: $(std(skipmissing(all_w_tracer_flat))) m/s")

# # Function to read the saved NetCDF files (for future use)
# function read_tracer_weighted_results(filename)
#     Dataset(filename, "r") do ds
#         return Dict(
#             "time" => ds["time"][:],
#             "w_tracer" => ds["w_tracer"][:]
#         )
#     end
# end



### plot all tracer weighted buoyancy B_c (tilt and flat) and B_p

# plot tracer weighted buoyancy equation: < bₜ > = 2 < ∇⋅(κ∇B) >, where < > is the tracer weighted average
using NCDatasets
using CairoMakie
using Statistics
include("/work/hdd/bcpi/cliu28/internal-tide-mixing/functions/mmintgrl.jl")

file = "output/tilt/tracer_weighted_vertical_velocity_tilt_458.5-462.0.nc"
ds = Dataset(file, "r")
w_tracer_tilt = ds["w_tracer"][:]
B_weighted_tilt = ds["B_weighted"][:]
t_tilt = ds["time"][:]
# Bt_weighted = diff(B_weighted) ./ diff(time)
two∇κ∇B_weighted_tilt = w_tracer_tilt.*ds["gradB_weighted"][:]
# cumulative integral 
int_2ω =  mmintgrl(t_tilt, two∇κ∇B_weighted_tilt)
# need to get B̄ᶜ(t) = B̄ᶜ(0) + ∫_0^t 2ω̄ dt, because the tracer wraps around the boundaries and the buoyancy would change dramatically
B_weighted_residual_tilt = int_2ω .+ B_weighted_tilt[1]

file_p = "output/tilt/concatenated_particle_data_z1000_all.nc"
ds_p_tilt = Dataset(file_p, "r")
# second part
Bp_tilt = ds_p_tilt["new_buoyancy"][:,:]
Bp_tilt_med = median(Bp_tilt, dims=1)
ΔBp_tilt = Bp_tilt_med[:].-Bp_tilt_med[1]
tp_tilt = ds_p_tilt["new_time"][:]

# first part
Bp_tilt_main = ds_p_tilt["main_buoyancy"][:,:]
Bp_tilt_med_main = median(Bp_tilt_main, dims=1)
ΔBp_tilt_main = Bp_tilt_med_main[:].-Bp_tilt_med_main[1]
tp_tilt_main = ds_p_tilt["main_time"][:]


file = "output/flat/tracer_weighted_vertical_velocity_flat_451.5-456.0.nc"
ds = Dataset(file, "r")
w_tracer_flat = ds["w_tracer"][:]
B_weighted_flat = ds["B_weighted"][:]
t_flat = ds["time"][:]
Bt_weighted = diff(B_weighted_flat) ./ diff(t_flat)
t_edge = t_flat[1:end-1] .+ diff(t_flat) ./ 2
two∇κ∇B_weighted_flat = w_tracer_flat.*ds["gradB_weighted"][:]
int_2ω =  mmintgrl(t_flat, two∇κ∇B_weighted_flat)
# need to get B̄ᶜ(t) = B̄ᶜ(0) + ∫_0^t 2ω̄ dt, it should be the same as B_weighted_flat
B_weighted_residual_flat = int_2ω .+ B_weighted_flat[1]

file_p = "output/flat/concatenated_particle_data_z_flat_1000_all.nc"
ds_p_flat = Dataset(file_p, "r")
Bp_flat = ds_p_flat["main_buoyancy"][:,:]
ΔBp_flat = Bp_flat[:,:].-Bp_flat[:,1]
ΔBp_flat_med = mean(ΔBp_flat, dims=1)
# Bp_flat_med = median(Bp_flat, dims=1)
# ΔBp_flat = Bp_flat_med[:].-Bp_flat_med[1]
tp_flat = ds_p_flat["main_time"][:]
# test plot dBdt and 2ω̄
# fig = CairoMakie.Figure(resolution=(1000, 600), fontsize=20)    
# # First panel: dB̄/dt and -2∇⋅𝔅
# ax1 = Axis(fig[1, 1],
#     xlabel="Tidal cycles",
#     ylabel=L"\Delta\bar{B}_c ~\mathrm{[m~s^{-2}]}",
#     xminorticksvisible = true,
#     yminorticksvisible = true,
#     xminorticks = IntervalsBetween(5),
#     yminorticks = IntervalsBetween(4),
#     limits = ((0, maximum(t_flat.-t_flat[1]) ./ (2*pi/1.4e-4)), nothing)
# )
# lines!(ax1, (t_edge.-t_edge[1]) ./ (2*pi/1.4e-4), Bt_weighted, linewidth=3, color=:red, label=L"d\bar{B}/dt")
# lines!(ax1, (t_flat.-t_flat[1]) ./ (2*pi/1.4e-4), two∇κ∇B_weighted_flat, linewidth=3, color=:blue, label=L"2\bar{ω}")
# axislegend(ax1, position=:rt)

# save("output/flat/tracer_weighted_buoyancy_equation_flat.png", fig)


# comapare B_weighted_flat and B_weighted_residual_flat first
# fig_compare = CairoMakie.Figure(resolution=(1000, 600), fontsize=20)

# ax1 = Axis(fig_compare[1, 1],
#     xlabel="Tidal cycles",
#     ylabel=L"\Delta\bar{B}_c ~\mathrm{[m~s^{-2}]}",
#     xminorticksvisible = true,
#     yminorticksvisible = true,
#     xminorticks = IntervalsBetween(5),
#     yminorticks = IntervalsBetween(4),
#     limits = ((0, maximum(t.-t[1]) ./ (2*pi/1.4e-4)), nothing)
# )

# lines!(ax1, (t_flat.-t_flat[1]) ./ (2*pi/1.4e-4), B_weighted_flat.-B_weighted_flat[1], linewidth=3, color=:blue, label="Flat: Direct")
# lines!(ax1, (t_flat.-t_flat[1]) ./ (2*pi/1.4e-4), B_weighted_residual_flat[:,1].-B_weighted_residual_flat[1,1], linewidth=3, color=:cyan, linestyle=:dash, label="Flat: Integrated")
# # lines!(ax1, (t_tilt.-t_tilt[1]) ./ (2*pi/1.4e-4), B_weighted_tilt.-B_weighted_tilt[1], linewidth=3, color=:red, label="Tilt: Direct")
# # lines!(ax1, (t_tilt.-t_tilt[1]) ./ (2*pi/1.4e-4), B_weighted_residual_tilt[:,1].-B_weighted_residual_tilt[1,1], linewidth=3, color=:orange, linestyle=:dash, label="Tilt: Integrated")

# axislegend(ax1, position=:rt)
# # fig_compare[1, 1].title = "Tracer-weighted Buoyancy Comparison (Direct vs Integrated)"

# save("output/flat/tracer_weighted_buoyancy_comparison.png", fig_compare)
# println("Saved tracer-weighted buoyancy comparison plot: output/flat/tracer_weighted_buoyancy_comparison.png")


using CairoMakie

fig = Figure(resolution=(800, 400), fontsize=22)

ax = Axis(fig[1, 1];
    xlabel="Time after release [tidal cycles]",
    ylabel="[10⁻⁵ m s⁻²]",
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorticks=IntervalsBetween(5),
    yminorticks=IntervalsBetween(4),
    limits=((0, 4), nothing),
    xticklabelsize=18,
    yticklabelsize=18,
    xlabelsize=22,
    ylabelsize=22,
    titlesize=24,
    xticksvisible=(:bottom, :top),
    yticksvisible=(:left, :right)
    # xticksvisible=:all,   # <-- show ticks on bottom and top
    # yticksvisible=:all   # <-- show ticks on left and right
)

# Tilt case
# Use visually distinct and colorblind-friendly colors
tilt_color = "#0072B2"      # Blue (colorblind safe)
tilt_dash_color = "#56B4E9" # Light blue (colorblind safe)
flat_color = "#D55E00"      # Orange/red (colorblind safe)
flat_dash_color = "#E69F00" # Yellow/orange (colorblind safe)

lines!(ax, (t_tilt.-t_tilt[1]) ./ (2*pi/1.4e-4), 1e5*(B_weighted_residual_tilt[:,1].-B_weighted_residual_tilt[1,1]),
    linewidth=3, color=tilt_color, label=L"\Delta\bar{B}^{c}_{tilt}")
lines!(ax, (tp_tilt.-tp_tilt[1]) ./ (2*pi/1.4e-4), 1e5*ΔBp_tilt[:],
    linewidth=3, color=tilt_dash_color, linestyle=:dash, label=L"\Delta\bar{B}^{p2}_{tilt}")
lines!(ax, (tp_tilt_main.-tp_tilt_main[1]) ./ (2*pi/1.4e-4), 1e5*ΔBp_tilt_main[:],
    linewidth=3, color=tilt_dash_color, linestyle=:dot, label=L"\Delta\bar{B}^{p1}_{tilt}")

# Flat case
lines!(ax, (t_flat.-t_flat[1]) ./ (2*pi/1.4e-4), 1e5*(B_weighted_residual_flat[:,1].-B_weighted_residual_flat[1,1]),
    linewidth=3, color=flat_color, label=L"\Delta\bar{B}^c_{flat}")
lines!(ax, (tp_flat.-tp_flat[1]) ./ (2*pi/1.4e-4), 1e5*ΔBp_flat_med[:],
    linewidth=3, color=flat_dash_color, linestyle=:dash, label=L"\Delta\bar{B}^p_{flat}")

hlines!(ax, [0.0], color=:black, linewidth=2)

axislegend(ax, position=:lt, framevisible=true, patchsize=(30, 10), labelsize=18)

# fig[1, 1].title = "Tracer-weighted and Particle Buoyancy Change"

save("output/tilt/tracer_weighted_buoyancy_particle_buoyancy_pubready.png", fig)

# # Second panel: w_tracer
# ax2_w = Axis(fig_w[2, 1],
#     xlabel="Tidal cycles",
#     ylabel=L"10^{-4}~ w_{tracer} ~\mathrm{[m ~s^{-1}]}",
#     xminorticksvisible = true,
#     yminorticksvisible = true,
#     xminorticks = IntervalsBetween(5),
#     yminorticks = IntervalsBetween(4),
#     limits = ((0, maximum(time.-time[1]) ./ (2*pi/1.4e-4)), (0, 3.6)))

# lines!(ax2_w, (time.-time[1]) ./ (2*pi/1.4e-4), w_tracer*1e4, 
#     linewidth=3, color=:green, label=L"w_{tracer}")

# # hlines!(ax2_w, [0.0], color=:black, linewidth=2, linestyle=:dash)
# # axislegend(ax2_w, position=:rt)

# # fig_w[1, 1].title = "Tracer-weighted Buoyancy Budget"
# # fig_w[2, 1].title = "Tracer-weighted Vertical Velocity"

# save("output/tilt/tracer_weighted_buoyancy.png", fig_w)
