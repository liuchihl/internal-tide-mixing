using NCDatasets
using Statistics

# compute mean w

tᶠ = 461.0
θ = 3.6e-3
if θ == 3.6e-3
    simname = "tilt"
else
    simname = "flat"
end

if tᶠ ≤ 10
    output_mode = "verification"
    Nt = 11      # number of time averages
    endtime = tᶠ
elseif tᶠ ≤ 450
    output_mode = "spinup"
    Nt = 4
    if tᶠ == 450
        endtime = ["$i" for i in 50:40:450]
        # endtime = tᶠ   # if you want to run only one case
    end
else
    output_mode = "analysis"
    Nt = 1    # only 1 tidal average
    endtime = tᶠ
end

# Define the time periods to process for 4 tidal periods (452.0 to 456.0)
# We need the final timestep from 452.0 chunk, then all timesteps from subsequent chunks
# time_chunks = 452.0:0.5:462.0
time_chunks = 451.5:0.5:461.0

# Get dimensions from the first data file
filename_first = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", time_chunks[2], "_analysis_round=all_threeD.nc")
ds_first = Dataset(filename_first, "r")

zC = ds_first["z_aac"][:]
zF = ds_first["z_aaf"][:]
xC = ds_first["x_caa"][:]
xF = ds_first["x_faa"][:]
yC = ds_first["y_aca"][:]
yF = ds_first["y_afa"][:]
Nz = length(zC[:])
Nx = length(xC[:])
dx = xF[end] - xF[end-1]
Ny = length(yC[:])
dy = yF[end] - yF[end-1]
z_face = zF

# Initialize sums with zeros of the correct size
what_sum = zeros(size(ds_first["what"][:, :, :, 1]))
B_sum = zeros(size(ds_first["B"][:, :, :, 1]))
uhat_sum = zeros(size(ds_first["uhat"][:, :, :, 1]))
v_sum = zeros(size(ds_first["v"][:, :, :, 1]))
eps_sum = zeros(size(ds_first["ε"][:, :, :, 1]))
∇κ∇B_sum = zeros(size(ds_first["∇κ∇B"][:, :, :, 1]))
Ri_sum = zeros(size(ds_first["Rig"][:, :, :, 1]))

close(ds_first)

# Count total timesteps processed
total_steps = 0

# Process each chunk
for i in 1:length(time_chunks)
    tf_current = time_chunks[i]
    println("Processing chunk ending at: ", tf_current)
    
    # Create filename for current chunk
    filename_current = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tf_current, "_analysis_round=all_threeD.nc")
    
    # Open dataset
    ds_current = Dataset(filename_current, "r")
    
    # Get number of timesteps in this dataset
    Nt_current = length(ds_current["time"][:])
    println("  Number of timesteps in chunk: ", Nt_current)
    
    timestep_range = 1:Nt_current
    println("  Taking all timesteps")
    
    # Process timesteps in this chunk
    for n in timestep_range
        # Accumulate data for this timestep
        what_sum .+= ds_current["what"][:, :, :, n]
        B_sum .+= ds_current["B"][:, :, :, n]
        uhat_sum .+= ds_current["uhat"][:, :, :, n]
        v_sum .+= ds_current["v"][:, :, :, n]
        eps_sum .+= ds_current["ε"][:, :, :, n]
        ∇κ∇B_sum .+= ds_current["∇κ∇B"][:, :, :, n]
        Ri_sum .+= ds_current["Rig"][:, :, :, n]
        
        global total_steps += 1
        
        # Print progress occasionally
        if total_steps % 10 == 0
            println("  Processed ", total_steps, " total timesteps")
        end
    end
    
    # Close current dataset
    close(ds_current)
    
    # Free memory
    GC.gc()
end

println("Total timesteps processed: ", total_steps)

# Calculate averages
what_avg = what_sum ./ total_steps
B_avg = B_sum ./ total_steps
uhat_avg = uhat_sum ./ total_steps
v_avg = v_sum ./ total_steps
eps_avg = eps_sum ./ total_steps
∇κ∇B_avg = ∇κ∇B_sum ./ total_steps
Ri_avg = Ri_sum ./ total_steps

# Save the averaged quantities
output_filename = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_10tidal_periods_avg.nc")

Dataset(output_filename, "c") do ds_out
    # Define dimensions
    defDim(ds_out, "x", Nx)
    defDim(ds_out, "y", Ny)
    defDim(ds_out, "z", Nz)
    defDim(ds_out, "zf", Nz+1)
    defDim(ds_out, "time", 1)
    
    # Define coordinate variables
    defVar(ds_out, "x_caa", xC, ("x",))
    defVar(ds_out, "y_aca", yC, ("y",))
    defVar(ds_out, "z_aac", zC, ("z",))
    defVar(ds_out, "x_faa", xF, ("x",))
    defVar(ds_out, "y_afa", yF, ("y",))
    defVar(ds_out, "z_aaf", zF, ("zf",))
    
    # Define and write the averaged variables
    defVar(ds_out, "uhat", uhat_avg, ("x", "y", "z", "time"))
    defVar(ds_out, "what", what_avg, ("x", "y", "zf", "time"))
    defVar(ds_out, "B", B_avg, ("x", "y", "z", "time"))
    defVar(ds_out, "v", v_avg, ("x", "y", "z", "time"))
    defVar(ds_out, "ε", eps_avg, ("x", "y", "z", "time"))
    defVar(ds_out, "∇κ∇B", ∇κ∇B_avg, ("x", "y", "z", "time"))
    defVar(ds_out, "Rig", Ri_avg, ("x", "y", "zf", "time"))
    
    # Add metadata
    ds_out.attrib["title"] = "10 tidal period averaged quantities (t=452.0 to t=462.0)"
    ds_out.attrib["time_chunks"] = string(time_chunks)
    ds_out.attrib["total_timesteps"] = total_steps
    ds_out.attrib["theta"] = θ
end

println("Saved averaged quantities to: ", output_filename)