using Printf
using PyPlot
using PyCall
using NCDatasets
using Statistics
using NaNStatistics
using Dates  # For potential time handling

include("/scratch/bcpi/cliu28/internal-tide-mixing/functions/mmderiv.jl")

# =============================================================================
# User Configuration
# =============================================================================
θ = 0.008  # Tilt angle (radians)
base_dir = "output/3D_idealized_tilt_$(θ)/"  # Base directory for files

# Time range for long-time averaging: Specify start and end tidal periods (TP)
# Example: start_tp = 10, end_tp = 40  # Averages over TP 10 to 40
start_tp = 180
end_tp = 180.0
file_pattern = "internal_tide_theta=$(θ)_Nx=2000_Nz=500_tᶠ=$(end_tp)_yavg"  # Base filename pattern
# Number of phase bins (e.g., 24 for 1/24 TP)
num_phases = 24  # Change to 8 if needed for coarser bins

# Tidal parameters
U₀ = 0.025  # Tidal velocity amplitude (m/s)
ω₀ = 1.4e-4  # Tidal frequency (rad/s)

# Output filename
output_fn = "$(base_dir)phase_avg_decomp_tp_$(start_tp)-$(end_tp).nc"

# =============================================================================
# Helper Functions
# =============================================================================
function load_files_and_time_range(base_dir, file_pattern, start_tp, end_tp)
    # For simplicity, assume one file per 10 TP block, e.g., "tp10-20.nc", "tp20-30.nc", etc.
    # User can adjust file loading logic based on actual naming.
    
    files = String[]
    all_t = Float64[]
    # Allow storing fields with different dimensionality (e.g., 1D "time", 3D data fields)
    all_data = Dict{String, Array{Float64,N} where N}()  # Key: varname, Val: Array with arbitrary dims
    
    # Load files for the range (assuming 10 TP per file; adjust num_tp_per_file if different)
    num_tp_per_file = 10
    num_files = ceil(Int, (end_tp - start_tp + 1) / num_tp_per_file)
    
    for i in 1:num_files
        file_start = start_tp + (i-1) * num_tp_per_file
        file_end = min(file_start + num_tp_per_file - 1, end_tp)
        fn = "$(base_dir)$(file_pattern).nc"  # Adjust naming convention
        
        if !isfile(fn)
            error("File not found: $fn. Adjust file_pattern or naming logic.")
        end
        
        ds = Dataset(fn, "r")
        t_local = ds["time"][:]
        push!(all_t, t_local...)
        
        # Load all variables (uhat_avg, what_avg, B, ∇κ∇B, div_uB, etc.)
        vars = ["uhat_avg", "what_avg", "B", "∇κ∇B", "div_uB"]  # Add "vhat_avg" if available
        if haskey(ds, "vhat_avg")
            push!(vars, "vhat_avg")
        end
        # Add others like "Bz", "ε" if needed
        
        for var in vars
            if haskey(ds, var)
                data_local = ds[var][:,:,:]
                if !haskey(all_data, var)
                    all_data[var] = zeros(size(data_local,1), size(data_local,2), length(all_t))
                end
                start_idx = length(all_t) - length(t_local) + 1
                all_data[var][:,:,start_idx:start_idx+length(t_local)-1] = data_local
            end
        end
        close(ds)
    end
    
    # Global time and data now cover start_tp to end_tp
    Nt = length(all_t)
    time_range = 1:Nt  # Full range for this block
    
    # Extract grid
    ds_sample = Dataset("$(base_dir)$(file_pattern).nc", "r")  # Sample for grid
    xC = ds_sample["x_caa"][:]
    zC = ds_sample["z_aac"][:]
    Nx, Nz = length(xC), length(zC)
    close(ds_sample)
    all_data["time"] = copy(all_t)

    return xC, zC, all_t, time_range, all_data, Nx, Nz
end

function compute_phase_indices(t, num_phases, ω₀)
    phases = mod.(ω₀ .* t, 2π)      # in [0, 2π)
    Δφ = 2π / num_phases
    k = round.(Int, phases ./ Δφ)   # integer, can be 0..num_phases
    # Wrap periodically into 0..num_phases-1, then shift to 1..num_phases
    bin_indices = 1 .+ mod.(k, num_phases)
    return bin_indices
end

function compute_long_time_means(all_data, time_range, Nx, Nz)
    # Compute <u>, <B>, etc. over time_range
    means =  Dict{String, Array{Float64,N} where N}()  # Store 2D long-time means (Nx_var, Nz_var)

    for (var, data) in all_data
        # Skip non-3D entries such as the 1D "time" vector
        if ndims(data) != 3
            continue
        end
        means[var] = nanmean(data[:,:,time_range], dims=3)[:,:,1]  # (Nx, Nz)
    end
    return means
end

function compute_phase_averages(all_data, time_range, num_phases, Nx, Nz, long_means)
    # Initialize phase arrays: (Nx, Nz, num_phases)
    phase_avgs =  Dict{String, Array{Float64,N} where N}()

    vars = keys(all_data)
    for var in vars
        # Only allocate for 3D fields; skip 1D entries like "time"
        var == "time" && continue
        data = all_data[var]
        ndims(data) == 3 || continue
        nxv, nzv, _ = size(data)
        phase_avgs[var] = zeros(nxv, nzv, num_phases)  # Match each field's own vertical dim
    end
    
    # For w (face-centered), we'll handle separately if needed
    
    bin_indices = compute_phase_indices(all_data["time"][time_range], num_phases, ω₀)  # Uses stored global time vector
    
    for i in 1:length(time_range)
        global_idx = time_range[i]
        phase_bin = bin_indices[i]
        
        for (var, data) in all_data
            # Only accumulate for 3D fields; skip "time" and any non-3D entries
            if ndims(data) != 3
                continue
            end
            phase_avgs[var][:,:,phase_bin] .+= data[:,:,global_idx]
        end
    end
    
    # Normalize by counts per bin (should be ~ equal for uniform sampling)
    counts = zeros(num_phases)
    for bin in bin_indices
        counts[bin] += 1
    end
    for p in 1:num_phases
        norm = counts[p]
        if norm > 0
            for (var, _) in all_data
                # Only normalize 3D phase-averaged fields
                haskey(phase_avgs, var) || continue
                phase_avgs[var][:,:,p] ./= norm
            end
        end
    end
    
    # Subtract long-time means to get zero-mean tilde components
    tilde_avgs =  Dict{String, Array{Float64,N} where N}()
    for (var, phase_avg) in phase_avgs
        if haskey(long_means, var)
            # long_means[var] is 2D (nx_var, nz_var); broadcast across phase dimension
            tilde_avgs[var] = phase_avg .- long_means[var]
        else
            tilde_avgs[var] = copy(phase_avg)
        end
    end
    
    return phase_avgs, tilde_avgs, bin_indices  # Return both if needed; tilde is the fixed zero-mean
end

function compute_budget_terms(all_data, bin_indices, time_range, tilde_u, tilde_w, tilde_B, long_u, long_w, long_B, xC, zC, θ, Nx, Nz, num_phases)
    """
    Computes flux divergence budget terms in the decomposed buoyancy equation:
      ∂B/∂t = -∂(uB)/∂x - ∂(wB)/∂z + ∇·(κ∇B)
    Decomposition: u = <u> + ũ + u',  B = <B> + B̃ + B'
    Terms computed (each split into x and z flux-divergence components):
      - Mean self:  ∂(<u><B>)/∂x,  ∂(<w><B>)/∂z
      - Tidal:      <∂(ũB̃)/∂x>,   <∂(w̃B̃)/∂z>   (phase-averaged)
      - Turb:       <∂(u'B')/∂x>,  <∂(w'B')/∂z>  (time-averaged)
    Slope-coordinate rotation applied to velocities before forming fluxes.
    Returns: terms dict (each entry (Nx,Nz)), fluxes dict (mean fluxes for saving).
    """

    # -------------------------------------------------------------------------
    # Step 1: Broadcast tilde fields to full time series (Nx, Nz, Nt) via phase bins
    # bin_indices[i] gives the phase bin (1..num_phases) for time index i in time_range
    # -------------------------------------------------------------------------
    Nt = length(time_range)
    u_tilde_full = zeros(Nx, Nz, Nt)
    w_tilde_full = zeros(Nx, Nz + 1, Nt)  # w face-centered in z
    B_tilde_full = zeros(Nx, Nz, Nt)
    for i in 1:Nt
        p = bin_indices[i]
        u_tilde_full[:,:,i] .= tilde_u[:,:,p]
        w_tilde_full[:,:,i] .= tilde_w[:,:,p]
        B_tilde_full[:,:,i] .= tilde_B[:,:,p]
    end

    # -------------------------------------------------------------------------
    # Step 2: Extract full fields and compute primes (Nx, Nz, Nt)
    # -------------------------------------------------------------------------
    u_full = all_data["uhat_avg"][:,:,time_range]
    w_full = all_data["what_avg"][:,:,time_range]
    B_full = all_data["B"][:,:,time_range]

    u_prime_full = u_full .- long_u .- u_tilde_full
    w_prime_full = w_full .- long_w .- w_tilde_full
    B_prime_full = B_full .- long_B .- B_tilde_full

    # -------------------------------------------------------------------------
    # Step 3: Center w fields from faces to cell centers
    # -------------------------------------------------------------------------
    long_w_cen        = 0.5 .* (long_w[:, 1:end-1]         .+ long_w[:, 2:end])          # (Nx, Nz)
    w_tilde_cen       = 0.5 .* (tilde_w[:, 1:end-1, :]     .+ tilde_w[:, 2:end, :])      # (Nx, Nz, num_phases)
    w_tilde_full_cen  = 0.5 .* (w_tilde_full[:, 1:end-1, :] .+ w_tilde_full[:, 2:end, :]) # (Nx, Nz, Nt)
    w_prime_cen       = 0.5 .* (w_prime_full[:, 1:end-1, :] .+ w_prime_full[:, 2:end, :]) # (Nx, Nz, Nt)

    # -------------------------------------------------------------------------
    # Step 4: Rotate to slope-coordinate velocities
    #   u_cross =  u·cos(θ) + w·sin(θ)
    #   w_cross = -u·sin(θ) + w·cos(θ)
    # -------------------------------------------------------------------------
    # Mean: (Nx, Nz)
    u_mean_cross = long_u .* cos(θ) .+ long_w_cen .* sin(θ)
    w_mean_cross = .-long_u .* sin(θ) .+ long_w_cen .* cos(θ)

    # Tilde per phase: (Nx, Nz, num_phases)
    u_tilde_cross = zeros(Nx, Nz, num_phases)
    w_tilde_cross = zeros(Nx, Nz, num_phases)
    for p in 1:num_phases
        u_tilde_cross[:,:,p] = tilde_u[:,:,p] .* cos(θ) .+ w_tilde_cen[:,:,p] .* sin(θ)
        w_tilde_cross[:,:,p] = .-tilde_u[:,:,p] .* sin(θ) .+ w_tilde_cen[:,:,p] .* cos(θ)
    end

    # Prime full time series: (Nx, Nz, Nt)
    u_prime_cross = u_prime_full .* cos(θ) .+ w_prime_cen .* sin(θ)
    w_prime_cross = .-u_prime_full .* sin(θ) .+ w_prime_cen .* cos(θ)

    # -------------------------------------------------------------------------
    # Step 5: Compute flux divergence terms
    #
    # Mean self: ∂(<u><B>)/∂x and ∂(<w><B>)/∂z
    #   Form the 2D mean flux, then differentiate once.
    # -------------------------------------------------------------------------
    flux_mean_x = u_mean_cross .* long_B                   # (Nx, Nz)
    flux_mean_z = w_mean_cross .* long_B                   # (Nx, Nz)
    mean_self_x = mmderiv(xC, flux_mean_x)                 # ∂(<u><B>)/∂x
    mean_self_z = mmderiv(zC, flux_mean_z')'               # ∂(<w><B>)/∂z

    # Tidal: <∂(ũB̃)/∂x> and <∂(w̃B̃)/∂z>
    #   For each phase: form flux, differentiate; then average over phases.
    div_tidal_x = zeros(Nx, Nz, num_phases)
    div_tidal_z = zeros(Nx, Nz, num_phases)
    for p in 1:num_phases
        flux_p_x = u_tilde_cross[:,:,p] .* tilde_B[:,:,p]  # ũB̃ at phase p
        flux_p_z = w_tilde_cross[:,:,p] .* tilde_B[:,:,p]  # w̃B̃ at phase p
        div_tidal_x[:,:,p] = mmderiv(xC, flux_p_x)
        div_tidal_z[:,:,p] = mmderiv(zC, flux_p_z')'
    end
    tidal_x = nanmean(div_tidal_x, dims=3)[:,:,1]          # phase-average
    tidal_z = nanmean(div_tidal_z, dims=3)[:,:,1]

    # Turbulent: <∂(u'B')/∂x> and <∂(w'B')/∂z>
    #   For each time step: form flux, differentiate; then time-average.
    div_turb_x = zeros(Nx, Nz, Nt)
    div_turb_z = zeros(Nx, Nz, Nt)
    for i in 1:Nt
        flux_i_x = u_prime_cross[:,:,i] .* B_prime_full[:,:,i]  # u'B' at time i
        flux_i_z = w_prime_cross[:,:,i] .* B_prime_full[:,:,i]  # w'B' at time i
        div_turb_x[:,:,i] = mmderiv(xC, flux_i_x)
        div_turb_z[:,:,i] = mmderiv(zC, flux_i_z')'
    end
    turb_x = nanmean(div_turb_x, dims=3)[:,:,1]            # time-average → (Nx, Nz)
    turb_z = nanmean(div_turb_z, dims=3)[:,:,1]

    # Phase-resolved turbulent term: average div_turb within each phase bin → (Nx, Nz, num_phases)
    # For each phase p, average only over time steps whose bin_indices == p.
    turb_x_phase = zeros(Nx, Nz, num_phases)
    turb_z_phase = zeros(Nx, Nz, num_phases)
    counts_phase  = zeros(Int, num_phases)
    for i in 1:Nt
        p = bin_indices[i]
        turb_x_phase[:,:,p] .+= div_turb_x[:,:,i]
        turb_z_phase[:,:,p] .+= div_turb_z[:,:,i]
        counts_phase[p] += 1
    end
    for p in 1:num_phases
        if counts_phase[p] > 0
            turb_x_phase[:,:,p] ./= counts_phase[p]
            turb_z_phase[:,:,p] ./= counts_phase[p]
        end
    end

    # -------------------------------------------------------------------------
    # Step 6: Package results
    #   *_phase fields are (Nx, Nz, num_phases) — phase-resolved
    #   others are (Nx, Nz) — long-time averaged
    # -------------------------------------------------------------------------
    terms = Dict(
        # Long-time averages (Nx, Nz)
        "mean_self_x"   => mean_self_x,    # phase-independent by construction
        "mean_self_z"   => mean_self_z,
        "tidal_x"       => tidal_x,
        "tidal_z"       => tidal_z,
        "turb_x"        => turb_x,
        "turb_z"        => turb_z,
        # Phase-resolved (Nx, Nz, num_phases)
        "tidal_x_phase" => div_tidal_x,    # ∂(ũB̃)/∂x at each phase
        "tidal_z_phase" => div_tidal_z,    # ∂(w̃B̃)/∂z at each phase
        "turb_x_phase"  => turb_x_phase,   # <∂(u'B')/∂x>_p conditional on phase
        "turb_z_phase"  => turb_z_phase,   # <∂(w'B')/∂z>_p conditional on phase
    )

    # Mean fluxes saved for diagnostics (can extend to tilde/prime if needed)
    fluxes = Dict(
        "flux_mean_x" => flux_mean_x,   # <u><B>  (Nx, Nz)
        "flux_mean_z" => flux_mean_z,   # <w><B>  (Nx, Nz)
    )

    return terms, fluxes
end

# =============================================================================
# Main Computation
# =============================================================================
xC, zC, t, time_range, all_data, Nx, Nz = load_files_and_time_range(base_dir, file_pattern, start_tp, end_tp)

# Compute long-time means < >
long_means = compute_long_time_means(all_data, time_range, Nx, Nz)

# Compute phase averages \bar{ }, then tilde = \bar{} - < >
phase_avgs, tilde_avgs, bin_indices = compute_phase_averages(all_data, time_range, num_phases, Nx, Nz, long_means)

# Extract key vars (tilde versions)
tilde_u = tilde_avgs["uhat_avg"]
tilde_w = tilde_avgs["what_avg"]  # Adjust if w-centered
tilde_w_cen = 0.5 .* (tilde_w[:, 1:end-1, :] .+ tilde_w[:, 2:end, :])  # (Nx, Nz, num_phases)
tilde_avgs["what_avg_cen"] = tilde_w_cen  # Store centered version
tilde_B = tilde_avgs["B"]

long_u = long_means["uhat_avg"]
long_w = long_means["what_avg"]
long_w_cen = 0.5 .* (long_w[:, 1:end-1] .+ long_w[:, 2:end])  # (Nx, Nz)
long_means["what_avg_cen"] = long_w_cen  # Store centered version
long_B = long_means["B"]

# Compute flux divergence budget terms
budget_terms, fluxes = compute_budget_terms(all_data, bin_indices, time_range,
                                            tilde_u, tilde_w, tilde_B,
                                            long_u, long_w, long_B,
                                            xC, zC, θ, Nx, Nz, num_phases)

# =============================================================================
# Save to NetCDF
# =============================================================================
ds_out = Dataset(output_fn, "c")
defDim(ds_out, "x", Nx)
defDim(ds_out, "z", Nz)
defDim(ds_out, "phase", num_phases)

defVar(ds_out, "x", xC, ("x",))
defVar(ds_out, "z", zC, ("z",))
defVar(ds_out, "phase", collect(0:num_phases-1) * 2π / num_phases, ("phase",))  # Phase values

# Save long-time means
for (var, data) in long_means
    if var != "what_avg"
        defVar(ds_out, "$(var)_long_mean", data, ("x", "z"))        
    end
end

# Save phase avgs (tilde = zero-mean)
for (var, data) in tilde_avgs
    if var == "what_avg_cen"
        defVar(ds_out, "$(var)_tilde", data, ("x", "z", "phase"))
    end
end

# Save budget terms: 2D long-time averages and 3D phase-resolved fields
phase_resolved_keys = Set(["tidal_x_phase", "tidal_z_phase", "turb_x_phase", "turb_z_phase"])
for (term, data) in budget_terms
    if term in phase_resolved_keys
        defVar(ds_out, term, data, ("x", "z", "phase"))
    else
        defVar(ds_out, term, data, ("x", "z"))
    end
end

# Save mean fluxes
defVar(ds_out, "flux_mean_x", fluxes["flux_mean_x"], ("x", "z"))  # <u><B>
defVar(ds_out, "flux_mean_z", fluxes["flux_mean_z"], ("x", "z"))  # <w><B>

close(ds_out)
println("Saved decomposed fields to $output_fn")

# =============================================================================
# Post-Processing (Load and Plot Example)
# =============================================================================

θ = 0.008  # Tilt angle (radians)
base_dir = "output/3D_idealized_tilt_$(θ)/"  # Base directory for files

# Time range for long-time averaging: Specify start and end tidal periods (TP)
# Example: start_tp = 10, end_tp = 40  # Averages over TP 10 to 40
start_tp = 180
end_tp = 180.0
file_pattern = "internal_tide_theta=$(θ)_Nx=2000_Nz=500_tᶠ=$(end_tp)_yavg"  # Base filename pattern
# Number of phase bins (e.g., 24 for 1/24 TP)
num_phases = 24  # Change to 8 if needed for coarser bins

# Tidal parameters
U₀ = 0.025  # Tidal velocity amplitude (m/s)
ω₀ = 1.4e-4  # Tidal frequency (rad/s)

# Output filename
output_fn = "$(base_dir)phase_avg_decomp_tp_$(start_tp)-$(end_tp).nc"

ds = Dataset(output_fn, "r")

# Load all terms for comparison
# Load only the 2D budget terms (x × z)

tidal_z            = ds["tidal_z"][:, :]
mean_self_z        = ds["mean_self_z"][:, :]
mean_self_x        = ds["mean_self_x"][:, :]
tidal_x            = ds["tidal_x"][:, :]
turb_z             = ds["turb_z"][:, :]
turb_x             = ds["turb_x"][:, :]
∇κ∇B_long_mean    = ds["∇κ∇B_long_mean"][:, :]
x = ds["x"][:]
z = ds["z"][:]

# Use buoyancy long-time mean to define topography mask (NaN below topo)

# Apply mask to all plotted budget fields
mean_self_x[mean_self_x .== 0.0] .= NaN  # Avoid plotting zeros if any
mean_self_z[mean_self_z .== 0.0] .= NaN
tidal_x[tidal_x .== 0.0] .= NaN
tidal_z[tidal_z .== 0.0] .= NaN
turb_x[turb_x .== 0.0] .= NaN
turb_z[turb_z .== 0.0] .= NaN
∇κ∇B_long_mean[∇κ∇B_long_mean .== 0.0] .= NaN
# Close the dataset
close(ds)

# =============================================================================
# 3x3 Budget Terms Plot
# =============================================================================

# Compute summed components for each row
mean_self_sum = mean_self_x .+ mean_self_z
tidal_sum     = tidal_x .+ tidal_z
turb_sum      = turb_x .+ turb_z
row1_clim = (-5e-9, 5e-9)  #(-row1_max, row1_max)
row2_clim = (-5e-9, 5e-9)
row3_clim = (-5e-9, 5e-9)
row4_clim = (-5e-9, 5e-9)

close("all")
fig, axs = subplots(4, 3, figsize=(14, 12), sharex=true, sharey=true)

function plot_budget_panel(ax, x, z, field, title; clim=nothing)
    C = transpose(field)  # (z,x) for pcolormesh
    pm = ax.pcolormesh(x, z, C, shading="auto", cmap="RdBu_r")
    pm.cmap.set_bad("white")  # NaNs (topography) in white
    if clim !== nothing
        pm.set_clim(clim...)
    end
    ax.set_title(title, fontsize=10)
    return pm
end

# Row 1: mean_self
pm11 = plot_budget_panel(axs[1, 1], x, z, -mean_self_x, L"-\langle u \rangle \partial_x \langle B \rangle"; clim=row1_clim)
pm12 = plot_budget_panel(axs[1, 2], x, z, -mean_self_z, L"-\langle w \rangle \partial_z \langle B \rangle"; clim=row1_clim)
pm13 = plot_budget_panel(axs[1, 3], x, z, -mean_self_sum, L"-\nabla\cdot (\langle\boldsymbol{u}\rangle \langle B \rangle)"; clim=row1_clim)

# Row 2: tidal
pm21 = plot_budget_panel(axs[2, 1], x, z, -tidal_x, L"-\langle \tilde{u} \partial_x \tilde{B} \rangle"; clim=row2_clim)
pm22 = plot_budget_panel(axs[2, 2], x, z, -tidal_z, L"-\langle \tilde{w} \partial_z \tilde{B} \rangle"; clim=row2_clim)
pm23 = plot_budget_panel(axs[2, 3], x, z, -tidal_sum, L"-\langle \nabla\cdot (\tilde{\boldsymbol{u}}\tilde{B}) \rangle"; clim=row2_clim)

# Row 3: turbulent
pm31 = plot_budget_panel(axs[3, 1], x, z, -turb_x, L"-\langle u' \partial_x B' \rangle"; clim=row3_clim)
pm32 = plot_budget_panel(axs[3, 2], x, z, -turb_z, L"-\langle w' \partial_z B' \rangle"; clim=row3_clim)
pm33 = plot_budget_panel(axs[3, 3], x, z, -turb_sum, L"-\langle \nabla\cdot (\boldsymbol{u}'B') \rangle"; clim=row3_clim)
pm43 = plot_budget_panel(axs[4, 3], x, z, ∇κ∇B_long_mean,
                         L"\langle \nabla\cdot (\kappa \nabla B) \rangle"; clim=row4_clim)# Axis labels
for r in 1:4
    axs[r, 1].set_ylabel("z (m)")
end
for c in 1:3
    axs[4, c].set_xlabel("x (m)")
end

# Single shared colorbar to the right of the 3×3 panels
fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])  # leave space on the right for colorbar
cbar3 = fig.colorbar(pm31, ax=axs[:, 3], location="right", fraction=0.046, pad=0.04)
cbar3.set_label("Buoyancy budget terms (m/s³)")

plot_fn = "$(base_dir)budget_3x3_$(θ)_t=$(start_tp)-$(end_tp).png"
savefig(plot_fn, dpi=150)
println("Saved 3x3 budget panels plot to $(plot_fn)")

