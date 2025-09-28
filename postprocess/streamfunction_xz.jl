using NCDatasets
using NaNStatistics

function average_fields(simname, tfs)
    θ = simname == "tilt" ? 0.0036 : 0
    sum_uhat = nothing
    sum_what = nothing
    sum_B = nothing
    nt_total = 0
    xC = zC = yC = zF = yF = nothing

    for tᶠ in tfs
        filename = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_analysis_round=all_threeD.nc")
        ds = Dataset(filename, "r")
        # Get grid info (once)
        if xC === nothing
            xC = ds["x_caa"][:]
            zC = ds["z_aac"][:]
            zF = ds["z_aaf"][:]
            yC = ds["y_aca"][:]
            yF = ds["y_afa"][:]
        end
        nt = size(ds["uhat"], 4)
        for t in 1:nt
            uhat = ds["uhat"][:, :, :, t]  # (x, y, z)
            what = ds["what"][:, :, :, t]
            B = ds["B"][:, :, :, t]
            if sum_uhat === nothing
                sum_uhat = zeros(size(uhat))
                sum_what = zeros(size(what))
                sum_B = zeros(size(B))
            end
            sum_uhat .+= uhat
            sum_what .+= what
            sum_B .+= B
            nt_total += 1
            println("Processed time slice $t of file $filename")
        end
        close(ds)
    end

    # Average over all time slices
    uhat_avg = sum_uhat ./ nt_total
    what_avg = sum_what ./ nt_total
    B_avg = sum_B ./ nt_total

    # Integrate in y (2nd dimension)
    dy = yF[end] - yF[end-1]
    uhat_int = nansum(uhat_avg * dy, dims=2)  # (x, z)
    what_int = nansum(what_avg * dy, dims=2)  # (x, z)

    # Take center slice in y for B
    B_avg_center = B_avg[:, div(size(B_avg, 2), 2), :]  # (x, z)

    Nz = length(zC)
    dz = abs.(zF[1:end-1] - zF[2:end])

    # Compute U, W, ψ
    what_cen = (what_int[:, 1, 1:end-1] .+ what_int[:, 1, 2:end]) ./ 2
    wtemp = (vcat(what_cen[end:end, :], what_cen[1:end-1, :]) .+ what_cen[:, :]) ./ 2
    U = (uhat_int[:,1,:] * cos(θ) .+ wtemp * sin(θ))
    W = (-uhat_int[:,1,:] * sin(θ) .+ wtemp * cos(θ))
    ψ = zeros(length(xC), Nz)
    for i in 1:length(xC)
        for j in 2:Nz
            ψ[i, j] = ψ[i, j-1] - U[i, j-1] * dz[j]
        end
    end

    return xC, zC, ψ, U, W, B_avg_center
end
# --- Specify time ranges ---
tfs_tilt = 457.5:0.5:462.0
tfs_flat = 451.5:0.5:456.0

# --- Compute averages ---
xC_tilt, zC_tilt, ψ_tilt, U_tilt, W_tilt, Bavg_tilt = average_fields("tilt", tfs_tilt)
xC_flat, zC_flat, ψ_flat, U_flat, W_flat, Bavg_flat = average_fields("flat", tfs_flat)


function save_to_nc(filename, xC, zC, ψ, U, W, Bavg)
    ds = Dataset(filename, "c")
    defDim(ds, "x", length(xC))
    defDim(ds, "z", length(zC))

    v_x = defVar(ds, "x", Float64, ("x",))
    v_z = defVar(ds, "z", Float64, ("z",))
    v_psi = defVar(ds, "psi", Float64, ("x", "z"))
    v_U = defVar(ds, "U", Float64, ("x", "z"))
    v_W = defVar(ds, "W", Float64, ("x", "z"))
    v_Bavg = defVar(ds, "Bavg", Float64, ("x", "z"))

    v_x[:] = xC
    v_z[:] = zC
    v_psi[:, :] = ψ
    v_U[:, :] = U
    v_W[:, :] = W
    v_Bavg[:, :] = Bavg

    close(ds)
end

save_to_nc("output/tilt/streamfunction_tilt_avg.nc", xC_tilt, zC_tilt, ψ_tilt, U_tilt, W_tilt, Bavg_tilt)
save_to_nc("output/flat/streamfunction_flat_avg.nc", xC_flat, zC_flat, ψ_flat, U_flat, W_flat, Bavg_flat)

using NCDatasets
using PyPlot
# read the data that has the mask of buoyancy
filename_b = string("output/", "tilt", "/internal_tide_theta=",0.0036,"_Nx=500_Nz=250_tᶠ=",10, "_threeD_timeavg.nc")
ds_b = Dataset(filename_b,"r")
b = ds_b["b"][:,500,:,1]

# Read tilt data
ds_tilt = Dataset("output/tilt/streamfunction_tilt_avg.nc", "r")
xC_tilt = ds_tilt["x"][:]
zC_tilt = ds_tilt["z"][:]
ψ_tilt = ds_tilt["psi"][:, :]
U_tilt = ds_tilt["U"][:, :]
W_tilt = ds_tilt["W"][:, :]
B_tilt_center = ds_tilt["Bavg"][:, :]
close(ds_tilt)

# Read flat data
ds_flat = Dataset("output/flat/streamfunction_flat_avg.nc", "r")
xC_flat = ds_flat["x"][:]
zC_flat = ds_flat["z"][:]
ψ_flat = ds_flat["psi"][:, :]
U_flat = ds_flat["U"][:, :]
W_flat = ds_flat["W"][:, :]
B_flat_center = ds_flat["Bavg"][:, :]
close(ds_flat)

ψ_tilt[ψ_tilt.==0] .= NaN
ψ_flat[ψ_flat.==0] .= NaN
U_tilt[U_tilt.==0] .= NaN
U_flat[U_flat.==0] .= NaN
W_tilt[W_tilt.==0] .= NaN
W_flat[W_flat.==0] .= NaN
B_tilt_center[b.==0] .= NaN
B_flat_center[b.==0] .= NaN
# topography data
# Interpolate topography
using MAT
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=500)
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=1000)

using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)
z_interp_y = nanmean(z_interp,dim=2)

close("all")
PyPlot.rc("font", size=20)
fig, ax = subplots(2, 1, figsize=(10, 8), sharex=true, constrained_layout=true)

# Tilt
c1 = ax[1].pcolor(xC_tilt, zC_tilt, ψ_tilt', cmap="coolwarm", vmin=-4e4, vmax=4e4)
c2 = ax[1].contour(xC_tilt[:], zC_tilt[:], ψ_tilt', levels=-1e7:2e4:1e7, colors=[63 43 43]./255, linewidth=0.8, linestyles="-")

# colorbar(c1, ax=ax[1], label="ψ [m³/s]")
ax[1].quiver(xC_tilt[1:12:end], zC_tilt[1:6:end], U_tilt[1:12:end, 1:6:end]', W_tilt[1:12:end, 1:6:end]', angles="xy", scale_units="xy")
c3 = ax[1].contour(xC_tilt, zC_tilt, B_tilt_center[:,:]', colors="green", levels=0.0007:0.0001:0.0025, linewidth=0.8, linestyles="-")
ax[1].plot(xC_tilt[:], z_interp_y, linewidth=2.5, color="brown")
ax[1].set_title("(a) tilt", loc="left", fontsize=18)
ax[1].set_facecolor("gray")
ax[1].set_ylabel("z [m]")
ax[1].set_xlabel("x [km]")
ax[1].set_xticks(0:2500:15000)

# Add reference vector on topography
# Calculate a reasonable reference velocity
ref_value = 600  # Use a more representative reference value
arrow_scale = 0.3
ax[1].quiver(xC_tilt[150], 60, ref_value/3, 0, 
             angles="xy", scale_units="xy", color="black", scale=arrow_scale, headwidth=1.6, headlength=2.5, headaxislength=2.5)
ax[1].quiver(xC_tilt[150], 60, 0, ref_value/10, 
             angles="xy", scale_units="xy", color="black", scale=arrow_scale, headwidth=1.6, headlength=2.5, headaxislength=2.5)
ax[1].text(xC_tilt[150]+850, 30, "$(round(ref_value/3, digits=0)) m² s⁻¹", fontsize=12)
ax[1].text(xC_tilt[150]+150, 180, "$(round(ref_value/10, digits=0)) m² s⁻¹", fontsize=12)


# Flat
c2 = ax[2].pcolor(xC_flat, zC_flat, ψ_flat', cmap="coolwarm", vmin=-4e4, vmax=4e4)
c5 = ax[2].contour(xC_flat[:], zC_flat[:], ψ_flat', levels=-1e7:2e4:1e7, colors=[63 43 43]./255, linewidth=0.8, linestyles="-")
colorbar(c2, ax=ax[2], label="ψ [m³/s]",shrink=0.8, pad=0.02, extend="both")
ax[2].quiver(xC_flat[1:12:end], zC_flat[1:6:end], U_flat[1:12:end, 1:6:end]', W_flat[1:12:end, 1:6:end]', angles="xy", scale_units="xy")
ax[2].plot(xC_tilt[:], z_interp_y, linewidth=2.5, color="brown")
c3 = ax[2].contour(xC_tilt, zC_tilt, B_flat_center[:,:]', colors="green", levels=0.0007:0.0001:0.0025, linewidth=0.8, linestyles="-")
ax[2].set_title("(b) flat", loc="left", fontsize=18)
ax[2].set_facecolor("gray")
ax[2].set_ylabel("z [m]")
ax[2].set_xlabel("x [km]")
ax[2].set_xticks(0:2500:15000)
ax[2].set_xticklabels(["0","2.5","5","7.5","10","12.5","15"])
savefig("output/comparison_streamfunction_tilt_vs_flat_avg.png", dpi=200, bbox_inches="tight")