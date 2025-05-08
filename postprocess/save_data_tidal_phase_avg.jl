basefile = "/scratch/bcpi/cliu28/internal-tide-mixing"
cd(basefile)
using Interpolations
using NCDatasets
using Oceananigans
using Oceananigans.Units
using NaNStatistics
function deriv(z, y)
    dydz = diff(y[:, :, :, :], dims=3) ./ reshape(diff(z[:]), 1, 1, length(zC) - 1)
    return dydz
end
function calculate_background_buoyancy(θ)
    θ = θ
    ĝ = (sin(θ), 0, cos(θ)) # the vertical unit vector in rotated coordinates
    N = 1e-3
    @inline ẑ(x, z, ĝ) = x * ĝ[1] + z * ĝ[3]
    @inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)

    # Create a background field
    B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N²=N^2))

    # Setup grid
    H = 2.25kilometers # vertical extent
    Lx = 15kilometers # along-canyon extent
    Ly = 30kilometers # cross-canyon extent
    Nx = 500
    Ny = 1000
    Nz = 250

    # Bottom-intensified stretching for vertical grid
    z_faces(k) = -H * ((1 + ((Nz + 1 - k) / Nz - 1) / 1.2) *
                       (1 - exp(-15 * (Nz + 1 - k) / Nz)) / (1 - exp(-15)) - 1)

    grid = RectilinearGrid(size=(Nx, Ny, Nz),
        x=(0, Lx),
        y=(0, Ly),
        z=z_faces,
        halo=(4, 4, 4),
        topology=(Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded))

    model = NonhydrostaticModel(
        grid=grid,
        background_fields=(; b=B̄_field),
        tracers=:b
    )

    return interior(compute!(Field(model.background_fields.tracers.b)))[:, :, :]
end

simname = "tilt"
tᶠ = 460
θ = 3.6e-3
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab, "r")
hab = ds_hab["hab"][:, :, :];
# for mask
filename_verification = string("output/", "tilt", "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", 10, "_threeD_timeavg.nc")
ds_verification = Dataset(filename_verification, "r")
b_ver = ds_verification["b"][:, :, :, 1:1];          # buoyancy perturbation
B̄ = calculate_background_buoyancy(θ)

# define grids
zC = ds_verification["zC"][:];
zF = ds_verification["zF"][:];
Nz = length(zC[:]);

xC = ds_verification["xC"][:];
xF = ds_verification["xF"][:];
Nx = length(xC[:]);
dx = xF[end] - xF[end-1];

yC = ds_verification["yC"][:];
yF = ds_verification["yF"][:];
Ny = length(yC[:]);
dy = yF[end] - yF[end-1];
# t = ds_velocity["time"][:];


# data_mode = "10TP average"   # "10TP average" or "tidal phase average"
data_mode = "tidal phase average"   # "10TP average" or "tidal phase average"

if data_mode == "10TP average"
    filename_velocity = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_timeavg_const_dt_u-v-w-Rig.nc")
    ds_velocity = Dataset(filename_velocity, "r")
    filename_B = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_timeavg_B-c.nc")
    ds_B = Dataset(filename_B, "r")

    what = nanmean(ds_velocity["what"][:, :, :, :], dim=4)    # 10TP average true w
    what_mag = nanmean(abs.(ds_velocity["what"][:, :, :, :]), dim=4)    # 10TP average true |w|
    # Interpolate each row
    what_center = zeros(size(what, 1), size(what, 2), length(zC))
    what_mag_center = zeros(size(what_mag, 1), size(what_mag, 2), length(zC))
    for i in 1:size(what, 1)
        for j in 1:size(what, 2)
            itp = linear_interpolation(zF[:], what[i, j, :], extrapolation_bc=Line())
            what_center[i, j, :] = itp(zC)
            itp_mag = linear_interpolation(zF[:], what_mag[i, j, :], extrapolation_bc=Line())
            what_mag_center[i, j, :] = itp(zC)
        end
    end
    B = nanmean(ds_B["B"][:, :, :, :], dim=4)                 # 10TP average B
    Bz = deriv(zC, B)
    Bz[b_ver[:, :, 1:end-1, 1:1].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
    # interpolate Bz from faces to center cell
    Bz_center = zeros(size(Bz, 1), size(Bz, 2), length(zC))
    for i in 1:size(Bz, 1)
        for j in 1:size(Bz, 2)
            itp = linear_interpolation(zF[2:end-1], Bz[i, j, :], extrapolation_bc=Line())
            Bz_center[i, j, :] = itp(zC)
        end
    end
    b = B .- B̄

elseif data_mode == "tidal phase average"
    filename_velocity = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_u-v-w-c.nc")
    ds_velocity = Dataset(filename_velocity, "r")
    filename_B = string("output/", simname, "/internal_tide_theta=", θ, "_Nx=500_Nz=250_tᶠ=", tᶠ, "_threeD_B-c.nc")
    ds_B = Dataset(filename_B, "r")
    # separate the data into two parts because it could be memory intensive
    part = 2
    if part==1
        ni = 1:6
    elseif part==2
        ni = 7:12
    end
    what = zeros(Nx, Ny, Nz + 1, length(ni))
    B = zeros(Nx, Ny, Nz, length(ni))
    for n in 1:length(ni)
        if ni[n] == 1   # first tidal phase
            what[:, :, :, n] = nanmean(ds_velocity["what"][:, :, :, 12:12:120], dim=4)
            B[:, :, :, n] = nanmean(ds_B["B"][:, :, :, 12:12:120], dim=4)
        else            # the rest of the tidal phases
            for i in 1:10
                what[:, :, :, n] += ds_velocity["what"][:, :, :, 12i-(12-(ni[n]-1))]    # true w
                B[:, :, :, n] += ds_B["B"][:, :, :, 12i-(12-(ni[n]-1))]    # true B
            end
            what[:, :, :, n] ./= 10
            B[:, :, :, n] ./= 10
        end
        @info "Tidal phase $ni[n] done"
    end
    b = B .- B̄
    b[abs.(b).<1e-15] .= 0
    Bz = deriv(zC, B)
    Bz[b[:, :, 1:end-1, :].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
    # interpolate Bz from faces to center cell
    Bz_center = zeros(size(Bz, 1), size(Bz, 2), length(zC), length(ni))
    for n in 1:length(ni)
        for i in 1:size(Bz, 1)
            for j in 1:size(Bz, 2)
                itp = linear_interpolation(zF[2:end-1], Bz[i, j, :, n], extrapolation_bc=Line())
                Bz_center[i, j, :, n] = itp(zC)
            end
        end
    end
    # Interpolate each row
    what_center = zeros(size(what, 1), size(what, 2), length(zC), length(ni))
    for n in 1:length(ni)
        for i in 1:size(what, 1)
            for j in 1:size(what, 2)
                itp = linear_interpolation(zF[:], what[i, j, :, n], extrapolation_bc=Line())
                what_center[i, j, :, n] = itp(zC)
            end
        end
    end
end



# Define dimensions based on the data mode
if data_mode == "10TP average"
    # save the data into netcdf file including what_center and Bz_center
    output_filename = string("output/", simname, "/10TP_average_w_Bz_theta=", θ, "_tᶠ=", tᶠ, ".nc")
    # Check if file exists and remove it if it does
    if isfile(output_filename)
        rm(output_filename)
        println("Removed existing file: $output_filename")
    end
    # Create a new NetCDF file
    ds_output = Dataset(output_filename, "c")
    # Define dimensions
    defDim(ds_output, "x", Nx)
    defDim(ds_output, "y", Ny)
    defDim(ds_output, "z", Nz)

    # Define variables
    v_what = defVar(ds_output, "what_center", Float64, ("x", "y", "z"))
    v_what_mag = defVar(ds_output, "what_mag_center", Float64, ("x", "y", "z"))
    v_Bz = defVar(ds_output, "Bz_center", Float64, ("x", "y", "z"))
    v_b = defVar(ds_output, "b", Float64, ("x", "y", "z"))

    # Write data
    v_what[:, :, :] = what_center
    v_what_mag[:, :, :] = what_mag_center
    v_Bz[:, :, :] = Bz_center
    v_b[:, :, :] = b

elseif data_mode == "tidal phase average"
    # save the data into netcdf file including what_center and Bz_center
    output_filename = string("output/", simname, "/tidal_phase_average_w_Bz_theta=", θ, "_tᶠ=", tᶠ, "_part-", part, ".nc")
    # Check if file exists and remove it if it does
    if isfile(output_filename)
        rm(output_filename)
        println("Removed existing file: $output_filename")
    end
    # Create a new NetCDF file
    ds_output = Dataset(output_filename, "c")
    # Define dimensions
    defDim(ds_output, "x", Nx)
    defDim(ds_output, "y", Ny)
    defDim(ds_output, "z", Nz)
    defDim(ds_output, "tidal_phase", length(ni))

    # Define variables
    v_what = defVar(ds_output, "what_center", Float64, ("x", "y", "z", "tidal_phase"))
    v_Bz = defVar(ds_output, "Bz_center", Float64, ("x", "y", "z", "tidal_phase"))

    # Write data
    v_what[:, :, :, :] = what_center
    v_Bz[:, :, :, :] = Bz_center

    # Add tidal phase information
    v_phase = defVar(ds_output, "tidal_phase", Int32, ("tidal_phase",))
    v_phase[:] = ni
end

# Add grid variables
v_xC = defVar(ds_output, "xC", Float64, ("x",))
v_yC = defVar(ds_output, "yC", Float64, ("y",))
v_zC = defVar(ds_output, "zC", Float64, ("z",))
v_xC[:] = xC
v_yC[:] = yC
v_zC[:] = zC

# Add global attributes
ds_output.attrib["data_mode"] = data_mode
ds_output.attrib["simulation"] = simname
ds_output.attrib["theta"] = θ
ds_output.attrib["tf"] = tᶠ

# Close the file
close(ds_output)
println("Data saved to: $output_filename")

