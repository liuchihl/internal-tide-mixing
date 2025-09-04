using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations


function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end
 


# include("functions/mmderiv.jl")
simname = "tilt"
tᶠ = 10

## load data
# filename_field = "output/internal_tide_3days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_xz.nc"
# ds_field = Dataset(filename_field,"r")
# bathy_data = "output/bathymetry.nc"
# Bathy = Dataset(bathy_data,"r")
# filename_slice = "output/supercritical_tilt/backgroundfluxdivergence_smagorinky/backgroundfluxdivergence_smagorinkyinternal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_0_5_xz.nc"
filename_slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_xz.nc")
# filename_slice = "output/no_tilt/internal_tide_5days-theta=0_realtopo3D_Nx500_Nz250_slices_30_50_xz.nc"
ds_slice = Dataset(filename_slice,"r")

# filename_mask = "output/higher_top_resolution/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_0-13_slices_xz.nc"
# ds_mask = Dataset(filename_mask,"r")


# grids
zC = ds_slice["zC"]; Nz=length(zC)
zF = ds_slice["zF"]; 
xC = ds_slice["xC"]; Nx=length(xC)
yC = ds_slice["yC"]; Ny=length(yC)
t = ds_slice["time"];

# load all data
B = ds_slice["B"].var;        B = B[:,:,:,:];
# Bz = ds_slice["Bz"].var;        Bz = Bz[:,:,:,:];
b = ds_slice["b"].var;        b = b[:,:,:,:];
# b = ds_mask["b"].var;        b = b[:,:,:,1];
# b = b.*ones(1,1,1,length(t));
uhat = ds_slice["uhat"].var; uhat = uhat[:,:,:,:];
# what = ds_slice["what"].var; #what = what[:,:,:,:];
# wb = ds_slice["wb"].var;     #wb = wb[:,:,:,:];
ε = ds_slice["ε"].var;       ε = ε[:,:,:,:];
χ = ds_slice["χ"].var;       χ = χ[:,:,:,:];


# set topography to NaN
uhat[uhat.==0] .= NaN
# what[what.==0] .= NaN
# wb[isnan.(uhat)] .= NaN
ε[isnan.(uhat)] .= NaN
B[b.==0] .= NaN
# Bz[b.==0] .= NaN
# B[isnan.(uhat)] .= NaN
χ[b.==0] .= NaN
# χ[isnan.(uhat)] .= NaN

# dBdz[isnan.(dBdz)] .= NaN
# dûdz[isnan.(dûdz)] .= NaN

# close(ds_field)


# derivatives of B and u. Only slices are used because it can be memory intensive
N = 1.e-3
Bz =  deriv(zC,B);
Bz[b[:,:,1:end-1,:].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
# interpolate Bz from faces to center cell
# Interpolate each row
Bz_center = zeros(size(Bz,1),size(Bz,2),length(zC),size(Bz,4))
for i in 1:size(Bz,1)
    for j in 1:size(Bz,2)
        for k in 1:size(Bz,4)
            itp = linear_interpolation(zF[2:end-1], Bz[i,j,:,k], extrapolation_bc=Line())
        Bz_center[i,j,:,k] = itp(zC)
        end
    end
end
χ = χ./Bz_center;

χ[χ.<0] .= 0

# plot
n = Observable(1)
ûₙ = @lift(uhat[:,1,:,$n])
εₙ = @lift(log10.(ε[:,1,:,$n]))
χₙ = @lift(log10.(abs.(χ[:,1,:,$n])))
Bₙ = @lift(B[:,1,:,$n])
# Bzₙ = @lift(Bz[:,1,:,$n])
# χₙ = @lift(log10.(χ[:,1,:,$n]))
# dBdzₙ = @lift interior(dBdz[$n], :, 1, :)
# dûdzₙ = @lift interior(dûdz[$n], :, 1, :)
ω₀ = 1.4e-4
M₂_period = 2π/ω₀



fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs = (xlabel = "Zonal distance x (m)",
                  ylabel = "Elevation z (m)",
                  limits = ((0, ds_slice["xF"][end]), (0, ds_slice["zF"][end])),
                  )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "u and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_uz = Axis(fig[1, 2]; title = "du/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_bz = Axis(fig[3, 1]; title = "dB/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_ε = Axis(fig[3, 1]; title = "TKE dissipation rate (log ε) and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_χ = Axis(fig[4, 1]; title = "Scalar variance dissipation rate (log χ) and equally-spaced buoyancy contours (B)", axis_kwargs...)


using ColorSchemes
U₀ = 0.025
hm_u = heatmap!(ax_u, xC[:], zC[:], ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_u = contour!(ax_u, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u; label = "m/s")

# hm_Bz = heatmap!(ax_bz, xC[:], zC[:], Bzₙ, colorrange=(-1e-6,1e-6),
#     colormap = :rainbow_bgyrm_35_85_c69_n256,
#     lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
#     nan_color = :gray)
# ct_bz = contour!(ax_bz, xC, zC, Bₙ,
#     levels=0.:1e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
# Colorbar(fig[3,2], hm_Bz)
hm_ε = heatmap!(ax_ε, xC[:], zC[:], εₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
ct_ε = contour!(ax_ε, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε)


hm_χ = heatmap!(ax_χ, xC[:], zC[:], χₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
ct_χ = contour!(ax_χ, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[4,2], hm_χ)

frames =  (1:1:length(t))

filename = join(split(filename_slice, ".")[1:end-1], ".")

record(fig, string(filename,".mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_slice)






## plot analysis periods
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations


function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end
 
function calculate_background_buoyancy(θ)
    θ = θ
    ĝ = (sin(θ), 0, cos(θ)) # the vertical unit vector in rotated coordinates
    N = 1e-3
    @inline ẑ(x, z, ĝ) = x*ĝ[1] + z*ĝ[3]
    @inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)

    # Create a background field
    B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

    # Setup grid
    H = 2.25kilometers # vertical extent
    Lx = 15kilometers # along-canyon extent
    Ly = 30kilometers # cross-canyon extent
    Nx = 500
    Ny = 1000
    Nz = 250

    # Bottom-intensified stretching for vertical grid
    z_faces(k) = - H * ((1 + ((Nz + 1 - k) / Nz - 1) / 1.2) * 
                    (1 - exp(-15 * (Nz + 1 - k) / Nz)) / (1 - exp(-15)) - 1)

    grid = RectilinearGrid(size=(Nx, Ny, Nz), 
    x = (0, Lx),
    y = (0, Ly), 
    z = z_faces,
    halo = (4, 4, 4),
    topology = (Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded))

    model = NonhydrostaticModel(
    grid = grid,
    background_fields = (; b=B̄_field),
    tracers = :b
    )

    return interior(compute!(Field(model.background_fields.tracers.b)))[:,:,:]
end

# include("functions/mmderiv.jl")
simname = "tilt"
tᶠ = 456.0

## load data
slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_analysis_round=all_slices_xz.nc")
ds = Dataset(slice,"r")
filename_slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_b = Dataset(filename_slice,"r")

B = ds["B"][:,:,:,:];
uhat = ds["uhat"][:,:,:,:];
eps = ds["ε"][:,:,:,:];
b = ds_b["b"][:,:,:,:]; # for mask

# grids
zC = ds["z_aac"][:]; Nz=length(zC)
zF = ds["z_aaf"][:]; 
xC = ds["x_caa"][:]; Nx=length(xC)
xF = ds["x_faa"][:];
yC = ds["y_aca"][:]; Ny=length(yC)
t = ds["time"];

# set topography to NaN
uhat[uhat.==0] .= NaN
eps[eps.==0] .= NaN

# derivatives of B and u. Only slices are used because it can be memory intensive
N = 1.e-3
Bz = deriv(zC,B);
Bz[b[:,:,1:end-1,1:12].==0] .= 0  # the grids are staggered, but this will effectively set the points inside and right above the immersed boundary to 0

Bz_center = (cat(zeros(Nx,1,1,length(t)), Bz, dims=3) .+ cat(Bz, zeros(Nx,1,1,length(t)), dims=3))/2 # average Bz at cell centers
# plot

n = Observable(1)
ûₙ = @lift(uhat[:,1,:,$n])
εₙ = @lift(log10.(eps[:,1,:,$n]))
Bzₙ = @lift(Bz_center[:,1,:,$n])
Bₙ = @lift(B[:,1,:,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = CairoMakie.Figure(resolution = (1000, 900), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, xF[end]), (0, zF[end])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "Velocity (u) with buoyancy contours", axis_kwargs...)
ax_ε = Axis(fig[3, 1]; title = "TKE dissipation rate (log ε) with buoyancy contours", axis_kwargs...)
ax_Bz = Axis(fig[4, 1]; title = "Stratification (dB/dz) with buoyancy contours", axis_kwargs...)

# Plot u with B contours
U₀ = 0.025
hm_u = heatmap!(ax_u, xC[:], zC[:], ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_u = contour!(ax_u, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u; label = "m/s")

# Plot ε with B contours
hm_ε = heatmap!(ax_ε, xC[:], zC[:], εₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
ct_ε = contour!(ax_ε, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε; label = "log₁₀(ε) [m²/s³]")

# Plot Bz with B contours
# Create a custom diverging colormap with yellow at zero
function custom_colormap(min_val=-5e-7, max_val=2e-6)
    # Calculate the zero point position in normalized space [0,1]
    zero_point = -min_val / (max_val - min_val)
    
    # Create arrays of colors for smooth transitions
    neg_colors = range(colorant"darkblue", colorant"yellow", length=100)
    pos_colors = range(colorant"yellow", colorant"darkred", length=100)
    
    # Combine the colors into a single array
    all_colors = vcat(neg_colors[1:end-1], pos_colors)
    
    # Create a colormap with proper scaling
    return cgrad(all_colors, zero_point)
end

hm_Bz = heatmap!(ax_Bz, xC[:], zC[:], Bzₙ,
    colorrange = (-5e-7, 2e-6),
    colormap = custom_colormap(-5e-7, 2e-6),
    nan_color = :gray)
ct_Bz = contour!(ax_Bz, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[4,2], hm_Bz; label = "dB/dz [1/s²]")

frames = 1:1:length(t)

filename = join(split(slice, ".")[1:end-1], ".")

record(fig, string(filename, "_uhat_eps_Bz.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_uvwB)
close(ds_eps)
close(ds_b)



## plot just b and w

## plot analysis periods
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations

function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end
 
function calculate_background_buoyancy(θ)
    θ = θ
    ĝ = (sin(θ), 0, cos(θ)) # the vertical unit vector in rotated coordinates
    N = 1e-3
    @inline ẑ(x, z, ĝ) = x*ĝ[1] + z*ĝ[3]
    @inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)

    # Create a background field
    B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

    # Setup grid
    H = 2.25kilometers # vertical extent
    Lx = 15kilometers # along-canyon extent
    Ly = 30kilometers # cross-canyon extent
    Nx = 500
    Ny = 1000
    Nz = 250

    # Bottom-intensified stretching for vertical grid
    z_faces(k) = - H * ((1 + ((Nz + 1 - k) / Nz - 1) / 1.2) * 
                    (1 - exp(-15 * (Nz + 1 - k) / Nz)) / (1 - exp(-15)) - 1)

    grid = RectilinearGrid(size=(Nx, Ny, Nz), 
    x = (0, Lx),
    y = (0, Ly), 
    z = z_faces,
    halo = (4, 4, 4),
    topology = (Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded))

    model = NonhydrostaticModel(
    grid = grid,
    background_fields = (; b=B̄_field),
    tracers = :b
    )

    return interior(compute!(Field(model.background_fields.tracers.b)))[:,:,:]
end

# include("functions/mmderiv.jl")
simname = "tilt"
tᶠ = 460

## load data
slice_uvwB = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_xz-u-v-w-B.nc")
ds_uvwB = Dataset(slice_uvwB,"r")

B = ds_uvwB["B"][:,:,:,:];
what = ds_uvwB["what"][:,:,:,:];
B̄ = calculate_background_buoyancy(0.0036)
b = B .- B̄[:,500:500,:]
b[abs.(b).<1e-15] .= 0
# grids
zC = ds_uvwB["zC"][:]; Nz=length(zC)
zF = ds_uvwB["zF"][:]; 
xC = ds_uvwB["xC"][:]; Nx=length(xC)
xF = ds_uvwB["xF"][:];
yC = ds_uvwB["yC"][:]; Ny=length(yC)
t = ds_uvwB["time"][:];

# set topography to NaN
what[what.==0] .= NaN
B[b[:,:,:,:].==0] .= NaN
b[b.==0] .= NaN
# plot

n = Observable(1)
ŵₙ = @lift(what[:,1,1:D,$n])
bₙ = @lift(b[:,1,1:D,$n])
Bₙ = @lift(B[:,1,1:D,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = CairoMakie.Figure(resolution = (1000, 900), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, ds_uvwB["xF"][end]), (0, ds_uvwB["zF"][D])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "w with total buoyancy contours", axis_kwargs...)
ax_b = Axis(fig[3, 1]; title = "buoyancy perturbation b with total buoyancy contours", axis_kwargs...)

# Plot u with B contours
function custom_colormap(min_val=-0.0002, max_val=0.0008)
    # Calculate the zero point position in normalized space [0,1]
    zero_point = -min_val / (max_val - min_val)
    
    # Create arrays of colors for smooth transitions to match Spectral colormap
    # Create transition through white at zero
    neg_colors = range(colorant"darkblue", colorant"blue", length=25)
    neg_colors = vcat(neg_colors, range(colorant"blue", colorant"lightblue", length=25))
    neg_colors = vcat(neg_colors, range(colorant"lightblue", colorant"white", length=50))
    pos_colors = range(colorant"white", colorant"yellow", length=25)
    pos_colors = vcat(pos_colors, range(colorant"yellow", colorant"orange", length=25))
    pos_colors = vcat(pos_colors, range(colorant"orange", colorant"red", length=25))
    pos_colors = vcat(pos_colors, range(colorant"red", colorant"darkred", length=25))
    # Combine the colors into a single array
    all_colors = vcat(neg_colors[1:end-1], pos_colors)
    
    # Create a colormap with proper scaling
    return cgrad(all_colors, zero_point)
end

U₀ = 0.01
hm_u = heatmap!(ax_u, xC[:], zF[1:D], ŵₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_u = contour!(ax_u, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u; label = "w m/s")

hm_b = heatmap!(ax_b, xC[:], zC[1:D], bₙ,
    # colorrange = (-0.0002,0.0008),
    colorrange = (0.0004,0.0008),
    colormap = :Spectral,
    # colormap = custom_colormap(0.0003,0.0008),
    nan_color = :gray)
ct_b = contour!(ax_b, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_b; label = "b [m/s²]")

frames = 1:length(t)

filename = join(split(slice_uvwB, ".")[1:end-1], ".")

record(fig, string(filename, "_w-b-zoomin.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_uvwB)






## plot just νₑ and b

## plot analysis periods
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations

function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end
 
# include("functions/mmderiv.jl")
simname = "tilt"
tᶠ = 452

## load data
slice_uvwB = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_xz-u-v-w-B.nc")
ds_uvwB = Dataset(slice_uvwB,"r")
slice_bnu = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_xz-b-nu.nc")
ds_bnu = Dataset(slice_bnu,"r")

B = ds_uvwB["B"][:,:,:,:];
what = ds_uvwB["what"][:,:,:,:];
b = ds_bnu["b"][:,:,:,:];
nu_const = 1.05e-6
nu = ds_bnu["νₑ"][:,:,:,:].+nu_const;
# grids
zC = ds_uvwB["zC"][:]; Nz=length(zC)
zF = ds_uvwB["zF"][:]; 
xC = ds_uvwB["xC"][:]; Nx=length(xC)
xF = ds_uvwB["xF"][:];
yC = ds_uvwB["yC"][:]; Ny=length(yC)
t = ds_uvwB["time"][:];

# set topography to NaN
what[what.==0] .= NaN
B[b[:,:,:,:].==0] .= NaN
nu[b.==0] .= NaN
b[b.==0] .= NaN
# plot
D = 250
n = Observable(1)
bₙ = @lift(b[:,1,1:D,$n])
Bₙ = @lift(B[:,1,1:D,$n])
nuₙ = @lift(nu[:,1,1:D,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀
fig = CairoMakie.Figure(resolution = (1000, 800), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, ds_uvwB["xF"][end]), (0, ds_uvwB["zF"][D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_b = Axis(fig[2, 1]; title = "buoyancy perturbation b with total buoyancy contours", axis_kwargs...)
ax_nu = Axis(fig[3, 1]; title = "eddy viscosity (log νₑ) with total buoyancy contours", axis_kwargs...)

hm_b = heatmap!(ax_b, xC[:], zC[1:D], bₙ,
    colorrange = (-0.0002,0.0008),
    # colorrange = (0.0004,0.0008),
    colormap = :Spectral,
    # colormap = custom_colormap(0.0003,0.0008),
    nan_color = :gray)
ct_b = contour!(ax_b, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_b; label = "b [m/s²]")

# Add log-scale plot of eddy viscosity
log_nuₙ = @lift(log10.($nuₙ))
hm_nu = heatmap!(ax_nu, xC[:], zC[1:D], log_nuₙ,
    colorrange = (-6, -2), colormap = :viridis,
    nan_color = :gray)
ct_nu = contour!(ax_nu, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_nu; label = "log₁₀(νₑ) [m²/s]")

frames = 1:length(t)

filename = join(split(slice_uvwB, ".")[1:end-1], ".")

record(fig, string(filename, "_b-nu.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_uvwB)




## zoomin

## plot analysis periods
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations

function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
end
 
# include("functions/mmderiv.jl")
simname = "tilt"
time_points = [452.0, 452.5, 453.0, 453.5, 454.0, 454.5, 455.0, 455.5, 456.0, 456.5, 457.0]

# Initialize arrays to hold combined data
B_all = []
uhat_all = []
what_all = []
eps_all = []
t_all = []

# Load reference file for mask
filename_mask = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_mask = Dataset(filename_mask,"r")
b_mask = ds_mask["b"][:,:,:,:]; # for mask

# Loop through each time point and load data
for tᶠ in time_points
    @info "Loading data for tᶠ = $tᶠ"
    slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_analysis_round=all_slices_xz.nc")
    ds = Dataset(slice,"r")
    
    # Load data for this time point
    B = ds["B"][:,:,:,:];
    uhat = ds["uhat"][:,:,:,:];
    what = ds["what"][:,:,:,:];
    eps = ds["ε"][:,:,:,:];
    
    # Get time values for this file
    t = ds["time"];
    
    # Set topography to NaN
    uhat[uhat.==0] .= NaN
    what[what.==0] .= NaN
    eps[eps.==0] .= NaN
    
    # Apply mask based on the reference file
    mask_size = min(size(b_mask, 4), size(B, 4))
    B[b_mask[:,:,:,1:mask_size].==0] .= NaN
    
    # Store data
    push!(B_all, B)
    push!(uhat_all, uhat)
    push!(what_all, what)
    push!(eps_all, eps)
    push!(t_all, t[:])
    
    close(ds)
end

# Concatenate all data arrays along the time dimension
B_combined = cat(B_all..., dims=4)
uhat_combined = cat(uhat_all..., dims=4)
what_combined = cat(what_all..., dims=4)
eps_combined = cat(eps_all..., dims=4)
t_combined = vcat(t_all...)

# Get grid information
zC = ds_mask["zC"][:]; Nz=length(zC)
zF = ds_mask["zF"][:]; 
xC = ds_mask["xC"][:]; Nx=length(xC)
xF = ds_mask["xF"][:];
yC = ds_mask["yC"][:]; Ny=length(yC)

# Set up visualization
D = 130 # Depth limit for zoomed view
n = Observable(1)
Bₙ = @lift(B_combined[:,1,1:D,$n])
wₙ = @lift(what_combined[:,1,1:D,$n])
εₙ = @lift(log10.(eps_combined[:,1,1:D,$n]))

ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = CairoMakie.Figure(resolution = (1000, 900), figure_padding=(10, 40, 10, 10), size=(750,750))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, xF[end]), (0, zF[D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t_combined[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_B = Axis(fig[2, 1]; title = "Total buoyancy (B) with contours", axis_kwargs...)
ax_w = Axis(fig[3, 1]; title = "Vertical velocity (w) with buoyancy contours", axis_kwargs...)
ax_ε = Axis(fig[4, 1]; title = "TKE dissipation rate (log ε) with buoyancy contours", axis_kwargs...)

# Plot B with contours
hm_B = heatmap!(ax_B, xC[:], zC[1:D], Bₙ,
    colorrange = (0.001,0.0012),
    colormap = reverse(cgrad(:Spectral)),
    nan_color = :gray)
ct_B = contour!(ax_B, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_B; label = "B [m/s²]")

# Plot w with B contours
U₀ = 0.01
hm_w = heatmap!(ax_w, xC[:], zC[1:D], wₙ,
    colorrange = (-3U₀, 3U₀), 
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_w = contour!(ax_w, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_w; label = "w [m/s]")

# Plot ε with B contours
hm_ε = heatmap!(ax_ε, xC[:], zC[1:D], εₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], 
    highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
ct_ε = contour!(ax_ε, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[4,2], hm_ε; label = "log₁₀(ε) [m²/s³]")

frames = 1:length(t_combined)

output_filename = string("output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=452-457_B_w_eps_zoomin.mp4")

record(fig, output_filename, frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_mask)



## zoomin B and Bz 
using Printf
using Oceananigans
using Oceananigans.Units
using CairoMakie
using NCDatasets
using Interpolations

function deriv(z,y)
    dydz = diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
end
 
simname = "tilt"
time_points = [452.0, 452.5, 453.0, 453.5, 454.0, 454.5, 455.0, 455.5, 456.0, 456.5, 457.0]

# Initialize arrays to hold combined data
B_all = []
t_all = []

# Load reference file for mask
filename_mask = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_mask = Dataset(filename_mask,"r")
b_mask = ds_mask["b"][:,:,:,:]; # for mask

# Loop through each time point and load data
for tᶠ in time_points
    @info "Loading data for tᶠ = $tᶠ"
    slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_analysis_round=all_slices_xz.nc")
    ds = Dataset(slice,"r")
    
    # Load data for this time point
    B = ds["B"][:,:,:,:];
    
    # Get time values for this file
    t = ds["time"];
    
    # Apply mask based on the reference file
    mask_size = min(size(b_mask, 4), size(B, 4))
    B[b_mask[:,:,:,1:mask_size].==0] .= NaN
    
    # Store data
    push!(B_all, B)
    push!(t_all, t[:])
    
    close(ds)
end

# Concatenate all data arrays along the time dimension
B_combined = cat(B_all..., dims=4)
t_combined = vcat(t_all...)

# Get grid information
zC = ds_mask["zC"][:]; Nz=length(zC)
zF = ds_mask["zF"][:]; 
xC = ds_mask["xC"][:]; Nx=length(xC)
xF = ds_mask["xF"][:];

# Calculate Bz (vertical derivative of B)
Bz = deriv(zC, B_combined);
Bz[b_mask[:,:,1:end-1,1].*ones(1,1,1,size(Bz,4)) .== 0] .= 0  # Apply mask to faces

Bz_center = (cat(zeros(Nx,1,1,length(t_combined)), Bz, dims=3) .+ cat(Bz, zeros(Nx,1,1,length(t_combined)), dims=3))/2 # average Bz at cell centers
Bz_center[b_mask[:,:,:,1:132] .== 0] .= NaN # set zero values to NaN
# Set up visualization
D = 130 # Depth limit for zoomed view
n = Observable(1)
Bₙ = @lift(B_combined[:,1,1:D,$n])
Bzₙ = @lift(Bz_center[:,1,1:D,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = CairoMakie.Figure(resolution = (1000, 600), figure_padding=(10, 40, 10, 10), size=(750,500))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, xF[end]), (0, zF[D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t_combined[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_B = Axis(fig[2, 1]; title = "Total buoyancy (B)", axis_kwargs...)
ax_Bz = Axis(fig[3, 1]; title = "Stratification (dB/dz)", axis_kwargs...)

# Plot B with contours
hm_B = heatmap!(ax_B, xC[:], zC[1:D], Bₙ,
    colorrange = (0.001,0.0012),
    colormap = reverse(cgrad(:Spectral)),
    nan_color = :gray)
ct_B = contour!(ax_B, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_B; label = "B [m/s²]")

# Custom colormap for Bz that centers at zero
function custom_colormap(min_val=-5e-7, max_val=2e-6)
    # Calculate the zero point position in normalized space [0,1]
    zero_point = -min_val / (max_val - min_val)
    
    # Create arrays of colors for smooth transitions
    neg_colors = range(colorant"darkblue", colorant"lightblue", length=50)
    neg_colors = vcat(neg_colors, range(colorant"lightblue", colorant"white", length=50))
    pos_colors = range(colorant"white", colorant"yellow", length=50)
    pos_colors = vcat(pos_colors, range(colorant"yellow", colorant"darkred", length=50))
    
    # Combine the colors into a single array
    all_colors = vcat(neg_colors[1:end-1], pos_colors)
    
    # Create a colormap with proper scaling
    return cgrad(all_colors, zero_point)
end

# Plot Bz with B contours
hm_Bz = heatmap!(ax_Bz, xC[:], zC[1:D], Bzₙ,
    colorrange = (-5e-7, 1e-6),
    colormap = custom_colormap(-5e-7, 1e-6),
    nan_color = :gray)
ct_Bz = contour!(ax_Bz, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_Bz; label = "dB/dz [1/s²]")

frames = 1:length(t_combined)

output_filename = string("output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=452-457_B_Bz_zoomin.mp4")

record(fig, output_filename, frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_mask)



##

## zoomin ν and Bz 
using Printf
using Oceananigans
using Oceananigans.Units
using CairoMakie
using NCDatasets
using Interpolations

function deriv(z,y)
    dydz = diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
end
 
simname = "tilt"
time_points = [452.0, 452.5, 453.0, 453.5, 454.0, 454.5, 455.0, 455.5, 456.0, 456.5, 457.0]

# Initialize arrays to hold combined data
B_all = []
nu_all = []
t_all = []

# Load reference file for mask
filename_mask = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_mask = Dataset(filename_mask,"r")
b_mask = ds_mask["b"][:,:,:,:]; # for mask

# Loop through each time point and load data
for tᶠ in time_points
    @info "Loading data for tᶠ = $tᶠ"
    slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_analysis_round=all_slices_xz.nc")
    ds = Dataset(slice,"r")
    
    # Load data for this time point
    B = ds["B"][:,:,:,:];
    nu = ds["νₑ"][:,:,:,:];
    
    # Get time values for this file
    t = ds["time"];
    
    # Apply mask based on the reference file
    mask_size = min(size(b_mask, 4), size(B, 4))
    B[b_mask[:,:,:,1:mask_size].==0] .= NaN
    nu[b_mask[:,:,:,1:mask_size].==0] .= NaN
    
    # Store data
    push!(B_all, B)
    push!(nu_all, nu)
    push!(t_all, t[:])
    
    close(ds)
end

# Concatenate all data arrays along the time dimension
B_combined = cat(B_all..., dims=4)
nu_combined = cat(nu_all..., dims=4)
t_combined = vcat(t_all...)

# Get grid information
zC = ds_mask["zC"][:]; Nz=length(zC)
zF = ds_mask["zF"][:]; 
xC = ds_mask["xC"][:]; Nx=length(xC)
xF = ds_mask["xF"][:];
# Calculate Bz (vertical derivative of B)
Bz = deriv(zC, B_combined);
Bz[b_mask[:,:,1:end-1,1].*ones(1,1,1,size(Bz,4)) .== 0] .= 0  # Apply mask to faces

Bz_center = (cat(zeros(Nx,1,1,length(t_combined)), Bz, dims=3) .+ cat(Bz, zeros(Nx,1,1,length(t_combined)), dims=3))/2 # average Bz at cell centers
Bz_center[b_mask[:,:,:,1:132] .== 0] .= NaN # set zero values to NaN

# Add constant viscosity to eddy viscosity
nu_const = 1.05e-6
nu_combined = nu_combined .+ nu_const

# Set up visualization
D = 130 # Depth limit for zoomed view
n = Observable(1)
Bzₙ = @lift(Bz_center[:,1,1:D,$n])
nuₙ = @lift(log10.(nu_combined[:,1,1:D,$n]))
Bₙ = @lift(B_combined[:,1,1:D,$n]) # Keep B for contour lines

ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = CairoMakie.Figure(resolution = (1000, 600), figure_padding=(10, 40, 10, 10), size=(750,500))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, xF[end]), (0, zF[D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t_combined[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_nu = Axis(fig[2, 1]; title = "Eddy viscosity (log₁₀(νₑ))", axis_kwargs...)
ax_Bz = Axis(fig[3, 1]; title = "Stratification (dB/dz)", axis_kwargs...)

# Plot nu with B contours
hm_nu = heatmap!(ax_nu, xC[:], zC[1:D], nuₙ,
    colorrange = (-4, -2),
    colormap = :viridis,
    nan_color = :gray)
ct_nu = contour!(ax_nu, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_nu; label = "log₁₀(νₑ) [m²/s]")

# Custom colormap for Bz that centers at zero
function custom_colormap(min_val=-5e-7, max_val=2e-6)
    # Calculate the zero point position in normalized space [0,1]
    zero_point = -min_val / (max_val - min_val)
    
    # Create arrays of colors for smooth transitions
    neg_colors = range(colorant"darkblue", colorant"lightblue", length=50)
    neg_colors = vcat(neg_colors, range(colorant"lightblue", colorant"white", length=50))
    pos_colors = range(colorant"white", colorant"yellow", length=50)
    pos_colors = vcat(pos_colors, range(colorant"yellow", colorant"darkred", length=50))
    
    # Combine the colors into a single array
    all_colors = vcat(neg_colors[1:end-1], pos_colors)
    
    # Create a colormap with proper scaling
    return cgrad(all_colors, zero_point)
end

# Plot Bz with B contours
hm_Bz = heatmap!(ax_Bz, xC[:], zC[1:D], Bzₙ,
    colorrange = (-5e-7, 1e-6),
    colormap = custom_colormap(-5e-7, 1e-6),
    nan_color = :gray)
ct_Bz = contour!(ax_Bz, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_Bz; label = "dB/dz [1/s²]")

frames = 1:length(t_combined)

output_filename = string("output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=452-457_nu_Bz_zoomin.mp4")

record(fig, output_filename, frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_mask)



## plot analysis periods
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
using BSplineKit
using LinearAlgebra
using Interpolations

function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
end
 
# include("functions/mmderiv.jl")
simname = "tilt"
# time_points = [452.0, 452.5, 453.0, 453.5, 454.0, 454.5, 455.0, 455.5, 456.0, 456.5, 457.0, 458.0, 458.5, 459.0, 459.5, 460.0, 460.5, 461.0, 461.5]
# time_points = [462.0]
time_points = [455.0, 455.5, 456.0, 456.5, 457.0, 457.5, 458.0, 458.5, 459.0, 459.5, 460.0, 460.5, 461.0, 461.5, 462.0]

# Initialize arrays to hold combined data
B_all = []
uhat_all = []
what_all = []
eps_all = []
t_all = []

# Load reference file for mask
filename_mask = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_mask = Dataset(filename_mask,"r")
b_mask = ds_mask["b"][:,:,:,:]; # for mask

# Loop through each time point and load data
for tᶠ in time_points
    @info "Loading data for tᶠ = $tᶠ"
    slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_analysis_round=all_slices_xz.nc")
    ds = Dataset(slice,"r")
    
    # Load data for this time point
    B = ds["B"][:,:,:,:];
    uhat = ds["uhat"][:,:,:,:];
    what = ds["what"][:,:,:,:];
    eps = ds["ε"][:,:,:,:];
    
    # Get time values for this file
    t = ds["time"];
    
    # Set topography to NaN
    uhat[uhat.==0] .= NaN
    what[what.==0] .= NaN
    eps[eps.==0] .= NaN
    
    # Apply mask based on the reference file
    mask_size = min(size(b_mask, 4), size(B, 4))
    B[b_mask[:,:,:,1:mask_size].==0] .= NaN
    
    # Store data
    push!(B_all, B)
    push!(uhat_all, uhat)
    push!(what_all, what)
    push!(eps_all, eps)
    push!(t_all, t[:])
    
    close(ds)
end

# Concatenate all data arrays along the time dimension
B_combined = cat(B_all..., dims=4)
uhat_combined = cat(uhat_all..., dims=4)
what_combined = cat(what_all..., dims=4)
eps_combined = cat(eps_all..., dims=4)
t_combined = vcat(t_all...)

# Get grid information
zC = ds_mask["zC"][:]; Nz=length(zC)
zF = ds_mask["zF"][:]; 
xC = ds_mask["xC"][:]; Nx=length(xC)
xF = ds_mask["xF"][:];
yC = ds_mask["yC"][:]; Ny=length(yC)

# Set up visualization
D = 250 # Depth limit for zoomed view
n = Observable(1)
uhatₙ = @lift(uhat_combined[:,1,1:D,$n])
εₙ = @lift(log10.(eps_combined[:,1,1:D,$n]))
Bₙ = @lift(B_combined[:,1,1:D,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = CairoMakie.Figure(resolution = (1000, 700), figure_padding=(10, 40, 10, 10), size=(750,600))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, xF[end]), (0, zF[D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t_combined[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "u with buoyancy contours", axis_kwargs...)
ax_ε = Axis(fig[3, 1]; title = "TKE dissipation rate (log ε) with buoyancy contours", axis_kwargs...)

U₀ = 0.025
hm_u = heatmap!(ax_u, xC[:], zC[1:D], uhatₙ,
    colorrange = (-0.09, 0.09), 
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_u = contour!(ax_u, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u; label = "u [m/s]")

hm_ε = heatmap!(ax_ε, xC[:], zC[1:D], εₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], 
    highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
ct_ε = contour!(ax_ε, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε; label = "log₁₀(ε) [m²/s³]")

frames = 1:length(t_combined)

output_filename = string("output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=455-462.0_uhat_eps.mp4")
println("Saving to $output_filename")
record(fig, output_filename, frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_mask)






## now that I found a strange plume near the bottom, I want to test this behavior in the beginning of the simulation, see where it arises.

using NCDatasets
using Printf
using CairoMakie
fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=10_slices_xz.nc"
ds = Dataset(fname,"r")
B = ds["B"][:,:,:,:];
b = ds["b"][:,:,:,:];
B[b.==0] .= NaN
# Load the data
zC = ds["zC"][:]; Nz=length(zC)
xC = ds["xC"][:]; Nx=length(xC)
t = ds["time"][:];

# Set up the plot
D = 130  # Depth limit for zoomed view
n = Observable(1)
Bₙ = @lift(B[:,1,1:D,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀
fig = CairoMakie.Figure(resolution = (1000, 600), size=(750,250))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, ds["xF"][end]), (0, ds["zF"][D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_b = Axis(fig[2, 1]; title = "Total buoyancy B with contours", axis_kwargs...)

hm_b = heatmap!(ax_b, xC[:], zC[1:D], Bₙ,
    colorrange = (0.0003, 0.0012),
    colormap = reverse(cgrad(:Spectral)),
    nan_color = :gray)
ct_b = contour!(ax_b, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_b; label = "B [m/s²]")

frames = 1:length(t)
filename = join(split(fname, ".")[1:end-1], ".")

record(fig, string(filename, "_verification_B_zoomin.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds)



## use the result from the CG simulation
using Oceananigans
using Oceananigans.Units
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

    return interior(compute!(Field(model.background_fields.tracers.b)))[:, 1:1, :]
end

using NCDatasets
using Printf
using CairoMakie
simname = "tilt"
solver = "FFT"
if solver=="CG"
    fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=452_slices_xz.nc"
elseif solver=="FFT"
    fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=460_slices_xz-u-v-w-B.nc"
end
ds_slice = Dataset(fname,"r")
filename_mask = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_mask = Dataset(filename_mask,"r")
# grids
zC = ds_mask["zC"]; Nz=length(zC)
zF = ds_mask["zF"]; 
xC = ds_mask["xC"]; Nx=length(xC); xF = ds_mask["xF"];
yC = ds_mask["yC"]; Ny=length(yC)
t = ds_slice["time"];

# load all data
B = ds_slice["B"][:,:,:,:];
uhat = ds_slice["uhat"][:,:,:,:]; 
b = ds_mask["b"][:,:,:,1:size(uhat,4)]; # b is the immersed boundary mask, 1 for the immersed boundary, 0 for the fluid


B[b.==0] .= NaN
# Set up the plot
D = 130  # Depth limit for zoomed view
n = Observable(1)
Bₙ = @lift(B[:,1,1:D,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀
fig = CairoMakie.Figure(resolution = (1000, 600), size=(750,250))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, ds_mask["xF"][end]), (0, ds_mask["zF"][D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_b = Axis(fig[2, 1]; title = "Total buoyancy B with contours", axis_kwargs...)

hm_b = heatmap!(ax_b, xC[:], zC[1:D], Bₙ,
    colorrange = (0.001,0.0012),
    colormap = reverse(cgrad(:Spectral)),
    nan_color = :gray)
ct_b = contour!(ax_b, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_b; label = "B [m/s²]")

frames = 25:48

filename = join(split(fname, ".")[1:end-1], ".")

record(fig, string(filename, "_B_zoomin_FFT.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_slice)




## plot B and Bz and w
## use the result from the CG simulation
using Oceananigans
using Oceananigans.Units
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

    return interior(compute!(Field(model.background_fields.tracers.b)))[:, 1:1, :]
end

using NCDatasets
using Printf
using CairoMakie
simname = "tilt"
solver = "FFT"
if solver=="CG"
    fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=452_slices_xz.nc"
elseif solver=="FFT"
    fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=460_slices_xz-u-v-w-B.nc"
end
ds_slice = Dataset(fname,"r")
filename_mask = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_mask = Dataset(filename_mask,"r")
# grids
zC = ds_mask["zC"]; Nz=length(zC)
zF = ds_mask["zF"]; 
xC = ds_mask["xC"]; Nx=length(xC); xF = ds_mask["xF"];
yC = ds_mask["yC"]; Ny=length(yC)
t = ds_slice["time"];

# load all data
B = ds_slice["B"][:,:,:,:];
uhat = ds_slice["uhat"][:,:,:,:]; 
b = ds_mask["b"][:,:,:,1:size(uhat,4)]; # b is the immersed boundary mask, 1 for the immersed boundary, 0 for the fluid


B[b.==0] .= NaN
# Set up the plot
D = 130  # Depth limit for zoomed view
n = Observable(1)
Bₙ = @lift(B[:,1,1:D,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀
fig = CairoMakie.Figure(resolution = (1000, 600), size=(750,250))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, ds_mask["xF"][end]), (0, ds_mask["zF"][D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_b = Axis(fig[2, 1]; title = "Total buoyancy B with contours", axis_kwargs...)

hm_b = heatmap!(ax_b, xC[:], zC[1:D], Bₙ,
    colorrange = (0.001,0.0012),
    colormap = reverse(cgrad(:Spectral)),
    nan_color = :gray)
ct_b = contour!(ax_b, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_b; label = "B [m/s²]")

frames = 25:48

filename = join(split(fname, ".")[1:end-1], ".")

record(fig, string(filename, "_B_zoomin_FFT.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_slice)



## Plot B, Bz and w in three panels to check everything looks right
using Printf
using Oceananigans
using Oceananigans.Units
using CairoMakie
using NCDatasets
using Interpolations
using Statistics

function deriv(z, y)
    dydz = diff(y[:,:,:,:], dims=3) ./ reshape(diff(z[:]), 1, 1, length(z)-1)
    return dydz
end

# Load data from both time periods
simname = "tilt"
tᶠ_1 = 451.5
tᶠ_2 = 452.0

# Load the first dataset
filename_1 = string("output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=", tᶠ_1, "_analysis_round=all_slices_xz.nc")
ds_1 = Dataset(filename_1, "r")

# Load the second dataset
filename_2 = string("output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=", tᶠ_2, "_analysis_round=all_slices_xz.nc")
ds_2 = Dataset(filename_2, "r")

# Load mask for immersed boundary from an earlier file
mask_file = string("output/", simname, "/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=", 10, "_slices_xz.nc")
ds_mask = Dataset(mask_file, "r")

# Get grid information (assuming identical between datasets)
zC = ds_mask["zC"][:]; Nz = length(zC)
zF = ds_mask["zF"][:]; 
xC = ds_mask["xC"][:]; Nx = length(xC)
xF = ds_mask["xF"][:];

# Load data from both files and combine
B_1 = ds_1["B"][:,:,:,:];
w_1 = ds_1["what"][:,:,:,:];
t_1 = ds_1["time"][:];

B_2 = ds_2["B"][:,:,:,:];
w_2 = ds_2["what"][:,:,:,:];
t_2 = ds_2["time"][:];

# Get mask for topography (assuming immersed boundary is where b=0)
b_mask = ds_mask["b"][:,:,:,1];

# Combine datasets
B = cat(B_1, B_2, dims=4);
w = cat(w_1, w_2, dims=4);
t = vcat(t_1, t_2);

# Apply mask - set points inside topography to NaN
B[b_mask[:,:,:,1].*ones(1,1,1,size(B,4)) .== 0] .= NaN
w[w .== 0] .= NaN
# Calculate Bz (vertical derivative of B)
Bz = deriv(zC, B);
Bz[b_mask[:,:,1:end-1,:].*ones(1,1,1,size(B,4)) .== 0] .= NaN  # Apply mask to faces

# Interpolate Bz from faces to centers for visualization
Bz_center = zeros(size(Bz,1), size(Bz,2), length(zC), size(Bz,4))
for i in 1:size(Bz,1)
    for j in 1:size(Bz,2)
        for k in 1:size(Bz,4)
            # Only interpolate if we have enough valid points
            if sum(.!isnan.(Bz[i,j,:,k])) > 2
                valid_indices = findall(.!isnan.(Bz[i,j,:,k]))
                valid_z = zF[2:end-1][valid_indices]
                valid_values = Bz[i,j,valid_indices,k]
                if length(valid_values) > 1
                    itp = linear_interpolation(valid_z, valid_values, extrapolation_bc=Line())
                    Bz_center[i,j,:,k] = itp(zC)
                end
            end
        end
    end
end
Bz_center[b_mask.*ones(1,1,1,size(B,4)) .== 0] .= NaN  # Apply mask to centers
# Set up visualization
D = 130  # Depth limit for zoomed view
n = Observable(1)

# Create the observables for the plots
Bₙ = @lift(B[:,1,1:D,$n])
Bzₙ = @lift(Bz_center[:,1,1:D,$n])
wₙ = @lift(w[:,1,1:D,$n])

# Time conversion for titles
ω₀ = 1.4e-4
M₂_period = 2π/ω₀

# Create the figure
fig =CairoMakie.Figure(resolution = (1000, 900), figure_padding=(10, 40, 10, 10), size=(750,750))
axis_kwargs = (xlabel = "Distance x (m)",
               ylabel = "Elevation z (m)",
               limits = ((0, xF[end]), (0, zF[D+1])),
              )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

# Create the three panels
ax_B = Axis(fig[2, 1]; title = "Total buoyancy (B)", axis_kwargs...)
ax_Bz = Axis(fig[3, 1]; title = "Stratification (dB/dz)", axis_kwargs...)
ax_w = Axis(fig[4, 1]; title = "Vertical velocity (w)", axis_kwargs...)

# Plot B with contours
hm_B = heatmap!(ax_B, xC[:], zC[1:D], Bₙ,
    colorrange = (0.001, 0.0012),
    colormap = reverse(cgrad(:Spectral)),
    nan_color = :gray)
ct_B = contour!(ax_B, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_B; label = "B [m/s²]")

# Plot Bz with B contours
hm_Bz = heatmap!(ax_Bz, xC[:], zC[1:D], Bzₙ,
    colorrange = (-2e-6, 2e-6),
    colormap = :balance,
    nan_color = :gray)
ct_Bz = contour!(ax_Bz, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_Bz; label = "dB/dz [1/s²]")

# Plot w with B contours
U₀ = 0.01
hm_w = heatmap!(ax_w, xC[:], zC[1:D], wₙ,
    colorrange = (-3U₀, 3U₀), 
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_w = contour!(ax_w, xC, zC[1:D], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[4,2], hm_w; label = "w [m/s]")

# Create frames for the animation
frames = 1:length(t)
output_filename = string("output/", simname, "/combined_B_Bz_w_tᶠ=", tᶠ_1, "_", tᶠ_2, "_zoomin.mp4")

# Record the animation
record(fig, output_filename, frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

# Close the datasets
close(ds_1)
close(ds_2)
close(ds_mask)
