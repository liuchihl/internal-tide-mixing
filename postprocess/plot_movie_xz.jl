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
tᶠ = 460

## load data
slice_uvwB = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_xz-u-v-w-B.nc")
ds_uvwB = Dataset(slice_uvwB,"r")
slice_eps_chi = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_xz-eps-chi.nc")
ds_eps = Dataset(slice_eps_chi,"r")
filename_slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_b = Dataset(filename_slice,"r")

B = ds_uvwB["B"][:,:,:,:];
uhat = ds_uvwB["uhat"][:,:,:,:];
eps = ds_eps["ε"][:,:,:,:];
b = ds_b["b"][:,:,:,:]; # for mask

# grids
zC = ds_eps["zC"][:]; Nz=length(zC)
zF = ds_eps["zF"][:]; 
xC = ds_eps["xC"][:]; Nx=length(xC)
xF = ds_eps["xF"][:];
yC = ds_eps["yC"][:]; Ny=length(yC)
t = ds_eps["time"];

# set topography to NaN
uhat[uhat.==0] .= NaN
eps[eps.==0] .= NaN

# derivatives of B and u. Only slices are used because it can be memory intensive
N = 1.e-3
Bz = deriv(zC,B);
Bz[b[:,:,1:end-1,1:240].==0] .= 0  # the grids are staggered, but this will effectively set the points inside and right above the immersed boundary to 0

# interpolate Bz from faces to center cell
Bz_center = zeros(size(Bz,1), size(Bz,2), length(zC), size(Bz,4))
for i in 1:size(Bz,1)
    for j in 1:size(Bz,2)
        for k in 1:size(Bz,4)
            itp = linear_interpolation(zF[2:end-1], Bz[i,j,:,k], extrapolation_bc=Line())
            Bz_center[i,j,:,k] = itp(zC)
        end
    end
end
Bz_center[b[:,:,:,1:240].==0] .= NaN
B[b[:,:,:,1:240].==0] .= NaN
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
              limits = ((0, ds_eps["xF"][end]), (0, ds_eps["zF"][end])),
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

filename = join(split(slice_uvwB, ".")[1:end-1], ".")

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
tᶠ = 460

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
tᶠ = 460

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
D = 130
n = Observable(1)
# bₙ = @lift(log10.(b[:,1,1:D,$n]))
Bₙ = @lift(B[:,1,1:D,$n])
nuₙ = @lift(nu[:,1,1:D,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀
fig = CairoMakie.Figure(resolution = (1000, 500), size=(750,500))
axis_kwargs = (xlabel = "Zonal distance x (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, ds_uvwB["xF"][end]), (0, ds_uvwB["zF"][D+1])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_b = Axis(fig[2, 1]; title = "Total buoyancy B with total buoyancy contours", axis_kwargs...)
ax_nu = Axis(fig[3, 1]; title = "eddy viscosity (log νₑ) with total buoyancy contours", axis_kwargs...)

hm_b = heatmap!(ax_b, xC[:], zC[1:D], Bₙ,
    colorrange = (0.001,0.0012),
    # colorrange = (0.0004,0.0008),
    # colorrange = (-4,-3.2),
    colormap = reverse(cgrad(:Spectral)),
    # colormap = custom_colormap(0.0003,0.0008),
    nan_color = :gray)
ct_b = contour!(ax_b, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_b; label = "log₁₀ B [m/s²]")

# Add log-scale plot of eddy viscosity
log_nuₙ = @lift(log10.($nuₙ))
hm_nu = heatmap!(ax_nu, xC[:], zC[1:D], log_nuₙ,
    colorrange = (-3.5, -2), colormap = :viridis,
    nan_color = :gray)
ct_nu = contour!(ax_nu, xC, zC[1:D], Bₙ,
    levels=0.:.05e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_nu; label = "log₁₀(νₑ) [m²/s]")

frames = 1:length(t)

filename = join(split(slice_uvwB, ".")[1:end-1], ".")

record(fig, string(filename, "_b_log-nu-zoomin.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end





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
fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=452_slices_xz.nc"
ds_slice = Dataset(fname,"r")
B = ds_slice["B"][:,:,:,:];

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

frames = 1:length(t)
filename = join(split(fname, ".")[1:end-1], ".")

record(fig, string(filename, "_B_zoomin_CG.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_slice)