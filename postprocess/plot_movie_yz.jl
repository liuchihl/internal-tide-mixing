using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics
# using BSplineKit
using LinearAlgebra
using Interpolations


function deriv(x,y)
spl = interpolate(x, y, BSplineOrder(6))
D1f = diff(spl, Derivative(1))   # change
    return D1f.(x)
end

# function nice_divergent_levels(c, clim; nlevels=20)
#     levels = range(-clim, stop=clim, length=nlevels)
#     cmax = maximum(abs, c)
#     clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
#     return (-clim, clim), levels
# end

## load data
# filename_field = "output/internal_tide_3days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_xz.nc"
# ds_field = Dataset(filename_field,"r")
# bathy_data = "output/bathymetry.nc"
# Bathy = Dataset(bathy_data,"r")
filename_slice = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_yz.nc"
ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_slice["zC"]; Nz=length(zC)
xC = ds_slice["xC"]; Nx=length(xC)
yC = ds_slice["yC"]; Ny=length(yC)
t = ds_slice["time"];

# load all data
B = ds_slice["B"].var;        B = B[:,:,:,:];
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
B[isnan.(uhat)] .= NaN
χ[isnan.(uhat)] .= NaN

# dBdz[isnan.(dBdz)] .= NaN
# dûdz[isnan.(dûdz)] .= NaN

# close(ds_field)


# derivatives of B and u. Only slices are used because it can be memory intensive
# dBdz = zeros(size(B[:,1,:,:])); dûdz = zeros(size(uhat[:,1,:,:]));

# for i in 1:length(t)
#     for j in 1:length(xC)
#       k = Int(round(length(yC)*0.5))
#         dBdz[j,:,i] = deriv(zC,B[j,k,:,i])
#         dûdz[j,:,i] = deriv(zC,uhat[j,k,:,i])
#     end
# end






# plot
n = Observable(1)
ûₙ = @lift(uhat[1,:,:,$n])
εₙ = @lift(log10.(ε[1,:,:,$n]))
Bₙ = @lift(B[1,:,:,$n])
χₙ = @lift(log10.(χ[1,:,:,$n]))
# dBdzₙ = @lift interior(dBdz[$n], :, 1, :)
# dûdzₙ = @lift interior(dûdz[$n], :, 1, :)
ω₀ = 1.4e-4
M₂_period = 2π/ω₀



fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs = (xlabel = "Cross canyon distance y (km)",
                  ylabel = "Elevation z (m)",
                  limits = ((0, ds_slice["yF"][end]), (0, ds_slice["zF"][end]-1000)),
                  )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "u and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_uz = Axis(fig[1, 2]; title = "du/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_bz = Axis(fig[2, 1]; title = "db/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_ε = Axis(fig[3, 1]; title = "TKE dissipation rate (log ε) and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_χ = Axis(fig[4, 1]; title = "Scalar variance dissipation rate (log χ) and equally-spaced buoyancy contours (B)", axis_kwargs...)


using ColorSchemes
U₀ = 0.025
hm_u = heatmap!(ax_u, yC[:], zC[:], ûₙ,
    colorrange = (-3U₀, 3U₀), colormap = :rainbow,
    lowclip=cgrad(:rainbow)[1], highclip=cgrad(:rainbow)[end],clim = (-3U₀,3U₀),
    nan_color = :gray)
ct_u = contour!(ax_u, yC, zC, Bₙ,
    levels=0.:.2e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u; label = "m/s")

hm_ε = heatmap!(ax_ε, yC[:], zC[:], εₙ,
    colorrange = (-10.5, -8.5), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    clim = (-10.5,-8.5),nan_color = :gray)
ct_ε = contour!(ax_ε, yC, zC, Bₙ,
    levels=0.:.2e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε)


hm_χ = heatmap!(ax_χ, yC[:], zC[:], χₙ,
    colorrange = (-17, -13), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    clim = (-17,-13),nan_color = :gray)
ct_χ = contour!(ax_χ, yC, zC, Bₙ,
    levels=0.:2e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[4,2], hm_χ)

frames = (1:length(t))

filename = join(split(filename_slice, ".")[1:end-1], ".")

record(fig, string(filename,".mp4"), frames, framerate=16) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_slice)



## plot analysis period

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
ds_B_Bz_eps = Dataset(slice_uvwB,"r")
slice_eps_chi = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_xz-eps-chi.nc")
ds_eps = Dataset(slice_eps_chi,"r")
filename_slice = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",10,"_slices_xz.nc")
ds_b = Dataset(filename_slice,"r")

B = ds_B_Bz_eps["B"][:,:,:,:];
uhat = ds_B_Bz_eps["uhat"][:,:,:,:];
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

close(ds_B_Bz_eps)
close(ds_eps)
close(ds_b)



## plot just B, Bz, and ε

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
slice_B_Bz_eps = string("output/",simname,"/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=",tᶠ,"_slices_yz-b-B-eps-chi-nu.nc")
ds_B_Bz_eps = Dataset(slice_B_Bz_eps,"r")

zC = ds_B_Bz_eps["zC"][:]; Nz=length(zC)
zF = ds_B_Bz_eps["zF"][:]; 
xC = ds_B_Bz_eps["xC"][:]; Nx=length(xC)
xF = ds_B_Bz_eps["xF"][:];
yC = ds_B_Bz_eps["yC"][:]; Ny=length(yC)
t = ds_B_Bz_eps["time"][:];

B = ds_B_Bz_eps["B"][:,:,:,:];
b = ds_B_Bz_eps["b"][:,:,:,:];
eps = ds_B_Bz_eps["ε"][:,:,:,:];
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


# set topography to NaN
eps[eps.==0] .= NaN
B[b[:,:,:,:].==0] .= NaN
Bz_center[b.==0] .= NaN
b[b.==0] .= NaN

# plot
n = Observable(1)
εₙ = @lift(log10.(eps[1,:,:,$n]))
bₙ = @lift(b[1,:,:,$n])
Bₙ = @lift(B[1,:,:,$n])
Bzₙ = @lift(Bz_center[1,:,:,$n])

ω₀ = 1.4e-4
M₂_period = 2π/ω₀

fig = CairoMakie.Figure(resolution = (1000, 900), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs = (xlabel = "y (m)",
              ylabel = "Elevation z (m)",
              limits = ((0, ds_B_Bz_eps["yF"][end]), (0, ds_B_Bz_eps["zF"][end])),
             )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_B = Axis(fig[2, 1]; title = "Total buoyancy", axis_kwargs...)
ax_Bz = Axis(fig[3, 1]; title = "buoyancy gradient dB/dz with total buoyancy contours", axis_kwargs...)
ax_eps = Axis(fig[4, 1]; title = "TKE dissipation rate ε with total buoyancy contours", axis_kwargs...)


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

hm_B = heatmap!(ax_B, yC[:], zC[:], Bₙ,
    colorrange = (0.001,0.0024),
    colormap = reverse(cgrad(:Spectral)),
    nan_color = :gray)
ct_B = contour!(ax_B, yC, zC[:], Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_B; label = "B [m/s²]")

hm_Bz = heatmap!(ax_Bz, yC[:], zC[:], Bzₙ,
    colorrange = (-5e-7, 2e-6),
    colormap = custom_colormap(-5e-7, 2e-6),
    nan_color = :gray)
ct_Bz = contour!(ax_Bz, yC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_Bz; label = "dB/dz [1/s²]")

hm_eps = heatmap!(ax_eps, yC[:], zC[:], εₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
ct_eps = contour!(ax_eps, yC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[4,2], hm_eps)

frames = 1:length(t)

filename = join(split(slice_B_Bz_eps, ".")[1:end-1], ".")

record(fig, string(filename, "-B-Bz-eps.mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end

close(ds_B_Bz_eps)