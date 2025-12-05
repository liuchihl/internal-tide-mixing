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
 

θ=0
# include("functions/mmderiv.jl")
simname = "2D_idealized_tilt_$(θ)"
tᶠ = 200.0
filename_slice = string("output/",simname,"/internal_tide_theta=$(θ)_Nx=2000_Nz=500_tᶠ=",tᶠ,"_snapshot.nc")
ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_slice["z_aac"][:]; Nz=length(zC)
zF = ds_slice["z_aaf"][:]; 
xC = ds_slice["x_caa"][:]; Nx=length(xC)
xF = ds_slice["x_faa"][:];

# time range
t = ds_slice["time"][:];
time_range = vcat(collect(1:div(length(t), 5)), collect(3 * div(length(t), 4):length(t))[1:div(length(t), 5)])
# time_range = 4801

# Function to load a single frame and mask topography
function load_frame(ds, idx)
    B_frame = ds["B"][:,:,idx]
    Bz_frame = ds["Bz"][:,:,idx]
    b_frame = ds["b"][:,:,idx]
    uhat_frame = ds["uhat"][:,:,idx]
    ε_frame = ds["ε"][:,:,idx]
    
    # set topography to NaN
    uhat_frame[uhat_frame.==0] .= NaN
    ε_frame[ε_frame.==0] .= NaN
    B_frame[b_frame.==0] .= NaN
    Bz_frame[b_frame.==0] .= NaN
    
    return uhat_frame, ε_frame, B_frame, Bz_frame
end

# Initialize with first frame
uhat_init, ε_init, B_init, Bz_init = load_frame(ds_slice, time_range[1])

# plot
n = Observable(1)
ûₙ = Observable(uhat_init)
εₙ = Observable(log10.(ε_init))
Bₙ = Observable(B_init)
Bzₙ = Observable(Bz_init)
ω₀ = 1.4e-4
M₂_period = 2π/ω₀

# Custom colormap for Bz that centers at zero (this is for Bz plot)
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


fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs = (xlabel = "Zonal distance x (m)",
                  ylabel = "Elevation z (m)",
                  limits = ((0, xF[end]), (0, zF[end])),
                  )

title = @lift @sprintf("t=%1.2f M₂ tidal periods", t[$n]/M₂_period)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "u and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_ε = Axis(fig[3, 1]; title = "TKE dissipation rate (log ε) and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_Bz = Axis(fig[4, 1]; title = "Bz and equally-spaced buoyancy contours (B)", axis_kwargs...)

using ColorSchemes
U₀ = 0.025
hm_u = heatmap!(ax_u, xC[:], zC[:], ûₙ,
    colorrange = (-4U₀, 4U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_u = contour!(ax_u, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_u; label = "m/s")

hm_ε = heatmap!(ax_ε, xC[:], zC[:], εₙ,
    colorrange = (-10, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    nan_color = :gray)
ct_ε = contour!(ax_ε, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_ε)

hm_Bz = heatmap!(ax_Bz, xC[:], zC[:], Bzₙ,
    colorrange = (-2e-6, 2e-6),
    colormap = custom_colormap(-2e-6, 2e-6),
    # lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
ct_Bz = contour!(ax_Bz, xC, zC, Bₙ,
    levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[4,2], hm_Bz; label = "m²/s³")

frames =  time_range

filename = string("output/",simname,"/xz_animation_u_epsilon_B_Bz_theta=$(θ)")

CairoMakie.record(fig, string(filename,".mp4"), frames, framerate=13) do i
    @info "Plotting frame $i of $(frames[end])..."
    
    # Load current frame data
    uhat_frame, ε_frame, B_frame, Bz_frame = load_frame(ds_slice, i)
    
    # Update observables
    ûₙ[] = uhat_frame
    εₙ[] = log10.(ε_frame)
    Bₙ[] = B_frame
    Bzₙ[] = Bz_frame
    n[] = findfirst(==(i), time_range)
end

close(ds_slice)