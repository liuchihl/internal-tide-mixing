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

# save(string(filename,".png"),fig)

# χ_log = χ
# χ_log[χ_log.<0] .= 0
# # χ_log = sign.(χ).*log10.(abs.(χ)+1)
# hm = heatmap(xC[:], zC[:], log10.(χ_log[:,1,:,50]),colorrange = (-10,-6))

    # colorrange = (-12, -6), colormap = :rainbow_bgyrm_35_85_c69_n256,
    # lowclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[1], highclip=cgrad(:rainbow_bgyrm_35_85_c69_n256)[end],
    # clim = (-12,-6),
# save("output/supercritical_tilt/backgroundfluxdivergence_smagorinky/chi.png",hm)

# ln = lines(χ_log[250,1,:,50],zC[:])
# save("output/supercritical_tilt/backgroundfluxdivergence_smagorinky/chi_vertical.png",ln)
# saving frame as png file
# frame = length(t)
# filename = join(split(filename_slice, ".")[1:end-1], ".")
# save(string(filename,".png"),fig)

# close(ds_slice)



# calculate the time series
# filename_slice_0_10 = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_xz.nc"
# filename_slice = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_slices_10_100_xz.nc"
# ds_slice_0_10 = Dataset(filename_slice_0_10,"r")
# ds_slice_10_100 = Dataset(filename_slice,"r")

# t_0_10 = ds_slice_0_10["time"];
# ε = ds_slice_0_10["ε"].var;       ε_0_10 = ε[:,:,:,:];
# t_10_100 = ds_slice_10_100["time"];
# ε = ds_slice_10_100["ε"].var;       ε_10_100 = ε[:,:,:,:];
# # t = 

# ff = lines(t_0_10/86400,log10.(dropdims(nanmean(ε_0_10,dims=(1,2,3)),dims=(1,2,3))))
# lines!(t_10_100/86400,log10.(dropdims(nanmean(ε_10_100,dims=(1,2,3)),dims=(1,2,3))))
# save("output/supercritical_tilt/epsilon.png", ff) # specify the path and filename 
