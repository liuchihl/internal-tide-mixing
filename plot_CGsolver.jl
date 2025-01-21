
## plotting animation to see where blows up

using CairoMakie
using NCDatasets
using Printf
fname = "output/tilt_conjugate_gradient_poisson_solver/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_0-1_slices_xz.nc"

ds = Dataset(fname,"r")

# grids
zC = ds["zC"]; Nz=length(zC)
zF = ds["zF"]; #Nz=length(zF)
xC = ds["xC"]; Nx=length(xC)
xF = ds["xF"];

yC = ds["yC"]; Ny=length(yC)
t = ds["time"];

u = ds["u"][:,:,:,:];
# w = ds["w"][:,:,:,:];
udiv = ds["udiv"][:,:,:,:];

u_center = (u[:,:,:,:].+vcat(u[2:end,:,:,:], u[1:1,:,:,:]))./2
# w_center = (w[:,:,1:end-1,:].+w[:,:,2:end,:])./2
u_center[u_center.==0].=NaN
# w_center[w_center.==0].=NaN
# w[w.==0].=NaN
u[u.==0].=NaN


# plot
n = Observable(1)
uₙ = @lift(u_center[:,1,:,$n])
# wₙ = @lift(w_center[:,1,:,$n])
udivₙ = @lift(udiv[:,1,:,$n])

fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800),fontsize=20)
axis_kwargs = (xlabel = "x (m)",
                  ylabel = "z (m)",
                  limits = ((0, ds["xF"][end]), (0, ds["zF"][end])),
                  )
title = @lift @sprintf("t=%1.2f hrs", t[$n]/3600)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
                  
                  
ax_u = Axis(fig[2, 1]; title = "u", axis_kwargs...)
# ax_w = Axis(fig[3, 1]; title = "w", axis_kwargs...)
ax_udiv = Axis(fig[3, 1]; title = L"∇⋅\vec{u}", axis_kwargs...)



using ColorSchemes
U₀ = 0.01
hm_u = heatmap!(ax_u, xC[:], zC[:], uₙ,
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
# hm_w = heatmap!(ax_w, xC[:], zC[:], wₙ,
#     colorrange = (-U₀, U₀), colormap = :diverging_bwr_20_95_c54_n256,
#     lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
#     nan_color = :gray)
# Colorbar(fig[3,2], hm_w; label = "m/s")

hm_udiv = heatmap!(ax_udiv, xC[:], zC[:], udivₙ,
    colorrange = (-1e-8,1e-8), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
Colorbar(fig[3,2], hm_udiv; label = "1/s")


frames =  (1:length(t))

filename = join(split(fname, ".")[1:end-1], ".")

record(fig, string(filename,".mp4"), frames, framerate=23) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end
