using CairoMakie
using NCDatasets
using MAT
using Statistics
using NaNStatistics
using Interpolations

slope = "tilt"
timerange = "0-20"
θ=3.6e-3

filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
t = ds_field["time"][:];

n=20
b = ds_field["b"][:,:,:,n:n];
Bz = ds_field["Bz"][:,:,:,n:n];
epsilon = ds_field["ε"][:,:,:,n:n];
uhat = ds_field["uhat"][:,:,:,n:n];    # true u
what = ds_field["what"][:,:,:,n:n];    # true w
chi = ds_field["χ"][:,:,:,n:n];        # -κ|∇b|²
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
u = uhat[:,:,:,1:1]*cos(θ) .+ wtemp*sin(θ)
u[uhat.==0].=NaN
what[what.==0].=NaN


# subscript indicates average in that dimension
u_z = dropdims(nanmean(u,dims=3),dims=(3,4))
u_y = dropdims(nanmean(u,dims=2),dims=(2,4))
u_x = dropdims(nanmean(u,dims=1),dims=(1,4))

what_z = dropdims(nanmean(what,dims=3),dims=(3,4))
what_y = dropdims(nanmean(what,dims=2),dims=(2,4))
what_x = dropdims(nanmean(what,dims=1),dims=(1,4))



# load topography 
topo_file = "topo.mat"
file = matopen(topo_file)
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
y_interp = range(y_topo[1],y_topo[end], length=Ny)


# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp.-minimum(z_interp)




fig = Figure(size=(200,1000))
axis_kwargs_hm_xy = (xlabel = "x (km)",
                  ylabel = "y (km)"
                  )
axis_kwargs_hm_xz = (xlabel = "x (km)",
                  ylabel = "z (m)"
                  )
axis_kwargs_hm_yz = (xlabel = "y (km)",
                  ylabel = "z (m)"
                  )
ax_xy = Axis(fig[1:2, 1]; title = "Vertically-integrated cross-slope velocity u", axis_kwargs_hm_xy...)
ax_xz = Axis(fig[1, 3]; title = "Cross-slope velocity ūʸ", axis_kwargs_hm_xz...)
ax_yz = Axis(fig[2, 3]; title = "Cross-slope velocity ūˣ", axis_kwargs_hm_yz...)

using ColorSchemes

hm_xy = heatmap!(ax_xy, xC[:]/1e3, yC[:]/1e3, u_z.*(zF[end].-z_interp),
    colorrange = (-15,15), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
Colorbar(fig[1:2,2], hm_xy)

contour!(ax_xy, xC[:]/1e3, yC[:]/1e3, z_interp, linewidth = 1.5, color =:black)

hm_xz = heatmap!(ax_xz, xC[:]/1e3, zC[:], u_y,
    colorrange = (-0.03,0.03), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
lines!(ax_xz, xC[:]/1e3, dropdims(mean(z_interp,dims=2),dims=2),linewidth=2.5,color=:black)
    
hm_yz = heatmap!(ax_yz, yC[:]/1e3, zC[:], u_x,
    colorrange = (-0.03,0.03), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
lines!(ax_yz, yC[:]/1e3, dropdims(mean(z_interp,dims=1),dims=1),linewidth=2.5,color=:black)
    Colorbar(fig[2,4], hm_yz)

save(string("output/",slope,"/horiztonal_avg_u_",timerange,".png"),fig)





fig = Figure(size=(200,1000))
axis_kwargs_hm_xy = (xlabel = "x (km)",
                  ylabel = "y (km)"
                  )
axis_kwargs_hm_xz = (xlabel = "x (km)",
                  ylabel = "z (m)"
                  )
axis_kwargs_hm_yz = (xlabel = "y (km)",
                  ylabel = "z (m)"
                  )
ax_xy = Axis(fig[1:2, 1]; title = "Vertically-integrated true vertical velocity", axis_kwargs_hm_xy...)
ax_xz = Axis(fig[1, 3]; title = "True vertical velocity", axis_kwargs_hm_xz...)
ax_yz = Axis(fig[2, 3]; title = "True vertical velocity", axis_kwargs_hm_yz...)

using ColorSchemes
U0 = 0.003
hm_xy = heatmap!(ax_xy, xC[:]/1e3, yC[:]/1e3, what_z.*(zF[end].-z_interp),
    colorrange = (-5,5), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
contour!(ax_xy, xC[:]/1e3, yC[:]/1e3, z_interp, linewidth = 1.5, color =:black)
Colorbar(fig[1:2,2], hm_xy)

hm_xz = heatmap!(ax_xz, xC[:]/1e3, zF[:], what_y,
    colorrange = (-U0,U0), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
lines!(ax_xz, xC[:]/1e3, dropdims(mean(z_interp,dims=2),dims=2),linewidth=2.5,color=:black)

hm_yz = heatmap!(ax_yz, yC[:]/1e3, zF[:], what_x,
    colorrange = (-U0,U0), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
lines!(ax_yz, yC[:]/1e3, dropdims(mean(z_interp,dims=1),dims=1),linewidth=2.5,color=:black)
    Colorbar(fig[2,4], hm_yz)

save(string("output/",slope,"/vertical_avg_w_",timerange,".png"),fig)
