# this script plots heatmaps of time averaged values
using CairoMakie
using NCDatasets
using MAT
using Statistics
using NaNStatistics
using Interpolations

θ = 0.0036
simname = "test_wta_4tidalperiod"
timerange = "0-2"
filename_field0_2 = string("output/", simname, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field0_2 = Dataset(filename_field0_2,"r")
t0_2 = ds_field0_2["time"][:];

timerange = "2-4"
filename_field2_4 = string("output/", simname, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field2_4 = Dataset(filename_field2_4,"r")
t2_4 = ds_field2_4["time"][:];

# grids
zC = ds_field0_2["zC"]; Nz=length(zC[:])
zF = ds_field0_2["zF"]; zF = zF[:];
xC = ds_field0_2["xC"]; Nx=length(xC[:])
xF = ds_field0_2["xF"]; 
yC = ds_field0_2["yC"]; Ny=length(yC[:])

# concatenate two simulations 0-2 and 2-4 tidal periods
b = cat(ds_field0_2["b"][:,Ny÷2,:,:],ds_field2_4["b"][:,Ny÷2,:,:],dims=3);
B = cat(ds_field0_2["B"][:,Ny÷2,:,:],ds_field2_4["B"][:,Ny÷2,:,:],dims=3);
uhat = cat(ds_field0_2["uhat"][:,Ny÷2,:,:],ds_field2_4["uhat"][:,Ny÷2,:,:],dims=3);
t = cat(t0_2,t2_4,dims=1)

uhat[uhat.==0].=NaN
B[b.==0].=NaN

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
T = 2π/1.4e-4  # tidal period



fig = Figure(size=(600,800))
axis_kwargs_hm_xz = (ylabel = "z (m)",)
ax1 = Axis(fig[1, 1]; title = string("t = ",Int(t[1]/T),"-",Int(t[2]/T)," tidal period"), axis_kwargs_hm_xz...)
ax2 = Axis(fig[2, 1]; title = string("t = ",Int(t[2]/T),"-",Int(t[3]/T)," tidal period"), axis_kwargs_hm_xz...)
ax3 = Axis(fig[3, 1]; title = string("t = ",Int(t[3]/T),"-",Int(t[4]/T)," tidal period"), axis_kwargs_hm_xz...)
ax4 = Axis(fig[4, 1]; title = string("t = ",Int(t[4]/T),"-",Int(t[5]/T)," tidal period"), xlabel = "x (km)",axis_kwargs_hm_xz...)

using ColorSchemes
U₀ = 0.025
heatmap!(ax1, xC[:]/1e3, zC[:], uhat[:,:,2],
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
contour!(ax1, xC[:]/1e3, zC[:], B[:,:,2], levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
heatmap!(ax2, xC[:]/1e3, zC[:], uhat[:,:,3],
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
contour!(ax2, xC[:]/1e3, zC[:], B[:,:,3], levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
heatmap!(ax3, xC[:]/1e3, zC[:], uhat[:,:,4],
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
contour!(ax3, xC[:]/1e3, zC[:], B[:,:,4], levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
hm = heatmap!(ax4, xC[:]/1e3, zC[:], uhat[:,:,4],
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
Colorbar(fig[4,2], hm, label="Tidally averaged zonal velocity")
ct_u = contour!(ax4, xC[:]/1e3, zC[:], B[:,:,5], levels=0.:.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)

save(string("output/",simname,"/horiztonal_avg_u_","0-4",".png"),fig)



## Hovmoller diagram
uhat = cat(ds_field0_2["uhat"][:,Ny÷2,:,:],ds_field2_4["uhat"][:,Ny÷2,:,:],dims=3);
uhat[uhat.==0].=NaN

fig = Figure(size=(600,500))
axis_kwargs_hm_xz = (xlabel = "t", ylabel = "z (m)",)
ax1 = Axis(fig[1, 1]; title = "", axis_kwargs_hm_xz...)
# ax2 = Axis(fig[2, 1]; title = string("t=",Int(t[3]/T)," tidal period"), axis_kwargs_hm_xz...)
# ax3 = Axis(fig[3, 1]; title = string("t=",Int(t[4]/T)," tidal period"), axis_kwargs_hm_xz...)
# ax4 = Axis(fig[4, 1]; title = string("t=",Int(t[5]/T)," tidal period"), xlabel = "x (km)",axis_kwargs_hm_xz...)

using ColorSchemes
U₀ = 0.025
heatmap!(ax1, t/T, zC[:], dropdims(nanmean((uhat[:,:,:]),dims=1),dims=1)',
    colorrange = (-3U₀, 3U₀), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:grey )
    Colorbar(fig[1,2], hm, label="Tidally averaged zonal velocity")

    save(string("output/",simname,"/Hovmoller","0-4",".png"),fig)
