# this script plots the cross sections of terrain following heatmaps
function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(zC)-1)
    return dydz
 end

# test
using Interpolations
slope = "tilt"
timerange = "80-120"
θ=3.6e-3
# load data

filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];

filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")
zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
Nz=length(zC[:]); 

xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
t = ds_field["time"][:];

new_bin_edges = 0:5:1500
bin_centers = @. (new_bin_edges[1:end-1] + new_bin_edges[2:end]) / 2
interpolated_what = zeros(Nx,Ny,size(bin_centers,1),11)
interpolated_uhat = zeros(Nx,Ny,size(bin_centers,1),11)
interpolated_Bz = zeros(Nx,Ny,size(bin_centers,1),11)
b = ds_field["b"][:,:,:,1:1];    # true w
for n in 30:40
    # uhat = ds_field["uhat"][:,:,:,n:n];    # true u
    # what = ds_field["what"][:,:,:,n:n];    # true w
    B = ds_field["B"][:,:,:,n:n];    # true w
    
    N = 1.e-3
        Bz =  deriv(zC,B);
        Bz[b[:,:,1:end-1,:].==0] .= 0      # the grids are staggered, but this will effectively set the points inside and right above the immersed boudary to 0
    # interpolate Bz from faces to center cell
    # Interpolate each row
    Bz_center = zeros(size(Bz,1),size(Bz,2),length(zC),1)
    for i in 1:size(Bz,1)
        for j in 1:size(Bz,2)
              itp = linear_interpolation(zF[2:end-1], Bz[i,j,:,1], extrapolation_bc=Line())
            Bz_center[i,j,:,:] = itp(zC)
        end
    end
    # Interpolate each row
    # what_center = zeros(size(what,1),size(what,2),length(zC),1)
    # for i in 1:size(what,1)
    #     for j in 1:size(what,2)
    #         itp = linear_interpolation(zF[:], what[i,j,:,1], extrapolation_bc=Line())
    #         what_center[i,j,:,:] = itp(zC)
    #     end
    # end

    # @time interpolated_what[:,:,:,n-29],_ =  interpolate_z_dimension(what_center, hab, new_bin_edges)
    # @time interpolated_uhat[:,:,:,n-29],_ =  interpolate_z_dimension(uhat, hab, new_bin_edges)
    @time interpolated_Bz[:,:,:,n-29],_ =  interpolate_z_dimension(Bz_center, hab, new_bin_edges)
end
# interpolated_what_tavg = nanmean(interpolated_what,dim =(4))
# interpolated_uhat_tavg = nanmean(interpolated_uhat,dim =(4))
interpolated_Bz_tavg = nanmean(interpolated_Bz,dim =(4))
# bathymetry
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
x_interp = range(x_topo[1],x_topo[end], length=Nx)
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=Ny)

using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)


using PyPlot
close(gcf())
PyPlot.plot(nanmean(interpolated_what_tavg[:,:,:],dim=(1,2)),bin_centers)
gcf()
savefig("output/tilt/interpolated_what_ln.png")


using PyPlot

close(gcf())
fig, (ax1, ax2) = subplots(1, 2, figsize=(15, 5))

c1 = ax1.pcolor(xC[:]*1e-3, yC[:]*1e-3, nanmean(interpolated_what_tavg[:,:,1:19],dim=(3))',
                cmap="RdYlBu_r",
                vmin=-0.01, vmax=0.01)
colorbar(c1, ax=ax1, extend="both")
ax1.set_title(L"\frac{1}{95}∫^{95}_{0} \bar{w} dHAB")
h1=ax1.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
PyPlot.clabel(h1, inline=true, fontsize=8)  # Add labels to contour lines
ax1.set_xlabel("x (km)")
ax1.set_ylabel("y (km)")

c2 = ax2.pcolor(xC[:]*1e-3, yC[:]*1e-3, nanmean(interpolated_uhat_tavg[:,:,1:19],dim=(3))',
                cmap="RdYlBu_r",
                vmin=-0.04, vmax=0.04)
colorbar(c2, ax=ax2, extend="both")
ax2.set_title(L"\frac{1}{95}∫^{95}_{0} \bar{u} dHAB")
h2=ax2.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
PyPlot.clabel(h2, inline=true, fontsize=8)  # Add labels to contour lines
ax2.set_xlabel("x (km)")
ax2.set_ylabel("y (km)")

savefig("output/tilt/interpolated_u_w_heatmap.png")


# plot only Bz 
using PyPlot

close(gcf())
fig, ax = subplots(1, 1, figsize=(5, 5))
    # mpl_colors = PyPlot.pyimport("matplotlib.colors")
    # linthresh = 1e-8
    
    # norm_log = mpl_colors["SymLogNorm"](linthresh=linthresh, vmin=-1e-7, vmax=1e-5, base=10)
c = ax.pcolor(xC[:]*1e-3, yC[:]*1e-3, (nanmean(interpolated_Bz_tavg[:,:,1:19],dim=(3))'),
                cmap="RdYlBu_r",
                vmin=0, vmax=2e-6)
colorbar(c, ax=ax, extend="both")
ax.set_title(L"\bar{B}_z (0-92.5 m)")
h=ax.contour(xC[:]*1e-3, yC[:]*1e-3, z_interp', linewidths=1.5, colors="black")
PyPlot.clabel(h, inline=true, fontsize=8)  # Add labels to contour lines
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
savefig("output/tilt/interpolated_Bz_heatmap_0-92.5.png")
