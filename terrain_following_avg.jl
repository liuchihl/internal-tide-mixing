# This script calculates the terrain-following averaged quantities by interpolation
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using BSplineKit
using Statistics
using NaNStatistics
using LinearAlgebra
using Interpolations
using MAT

function deriv(x,y)
   dydx =  diff(y[:,:,:,:],dims=3)./reshape(diff(x[:]),1,1,length(zC)-1)
   return dydx
end


# terrain following average algorithms
@views function terrain_following_fast(hab, ϕ, new_height, Nx, Ny, Lz)
    mean_values = zeros(length(new_height), size(ϕ, 4))
    hab_interp = zeros(size(ϕ,3))
    value = zeros(Nx,Ny,length(new_height))
    for i in 1:Nx
        for j in 1:Ny
            hab_interp = hab[i, j, :]
            for tt = 1:size(ϕ,4)
            itp = Interpolations.interpolate((hab_interp,), ϕ[i, j, :, tt], Gridded(Linear()))
            itp_extrapolated = Interpolations.extrapolate(itp, Interpolations.Flat())
            h = new_height
            h = Float64.(h)
            h[h .+ z_interp[i,j] .> Lz] .= NaN
            value[i,j,:] = itp_extrapolated.(h)
            end
        end
    end
    mean_values = dropdims(nanmean(value,dims=(1,2)),dims=(1,2))
    return mean_values
end


slope = "notilt"
timerange = "0-20"
θ=0#3.6e-3

###################################
# load data
# filename_field = "output/no_tilt/internal_tide_5days-theta=0_realtopo3D_Nx500_Nz250_fields_timeavg_0_20.nc"
# ds_field_no_0_20 = Dataset(filename_field,"r")
filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
t = ds_field["time"][:];

new_height = 0:5:1500#0

# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"].var[:,:,:];
# preallocate
b_avg = zeros(length(new_height),length(t))
Bz_avg = zeros(length(new_height),length(t))
uhat_avg = zeros(length(new_height),length(t))
u_avg = zeros(length(new_height),length(t))
what_avg = zeros(length(new_height),length(t))
epsilon_avg = zeros(length(new_height),length(t))
chi_avg = zeros(length(new_height),length(t))



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




# θ = 0# 3.6e-3
for n in 1:length(t)
b = ds_field["b"][:,:,:,n:n];
Bz = ds_field["Bz"][:,:,:,n:n];
epsilon = ds_field["ε"][:,:,:,n:n];
uhat = ds_field["uhat"][:,:,:,n:n];    # true u
what = ds_field["what"][:,:,:,n:n];    # true w
chi = ds_field["χ"][:,:,:,n:n];        # -κ|∇b|²
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ)

# mask out
# Bz[b.==0] .= NaN
# epsilon[uhat.==0] .= NaN;
# chi[b.==0] .= NaN;
# uhat[uhat.==0] .= NaN;
# b[b.==0] .= NaN;
# u[u.==0] .= NaN;
# what[what.==0] .= NaN;


N = 1.e-3
# bz_no_0_20 =  deriv(zC,b_no_0_20);
# bz_no_20_30 =  deriv(zC,b_no_20_30);
# θ = 0; 
# bz_bc_0_20 = -N^2*cos(θ).*ones(Nx,Ny,1,length(t_no_0_20));
# bz_bc_20_30 = -N^2*cos(θ).*ones(Nx,Ny,1,length(t_no_20_30));

# bz_no_0_20 = cat(bz_no_0_20, bz_bc_0_20, dims=3);
# bz_no_20_30 = cat(bz_no_20_30, bz_bc_20_30, dims=3);

# bz =  deriv(zC,b);
# θ = 3.6e-3; 
# bz_bc = -N^2*cos(θ).*ones(Nx,Ny,1,1);
# bz = cat(bz, bz_bc, dims=3);


### terrain following quantities:

# call the function to get the terrain following averaged values
Lz = Int(zF[end])
@time b_avg[:,n] = terrain_following_fast(hab, b[:,:,:,1], new_height, Nx, Ny,Lz);
@time Bz_avg[:,n] = terrain_following_fast(hab, Bz[:,:,:,1], new_height, Nx, Ny ,Lz);
@time uhat_avg[:,n] = terrain_following_fast(hab, uhat[:,:,:,1], new_height, Nx, Ny, Lz);
@time what_avg[:,n] = terrain_following_fast(hab, what[:,:,1:end-1,1], new_height, Nx, Ny,Lz);
@time u_avg[:,n] = terrain_following_fast(hab, u[:,:,:,1], new_height, Nx, Ny,Lz);
@time epsilon_avg[:,n] = terrain_following_fast(hab, epsilon[:,:,:,1], new_height, Nx, Ny,Lz);
@time chi_avg[:,n] = terrain_following_fast(hab, chi[:,:,:,1], new_height, Nx, Ny,Lz);
 
end

close(ds_field)
close(ds_hab)

# This creates a new NetCDF file 
# The mode "c" stands for creating a new file

ds_create = Dataset(string("output/",slope,"/TF_avg_",timerange,".nc"),"c")
# Define the dimension
defDim(ds_create,"z_TF",length(new_height))
defDim(ds_create,"t",length(t))

# Define a global attribute
ds_create.attrib["title"] = "Terrain-following averages"
# Define the variables
v1 = defVar(ds_create,"b_avg",Float64,("z_TF","t"))
v1[:,:] = b_avg
v2 = defVar(ds_create,"Bz_avg",Float64,("z_TF","t"))
v2[:,:] = Bz_avg
v3 = defVar(ds_create,"uhat_avg",Float64,("z_TF","t"))
v3[:,:] = uhat_avg
v4 = defVar(ds_create,"u_avg",Float64,("z_TF","t"))
v4[:,:] = u_avg
v5 = defVar(ds_create,"what_avg",Float64,("z_TF","t"))
v5[:,:] = what_avg
v6 = defVar(ds_create,"chi_avg",Float64,("z_TF","t"))
v6[:,:] = chi_avg
v7 = defVar(ds_create,"epsilon_avg",Float64,("z_TF","t"))
v7[:,:] = epsilon_avg
v8 = defVar(ds_create,"new_height",Float64,("z_TF",))
v8[:,1] = new_height
v9 = defVar(ds_create,"t",Float64,("t",))
v9[:,1] = t   

# write attributes
v1.attrib["units"] = "m/s²"
v2.attrib["units"] = "1/s²"
v3.attrib["units"] = "m/s"
v4.attrib["units"] = "m/s"
v5.attrib["units"] = "m/s"
v6.attrib["units"] = ""
v7.attrib["units"] = "WKg"
v8.attrib["units"] = "m"
v9.attrib["units"] = "s"

close(ds_create)


########################## plot the hab profiles

## plot b, and Bz
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t = ds["t"][:]/86400
z = ds["new_height"][:]
Bz_avg = ds["Bz_avg"][:,:]
b_avg = ds["b_avg"][:,:]

fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs_hm = (xlabel = "time (day)",
                  ylabel = "hab (m)",
                  limits = ((t[1], t[end]), (0, z[end]))
                  )

axis_kwargs_line_b = (ylabel = "hab (m)",limits = ((0,maximum(b_avg)),(0, 500)) )   
axis_kwargs_line_Bz = (ylabel = "hab (m)",limits = (nothing,(0, 500)) ) 

ax_b = Axis(fig[1, 1]; title = "b (buoyancy perturbation)", axis_kwargs_hm...)
ax_Bz = Axis(fig[2, 1]; title = "dB/dz (Total buoyancy gradient)", axis_kwargs_hm...)

ax_b_ln = Axis(fig[1, 3]; title = "b (buoyancy perturbation)", axis_kwargs_line_b...)
ax_Bz_ln = Axis(fig[2, 3]; title = "dB/dz (Total buoyancy gradient)", axis_kwargs_line_Bz...)

using ColorSchemes
U₀ = 0.025
hm_b = heatmap!(ax_b, t[:], z[:], b_avg',
    colorrange = (minimum(filter(!isnan,b_avg)),maximum(filter(!isnan,b_avg))*0.8), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[1,2], hm_b )
hm_Bz = heatmap!(ax_Bz, t[:], z[:], Bz_avg',
    colormap = :diverging_bwr_20_95_c54_n256, colorrange=(8.e-7,10.e-7),
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
    Colorbar(fig[2,2], hm_Bz)

ind = argmin(abs.(t .- 20))   # 1300 m height above bottom
lines!(ax_b_ln, b_avg[:,ind], z[:], linewidth=3)
ln1 = lines!(ax_Bz_ln, Bz_avg[:,ind], z[:], linewidth=3)
lines!(ax_b_ln, b_avg[:,end], z[:], linewidth=3)
ln2 = lines!(ax_Bz_ln, Bz_avg[:,end], z[:], linewidth=3)
axislegend(ax_Bz_ln, [ln1,ln2],["Day 20","Day 40"], position = :lt)

save(string("output/",slope,"/hab_buoyancy_",timerange,".png"),fig)



## plot velocities (along slope u, and true vertical velocity w)
u_avg = ds["u_avg"][:,:]
uhat_avg = ds["uhat_avg"][:,:]
what_avg = ds["what_avg"][:,:]

u_avg_15_20 = dropdims(mean(u_avg[:,30:39],dims=2),dims=2)
what_avg_15_20 = dropdims(mean(what_avg[:,30:39],dims=2),dims=2)


    fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(1000,800))
    axis_kwargs_hm = (xlabel = "time (day)",
                      ylabel = "hab (m)",
                      limits = ((t[1], t[end]), (0, z[end]))
                      )
    axis_kwargs_ln = (ylabel = "hab (m)",limits = (nothing,(0, 1500)) )   
    
ax_u = Axis(fig[1, 1]; title = "Cross-slope velocity", axis_kwargs_hm...)
# ax_uhat = Axis(fig[2, 1]; title = "True zonal velocity", axis_kwargs_hm...)
ax_what = Axis(fig[2, 1]; title = "True vertical velocity", axis_kwargs_hm...)
ax_u_ln = Axis(fig[1, 3]; title = "Cross-slope velocity", axis_kwargs_ln...)
# ax_uhat_ln = Axis(fig[2, 3]; title = "True zonal velocity", axis_kwargs_ln...)
ax_what_ln = Axis(fig[2, 3]; title = "True vertical velocity", axis_kwargs_ln...)


hm_u = heatmap!(ax_u, t[:], z[:], u_avg',
    colorrange = (-0.02, 0.02), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],clim = (-3U₀,3U₀),
    nan_color = :gray)
    Colorbar(fig[1,2], hm_u)
# hm_uhat = heatmap!(ax_uhat, t[:], z[:], uhat_avg',
#     colorrange = (-0.02, 0.02), colormap = :diverging_bwr_20_95_c54_n256,
#     lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],clim = (-3U₀,3U₀),
#     nan_color = :gray)
#     Colorbar(fig[2,2], hm_uhat)
hm_what = heatmap!(ax_what, t[:], z[:], what_avg',
    colorrange = (-0.001, 0.001), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],clim = (-3U₀,3U₀),
    nan_color = :gray)
    Colorbar(fig[2,2], hm_what)

ind = argmin(abs.(t .- 20))   # 1300 m height above bottom
lines!(ax_u_ln, u_avg[:,ind], z[:], linewidth=3,color=:black)
# lines!(ax_uhat_ln, uhat_avg[:,ind], z[:], linewidth=3,color=:black)
ln1 = lines!(ax_what_ln, what_avg[:,ind], z[:], linewidth=3,color=:black)
lines!(ax_u_ln, u_avg[:,end], z[:], linewidth=3,color=:red)
lines!(ax_u_ln, u_avg_15_20[:], z[:], linewidth=3,color=:blue)

# lines!(ax_uhat_ln, uhat_avg[:,end], z[:], linewidth=3,color=:red)
ln2 = lines!(ax_what_ln, what_avg[:,end], z[:], linewidth=3,color=:red)
lines!(ax_what_ln, what_avg_15_20[:], z[:], linewidth=3,color=:blue)
# axislegend(ax_what_ln, [ln1,ln2],["Day 20","Day 40"], position = :lt)
#     orientation = :vertical)

save(string("output/",slope,"/hab_velocities_",timerange,".png"),fig)







## plot mixing (epsilon, chi, Gamma)

t = ds["t"][:]/86400
z = ds["new_height"][:]
epsilon_avg = ds["epsilon_avg"][:,:]
Bz_avg = ds["Bz_avg"][:,:]
chi_avg = ds["chi_avg"][:,:]./Bz_avg    # -κ|∇b|²/Bz
Gamma_avg = chi_avg./epsilon_avg
# eta_avg = chi_avg./(epsilon_avg.+chi_avg)
fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,300))
axis_kwargs_hm = (xlabel = "time (day)",
                  ylabel = "hab (m)",
                  limits = ((t[1], t[end]), (0, z[end]))
                  )
axis_kwargs_ln = (ylabel = "hab (m)",limits = (nothing,(0, 500)) ) 
axis_kwargs_ln_Gamma = (ylabel = "hab (m)",limits = ((0,3),(0, 500)) ) 

ax_epsilon = Axis(fig[1, 1]; title = "log ε", axis_kwargs_hm...)
ax_chi = Axis(fig[2, 1]; title = "log χ", axis_kwargs_hm...)
ax_gamma = Axis(fig[3, 1]; title = "Γ mixing coefficient", axis_kwargs_hm...)
ax_epsilon_ln = Axis(fig[1, 3]; title = "log ε", axis_kwargs_ln...)
ax_chi_ln = Axis(fig[2, 3]; title = "log χ", axis_kwargs_ln...)
ax_gamma_ln = Axis(fig[3, 3]; title = "Γ mixing coefficient", axis_kwargs_ln_Gamma...)
χ_hm = chi_avg[:,:]
χ_hm[χ_hm.<0] .= NaN
using ColorSchemes
U₀ = 0.025
hm_epsilon = heatmap!(ax_epsilon, t[:], z[:], log10.(epsilon_avg)',
    colorrange = (-10,-7), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[1,2], hm_epsilon )
hm_chi = heatmap!(ax_chi, t[:], z[:], log10.(χ_hm)', colorrange=(-10,-7),
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:blue )
    Colorbar(fig[2,2], hm_chi )
hm_gamma = heatmap!(ax_gamma, t[:], z[:], Gamma_avg',
    colorrange = (0,2), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[3,2], hm_gamma )

lines!(ax_epsilon_ln, log10.(epsilon_avg[:,end]), z[:], linewidth=3)
lines!(ax_chi_ln, log10.(chi_avg[:,end]), z[:], linewidth=3)
lines!(ax_gamma_ln, Gamma_avg[:,end], z[:], linewidth=3)

save(string("output/",slope,"/hab_mixing_",timerange,".png"),fig)

close(ds)




## plot x-z slices 

slope = "tilt"
timerange = "20-40"
filename_field = string("output/", slope, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
t = ds_field["time"][:];
n=38;
b = ds_field["b"][:,:,:,n:n];
Bz = ds_field["Bz"][:,:,:,n:n];
epsilon = ds_field["ε"][:,:,:,n:n];
uhat = ds_field["uhat"][:,:,:,n:n];    # true u
what = ds_field["what"][:,:,:,n:n];    # true w
chi = ds_field["χ"][:,:,:,n:n];        # -κ|∇b|²
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
θ = 0.0036
u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ)

# ----
# plot x-z slices with varying y
n = Observable(1)
uₙ = @lift(u[:,$n,:])
whatₙ = @lift(what_cen[:,$n,:])


    fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(1000,800))
    axis_kwargs_hm = (xlabel = "x (km)",
                      ylabel = "z (m)"
                      )
title = @lift @sprintf("y=%1.0f (m)", yC[$n])
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
                      
ax_u = Axis(fig[2, 1]; title = "Cross-slope velocity", axis_kwargs_hm...)
ax_what = Axis(fig[3, 1]; title = "True vertical velocity", axis_kwargs_hm...)
# ax_Bz = Axis(fig[3, 1]; title = "Buoyancy Gradient", axis_kwargs_hm...)

hm_u = heatmap!(ax_u, xC[:], zC[:], uₙ,
    colorrange = (-0.04, 0.04), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
    Colorbar(fig[2,2], hm_u)
hm_what = heatmap!(ax_what, xC[:], zC[:], whatₙ,
    colorrange = (-0.005, 0.005), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
    Colorbar(fig[3,2], hm_what)

    frames =  (1:5:length(yC))

    filename_field = string("output/", slope, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
    filename = join(split(filename_field, ".")[1:end-1], ".")
    
    record(fig, string(filename,".mp4"), frames, framerate=13) do i
        @info "Plotting frame $i of $(frames[end])..."
        n[] = i
    end
    
# save(string("output/",slope,"/xz_sections_velocities_",timerange,".png"),fig)



## plot difference between flat and tilted case

slope = "tilt"
timerange = "0-20"
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t = ds["t"][:]/86400
z = ds["new_height"][:]
# average between day 15-20
u_avg_tilt = dropdims(mean(ds["u_avg"][:,30:39],dims=2),dims=2)
what_avg_tilt = dropdims(mean(ds["what_avg"][:,30:39],dims=2),dims=2)
Bz_avg_tilt = dropdims(mean(ds["Bz_avg"][:,30:39],dims=2),dims=2)
b_avg_tilt = dropdims(mean(ds["b_avg"][:,30:39],dims=2),dims=2)


slope = "notilt"
timerange = "0-20"
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t = ds["t"][:]/86400
z = ds["new_height"][:]
# average between day 15-20
u_avg_notilt = dropdims(mean(ds["u_avg"][:,30:39],dims=2),dims=2)
what_avg_notilt = dropdims(mean(ds["what_avg"][:,30:39],dims=2),dims=2)
Bz_avg_notilt = dropdims(mean(ds["Bz_avg"][:,30:39],dims=2),dims=2)
b_avg_notilt = dropdims(mean(ds["b_avg"][:,30:39],dims=2),dims=2)

diff_u_avg = u_avg_tilt.-u_avg_notilt
diff_what_avg = what_avg_tilt.-what_avg_notilt


fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,300))

axis_kwargs_ln = (ylabel = "hab (m)",limits = (nothing,(0, 1500)) ) 

ax_u_ln = Axis(fig[1, 1]; title = "u", axis_kwargs_ln...)
ax_u_diff_ln = Axis(fig[1, 2]; title = "uₛ-uₙ", axis_kwargs_ln...)
ax_what_ln = Axis(fig[2, 1]; title = "ŵ", axis_kwargs_ln...)
ax_what_diff_ln = Axis(fig[2, 2]; title = "ŵₛ-ŵₙ", axis_kwargs_ln...)
ax_b_ln = Axis(fig[1, 3]; title = "b", axis_kwargs_ln...)
ax_Bz_ln = Axis(fig[2, 3]; title = "Bz", axis_kwargs_ln...)

ln1=lines!(ax_u_ln, u_avg_tilt, z[:], linewidth=3)
ln2=lines!(ax_u_ln, u_avg_notilt, z[:], linewidth=3)
lines!(ax_u_diff_ln, diff_u_avg, z[:], linewidth=3)
lines!(ax_what_ln, what_avg_tilt, z[:], linewidth=3)
lines!(ax_what_ln, what_avg_notilt, z[:], linewidth=3)
lines!(ax_what_diff_ln, diff_what_avg, z[:], linewidth=3)
lines!(ax_b_ln, b_avg_tilt, z[:], linewidth=3)
lines!(ax_b_ln, b_avg_notilt, z[:], linewidth=3)
lines!(ax_Bz_ln, Bz_avg_tilt, z[:], linewidth=3)
lines!(ax_Bz_ln, Bz_avg_notilt, z[:], linewidth=3)


axislegend(ax_u_ln, [ln1,ln2],["slope","no slope"], position = :rt)

save(string("output/",slope,"/hab_velocity_diff_",timerange,".png"),fig)




## plot tilted case from 0-20 day and 20-40 together

slope = "tilt"
timerange = "0-20"
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t1 = ds["t"][:]/86400
z = ds["new_height"][:]
u_avg1 = ds["u_avg"][:,:]
what_avg1 = ds["what_avg"][:,:]
Bz_avg1 = ds["Bz_avg"][:,:]
b_avg1 = ds["b_avg"][:,:]

slope = "tilt"
timerange = "20-40"
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t2 = ds["t"][:]/86400
z = ds["new_height"][:]
u_avg2 = ds["u_avg"][:,:]
what_avg2 = ds["what_avg"][:,:]
Bz_avg2 = ds["Bz_avg"][:,:]
b_avg2 = ds["b_avg"][:,:]


fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs_hm = (xlabel = "time (day)",
                  ylabel = "hab (m)",
                  limits = ((t1[1], t2[end]), (0, z[end]))
                  )


ax_u = Axis(fig[1, 1]; title = "u", axis_kwargs_hm...)
ax_what = Axis(fig[2, 1]; title = "ŵ", axis_kwargs_hm...)

ax_b = Axis(fig[1, 3]; title = "b (buoyancy perturbation)", axis_kwargs_hm...)
ax_Bz = Axis(fig[2, 3]; title = "dB/dz (Total buoyancy gradient)", axis_kwargs_hm...)

using ColorSchemes

hm_u = heatmap!(ax_u, [t1;t2], z[:], [u_avg1 u_avg2]',
    colorrange = (-0.02,0.02), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[1,2], hm_u )
hm_what = heatmap!(ax_what, [t1;t2], z[:], [what_avg1 what_avg2]',
    colormap = :diverging_bwr_20_95_c54_n256, colorrange=(-0.001, 0.001),
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
    Colorbar(fig[2,2], hm_what)
hm_b = heatmap!(ax_b, [t1;t2], z[:], [b_avg1 b_avg2]',
    colorrange = (minimum(filter(!isnan,b_avg2)),maximum(filter(!isnan,b_avg2))*0.8), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[1,4], hm_b )
hm_Bz = heatmap!(ax_Bz, [t1;t2], z[:], [Bz_avg1 Bz_avg2]',
    colormap = :diverging_bwr_20_95_c54_n256, colorrange=(8.e-7,10.e-7),
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
    Colorbar(fig[2,4], hm_Bz)

save(string("output/",slope,"/hab_tilt_0-40.png"),fig)






# # find the grid that is above z_interp at x-y plane
# inx = zeros(Nx,Ny)  # Preallocate inx array to store the indices
# # create an array of indices that captures the frist element above the topography
# for i in 1:Nx
#    for j in 1:Ny
# inx[i,j] = findfirst(x -> x > z_interp[i,j], zC[:])
#    end
# end


# # using Plots, StatsBase

# function terrain_follow_average(ϕ)

# ## creating terrain-aligned horizontal average

# # Find the tallest point and use histogram to bin the vertical grids first
# binsize = ceil(maximum(diff(zF)));  # smallest binsize has to be larger than the maximun Δz
# row, col = findmax(z_interp)[2][1],findmax(z_interp)[2][2]
# h = fit(Histogram, zC[Int(inx[row,col]):end],[zC[Int(inx[row,col])]:binsize:maximum(zC);])
# bins = diff(h.edges[1])/2 .+ h.edges[1][1:end-1]   # central bin
# # preallocation
# temp = zeros(Nx,Ny,length(h.weights));
# u_TFM=zeros(1,length(h.weights))

# # loop over the entire x-y plane to get the depth histogram from the topography to the surface
# using Distributed
# Threads.@threads for k in 1:length(h.weights)
#     for i in 1:Nx
#         for j in 1:Ny
#             h = fit(Histogram, zC[Int(inx[i,j]):end],[zC[Int(inx[i,j])]:binsize:maximum(zC);])
#             window = 0; 
#             # lg and ug are the lowest and upmost grids within the bin.
#             # For example, if zF = 0, 0.5, 1, 1.5, 2, 2.5, 3, and bins = 0.1, 2.1, 4.1. Within the first bin, lg=0 and ug=2
#             lg = Int(inx[i,j])+window # lowest grid in the bin
#             ug = Int(inx[i,j])+window+h.weights[k]-1 #upmost grid in the bin
#             # temp[i,j,k] = sum(ϕ[i,j,lg:ug].*diff(zF)[lg:ug]) ./ (zF[lg]-zF[ug])
#             temp[i,j,k] = sum(ϕ[i,j,lg:ug].*diff(zF)[lg:ug]) ./ (zF[lg]-zF[ug])
#             window = window + h.weights[k]
#         end
#     end
# end


# ϕ̄ = vec(mean(temp,dims=(1,2))) 
# # uu = vec(mean(temp,dims=(1,2))) 
# # shift the bins to 0
# return ϕ̄, bins.-minimum(bins)    
# end

# ε_avg = terrain_follow_average(ε[:,:,:,73])
# χ_avg = terrain_follow_average(χ[:,:,:,73])
# # uhat_avg = terrain_follow_average(ε[:,:,:,73])

# # calculate without a loop
# # h = fit(Histogram, zC[Int(inx[i,j]):end],[zC[Int(inx[i,j])]:binsize:maximum(zC);],weights = )
# bins = 0:binsize:maximum(z_interp)

# h = fit.(Histogram, eachrow(z_interp), Ref(bins), weights = grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ*grid.Δzᵃᵃᶜ[1:end-4])
# hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]
# hh = hab .* (hab .> 0)    # mask hab



# u.*reshape(grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ*grid.Δzᵃᵃᶜ[1:end-4],1,1,Nz)


# epsilon[epsilon.==0].=NaN
# ln=lines(log10.(epsilon[1,1,:,1]),zC[:])
# for i=1:500:Nx
#     for j=1:500:Ny
# lines!(log10.(epsilon[i,j,:,1]),zC[:])
#     end
# end
# save("output/higher_top_resolution/profile_all_epsilon.png",ln)













# make plots
# fig = Figure()#resolution = (1000, 1000), figure_padding=(10, 4, 10, 10))
#     axis_kwargs1 = (xlabel = "b'",
#                       ylabel = "hab (m)",
#                       )
#     axis_kwargs2 = (xlabel = "b'",)
#     axis_kwargs3 = (xlabel = "∂b'/∂z",
#                       ylabel = "hab (m)",
#                       )
#     axis_kwargs4 = (xlabel = "∂b'/∂z",)

#     ax_1 = Axis(fig[1, 1]; axis_kwargs1...)
#     ax_1.title = "θ=0"
#     ax_2 = Axis(fig[1, 2]; axis_kwargs2...)
#     ax_2.title = "θ=3.6e-3"
#     ax_3 = Axis(fig[2, 1]; axis_kwargs3...)
#     ax_3.title = "θ=0"
#     ax_4 = Axis(fig[2, 2]; axis_kwargs4...)
#     ax_4.title = "θ=3.6e-3"
# lines!(ax_1, b_avg_no[:,1], new_height,linewidth=2.5)
# lines!(ax_1, b_avg_no[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# lines!(ax_2, b_avg[:,1], new_height,linewidth=2.5)
# lines!(ax_2, b_avg[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# lines!(ax_3, bz_avg_no[:,1], new_height,linewidth=2.5)
# lines!(ax_3, bz_avg_no[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# a = lines!(ax_4, bz_avg[:,1], new_height,linewidth=2.5)
# b = lines!(ax_4, bz_avg[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)

# # legend(fig[3,2],[a,b],["ε", "ε_oceanostics"])
# # axislegend("Titled Legend", position = :lb)
# axislegend(ax_4, [a,b],["Day 10","Day 20"], position = :rt,
#     orientation = :vertical)

# save("buoyancy_pert.png",fig)





# fig = Figure()#resolution = (1000, 1000), figure_padding=(10, 4, 10, 10))
#     axis_kwargs1 = (xlabel = "b'x 10⁻⁶",
#                       ylabel = "hab (m)",
#                       )
#     axis_kwargs2 = (xlabel = "∂b'/∂z",
#                       )

#     ax_1 = Axis(fig[1, 1]; axis_kwargs1...)
#     ax_2 = Axis(fig[1, 2]; axis_kwargs2...)
#     # ax_2.title = "θ=3.6e-3"
#     # ax_3 = Axis(fig[2, 1]; axis_kwargs3...)
#     # ax_3.title = "θ=0"
#     # ax_4 = Axis(fig[2, 2]; axis_kwargs4...)
#     # ax_4.title = "θ=3.6e-3"
# # lines!(ax_1, b_avg_no_0_20[:,1]*1e6, new_height,linewidth=2.5,color=:blue)
# # lines!(ax_1, b_avg_no_0_20[:,2]*1e6, new_height,linewidth=2.5,linestyle=:dot,color=:blue)
# # lines!(ax_1, b_avg_no_20_30[:,1]*1e6, new_height,linewidth=2.5,linestyle=:dash,color=:blue)
# lines!(ax_1, b_avg[:,1]*1e6, new_height,linewidth=2.5,color=:red)
# # lines!(ax_1, b_avg[:,2]*1e6, new_height,linewidth=2.5,linestyle=:dot,color=:red)
# # lines!(ax_1, b_avg[:,3]*1e6, new_height,linewidth=2.5,linestyle=:dash,color=:red)
# a=lines!(ax_2, bz_avg_no_0_20[:,1], new_height,linewidth=2.5,color=:blue)
# b=lines!(ax_2, bz_avg_no_0_20[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:blue)
# c=lines!(ax_2, bz_avg_no_20_30[:,1], new_height,linewidth=2.5,linestyle=:dash,color=:blue)
# d = lines!(ax_2, bz_avg[:,1], new_height,linewidth=2.5,color=:red)
# e = lines!(ax_2, bz_avg[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# f = lines!(ax_2, bz_avg[:,3], new_height,linewidth=2.5,linestyle=:dash,color=:red)
# # legend(fig[3,2],[a,b],["ε", "ε_oceanostics"])
# # axislegend("Titled Legend", position = :lb)
# axislegend(ax_1, [a,b,c,d,e,f],["Day 10, θ=0","Day 20, θ=0","Day 30, θ=0",
#  "Day 10, θ=3.6e-3","Day 20, θ=3.6e-3","Day 30, θ=3.6e-3"], position = :rt,
#     orientation = :vertical)

# save("buoyancy_pert_together_3times.png",fig)


