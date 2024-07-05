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
using NaNStatistics

# function deriv(x,y)
# spl = interpolate(x, y, BSplineOrder(6))
# D1f = diff(spl, Derivative(1))   # change
#     return D1f.(x)
# end

function deriv(x,y,dims)
    # dydx =  diff(y[:,:,:,:],dims=3)./reshape(diff(x[:]),1,1,length(y)-1)
    dydx =  diff(y)./diff(x,dims=dims)
    return dydx
 end
# function nice_divergent_levels(c, clim; nlevels=20)
#     levels = range(-clim, stop=clim, length=nlevels)
#     cmax = maximum(abs, c)
#     clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
#     return (-clim, clim), levels
# end

## load data

filename_field = "output/nonconstantdiffusivity8days-theta=0.2_Nx4_Ny4_immersed_nocrossflux_3Dfields.nc"
ds_field = Dataset(filename_field,"r")
bathy_data = "output/supercritical_slope/bathymetry.nc"
Bathy = Dataset(bathy_data,"r")
# filename_slice = "internal_tide_0.1days-theta=0.002.jld2.nc"
# ds_slice = Dataset(filename_slice,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC)
# zF = ds_field["zF"]; Nz=length(zF)
xC = ds_field["xC"]; Nx=length(xC)
xF = ds_field["xF"]; Nx=length(xF)
yC = ds_field["yC"]; Ny=length(yC)
t = ds_field["time"];

# load all data
B = ds_field["B"].var;          B = B[:,:,:,:];
Bz = ds_field["Bz"].var;        Bz = Bz[:,:,:,:]*1e6;
b = ds_field["b"].var;          b = b[:,:,:,:];
u = ds_field["u"].var;          u = u[:,:,:,:];
# what = ds_field["what"].var; what = what[:,:,:,:];
# wb = ds_field["wb"].var;     wb = wb[:,:,:,:];
# ε = ds_field["ε"].var; ε = ε[:,:,:,:];
# χ = ds_field["χ"].var; χ = χ[:,:,:,:];
# Bz = deriv(zC[:],B,dims=3)

# set topography to NaN
# uhat[uhat.==0] .= NaN
# what[what.==0] .= NaN
# wb[isnan.(uhat)] .= NaN
# ε[isnan.(uhat)] .= NaN

# Bz =  diff(B,dims=3)./reshape(diff(zC[:]),1,1,length(diff(zC[:])))
# N = 1e-3;
# θ = 0#3.6e-3;
# Bz_bc = ones(size(Bz,1),size(Bz,2),1,size(Bz,4)).*-N^2*cos(θ) 
# Bz = cat(Bz,Bz_bc,dims=3)
# Bz_mean = dropdims(nanmean(Bz,dims=1),dims=(1,2))
# u_mean = dropdims(nanmean(u,dims=1),dims=(1,2))

topog = [0 0 0 0; 0 0 0 0; 200 200 200 200; 0 0 0 0]
hab = ones(Nx,Ny).*reshape(zC[:],1,1,Nz) .- topog  
new_height = 0:2:2000   # desired height above bottom grids

@views function terrain_following_fast(hab, u, new_height, Nx, Ny)
    # u_interp = Matrix{Int}(undef, 3)

    # u_interp = zeros(Nx, Ny, length(new_height), size(u,4))
    mean_values = zeros(length(new_height), size(u, 4))
    hab_interp = zeros(size(zC))
    for i in 1:Nx
        for j in 1:Ny
            hab_interp = hab[i, j, :]
            for tt = 1:size(u,4)
            itp = Interpolations.interpolate((hab_interp,), u[i, j, :, tt], Gridded(Linear()))
            itp_extrapolated = Interpolations.extrapolate(itp, Interpolations.Flat())
            
            # Directly accumulate the interpolated values into the mean array
            mean_values[:, tt] .+= itp_extrapolated.(new_height)
            end
        end
    end
    mean_values ./= (Nx * Ny)

    return mean_values
end
# call the function to get the terrain following averaged velocity (assuming u is a 4D matrix) 

 u_avg = terrain_following_fast(hab, u, new_height, Nx, Ny);
 Bz_avg = terrain_following_fast(hab, Bz, new_height, Nx, Ny);

 u[u.==0] .=NaN
 B[b.==0] .= NaN
 Bz[b.==0] .= NaN
 b[b.==0] .= NaN

 
## analytical steady solution from Callies 2018
# Environmental parameters
N = 1.3e-3              # Brunt-Väisälä buoyancy frequency        
f₀ = -5.5e-5            # Coriolis frequency
θ = 2e-1#2e-1#2e-3                # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates
κ₀ = 6e-3 #5.2e-5 #  # Far-Field diffusivity
κ₁ = 1.8e-3             # Bottom enhancement of diffusivity
h = 230meter            # decay scale of diffusivity
σ = 1                   # Prandtl number
ν₀ = κ₀
ν₁ = κ₁
z = zC[:]
z_anal = range(0,zC[end],2000)
S = N^2*tan(θ)^2/f₀^2
q = (f₀^2*cos(θ)^2*(1+S*σ) / (4*(ν₀+ν₁)^2))^(1/4)

Bz_analytical = @. N^2*cos(θ)*(κ₀./(κ₀+κ₁*exp(-z_anal/h)) + 
       κ₁*exp(-z_anal/h)/(κ₀+κ₁*exp(-z_anal/h))*S*σ/(1+S*σ) -
       (κ₀/(κ₀+κ₁)+κ₁/(κ₀+κ₁)*S*σ/(1+S*σ))*exp(-q*z_anal)*(cos(q*z_anal)+sin(q*z_anal))
     )
u_analytical = @. -κ₁*cot(θ)*exp(-z_anal/h)/h *(S*σ/(1+S*σ)) +
                2*q*cot(θ)*(κ₀+κ₁*S*σ/(1+S*σ))*exp(-q*z_anal)*sin(q*z_anal) 

# close(ds_field)
# close(ds_slice)


# derivatives of B and u. Only slices are used because it can be memory intensive
# dBdz = zeros(size(B[:,1,:,:])); dûdz = zeros(size(uhat[:,1,:,:]));

# for i in 1:length(t)
#     for j in 1:length(xC)
#       k = Int(round(length(yC)*0.5))
#         dBdz[j,:,i] = deriv(zC,B[j,k,:,i])
#         dûdz[j,:,i] = deriv(zC,uhat[j,k,:,i])
#     end
# end

n = Observable(1)

# εₙ = @lift(ε[:,1,:,$n])
Bₙ = @lift(B[:,1,:,$n])
# B_profileₙ = @lift(B[15,1,:,$n])
Bzₙ = @lift(Bz[:,1,:,$n])
bₙ = @lift(b[:,1,:,$n])
Bz_profileₙ = @lift(Bz_avg[:,$n])
u_profileₙ = @lift(u_avg[:,$n])
# dBdzₙ = @lift interior(dBdz[$n], :, 1, :)
# dûdzₙ = @lift interior(dûdz[$n], :, 1, :)


fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,600));
fs = 20
axis_kwargs = (xlabel = "zonal distance (x)",
                  ylabel = "elevation (m)",
                  limits = ((0, 500), (0, 500)), 
                  xlabelsize = fs,
                  ylabelsize = fs,
                  xticklabelsize = fs,
                  yticklabelsize = fs,
                  titlesize = fs
                  )
axis_kwargs_Bz = (ylabel = "hab (m)",
                  limits = ((0, 2), (0, 100)), 
                  xlabelsize = fs,
                  ylabelsize = fs,
                  xticklabelsize = fs,
                  yticklabelsize = fs,
                  titlesize = fs
                  )
axis_kwargs_u = (ylabel = "hab (m)",
                  limits = ((-.005, .005), (0, 100)), 
                  xlabelsize = fs,
                  ylabelsize = fs,
                  xticklabelsize = fs,
                  yticklabelsize = fs,
                  titlesize = fs
                  )

# ax_u = Axis(fig[1, 1]; title = "u and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_uz = Axis(fig[1, 2]; title = "du/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_bz = Axis(fig[2, 1]; title = "db/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
# ax_ε = Axis(fig[2, 1]; title = "TKE dissipation rate (ε) and equally-spaced buoyancy contours (B)", axis_kwargs...)
title = @lift @sprintf("t=%1.2f day", t[$n]/86400)
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
ax_b = Axis(fig[2, 1]; title = "b", axis_kwargs...)
ax_Bz = Axis(fig[3, 1]; title = L"B_z×10^{-6}", axis_kwargs...)
ax_Bprofile = Axis(fig[2, 3]; title = L"B_z×10^{-6}", axis_kwargs_Bz...)
ax_u = Axis(fig[3, 3]; title = "u", axis_kwargs_u...)



# ax_B = Axis(fig[1, 3]; title = "B")

# U₀ = 0.025
# hm_u = heatmap!(ax_u, xC[:], zC[:], ûₙ,
#     colorrange = (-3U₀, 3U₀), colormap = :balance,
#     lowclip=cgrad(:balance)[1], highclip=cgrad(:balance)[end])
# ct_u = contour!(ax_u, xC, zC, Bₙ,
#     levels=0.:2e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
# Colorbar(fig[1,2], hm_u)
# ylims!(ax_u,(0,1000))

hm_b = heatmap!(ax_b, xC[:], zC[:], bₙ,
    colormap = :matter,
    lowclip=cgrad(:matter)[1], highclip=cgrad(:matter)[end], colorrange=(0,1e-4), nancolor=:grey)
ct_b = contour!(ax_b, xC, zC, Bₙ,
    levels=0.:0.25e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[2,2], hm_b)
# ylims!(ax_B,(0,2000))

hm_Bz = heatmap!(ax_Bz, xC[:], zC[:], Bzₙ,
    colorrange = (minimum(filter(!isnan,Bz)),maximum(filter(!isnan,Bz)) ),colormap = :balance,
    lowclip=cgrad(:matter)[1], highclip=cgrad(:matter)[end])
ct_Bz = contour!(ax_Bz, xC, zC, Bₙ,
    levels=0.:0.25e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
Colorbar(fig[3,2], hm_Bz)


L1 = lines!(ax_Bprofile,Bz_profileₙ,new_height,linestyle=:solid,color=:black,linewidth=2)
L2 = lines!(ax_Bprofile,Bz_analytical*1e6,z_anal,color=:red,linewidth=2,linestyle=:dash)

axislegend(ax_Bprofile,[L1, L2],
    [L"\langle B_z \rangle_x",
     "Analytical Sol. (Callies 2018)"],
     position = :lt,framevisible = false, fontsize=25
     )
lines!(ax_u,u_profileₙ,new_height,linestyle=:solid,color=:black,linewidth=2)
lines!(ax_u,u_analytical,z_anal,color=:red,linewidth=2,linestyle=:dash)

frames = (1:length(t))
filename = join(split(filename_field, ".")[1:end-1], ".")
record(fig, string(filename,".mp4"), frames, framerate=25) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i 
end

# close(ds_field)
# ### calculate the terrain following avg
# filename = "output/internal_tide_1days-theta=0.0036_realtopo3D_Nx20_fields.nc"
# ds = Dataset(filename)
# B = ds["B"].var; 
# zC = ds["zC"].var; Nz=length(zC)
# zF = ds["zF"].var; 
# yC = ds["yC"]; Ny=length(yC)
# xC = ds["xC"]; Nx=length(xC)
# hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]
# # hh = hab .* (hab .> 0)    # mask hab
# # hh[hh.==0] .= NaN
# # hab_interp = 0:20:3000 # define the desired vertical grids to be interpolated on
# # itp = interpolate((xC,yC,zC), u[1:end-4,1:end-4,1:end-4], Gridded(Linear()))
# # u_interp = itp(xC,yC,15:40:3000)
# new_height = 0:40:3000
# B_interp = zeros(Nx,Ny,length(new_height))
# hab_interp = zeros(Nx,Ny,Nz)
# # hab_interp = 0:20:3000


# for i in 1:Nx
#     for j in 1:Ny
# hab_interp = hab[i,j,:] # define the desired vertical grids to be interpolated on
# itp = interpolate((hab_interp,), B[i,j,:,25], Gridded(Linear()))
# itp_extrapolated = extrapolate(itp, Interpolations.Flat())
#         for inx in 1:length(new_height)
#             B_interp[i,j,inx] = itp_extrapolated(new_height[inx])
#         end
#     end
# end

# B_interp_avg = mean(B_interp,dims=(1,2))
# lines(vec(B_interp_avg),new_height,linewidth=3,color=:red)
# for i in 1:20
#     for j in 1:20
#     lines!(B[i,j,:,25], zC[:])
#     end
# end





# using Interpolations
# hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]
# new_height = 0:40:3000
# B_interp = zeros(Nx,Ny,length(new_height))
# hab_interp = zeros(Nx,Ny,Nz)
# # Assuming hab, B, and new_height are predefined.

# function interpolate_B(hab, B, new_height)
#     Nx, Ny, Nz = size(hab)
#     B_interp = zeros(Nx, Ny, length(new_height), size(B,4))

#     for i in 1:Nx, j in 1:Ny
#         for tt = 1:size(B,4)
#         hab_interp = hab[i, j, :]
#         itp = interpolate((hab_interp,), B[i, j, :, tt], Gridded(Linear()))
#         itp_extrapolated = extrapolate(itp, Interpolations.Flat())
        
#         # Instead of looping over new_height, broadcast the extrapolation function over it
#         B_interp[i, j, :, tt] .= itp_extrapolated.(new_height)
#         end
#     end
#     return dropdims(mean(B_interp,dims=(1,2)) , dims=(1,2))
# end

# BB = interpolate_B(hab, B, new_height);



# f = Figure(figure_padding=(10, 100, 10, 10))

# Axis(f[1, 1], title = "My column has size Relative(2/3)")
# Axis(f[1, 2], title = "My column has size Auto()")
# Colorbar(f[2, 1],vertical=false)
# colsize!(f.layout, 1, Aspect(1,0.5))

# f


# This creates a new NetCDF file /tmp/test.nc.
# The mode "c" stands for creating a new file (clobber)
# ds_create = NCDataset("output/TF_avg.nc","c")
# # Define the dimension
# defDim(ds_create,"z_TF",length(new_height))

# # Define a global attribute
# ds_create.attrib["title"] = "Terrain-following averages"
# # Define the variables
# vv = defVar(ds_create,"Buoyancy",Float64,("z_TF",))

# # write a single column
# vv[:] = B_interp
# # write attributes
# vv.attrib["units"] = "m/s2"







# # using broadcasting to calculate terrain-following averages
# using MAT
# using Interpolations

# # load topography 
# file = matopen("/Users/chihlunliu/Library/CloudStorage/OneDrive-SharedLibraries-UCIrvine/Chih-Lun - Documents/UC Irvine/research/topo.mat")
# z_topo = read(file, "z_noslope_periodic") 
# x_topo = read(file, "x_domain")
# y_topo = read(file, "y_domain")
# # grids has to be evenly spaced
# x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
# y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
# close(file)
# # high-resolution grids
# x_interp = range(x_topo[1],x_topo[end], length=Nx)
# #Ny=2Nx
# y_interp = range(y_topo[1],y_topo[end], length=Ny)

# # Interpolation object (caches coefficients and such)
# itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# # Interpolate z_topo onto a higher-resolution grid
# # itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
# z_interp = z_interp.-minimum(z_interp)

#                 hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]

#                 itp = interpolate((hab,), v1[i, j, :, inx], Gridded(Linear()))
#                 itp_extrapolated = extrapolate(itp, Interpolations.Flat())
#                 # broadcast the extrapolation function over it
#                 v1_interp[i, j, :] .= itp_extrapolated.(new_height)
#                 v2_interp[i, j, :] .= itp_extrapolated.(new_height)
#                 v3_interp[i, j, :] .= itp_extrapolated.(new_height)
#                 v4_interp[i, j, :] .= itp_extrapolated.(new_height)
#                 v5_interp[i, j, :] .= itp_extrapolated.(new_height)


# return dropdims(mean(v1_interp,dims=(1,2)) , dims=(1,2))

# # call the function to calculate terrain-following average
# new_height = 0:40:3000
# inx = 6
# ε_interp = terrain_following(ε, new_height, inx);
