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
using LinearAlgebra
using Interpolations
using MAT


# function deriv(x,y)
#     spl = BSplineKit.interpolate(x, y, BSplineOrder(6))
#     D1f = diff(spl, Derivative(1))   # change
#         return D1f.(x)
# end

function deriv(x,y)
   dydx =  diff(y[:,:,:,:],dims=3)./reshape(diff(x[:]),1,1,length(zC)-1)
   return dydx
end
# function tracer_derivative(z,var)
#     Nx = size(var,1)
#     Ny = size(var,2)
#     dvardz = zeros(size(var));
#     for i in 1:Nx
#         for j in 1:Ny
#             for k = 1:size(var,4)
#             dvardz[i,j,:,k] = deriv(z,vec(var[i,j,:,k]))
#             end
#         end
#     end
#     return dvardz
# end

# @time bz =  tracer_derivative(zC[:],b)


# terrain following average algorithms
@views function terrain_following_fast(hab, u, new_height, Nx, Ny)

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



###################################
# load data
# filename_field = "output/no_tilt/internal_tide_5days-theta=0_realtopo3D_Nx500_Nz250_fields_timeavg_0_20.nc"
# ds_field_no_0_20 = Dataset(filename_field,"r")
filename_field = "output/supercritical_tilt/backgroundfluxdivergence_smagorinky/backgroundfluxdivergence_smagorinkyinternal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_fields_timeavg_0_5.nc"
# ds_field_no_20_30 = Dataset(filename_field,"r")
# filename_field = "output/supercritical_tilt/internal_tide_5days-theta=0.0036_realtopo3D_Nx500_Nz250_fields_timeavg_50_60.nc"
ds_field_super = Dataset(filename_field,"r")

# grids
zC = ds_field_super["zC"]; Nz=length(zC[:])
zF = ds_field_super["zF"]; zF = zF[:];
xC = ds_field_super["xC"]; Nx=length(xC[:])
yC = ds_field_super["yC"]; Ny=length(yC[:])
t_super = ds_field_super["time"][:];

new_height = 0:10:1500#0

# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"].var[:,:,:];
b_avg_super = zeros(length(new_height),length(t_super))
bz_avg_super = zeros(length(new_height),length(t_super))
uhat_avg_super = zeros(length(new_height),length(t_super))
what_avg_super = zeros(length(new_height),length(t_super))

for n in 1:length(t_super)
# t_no_0_20 = ds_field_no_0_20["time"];
# t_no_20_30 = ds_field_no_20_30["time"];

# b_no_0_20 = ds_field_no_0_20["b"][:,:,:,:];
# b_no_20_30 = ds_field_no_20_30["b"][:,:,:,:];
# uhat_no = ds_field_no["uhat"];
b_super = ds_field_super["b"][:,:,:,n:n];
# B_super = ds_field_super["B"][:,:,:,:];
uhat_super = ds_field_super["uhat"][:,:,:,n:n];
what_super = ds_field_super["what"][:,:,:,n:n];

# b_no_0_20[b_no_0_20.==0] .= NaN;
# b_no_20_30[b_no_20_30.==0] .= NaN;
# b_super[b_super.==0] .= NaN;
# uhat_super[uhat_super.==0] .= NaN;
# what_super[what_super.==0] .= NaN;

N = 1.e-3
# bz_no_0_20 =  deriv(zC,b_no_0_20);
# bz_no_20_30 =  deriv(zC,b_no_20_30);
# θ = 0; 
# bz_bc_0_20 = -N^2*cos(θ).*ones(Nx,Ny,1,length(t_no_0_20));
# bz_bc_20_30 = -N^2*cos(θ).*ones(Nx,Ny,1,length(t_no_20_30));

# bz_no_0_20 = cat(bz_no_0_20, bz_bc_0_20, dims=3);
# bz_no_20_30 = cat(bz_no_20_30, bz_bc_20_30, dims=3);

bz_super =  deriv(zC,b_super);
θ = 3.6e-3; 
bz_bc = -N^2*cos(θ).*ones(Nx,Ny,1,1);
bz_super = cat(bz_super, bz_bc, dims=3);


### terrain following quantities:


# call the function to get the terrain following averaged velocity (assuming u is a 4D matrix) 
#  @time u_avg = terrain_following_fast(hab, uhat[:,:,:,2:3], new_height, Nx, Ny);
# @time b_avg_no_0_20 = terrain_following_fast(hab, b_no_0_20[:,:,:,2:3], new_height, Nx, Ny);
# @time b_avg_no_20_30 = terrain_following_fast(hab, b_no_20_30[:,:,:,2], new_height, Nx, Ny);
@time b_avg_super[:,n] = terrain_following_fast(hab, b_super[:,:,:,:], new_height, Nx, Ny);
# @time bz_avg_no_0_20 = terrain_following_fast(hab, bz_no_0_20[:,:,:,2:3], new_height, Nx, Ny);
# @time bz_avg_no_20_30 = terrain_following_fast(hab, bz_no_20_30[:,:,:,2], new_height, Nx, Ny);
@time bz_avg_super[:,n] = terrain_following_fast(hab, bz_super[:,:,:,:], new_height, Nx, Ny);
@time uhat_avg_super[:,n] = terrain_following_fast(hab, uhat_super[:,:,:,:], new_height, Nx, Ny);
@time what_avg_super[:,n] = terrain_following_fast(hab, what_super[:,:,1:end-1,:], new_height, Nx, Ny);
 
end
close(ds_field_super)
close(ds_hab)

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
# lines!(ax_2, b_avg_super[:,1], new_height,linewidth=2.5)
# lines!(ax_2, b_avg_super[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# lines!(ax_3, bz_avg_no[:,1], new_height,linewidth=2.5)
# lines!(ax_3, bz_avg_no[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# a = lines!(ax_4, bz_avg_super[:,1], new_height,linewidth=2.5)
# b = lines!(ax_4, bz_avg_super[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)

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
# lines!(ax_1, b_avg_super[:,1]*1e6, new_height,linewidth=2.5,color=:red)
# # lines!(ax_1, b_avg_super[:,2]*1e6, new_height,linewidth=2.5,linestyle=:dot,color=:red)
# # lines!(ax_1, b_avg_super[:,3]*1e6, new_height,linewidth=2.5,linestyle=:dash,color=:red)
# a=lines!(ax_2, bz_avg_no_0_20[:,1], new_height,linewidth=2.5,color=:blue)
# b=lines!(ax_2, bz_avg_no_0_20[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:blue)
# c=lines!(ax_2, bz_avg_no_20_30[:,1], new_height,linewidth=2.5,linestyle=:dash,color=:blue)
# d = lines!(ax_2, bz_avg_super[:,1], new_height,linewidth=2.5,color=:red)
# e = lines!(ax_2, bz_avg_super[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# f = lines!(ax_2, bz_avg_super[:,3], new_height,linewidth=2.5,linestyle=:dash,color=:red)
# # legend(fig[3,2],[a,b],["ε", "ε_oceanostics"])
# # axislegend("Titled Legend", position = :lb)
# axislegend(ax_1, [a,b,c,d,e,f],["Day 10, θ=0","Day 20, θ=0","Day 30, θ=0",
#  "Day 10, θ=3.6e-3","Day 20, θ=3.6e-3","Day 30, θ=3.6e-3"], position = :rt,
#     orientation = :vertical)

# save("buoyancy_pert_together_3times.png",fig)




# This creates a new NetCDF file 
# The mode "c" stands for creating a new file

ds_create = Dataset("output/supercritical_tilt/backgroundfluxdivergence_smagorinky/TF_avg.nc","c")
# Define the dimension
defDim(ds_create,"z_TF",length(new_height))
defDim(ds_create,"t",length(t_super))

# Define a global attribute
ds_create.attrib["title"] = "Terrain-following averages"
# Define the variables
v1 = defVar(ds_create,"b_avg",Float64,("z_TF","t"))
v1[:,:] = b_avg_super
v2 = defVar(ds_create,"bz_avg",Float64,("z_TF","t"))
v2[:,:] = bz_avg_super
v3 = defVar(ds_create,"uhat_avg",Float64,("z_TF","t"))
v3[:,:] = uhat_avg_super
v4 = defVar(ds_create,"what_avg",Float64,("z_TF","t"))
v4[:,:] = what_avg_super
v5 = defVar(ds_create,"new_height",Float64,("z_TF",))
v5[:,1] = new_height
v6 = defVar(ds_create,"t",Float64,("t",))
v6[:,1] = t_super   

# write attributes
v1.attrib["units"] = "m/s²"
v2.attrib["units"] = "1/s²"
v3.attrib["units"] = "m/s"
v4.attrib["units"] = "m/s"
v5.attrib["units"] = "m"
v6.attrib["units"] = "s"

close(ds_create)


## plot the hab profiles
file = "output/supercritical_tilt/backgroundfluxdivergence_smagorinky/TF_avg.nc"
ds = Dataset(file,"r")

t = ds["t"][:]/86400
z = ds["new_height"][:]
bz_avg = ds["bz_avg"][:,:]
b_avg = ds["b_avg"][:,:]
# BC: 
θ = 3.6e-3; 
N = 1.e-3
bz_avg[1,:] .= -N^2*cos(θ)
uhat_avg = ds["uhat_avg"][:,:]
uhat_avg[1,:] .= 0

fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs_hm = (xlabel = "time (day)",
                  ylabel = "hab (m)",
                  limits = ((0, t[end]), (0, z[end]))
                  )
axis_kwargs_line = (ylabel = "hab (m)",limits = (nothing,(0, 500)) ) 

ax_b = Axis(fig[1, 1]; title = "b", axis_kwargs_hm...)
ax_bz = Axis(fig[2, 1]; title = "db/dz", axis_kwargs_hm...)
ax_u = Axis(fig[3, 1]; title = "û", axis_kwargs_hm...)

ax_b_z = Axis(fig[1, 3]; title = "b", axis_kwargs_line...)
ax_bz_z = Axis(fig[2, 3]; title = "db/dz", axis_kwargs_line...)
ax_u_z = Axis(fig[3, 3]; title = "û", axis_kwargs_line...)


# ax_χ = Axis(fig[4, 1]; title = "Scalar variance dissipation rate (log χ) and equally-spaced buoyancy contours (B)", axis_kwargs...)


using ColorSchemes
U₀ = 0.025
hm_b = heatmap!(ax_b, t[:], z[:], b_avg',
    colorrange = (minimum(b_avg),maximum(b_avg)*0.8), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[1,2], hm_b )
hm_bz = heatmap!(ax_bz, t[:], z[:], bz_avg',
    colormap = :diverging_bwr_20_95_c54_n256, colorrange=(-3e-8, 3e-8),
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
    Colorbar(fig[2,2], hm_bz)
hm_u = heatmap!(ax_u, t[:], z[:], uhat_avg',
    colorrange = (-0.001, 0.001), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],clim = (-3U₀,3U₀),
    nan_color = :gray)
    Colorbar(fig[3,2], hm_u)

ln_b = lines!(ax_b_z, b_avg[:,end], z[:], linewidth=3)
ln_bz = lines!(ax_bz_z, bz_avg[:,end], z[:], linewidth=3)
ln_u = lines!(ax_u_z, uhat_avg[:,end], z[:], linewidth=3)


save("output/supercritical_tilt/backgroundfluxdivergence_smagorinky/hab.png",fig)
close(ds)





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
