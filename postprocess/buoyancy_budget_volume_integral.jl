
## plot the volume integrated quantities of the buoyancy budget 
# <∂B/∂t> = <-ũ⋅∇B> - <∇⋅(-κ∇B)>

using NCDatasets
using CairoMakie
using Statistics
using NaNStatistics
using Oceananigans
using Oceananigans.Units
include("../functions/bins.jl")
include("../functions/mmderiv.jl")
include("../functions/mmintgrl.jl")

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

simname = "flat"
θ = simname == "tilt" ? 0.0036 : 0
tᶠ = 460
N=0.001

Lx = 15e3
Ly = 30e3
Lz = 2250
n = 2
m = 10
k = 250  #argmin(abs.(zC .- 2000))


# take the average over 451-460 TP (2-10) from time averaged data

u_10avg = zeros((length(xC), length(yC), k))
w = zeros((length(xC), length(yC), k))
wB_10avg = zeros((length(xC), length(yC), k))
for i in n:m
        uhat = ds_field["uhat"][:,:,1:k,i]       # true u (tidal averaged)
        what = ds_field["what"][:,:,1:k+1,i];    # true w (tidal averaged)
        what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
        # b = ds_timeavg["b"][:,:,1:k,n];    # b
        # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
        wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
        # convert velocities from true coordinates to slope coordinates
        u_face = uhat[:,:,:]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity [face,center,center]
        w_face = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ) # slope-normal velocity [face,center,center]
        # Convert from [face,center,center] to [center,center,center]
        u_10avg .+= (u_face[1:end,:,:] .+ vcat(u_face[2:end,:,:],u_face[1:1,:,:])) ./ 2
        w = (w_face[1:end,:,:] .+ vcat(w_face[2:end,:,:],w_face[1:1,:,:])) ./ 2
        if simname == "tilt"
                B = ds_B["B"][:,:,1:k,i]
        else
                B = ds_field["B"][:,:,1:k,i]
        end
        wB_10avg .+= w.*B     # sum up all 1TP-averaged wB
        @info i
end
wB_10avg ./= (m-n+1)
u_10avg ./= (m-n+1)
if simname == "tilt"
        ∇κ∇B = nanmean(ds_const_dt["∇κ∇B"][:,:,1:k,n:m],dim=4)
        div_uB = nanmean(ds_const_dt["div_uB"][:,:,1:k,n:m],dim=4)        
else
        ∇κ∇B = nanmean(ds_budget["∇κ∇B"][:,:,1:k,n:m],dim=4)
        div_uB = nanmean(ds_budget["div_uB"][:,:,1:k,n:m],dim=4)        
end

# Calculate background buoyancy
B̄ = calculate_background_buoyancy(θ)
b = simname=="tilt" ? ds_B["B"][:,:,1:k,1] .- B̄ : ds_field["B"][:,:,1:k,1] .- B̄
b[abs.(b).<1e-15].=0
# mask
u_10avg[u_10avg.==0].=NaN
∇κ∇B[b.==0].=NaN
div_uB[b.==0].=NaN
wB_10avg[wB_10avg.==0].=NaN
b[b.==0].=NaN

# compute all terms
# mean flow advective flux (east + west)
meanflow_transport = -N^2*Lx*sin(θ)*mmintgrl(zC[1:k],nansum(u_10avg[1,:,:].*dy,dim=1))
wb_int_total = mmintgrl(zC[1:k],nansum(div_uB*dx*dy,dim=(1,2))) .+ meanflow_transport
w_avgb_avg_int = nansum(wB_10avg*dx*dy,dim=(1,2))
w_primeb_prime_int = wb_int_total .- w_avgb_avg_int

∇κ∇B_int = mmintgrl(zC[1:k],nansum(∇κ∇B*dx*dy,dim=(1,2)))
t_diff = diff(t)
end_time = argmin(abs.(t_3D.-t[m]))   
begin_time = argmin(abs.(t_3D./(2*pi/1.4e-4).-(Int(round(t[n]/(2*pi/1.4e-4))).-1))) 
B2 = ds_3D["B"][:,:,1:k,[begin_time,end_time]];          # total buoyancy with two timesteps
b2 = B2 .- B̄
# b_mask = cat(b2, b2, dims=4)
# Apply to B2
# B2[b_mask.==0] .= NaN
# dBdt_int = mmintgrl(zC[1:k],nansum(((B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ t_diff[n-1])*dx*dy ,dim=(1,2) ))
# dBdt_int = mmintgrl(zC[1:k],nansum(((B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ (t_3D[end_time]-t_3D[begin_time]))*dx*dy ,dim=(1,2) ))
dbdt_int = mmintgrl(zC[1:k],nansum(((b2[:, :, :, 2] .- b2[:, :, :, 1]) ./ (t_3D[end_time]-t_3D[begin_time]))*dx*dy ,dim=(1,2) ))
rhs = meanflow_transport - wb_int_total + ∇κ∇B_int
factor = θ==0.0036 ? -N^2*Lx*sin(θ) : 1

topo_top = argmin(abs.(zC[:].-1482.2918509387632))
using PyPlot
close("all")
fig = figure(figsize=(6,5))
# Set font size
plt.rcParams["font.size"] = 20
# if simname == "tilt"
#         # Plot each time series
#         plt.plot(dbdt_int/factor/1e3, zC[1:k], 
#         label = L"tendency", color = "gray", linewidth=2.0)
#         plt.plot(rhs/factor/1e3, zC[1:k], 
#         label = L"RHS", color = "black", linestyle="--",linewidth=2.0)
#         plt.plot(-w_avgb_avg_int/factor/1e3, zC[1:k], 
#         label = "subtidal flux", color = [150,148,255]/255, linestyle="-",linewidth=2.0)
#         plt.plot(-w_primeb_prime_int/factor/1e3, zC[1:k], 
#         label = "instability", color = [136,194,115]/255, linestyle="-",linewidth=2.0)
#         plt.plot(-wb_int_total/factor/1e3, zC[1:k], 
#         label = "buoyancy flux", color = "red", linewidth=2.0)
#         plt.plot(∇κ∇B_int/factor/1e3, zC[1:k], 
#         label = "mixing", color = "orange", linestyle="-", linewidth=2.0)
#         plt.plot(meanflow_transport/factor/1e3, zC[1:k], 
#         label = "mean flow", color = "purple", linestyle="-", linewidth=2.0)
# else 
        # Plot each time series
        plt.plot(dbdt_int, zC[1:k], 
        label = L"tendency", color = "gray", linewidth=2.0)
        plt.plot(rhs, zC[1:k], 
        label = L"RHS", color = "black", linestyle="--",linewidth=2.0)
        plt.plot(-w_avgb_avg_int, zC[1:k], 
        label = "subtidal flux", color = [150,148,255]/255, linestyle="-",linewidth=2.0)
        plt.plot(-w_primeb_prime_int, zC[1:k], 
        label = "instability", color = [136,194,115]/255, linestyle="-",linewidth=2.0)
        plt.plot(-wb_int_total, zC[1:k], 
        label = "buoyancy flux", color = "red", linewidth=2.0)
        plt.plot(∇κ∇B_int, zC[1:k], 
        label = "diffusion", color = "orange", linestyle="-", linewidth=2.0)
        plt.plot(meanflow_transport, zC[1:k], 
        label = "mean flow", color = "purple", linestyle="-", linewidth=2.0)
# end
# Create a shaded patch for the topography
        xlims = plt.xlim()  # Get current x limits
        plt.fill_between([xlims[1], xlims[2]], [0, 0], [zC[topo_top], zC[topo_top]], 
                        color="brown", alpha=0.3) 
legend(loc="center", bbox_to_anchor=(.75, .25), frameon=true, ncol=1, fontsize=10,
        handlelength=1.4, columnspacing=.9, framealpha=.93)
# legend(loc="center", bbox_to_anchor=(.5, -.15), frameon=true, ncol=1, fontsize=12,
#         handlelength=1.4, columnspacing=.9, framealpha=.93)


# ax.plot([0,0],[0,200], color = "black", linewidth=1.5)
grid("on")
# if simname =="tilt"
#         xlabel("Transport [mSv]")
# else
        xlabel(L"[m^4s^{-3}]")
# end
ylabel("Height above mean slope (m)")
ylim(0,2250)
xlim(xlims[1],xlims[end])
# xlim(-10,10)
# Set the y-limits and labels
# set_ylim(0, 200)
# plt.set_ylabel("Height Above  (m)")
title("Integrated buoyancy budget")
# plt.xscale("symlog", linthresh=1)

plt.minorticks_on()
t_begin = Int(round(t_3D[begin_time]./(2*pi/1.4e-4)))
t_end = Int(round(t_3D[end_time]./(2*pi/1.4e-4)))
savefig(string("output/", simname, "/buoyancy_budget_volume_integral_testintegral_nofactor_const_dt",t_begin,"-",t_end,"TP.png"))

################
# this part of the script shows that wB calculated from the residual and directly from the snapshot data match well

# use the 3D snapshot to plot the buoyancy budget (time average might be untrustworthy)
# <∂B/∂t> = <-ũ⋅∇B> - <∇⋅(-κ∇B)>



# simname = "tilt"
# θ = simname == "tilt" ? 0.0036 : 0
# tᶠ = 460
# N=0.001
# # Decompose advection term
# # <∇⋅(uB)> = <u⋅∇B> = <u>⋅∇<B> + <u'⋅∇B'>
# filename_budget = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_Bbudget.nc")
# ds_budget = Dataset(filename_budget,"r")

# filename_B = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_B-c.nc")
# ds_B = Dataset(filename_B,"r")

# filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_u-v-w-c.nc")
# ds_field = Dataset(filename_field,"r")

# t = ds_field["time"][:];

# xC = ds_field["xC"][:]; dx = diff(xC)[1]
# yC = ds_field["yC"][:]; dy = diff(yC)[1]
# zC = ds_field["zC"][:]

# Lx = 15e3
# Ly = 30e3
# Lz = 2250
# n = 1
# m = 2#length(t)

# # take the average over 451-460 TP (2-10) from time averaged data
# ∇κ∇B = nanmean(ds_budget["∇κ∇B"][:,:,:,n:m],dim=4)
# div_uB = nanmean(ds_budget["div_uB"][:,:,:,n:m],dim=4)

# uhat = nanmean(ds_field["uhat"][:,:,:,n:m],dim=4)
# what = nanmean(ds_field["what"][:,:,:,n:m],dim=4);    # true w
# what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
# B = nanmean(ds_B["B"][:,:,:,n:m],dim=4)

# # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
# wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
# # Calculate u and w at [face,center,center] first
# u_face = uhat[:,:,:]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity [face,center,center]
# w_face = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ) # slope-normal velocity [face,center,center]

# # Convert from [face,center,center] to [center,center,center]
# u = (u_face[1:end,:,:] .+ vcat(u_face[2:end,:,:],u_face[1:1,:,:])) ./ 2
# w = (w_face[1:end,:,:] .+ vcat(w_face[2:end,:,:],w_face[1:1,:,:])) ./ 2


# # Calculate background buoyancy
# B̄ = calculate_background_buoyancy(θ)
# b = B[:,:,:] .- B̄
# b[abs.(b).<1e-15].=0

# wB_avg = zeros(size(B̄))
# count = 0
# for i in 1:2#length(t)
#         uhat = ds_field["uhat"][:,:,:,i]
#         what_cen = (ds_field["what"][:,:,1:end-1,i] .+ ds_field["what"][:,:,2:end,i])./2 # what at center
#         wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
#         # Calculate u and w at [face,center,center] first
#         u_face = uhat[:,:,:]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity [face,center,center]
#         w_face = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ) # slope-normal velocity [face,center,center]

#         # Convert from [face,center,center] to [center,center,center]
#         # u = (u_face[1:end,:,:] .+ vcat(u_face[2:end,:,:],u_face[1:1,:,:])) ./ 2
#         w_snapshot = (w_face[1:end,:,:] .+ vcat(w_face[2:end,:,:],w_face[1:1,:,:])) ./ 2


#         global wB_avg .+= w_snapshot .* (ds_B["B"][:,:,:,i])
#         global count += 1 
#         @info count
# end
# wB_avg ./= count

# # mask
# u[u.==0].=NaN
# ∇κ∇B[b.==0].=NaN
# div_uB[b.==0].=NaN
# # B[b.==0].=NaN
# w[w.==0].=NaN
# wB_avg[b.==0].=NaN

# # compute all terms
# # mean flow advective flux (east + west)
# meanflow_transport = -N^2*Lx*sin(θ)*mmintgrl(zC[:],nansum(u[1,:,:].*dy,dim=1))
# wb_int_total = mmintgrl(zC[:],nansum(div_uB*dx*dy,dim=(1,2))) .+ meanflow_transport
# wb_int_total_direct_calculation = nansum(wB_avg*dx*dy,dim=(1,2))
# w_avgb_avg_int = nansum(w.*b*dx*dy,dim=(1,2))
# w_primeb_prime_int = wb_int_total .- w_avgb_avg_int

# ∇κ∇B_int = mmintgrl(zC[:],nansum(∇κ∇B*dx*dy,dim=(1,2)))

# B2 = ds_B["B"][:,:,:,[1,2]];          # total buoyancy with two timesteps
# dbdt_int = mmintgrl(zC[:],nansum(((B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ (t[2]-t[1]))*dx*dy ,dim=(1,2) ))
# rhs = meanflow_transport - wb_int_total + ∇κ∇B_int
# factor = θ==0.0036 ? -N^2*Lx*sin(θ) : 1

# ## save the budget into a netcdf file
# # Define variables for the netCDF file
# # k = length(zC)
# # budget_file = string("output/", simname, "/buoyancy_budget_volume_integral_snapshot_avg.nc")
# # # Create the netCDF file
# # ds_out = Dataset(budget_file, "c")
# # # Define dimensions
# # defDim(ds_out, "z", k)

# # # Define variables
# # defVar(ds_out, "zC", zC[1:k], ("z",))
# # defVar(ds_out, "dbdt", dbdt_int, ("z",))
# # defVar(ds_out, "meanflow_transport", meanflow_transport, ("z",))
# # defVar(ds_out, "wb_int_total", wb_int_total, ("z",))
# # defVar(ds_out, "wb_int_total_direct_calculation", wb_int_total_direct_calculation, ("z",))
# # defVar(ds_out, "diffusion", ∇κ∇B_int, ("z",))
# # defVar(ds_out, "w_avgb_avg", w_avgb_avg_int, ("z",))
# # defVar(ds_out, "w_primeb_prime", w_primeb_prime_int, ("z",))
# # defVar(ds_out, "rhs", rhs, ("z",))

# # # Add attributes for better documentation
# # ds_out.attrib["description"] = "Volume integrated buoyancy budget terms (from snapshots)"
# # # Close the file
# # close(ds_out)
# # println("Saved budget data to $budget_file")


# using PyPlot
# close("all")
# fig = figure(figsize=(6,5))
# # Set font size
# plt.rcParams["font.size"] = 20
# plt.plot(dbdt_int, zC[:], 
# label = L"tendency", color = "gray", linewidth=2.0)
# plt.plot(rhs, zC[:], 
# label = L"RHS", color = "black", linestyle="--",linewidth=2.0)
# plt.plot(-w_avgb_avg_int, zC[:], 
# label = "subtidal flux", color = [150,148,255]/255, linestyle="-",linewidth=2.0)
# plt.plot(-w_primeb_prime_int, zC[:], 
# label = "instability", color = [136,194,115]/255, linestyle="-",linewidth=2.0)
# plt.plot(-wb_int_total, zC[:], 
# label = "buoyancy flux", color = "red", linewidth=2.0)
# plt.plot(-wb_int_total_direct_calculation, zC[:], 
# label = "buoyancy flux direct", color = "green", linewidth=2.0, linestyle="--")
# plt.plot(∇κ∇B_int, zC[:], 
# label = "diffusion", color = "orange", linestyle="-", linewidth=2.0)
# plt.plot(meanflow_transport, zC[:], 
# label = "mean flow", color = "purple", linestyle="-", linewidth=2.0)
# # Create a shaded patch for the topography
# xlims = plt.xlim()  # Get current x limits
# topo_top = 194
# plt.fill_between([xlims[1], xlims[2]], [0, 0], [zC[topo_top], zC[topo_top]], 
#                 color="brown", alpha=0.3) 
# # end
# legend(loc="lower right", fontsize=8, framealpha=.93)
# # legend(loc="best", bbox_to_anchor=(.75, .25), frameon=true, ncol=1, fontsize=12,
# #         handlelength=1.4, columnspacing=.9, framealpha=.93)
# grid("on")
# xlabel(L"[m^4s^{-3}]")
# # end
# ylabel("Height above mean slope (m)")
# ylim(0,2250)
# title("Integrated buoyancy budget")

# plt.minorticks_on()
# t_begin = Int(round(t[1]./(2*pi/1.4e-4)))
# t_end = Int(round(t[2]./(2*pi/1.4e-4)))
# savefig(string("output/", simname, "/buoyancy_budget_volume_integral_snapshot_avg_wB",t_begin,"-",t_end,"TP.png"))











# create a plot that only plots for the tendency, advection and diffusion terms
# dbdt = (b2[:, :, :, 2] .- b2[:, :, :, 1]) ./ (t_3D[end_time]-t_3D[begin_time])

# using PyPlot
# close("all")
# fig = figure(figsize=(8, 6))
# plt.rcParams["font.size"] = 16

# # Extract the data points
# div_uB_point = div_uB[250, 100, :]
# diff_point = ∇κ∇B[250, 100, :]
# dbdt_point = dbdt[250, 100, :]

# # Get the z-coordinates for the specific point
# z_coords = zC[1:k]

# # Plot all three variables
# plt.plot(div_uB_point, z_coords, label="Advection flux", linewidth=2.0, color="red")
# plt.plot(diff_point, z_coords, label="Diffusion", linewidth=2.0, color="orange")
# plt.plot(diff_point.-div_uB_point, z_coords, label="rhs", linewidth=2.0, color="black")
# plt.plot(dbdt_point, z_coords, label="Tendency", linewidth=2.0, color="gray",linestyle="--")

# plt.xlabel(L"[m^2s^{-3}]")
# plt.ylabel("Height above mean slope (m)")
# plt.grid(true)
# plt.legend(loc="best")
# plt.title("Buoyancy budget at point (250, 500)")


# plt.tight_layout()
# savefig(string("output/", simname, "/buoyancy_budget_point_profile",t_begin,"-",t_end,".png"))


## why is w_avgb_avg_int so large?
# test w̄, B̄, b̄, w̄B̄, w̄b̄
# k=67
# uhat = nanmean(ds_timeavg["uhat"][:,:,1:k,n:n+10],dim=4)
# what = nanmean(ds_timeavg["what"][:,:,1:k+1,n:n+10],dim=4);    # true w
# what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
# b = ds_timeavg["b"][:,:,1:k,n];    # b
# B = nanmean(ds_timeavg["B"][:,:,1:k,n:n+10],dim=4)

# # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
# wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
# u = uhat[:,:,:]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
# w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity

# # mask
# u[u.==0].=NaN
# ∇κ∇B[b.==0].=NaN
# div_uB[b.==0].=NaN
# B[b.==0].=NaN
# b[b.==0].=NaN
# w[w.==0].=NaN


# using PyPlot

# # Create a figure with 5 subplots in a single row
# fig, axs = plt.subplots(1, 5, figsize=(20, 4))

# # Variable names
# var_names = [L"$\overline{w}$", L"$\overline{B}$", L"$\overline{b}$", L"$\overline{w}\overline{B}$", L"$\overline{w}\overline{b}$"]

# # Variables to plot (replace these with your actual variables)
# variables = [w, B, b, w.*B, w.*b]
# clim_ranges = [
# (-0.01,0.01),
# (0,0.001),
# (0,0.0005),
# (-1e-5,1e-5),
# (-4e-6,4e-6)
# ]
# # Plotting loop
# for (i, (var, name, clims)) in enumerate(zip(variables, var_names, clim_ranges))
#     # Select a specific z-level (e.g., 40)
#     im = axs[i].pcolor(xC[:], yC[:], var[:,:,67]', 
#         cmap="gnuplot2",  # You can change the colormap
#         shading="auto",   # Adjust shading as needed
#         vmin=clims[1],  # Set minimum of colorbar
#         vmax=clims[2]   # Set maximum of colorbar
#     )
#     # Add colorbar to each subplot    
#     fig.colorbar(im, ax=axs[i])
#     axs[i].set_title(name)
#     axs[i].set_xlabel("x")
#     axs[i].set_ylabel("y")
# end

# # Adjust layout and display
# plt.tight_layout()
# plt.show()
# savefig("output/tilt/buoyancyflux_w_B_b.png")





# ## this section tests time series of idividual points
#         k=150
#         ∇κ∇B = nanmean(ds_Bbudget["∇κ∇B"][:,:,k,:],dim=(1,2))
#         div_uB = nanmean(ds_Bbudget["div_uB"][:,:,k,:],dim=(1,2))
        
#         # mask
#         uhat[uhat.==0].=NaN
#         ∇κ∇B[b.==0].=NaN
#         div_uB[b.==0].=NaN
#         # compute all terms
        
#         t_diff = diff(t)
#         dBdt = zeros(length(t_diff))
#         for n in 1:length(t)-1
#         B2 = nanmean(ds_3D["B"][:,:,k,n:n+1],dim=(1,2));          # total buoyancy with two timesteps
#         dBdt[n] = (B2[2] .- B2[1]) ./ t_diff[n]
#         @info n
#         end



#         using PyPlot
#     close(gcf())
#     fig = figure(figsize=(8,5))
#     # Set font size
#     plt.rcParams["font.size"] = 20

#     # Plot each time series
#     plt.plot(t[2:end], dBdt[1:end], color = "black", linestyle="-", linewidth=2.5, 
#             label = L"d\langle B\rangle/dt")
#     plt.plot(t[2:end], -div_uB[2:end]+∇κ∇B[2:end], 
#             label = "RHS", color = "red", linestyle="--",linewidth=2.5)
#     savefig("output/tilt/buoyancy_budget_pointwise_test.png")
