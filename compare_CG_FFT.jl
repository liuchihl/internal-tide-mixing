using NCDatasets
using PyPlot

include("functions/bins.jl")

# Read the NetCDF files
ds_FFT = Dataset("output/tilt_FFT_poisson_solver/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_0-1_threeD_timeavg.nc")
ds_CG = Dataset("output/tilt_conjugate_gradient_poisson_solver/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_0-1_threeD_timeavg.nc")

θ = 0.0036
zC = ds_CG["zC"][:]; zF = ds_CG["zF"][:];
Nz=length(zC[:]); 

xC = ds_CG["xC"][:]; xF = ds_CG["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_CG["yC"][:]; yF = ds_CG["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
z_face = xF * sin(θ) .+ zF' * cos(θ)     # Cartesian coordinate

t = ds_CG["time"][:];

bin_edge = 0:5:1500#0
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2

# load hab
filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];
bin_mask = hab

# preallocate
Bz_FFT_avg = zeros(length(bin_edge)-1,length(t))
Bz_CG_avg = zeros(length(bin_edge)-1,length(t))
what_FFT_avg = zeros(length(bin_edge)-1,length(t))
what_CG_avg = zeros(length(bin_edge)-1,length(t))
for n in 1:length(t)
     Bz_FFT = ds_FFT["Bz"][:,:,:,n:n];          # Bz from FFT
     what_FFT = ds_FFT["what"][:,:,:,n:n];          # true w FFT
     what_FFT_cen = (what_FFT[:,:,1:end-1,1] .+ what_FFT[:,:,2:end,1])./2 # what at center FFT
     Bz_CG = ds_CG["Bz"][:,:,:,n:n];          # Bz from CG
     what_CG = ds_CG["what"][:,:,:,n:n];          # true w CG
     what_CG_cen = (what_CG[:,:,1:end-1,1] .+ what_CG[:,:,2:end,1])./2 # what at center CG
    # terrain following quantities:

    @time Bz_FFT_avg[:,n], _ = bins(Bz_FFT,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time what_FFT_avg[:,n], _ = bins(what_FFT_cen,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time Bz_CG_avg[:,n], _ = bins(Bz_CG,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time what_CG_avg[:,n], _ = bins(what_CG_cen,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
end

close(ds_FFT)
close(ds_CG)
close(ds_hab)

# This creates a new NetCDF file 
# The mode "c" stands for creating a new file

ds_create = Dataset(string("output/compare_CG_FFT.nc"),"c")
# Define the dimension
defDim(ds_create,"z_TF",length(bin_center))
defDim(ds_create,"t",length(t))
defDim(ds_create, "t_diff", length(t) - 1)  # Define a separate dimension for the derivative

# Define a global attribute
ds_create.attrib["title"] = "Terrain-following averages"
# Define the variables
v1 = defVar(ds_create,"Bz_FFT_avg",Float64,("z_TF","t"))
v1[:,:] = Bz_FFT_avg
v2 = defVar(ds_create,"Bz_CG_avg",Float64,("z_TF","t"))
v2[:,:] = Bz_CG_avg
v3 = defVar(ds_create,"what_FFT_avg",Float64,("z_TF","t"))
v3[:,:] = what_FFT_avg
v4 = defVar(ds_create,"what_CG_avg",Float64,("z_TF","t"))
v4[:,:] = what_CG_avg
v5 = defVar(ds_create,"bin_center",Float64,("z_TF",))
v5[:,1] = bin_center
v6 = defVar(ds_create,"t",Float64,("t",))
v6[:,1] = t   

# write attributes
v1.attrib["units"] = "1/s²"
v2.attrib["units"] = "1/s²"
v3.attrib["units"] = "m/s"
v4.attrib["units"] = "m/s"
v5.attrib["units"] = "m"
v6.attrib["units"] = "s"

close(ds_create)



    

using PyPlot
ds = Dataset("output/compare_CG_FFT.nc")
what_FFT_avg = ds["what_FFT_avg"][:,:]
Bz_FFT_avg = ds["Bz_FFT_avg"][:,:]
what_CG_avg = ds["what_CG_avg"][:,:]
Bz_CG_avg = ds["Bz_CG_avg"][:,:]
t = ds["t"][:]
bin_center = ds["bin_center"][:]
# Create 2x2 Hovmoller plots with shared colorbars
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plt.tight_layout()

# Calculate common color limits
w_max = maximum(abs.([what_FFT_avg; what_CG_avg]))
Bz_max = maximum(abs.([Bz_FFT_avg; Bz_CG_avg]))

# Plot what comparisons
im1 = axs[1,1].pcolor(t, bin_center, what_FFT_avg, 
                         cmap="RdBu_r", 
                         vmin=-0.001, vmax=0.001)
axs[1,1].set_title("True w - FFT")
axs[1,1].set_ylabel("Height above bottom [m]")

axs[1,2].pcolor(t, bin_center, what_CG_avg, 
                   cmap="RdBu_r", 
                   vmin=-0.001, vmax=0.001)
axs[1,2].set_title("True w - CG")

# Plot Bz comparisons
axs[2,1].pcolor(t, bin_center, Bz_FFT_avg, 
                   cmap="RdBu_r", 
                   vmin=0, vmax=Bz_max)
axs[2,1].set_title("dB/dz - FFT")
axs[2,1].set_xlabel("Time [s]")
axs[2,1].set_ylabel("Height above bottom [m]")

im2 = axs[2,2].pcolor(t, bin_center, Bz_CG_avg, 
                        cmap="RdBu_r", 
                        vmin=0, vmax=Bz_max)
axs[2,2].set_title("dB/dz - CG")
axs[2,2].set_xlabel("Time [s]")

# Add colorbars
cbar1 = plt.colorbar(im1, ax=[axs[1,1], axs[1,2]], label="w [m/s]")
cbar1.set_ticks(-0.001:0.0005:0.001)
cbar2 = plt.colorbar(im2, ax=[axs[2,1], axs[2,2]], label="dB/dz [1/s²]")

# plt.tight_layout()
plt.savefig("output/hovmoller_comparison.png", dpi=300, bbox_inches="tight")











# Create 2x2 plot with pcolor and profiles
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))
# plt.tight_layout()

# Calculate color limits for CG plots
w_max_CG = maximum(abs.(what_CG_avg))
Bz_max_CG = maximum(abs.(Bz_CG_avg))

# Left column: pcolor plots
# Plot what CG
im1_CG = axs2[1,1].pcolor(t/3600, bin_center, what_CG_avg, 
                     cmap="RdBu_r", 
                     vmin=-w_max_CG, vmax=w_max_CG)
axs2[1,1].set_title("True w - CG")
axs2[1,1].set_ylabel("Height above bottom [m]")

# Plot Bz CG
im2_CG = axs2[2,1].pcolor(t/3600, bin_center, Bz_CG_avg, 
                     cmap="RdBu_r", 
                     vmin=0, vmax=Bz_max_CG)
axs2[2,1].set_title("dB/dz - CG")
axs2[2,1].set_xlabel("Time [hr]")
axs2[2,1].set_ylabel("Height above bottom [m]")

# Add colorbars for pcolor plots
cbar1_CG = plt.colorbar(im1_CG, ax=axs2[1,1])
# cbar1_CG.set_ticks(-1e-4:5e-5:1e-4)
cbar2_CG = plt.colorbar(im2_CG, ax=axs2[2,1])

# Right column: vertical profiles at final timestep
axs2[1,2].plot(what_CG_avg[:,end], bin_center, "k-", linewidth=2)
axs2[1,2].plot([0,0], [0,1500], "k-", linewidth=2)
axs2[1,2].set_title("w profile at t = 1 tidal period")
# axs2[1,2].set_xlabel("w [m/s]")
axs2[1,2].grid(true)
axs2[1,2].set_ylim(0,1500)

axs2[2,2].plot(Bz_CG_avg[:,end], bin_center, "k-", linewidth=2)
axs2[2,2].set_title("dB/dz profile at t = 1 tidal period")
# axs2[2,2].set_xlabel("dB/dz [1/s²]")
axs2[2,2].grid(true)
axs2[2,2].set_ylim(0,1500)

plt.savefig("output/hovmoller_CG_profiles.png", dpi=300, bbox_inches="tight")



# Create 1x2 plot comparing FFT and CG profiles
fig3, axs3 = plt.subplots(1, 2, figsize=(12, 6))
plt.tight_layout()

# Left subplot: w comparison
axs3[1].plot(what_FFT_avg[:,end], bin_center, "b-", linewidth=2, label="FFT")
axs3[1].plot(what_CG_avg[:,end], bin_center, "r--", linewidth=2, label="CG")
axs3[1].plot([0,0], [0,1500], "k-", linewidth=1)
axs3[1].set_title("w profile at t = 1 tidal period")
axs3[1].set_xlabel("w [m/s]")
axs3[1].set_ylabel("Height above bottom [m]")
axs3[1].grid(true)
axs3[1].set_ylim(0,1500)
axs3[1].legend()

# Right subplot: Bz comparison
axs3[2].plot(Bz_FFT_avg[:,end], bin_center, "b-", linewidth=2, label="FFT")
axs3[2].plot(Bz_CG_avg[:,end], bin_center, "r--", linewidth=2, label="CG")
axs3[2].set_title("dB/dz profile at t = 1 tidal period")
axs3[2].set_xlabel("dB/dz [1/s²]")
axs3[2].grid(true)
axs3[2].set_ylim(0,1500)
axs3[2].legend()

plt.savefig("output/compare_CG_FFT_profiles.png", dpi=300, bbox_inches="tight")

close(ds)


