# this is a Julia script that shows the PDF of Rig in terms of HAB
using NCDatasets
include("functions/bins_pdf.jl") 
fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=460_threeD.nc"
ds = Dataset(fname,"r")
t = ds["time"][:]
# grids
zC = ds["zC"][:]; Nz=length(zC[:])
zF = ds["zF"][:]; zF = zF[:];
xC = ds["xC"][:]; Nx=length(xC[:])
xF = ds["xF"][:];
yC = ds["yC"][:]; Ny=length(yC[:])
yF = ds["yF"][:];

filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:]
bin_edge1 = 0:8:1500#0
bin_center1 = (bin_edge1[1:end-1] .+ bin_edge1[2:end]) ./ 2
bin_mask1 = hab
z_face = zF

dx = 30
dy = 30
hab = hab[:,1:50:950,:]  # Select specific time steps

# i=1
iter=[1,4,7,10,13]
Rig = ds["Rig"][:,1:50:950,:,iter]  
Rig_cen = (Rig[:,:,1:end-1,:] .+ Rig[:,:,2:end,:])/2

Rig_cen[Rig_cen .>10 .|| Rig_cen .< -10 .|| abs.(Rig_cen).<1e-15] .= NaN  # Set values outside the range to NaN
pdf_values, hab_centers, rig_centers = bins_pdf(Rig_cen, bin_edge1, hab; 
                                             dx=dx, dy=dy, z_face=z_face,
                                             n_pdf_bins=900, 
                                             pdf_limits=(-10, 10))

# For plotting with PyPlot:
using PyPlot
close("all")
figure(figsize=(6, 4))
pcolormesh(rig_centers, hab_centers, (pdf_values[:,:,1]), cmap="cividis")
# clim(-2.5,-1.5)
colorbar(label="PDF")
# contour(rig_centers, hab_centers, (pdf_values[:,:,1]), levels=4, colors="black", linewidths=0.5)

plot([0.25, 0.25], [0, 1500], color="yellow", linestyle="--",linewidth=1.3)
xlabel(L"Ri_g")
ylabel("hab [m]")
xlim(-1, 5)
ylim(0, 1500)
minorticks_on()
# title("PDF of Richardson Number vs Height Above Bottom")
savefig(string("output/tilt/Rig_hab_PDF_",iter,".png"), dpi=300, bbox_inches="tight")





##### focus on canyon
hab = ds_hab["hab"][:,:,:]
lower_bound = argmin(abs.(yC .- 10e3))
higher_bound = argmin(abs.(yC .- 20e3))
hab = hab[:,lower_bound:10:higher_bound,:]  # Select specific time steps

# i=1
iter=[1,4,7,10,13] # including 1 tidal cycle, and other tidal cycles look similar
Rig = ds["Rig"][:,lower_bound:10:higher_bound,:,iter]  
Rig_cen = (Rig[:,:,1:end-1,:] .+ Rig[:,:,2:end,:])/2

Rig_cen[Rig_cen .>10 .|| Rig_cen .< -10 .|| abs.(Rig_cen).<1e-15] .= NaN  # Set values outside the range to NaN
pdf_values_canyon, hab_centers, rig_centers = bins_pdf(Rig_cen, bin_edge1, hab; 
                                             dx=dx, dy=dy, z_face=z_face,
                                             n_pdf_bins=900, 
                                             pdf_limits=(-10, 10))


##### focus on flanks
hab = ds_hab["hab"][:,:,:]
lower_bound = argmin(abs.(yC .- 10e3))
higher_bound = argmin(abs.(yC .- 20e3))
hab = hab[:,vcat(1:10:lower_bound,higher_bound:10:end),:]  # Select specific time steps

# i=1
iter=[1,4,7,10,13] # including 1 tidal cycle, and other tidal cycles look similar
Rig = ds["Rig"][:,vcat(1:10:lower_bound,higher_bound:10:end),:,iter]  
Rig_cen = (Rig[:,:,1:end-1,:] .+ Rig[:,:,2:end,:])/2

Rig_cen[Rig_cen .>10 .|| Rig_cen .< -10 .|| abs.(Rig_cen).<1e-15] .= NaN  # Set values outside the range to NaN
pdf_values_flanks, hab_centers, rig_centers = bins_pdf(Rig_cen, bin_edge1, hab; 
                                             dx=dx, dy=dy, z_face=z_face,
                                             n_pdf_bins=900, 
                                             pdf_limits=(-10, 10))
# Create a figure with two subplots for canyon and flanks
using PyPlot
close("all")
fig, (ax1, ax2) = subplots(1, 2, figsize=(8, 4), sharey=true)

# Canyon subplot
pcm1 = ax1.pcolormesh(rig_centers, hab_centers, (pdf_values_canyon[:,:,1]), cmap="magma")
ax1.contour(rig_centers, hab_centers, (pdf_values_canyon[:,:,1]), levels=[0,0.002,0.004,0.006,0.008,0.01], colors="black", linewidths=0.5)
ax1.plot([0.25, 0.25], [0, 1500], color="yellow", linestyle="--", linewidth=1.3)
ax1.set_xlabel(L"Ri_g")
ax1.set_ylabel("hab [m]")
ax1.set_xlim(-1, 5)
ax1.set_ylim(0, 1000)
ax1.minorticks_on()
ax1.set_title("Canyon")

# Flanks subplot
pcm2 = ax2.pcolormesh(rig_centers, hab_centers, (pdf_values_flanks[:,:,1]), cmap="magma")
ax2.contour(rig_centers, hab_centers, (pdf_values_flanks[:,:,1]), levels=[0,0.002,0.004,0.006,0.008,0.01], colors="black", linewidths=0.5)
ax2.plot([0.25, 0.25], [0, 1500], color="yellow", linestyle="--", linewidth=1.3)
ax2.set_xlabel(L"Ri_g")
ax2.set_xlim(-1, 5)
ax2.set_ylim(0, 1000)
ax2.minorticks_on()
ax2.set_title("Flanks")

# Set same color limits for both plots
pcm1.set_clim(0, 0.008)
pcm2.set_clim(0, 0.008)

# Add a single colorbar for both subplots
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(pcm2, cax=cbar_ax, label="PDF")

# Adjust layout to make room for colorbar
fig.subplots_adjust(right=0.9)
savefig(string("output/tilt/Rig_hab_PDF_canyon_flanks_",iter,".png"), dpi=300, bbox_inches="tight")



##### analyze in terms of dissipation rate

##### focus on canyon
hab = ds_hab["hab"][:,:,:]
lower_bound = argmin(abs.(yC .- 10e3))
higher_bound = argmin(abs.(yC .- 20e3))
hab = hab[:,lower_bound:10:higher_bound,:]  # Select specific time steps

# i=1
iter=[1,4,7,10,13] # including 1 tidal cycle, and other tidal cycles look similar
Rig = ds["Rig"][:,lower_bound:10:higher_bound,:,iter]  
Rig_cen = (Rig[:,:,1:end-1,:] .+ Rig[:,:,2:end,:])/2

Rig_cen[Rig_cen .>10 .|| Rig_cen .< -10 .|| abs.(Rig_cen).<1e-15] .= NaN  # Set values outside the range to NaN
pdf_values_canyon, hab_centers, rig_centers = bins_pdf(Rig_cen, bin_edge1, hab; 
                                             dx=dx, dy=dy, z_face=z_face,
                                             n_pdf_bins=900, 
                                             pdf_limits=(-10, 10))


##### focus on flanks
hab = ds_hab["hab"][:,:,:]
lower_bound = argmin(abs.(yC .- 10e3))
higher_bound = argmin(abs.(yC .- 20e3))
hab = hab[:,vcat(1:10:lower_bound,higher_bound:10:end),:]  # Select specific time steps

# i=1
iter=[1,4,7,10,13] # including 1 tidal cycle, and other tidal cycles look similar
Rig = ds["Rig"][:,vcat(1:10:lower_bound,higher_bound:10:end),:,iter]  
Rig_cen = (Rig[:,:,1:end-1,:] .+ Rig[:,:,2:end,:])/2

Rig_cen[Rig_cen .>10 .|| Rig_cen .< -10 .|| abs.(Rig_cen).<1e-15] .= NaN  # Set values outside the range to NaN
pdf_values_flanks, hab_centers, rig_centers = bins_pdf(Rig_cen, bin_edge1, hab; 
                                             dx=dx, dy=dy, z_face=z_face,
                                             n_pdf_bins=900, 
                                             pdf_limits=(-10, 10))
# Create a figure with two subplots for canyon and flanks
using PyPlot
close("all")
fig, (ax1, ax2) = subplots(1, 2, figsize=(8, 4), sharey=true)

# Canyon subplot
pcm1 = ax1.pcolormesh(rig_centers, hab_centers, (pdf_values_canyon[:,:,1]), cmap="magma")
ax1.contour(rig_centers, hab_centers, (pdf_values_canyon[:,:,1]), levels=[0,0.002,0.004,0.006,0.008,0.01], colors="black", linewidths=0.5)
ax1.plot([0.25, 0.25], [0, 1500], color="yellow", linestyle="--", linewidth=1.3)
ax1.set_xlabel(L"Ri_g")
ax1.set_ylabel("hab [m]")
ax1.set_xlim(-1, 5)
ax1.set_ylim(0, 1000)
ax1.minorticks_on()
ax1.set_title("Canyon")

# Flanks subplot
pcm2 = ax2.pcolormesh(rig_centers, hab_centers, (pdf_values_flanks[:,:,1]), cmap="magma")
ax2.contour(rig_centers, hab_centers, (pdf_values_flanks[:,:,1]), levels=[0,0.002,0.004,0.006,0.008,0.01], colors="black", linewidths=0.5)
ax2.plot([0.25, 0.25], [0, 1500], color="yellow", linestyle="--", linewidth=1.3)
ax2.set_xlabel(L"Ri_g")
ax2.set_xlim(-1, 5)
ax2.set_ylim(0, 1000)
ax2.minorticks_on()
ax2.set_title("Flanks")

# Set same color limits for both plots
pcm1.set_clim(0, 0.008)
pcm2.set_clim(0, 0.008)

# Add a single colorbar for both subplots
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(pcm2, cax=cbar_ax, label="PDF")

# Adjust layout to make room for colorbar
fig.subplots_adjust(right=0.9)
savefig(string("output/tilt/Rig_hab_PDF_canyon_flanks_",iter,".png"), dpi=300, bbox_inches="tight")

















# # Add 1D PDF of Rig for all data points
# using PyPlot

# # Collect all Rig values across all time steps
# all_rig_values = Float64[]
# for i in 1:10#:2#length(t)
# # i=
#     Rig = ds["Rig"][:,:,:,i]
#     Rig[Rig .>10 .|| Rig .< -10] .= NaN  # Set values outside the range to NaN
#     # Filter out NaNs and Infs
#     valid_rig = filter(x -> isfinite(x), Rig[:])
#     append!(all_rig_values, valid_rig)
#     @info i
# end

# # Create figure for 1D PDF
# figure(figsize=(10, 6))
# # Use histogram with density=true to get PDF
# hist(all_rig_values, bins=100, density=true, alpha=0.7)
# xlabel("Richardson Number")
# ylabel("Probability Density")
# title("1D PDF of Richardson Number")
# grid(true, alpha=0.3)

# # Add vertical line at Rig=0.25 (critical value)
# axvline(x=0.25, color="r", linestyle="--", label="Critical Rig=0.25")
# legend()

# # Set reasonable x-limits if needed
# xlim(-10, 10)

# # Save the figure
# savefig("output/tilt/Rig_1D_PDF.png", dpi=300, bbox_inches="tight")
# println("1D PDF successfully created and saved.")
