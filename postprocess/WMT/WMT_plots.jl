## this script plots water mass transformation results computed in water_mass_transformation.jl


using NCDatasets
using NaNStatistics
# load data
simname = "tilt"
θ = simname == "tilt" ? 3.6e-3 : 0


# 10 TP average

# total region
# there are two files to be loaded and concatenated (because I forgot to save tf = 451.5, so I save that separately as part 1)

ds_total_p1 = Dataset(string("output/",simname,"/WMT_total_tᶠ=",451.5,"_#bintervals=17_instantaneous.nc"),"r") # from 451.0 to 451.5
ds_total_p2 = Dataset(string("output/",simname,"/WMT_total_tᶠ=",461.0,"_#bintervals=17_instantaneous.nc"),"r") # from 452.0 to 461.0

# Concatenate data along the time dimension
∇κ∇B_t_p1 = ds_total_p1["∇κ∇B_t"][:,:,:]  # [z_TF, b_bin, time]
∇κ∇B_t_p2 = ds_total_p2["∇κ∇B_t"][:,:,:]  # [z_TF, b_bin, time]
∇κ∇B_t = cat(∇κ∇B_t_p1, ∇κ∇B_t_p2, dims=3)  # Concatenate along the third dimension

∇κ∇B_b_t_p1 = ds_total_p1["∇κ∇B_b_t"][:,:]  # [b_bin, time]
∇κ∇B_b_t_p2 = ds_total_p2["∇κ∇B_b_t"][:,:]  # [b_bin, time]
∇κ∇B_b_t = cat(∇κ∇B_b_t_p1, ∇κ∇B_b_t_p2, dims=2)  # Concatenate along the second dimension

# Compute mean and other metrics
∇κ∇B_t_mean = nanmean(∇κ∇B_t, dim=3)  # [z_TF, b_bin]
∇κ∇B_b_t_mean = nanmean(∇κ∇B_b_t, dim=2)  # [b_bin]

# Finding the positive values of ∇κ∇B_t
SML_first = findfirst(nanmean(∇κ∇B_t_mean, dim=2) .< 0.0)  # Find the first negative value in the mean profile
SML_end = SML_first + findfirst(nanmean(∇κ∇B_t_mean[SML_first:end, :], dim=2) .> 0) - 2
E_BBL_t = nansum(nanmean(∇κ∇B_t_mean[1:SML_first-1, :], dim=2)) * 1e-3
E_SML_t = nansum(nanmean(∇κ∇B_t_mean[SML_first:SML_end, :], dim=2)) * 1e-3
z_TF = ds_total_p1["bin_center1"][:]


## profiles of WMT vs HAB 
Ly = 30e3
Lx = Ly/2
A = Lx*Ly
## plot diffusive and advective terms together with respect to HAB with three different regions
using PyPlot 
using Statistics
close(gcf())
# Create a figure and an array of subplots
fig, axs = subplots(1, 1, figsize=(4, 3))  # 1 row, 1 column

axs.plot(nanmean(∇κ∇B_t_mean,dim=2)*1e-3, z_TF, color="black", marker=".",markersize=4)
axs.set_ylabel("HAB [m]")
axs.set_xlabel(L"\langle\overline{\delta\mathcal{E}}\rangle_B~\mathrm{[mSv~per~8~m~bin]}")
# axs[1].legend([line1, line3], title=L"\mathcal{E}^{diff}",
# ["Total","Canyon"], loc="upper right")
axs.grid(true)
axs.set_ylim(0,800)
axs.minorticks_on()
axs.set_xlim(-1e2,1e2)
# axs[1].minorticks_on()
axs.set_xscale("symlog", linthresh=10^(0))
# Add (a) label to the first subplot
# axs[1].text(0.05, 0.95, "(a)", transform=axs[1].transAxes, fontsize=10, va="top")
tight_layout()
gcf()   
savefig(string("output/",simname,"/water_mass_transformation_diffusive_flux_HAB_totalregion",".png"),dpi=250)
println(string("output/",simname,"/water_mass_transformation_diffusive_flux_HAB_totalregion",".png"))










using NCDatasets
using NaNStatistics
# load data
simname = "tilt"
θ = simname == "tilt" ? 3.6e-3 : 0
tᶠ = 457.0 

# 5 TP average
# total
ds_total = Dataset(string("output/",simname,"/WMT_total_tᶠ=",tᶠ,"_#bintervals=17_instantaneous.nc"),"r")
∇κ∇B_t = nanmean(ds_total["∇κ∇B_t"][:,:,1:end],dim=3)  # [z_TF, b_bin]
∇κ∇B_b_t = nanmean(ds_total["∇κ∇B_b_t"][:,:],dim=2)    # [b_bin]
#1) finding the positive values of ∇κ∇B_t would be one way 
SML_first = findfirst(nanmean(∇κ∇B_t,dim=2) .< 0.0)  # find the first negative value in the mean profile
SML_end = SML_first + findfirst(nanmean(∇κ∇B_t[SML_first:end,:],dim=2) .>0) - 2
E_BBL_t = nansum(nanmean(∇κ∇B_t[1:SML_first-1,:],dim=2))*1e-3
E_SML_t = nansum(nanmean(∇κ∇B_t[SML_first:SML_end,:],dim=2))*1e-3
#2) 
# E_BBL_t = sum(nanmean(∇κ∇B_t[1:3,:],dim=2))*1e-3
# E_SML_t = sum(nanmean(ifelse.(∇κ∇B_t[4:29,:].<0.0, ∇κ∇B_t[4:29,:], 0.0),dim=2))*1e-3

ds_canyon = Dataset(string("output/",simname,"/WMT_canyon_tᶠ=",tᶠ,"_#bintervals=17_instantaneous.nc"),"r")
∇κ∇B_c = nanmean(ds_canyon["∇κ∇B_t"][:,:,1:end],dim=3)
∇κ∇B_b_c = nanmean(ds_canyon["∇κ∇B_b_t"][:,:],dim=2)
SML_first = findfirst(nanmean(∇κ∇B_c,dim=2) .< 0.0)  # find the first negative value in the mean profile
SML_end = SML_first + findfirst(nanmean(∇κ∇B_c[SML_first:end,:],dim=2) .>0) - 2
E_BBL_c = nansum(nanmean(∇κ∇B_c[1:SML_first-1,:],dim=2))*1e-3
E_SML_c = nansum(nanmean(∇κ∇B_c[SML_first:SML_end,:],dim=2))*1e-3

ds_flank = Dataset(string("output/",simname,"/WMT_flanks_tᶠ=",tᶠ,"_#bintervals=17_instantaneous.nc"),"r")
∇κ∇B_f = nanmean(ds_flank["∇κ∇B_t"][:,:,1:end],dim=3)
∇κ∇B_b_f = nanmean(ds_flank["∇κ∇B_b_t"][:,:],dim=2)
SML_first = findfirst(nanmean(∇κ∇B_f,dim=2) .< 0.0)  # find the first negative value in the mean profile
SML_end = SML_first + findfirst(nanmean(∇κ∇B_f[SML_first:end,:],dim=2) .>0) - 2
E_BBL_f = nansum(nanmean(∇κ∇B_f[1:SML_first-1,:],dim=2))*1e-3
E_SML_f = nansum(nanmean(∇κ∇B_f[SML_first:SML_end,:],dim=2))*1e-3

t = ds_total["t"][:]
z_TF = ds_total["bin_center1"][:]
b_bin = ds_total["bin_center2"][:]



## plot pcolor of ∇κ∇B_t
using PyPlot
close(gcf())
# Create a figure with three subplots
fig, axs = subplots(1, 3, figsize=(9, 3), sharey=true, constrained_layout=true)

# Total plot
pc1 = axs[1].pcolor(b_bin*1e3, z_TF, ∇κ∇B_t[:,:]*1e-3, cmap="RdBu_r", shading="auto")
pc1.set_clim((-5e2, 5e2).*1e-3)
axs[1].set_title("Total")
axs[1].set_xlabel("Buoyancy (×10⁻³ m/s²)")
axs[1].set_ylabel("HAB (m)")

# Canyon plot
pc2 = axs[2].pcolor(b_bin*1e3, z_TF, ∇κ∇B_c[:,:]*1e-3, cmap="RdBu_r", shading="auto")
pc2.set_clim((-5e2, 5e2).*1e-3)
axs[2].set_title("Canyon")
axs[2].set_xlabel("Buoyancy (×10⁻³ m/s²)")

# Flank plot
pc3 = axs[3].pcolor(b_bin*1e3, z_TF, ∇κ∇B_f[:,:]*1e-3, cmap="RdBu_r", shading="auto")
pc3.set_clim((-5e2, 5e2).*1e-3)
axs[3].set_title("Flanks")
axs[3].set_xlabel("Buoyancy (×10⁻³ m/s²)")

# Add colorbar
cbar = fig.colorbar(pc1, ax=axs, orientation="vertical", pad=0.01, shrink=0.8)
cbar.set_label(L"\mathcal{E}~\mathrm{(HAB,B)} [m Sv]")
subplots_adjust(left=0.08, right=0.85, wspace=0.1)

gcf()
savefig(string("output/", simname, "/∇κ∇B_comparison.png"), dpi=250)



### plot profiles of WMT vs buoyancy 
using PyPlot
using Statistics
close(gcf())
# Create a figure and an array of subplots
figure(figsize=(7, 3.2))  # 1 row, 2 columns
plot(∇κ∇B_b_t*1e-3, b_bin, label="", color=[71,147,175]./255, marker=".",markersize=4)
plot(∇κ∇B_b_c*1e-3, b_bin, label="", color=[136,194,115]./255, linestyle="--", marker=".",markersize=4)
plot(∇κ∇B_b_f*1e-3, b_bin, label="", color=[255,41,41]./255, linestyle=":", marker=".",markersize=4)
xlabel("∇κ∇B [m³/s]")
ylabel("Buoyancy [m/s²]")
legend(["Total","Canyon","Flanks"], loc="upper left")
grid(true)
# ylim(-0.001,0.001)
tight_layout()
gcf()
savefig(string("output/",simname,"/water_mass_transformation_buoyancy",".png"),dpi=250)



## profiles of WMT vs HAB 
Ly = 30e3
Lx = Ly/2
A = Lx*Ly
## plot diffusive and advective terms together with respect to HAB with three different regions
using PyPlot 
using Statistics
close(gcf())
# Create a figure and an array of subplots
fig, axs = subplots(1, 2, figsize=(7, 3.2))  # 1 row, 2 columns

colors = [150 148 255;136 194 115;255 41 41]./255

line1,=axs[1].plot(nanmean(∇κ∇B_t,dim=2)*1e-3, z_TF, label="", color=colors[1,:], marker=".",markersize=4)
line2,=axs[1].plot(nanmean(∇κ∇B_c,dim=2)*1e-3, z_TF, label="", color=colors[2,:], linestyle="--", marker=".",markersize=4)
line3,=axs[1].plot(nanmean(∇κ∇B_f,dim=2)*1e-3, z_TF, label="", linestyle=":", color=colors[3,:], marker=".",markersize=4)
axs[1].set_ylabel("HAB [m]")
axs[1].set_xlabel(L"\langle\overline{\delta\mathcal{E}}\rangle_B~\mathrm{[mSv~per~8~m~bin]}")
# axs[1].legend([line1, line3], title=L"\mathcal{E}^{diff}",
# ["Total","Canyon"], loc="upper right")
axs[1].legend([line1, line2, line3],
["Total","Canyon","Flanks"], loc="upper right", fontsize=8)
axs[1].grid(true)
axs[1].set_ylim(0,800)
# axs[1].minorticks_on()
axs[1].set_xscale("symlog", linthresh=10^(0))
# Add (a) label to the first subplot
axs[1].text(0.05, 0.95, "(a)", transform=axs[1].transAxes, fontsize=10, va="top")

# Calculate BBL contribution after the plot
# ∇κ∇B_t_BBL = mean(sum(mean(∇κ∇B_t[1:2,:,:],dims=3)*1e-3,dims=1))
# ∇κ∇B_c_BBL = mean(sum(mean(∇κ∇B_c[1:2,:,:],dims=3)*1e-3,dims=1))
# ∇κ∇B_f_BBL = mean(sum(∇κ∇B_f[1:3,:]*1e-3,dims=1))

# Create data arrays for BBL and SML
bbl_data = [E_BBL_t, E_BBL_c*3, E_BBL_f*3/2]
sml_data = .-[E_SML_t, E_SML_c*3, E_SML_f*3/2]

# Set up x positions for the bars
x = [1, 2, 3]
width = 0.26
labels = ["Total", "Canyon", "Flanks"]
# Create the side-by-side bar plot
bars1 = axs[2].bar(x .- width/2, bbl_data, width, edgecolor="black", color=colors, alpha=1)
bars2 = axs[2].bar(x .+ width/2, sml_data, width, edgecolor="black", color=colors, alpha=0.5)

using PyCall
patches = pyimport("matplotlib.patches")

# Create legend patches 
patch1 = patches.Patch(color="black", alpha=1, edgecolor="none")
patch2 = patches.Patch(color="black", alpha=0.4, edgecolor="none")

# Configure the plot
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels)
axs[2].legend([patch1, patch2], 
[L"\sum_{\substack{\text{HAB} \\ \text{bins}}}\langle\overline{\delta\mathcal{E}_{BBL}}\rangle_B\times \frac{\mathcal{A}}{\mathcal{A}_\mathcal{R}}",
 L"-\sum_{\substack{\text{HAB} \\ \text{bins}}}\langle\overline{\delta\mathcal{E}_{SML}}\rangle_B\times \frac{\mathcal{A}}{\mathcal{A}_\mathcal{R}}"],              loc="upper center", fontsize=8, edgecolor="black", facecolor="none", frameon=false, framealpha=1)
axs[2].set_ylabel("[mSv]")
# Add (b) label to the second subplot
axs[2].text(0.05, 0.95, "(b)", transform=axs[2].transAxes, fontsize=10, va="top")

# labels = ["Total","Canyon"]
# bar1=axs[2].bar(labels, data2, edgecolor = "black",color=colors,alpha=0.2)
# bar2=axs[2].bar(labels, data1, edgecolor = "black",color=colors,alpha=1, width=0.5)
# axs[2].set_title(L"\mathcal{E}^{diff} (A/A_\mathcal{R}) [mSv]")
# axs[2].set_ylim(0,220)
# axs[2].legend([bar1[1],bar2[1]],[L"\mathcal{E}_{BBL}^{diff}\times A_R/A",L"\mathcal{E}_{BBL}^{diff}"],loc="upper left")

axs[2].set_ylim(0, 200)
tight_layout()
gcf()   
savefig(string("output/",simname,"/water_mass_transformation_diffusive_flux",".png"),dpi=250)

    # Create a quick pcolor plot of ∇κ∇B_t[:,:,1]
    # bin_center2 = (b_bin[1:end-1] .+ b_bin[2:end]) ./ 2
    # close("all")
    # figure(figsize=(4, 3))
    # pc = pcolor(b_bin*1e3, hab, ∇κ∇B_t[:,:], cmap="RdBu_r", shading="auto")
    # colorbar(pc)
    # xlabel("Buoyancy (×10⁻³ m/s²)")
    # ylabel("HAB (m)")
    # ylim(0,100)
    # title(L"\overline{\frac{\partial}{\partial B}∭_{V(B'≤B)}\nabla\cdot(\kappa\nabla B)~dV}")
    # tight_layout()
    # gcf()
    # savefig("output/tilt/∇κ∇B_t.png", dpi=250)


## plot diffusive and advective terms together with respect to HAB
    using PyPlot 
    using Statistics
    close(gcf())
    # Create a figure and an array of subplots
    fig, axs = subplots(1, 1, figsize=(3.5, 3.5))  # 1 row, 2 columns

    line1,=axs.plot(dropdims(mean(∇κ∇B_t[:,:,1],dims=2),dims=(2))*1e-3, hab, label="", color=[71,147,175]./255, marker=".",markersize=6)
    line2,=axs.plot(dropdims(mean(div_uB_t[:,:,1],dims=2),dims=(2))*1e-3, hab, label="", color="red", marker=".",markersize=6)
    axs.set_ylabel("HAB (m)")
    axs.set_xlabel("mSv")
    axs.legend([line1, line2], 
    [L"\frac{\partial}{\partial B}\iiint_{V(B'\leq B)} ∇⋅κ∇B ~dV", 
     L"\frac{\partial}{\partial B}\iiint_{V(B'\leq B)} ∇⋅(\mathbf{u}B) ~dV"],
    loc="upper right")
    axs.grid(true)
    axs.set_ylim(0,200)
    axs.set_xlim(-30,120)
    minorticks_on()
    tight_layout()
    gcf()   
    savefig(string("output/",simname,"/water_mass_transformation_terms",".png"),dpi=250)

    
## 
    close(gcf())
    fig, axs = subplots(1, 2, figsize=(8, 5))  # 1 row, 2 columns

    cmap = PyPlot.get_cmap("coolwarm") 
    pc1 = axs[1].pcolor(b_bin,hab,∇κ∇B_t,cmap=cmap)
    pc1.set_clim(-1000,1000)
    axs[1].set_ylim(0,250)
    axs[1].set_xlabel("B")
    axs[1].set_ylabel("HAB (m)")
    axs[1].set_title(L"\frac{\partial}{\partial B}\iiint_V ∇⋅κ∇B ~dV")
        # colorbar(pc1, ax=axs[1])  # Add colorbar to first subplot

    cmap = PyPlot.get_cmap("coolwarm") 
    pc2 = axs[2].pcolor(b_bin,hab,div_uB_t,cmap=cmap)
    pc2.set_clim(-1500,1500)
    axs[2].set_ylim(0,250)
    colorbar(pc2, ax=axs[2])  # Add colorbar to first subplot
    axs[2].set_xlabel("B")
    axs[2].set_title(L"\frac{\partial}{\partial B}\iiint_V ∇⋅(\mathbf{u}B) ~dV")
    gcf()
    savefig(string("output/",simname,"/water_mass_transformation_terms_pcolor",".png"),dpi=100)
    
    
    
## plot the heatmap of ∇κ∇B 
    using NCDatasets
    simname = "tilt"
    timerange = "100-120"
    θ=3.6e-3
    filename_3D = string("output/", simname, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD.nc")
    ds_3D = Dataset(filename_3D,"r")
    filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
    ds_field = Dataset(filename_field,"r")
   
    # grids
    zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
    Nz=length(zC[:]); 
    xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
    Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

    yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
    Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
    t = ds_3D["time"][:];
    n=40
    j=1
    ∇κ∇B = ds_3D["∇κ∇B"][:,j:j,:,n:n]; 
    # b = ds_field["b"][:,j:j,:,n:n]; 
    ∇κ∇B[∇κ∇B.==0].=NaN


    using PyPlot
    PyPlot.rc("font", size=14)  # Sets base font size
    mpl_colors = PyPlot.pyimport("matplotlib.colors")
    linthresh = 1.0
    close(gcf())
    norm_log = mpl_colors["SymLogNorm"](linthresh=linthresh, vmin=-1e-14, vmax=1e-14, base=10)
    fig = figure(figsize=(13, 6))  # 10 inches wide by 6 inches tall
 
    
    # ax = subplots()
    c = PyPlot.pcolor(xC./1000, zC, ∇κ∇B[:,1,:,1]',cmap="seismic",norm=norm_log)
    cbar = PyPlot.colorbar(c, extend="both")
    cbar.set_label("∇⋅(κ∇B)")
    # tick_locations = [-1e-9, -1e-10, 0, 1e-10, 1e-9]
    # Set the tick locations
    # cbar.set_ticks(tick_locations)
    # Set the background color of the axes to gray
    ax = gca()
    ax.set_facecolor("gray")


    xlabel("x (km)")
    ylabel("z (m)")
    minorticks_on()
    tight_layout()
    PyPlot.draw()
    gcf()

    savefig("output/tilt/heatmap_∇κ∇B_test_constant_kappa.png",dpi=100)

    using PyPlot
    close(gcf)
    fig, ax = subplots()  # Create a new figure and axes
    ax.set_facecolor("gray")  # Set the axes background color
    
    # Plot something simple
    ax.plot([0, 1], [0, 1])
    gcf()
    # Show the plot
    show()
