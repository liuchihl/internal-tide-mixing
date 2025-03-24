## this script plots water mass transformation results computed in water_mass_transformation.jl
     using NCDatasets
     # load data
     simname = "tilt"
     θ = simname == "tilt" ? 3.6e-3 : 0
     tᶠ = 460 
     
     ds_total = Dataset(string("output/",simname,"/WMT_total_tᶠ=",tᶠ,".nc"),"r")
     ∇κ∇B_t = ds_total["∇κ∇B_t"][:,:,1:2:end]
     div_uB_t = ds_total["div_uB_t"][:,:,1:2:end]
     
    #  ds_canyon = Dataset(string("output/",simname,"/WMT_canyon_120",".nc"),"r")
    #  ∇κ∇B_c = ds_canyon["∇κ∇B_t"][:,:,1]
    #  div_uB_c = ds_canyon["div_uB_t"][:,:,1]
     
    #  ds_flank = Dataset(string("output/",simname,"/WMT_flanks_120",".nc"),"r")
    #  ∇κ∇B_f = ds_flank["∇κ∇B_t"][:,:,1]
    #  div_uB_f = ds_flank["div_uB_t"][:,:,1]
     
     hab = ds_total["bin_center1"][:]
     b_bin = ds_total["bin_center2"][:]
     Ly = 30e3
     Lx = Ly/2
     A = Lx*Ly
## plot diffusive and advective terms together with respect to HAB with three different regions
    using PyPlot 
    using Statistics
    close(gcf())
    # Create a figure and an array of subplots
    fig, axs = subplots(1, 2, figsize=(6, 3.5))  # 1 row, 2 columns

    colors = [150 148 255;136 194 115;255 41 41]./255

    # line2,=axs.plot(-dropdims(mean(div_uB_t,dims=2),dims=(2))*1e-3, hab, label="", color="red", linestyle="-",marker="",markersize=3)
    line1,=axs[1].plot(dropdims(mean(∇κ∇B_t,dims=2),dims=(2))*1e-3, hab, label="", color=colors[1,:], marker=".",markersize=2, alpha=0.5)
    line1,=axs[1].plot(mean(dropdims(mean(∇κ∇B_t,dims=2),dims=(2)),dims=2)*1e-3, hab, label="", color=:black, marker=".",markersize=6)
    
    # # line4,=axs.plot(-dropdims(mean(div_uB_c,dims=2),dims=(2))*1e-3*1/3, hab, label="", color="red", linestyle="--", marker="",markersize=3)
    # line3,=axs[1].plot(dropdims(mean(∇κ∇B_c,dims=2),dims=(2))*1e-3, hab, label="", color=colors[2,:], linestyle="--", marker=".",markersize=6)
    
    # # line6,=axs.plot(-dropdims(mean(div_uB_f,dims=2),dims=(2))*1e-3./(2/3*A), hab, label="", color="red", linestyle=":", marker="",markersize=3)
    # line5,=axs[1].plot(dropdims(mean(∇κ∇B_f,dims=2),dims=(2))*1e-3, hab, label="", linestyle=":", color=colors[3,:], marker=".",markersize=6)
    axs[1].set_ylabel("HAB (m)")
    axs[1].set_xlabel("mSv")

    # axs[1].legend([line1, line3, line5], title=L"\mathcal{E}^{diff}",
    # ["Total","Canyon","Flanks"], loc="upper right")
    axs[1].grid(true)
    axs[1].set_ylim(0,70)
    axs[1].minorticks_on()

    # plot histogram
    # ∇κ∇B_t_BBL = mean(sum(∇κ∇B_t[1:3,:]*1e-3,dims=1))
    # # ∇κ∇B_c_BBL = mean(sum(∇κ∇B_c[1:3,:]*1e-3,dims=1))
    # # ∇κ∇B_f_BBL = mean(sum(∇κ∇B_f[1:3,:]*1e-3,dims=1))

    # data1 = [∇κ∇B_t_BBL,∇κ∇B_c_BBL,∇κ∇B_f_BBL]
    # data2 = [∇κ∇B_t_BBL,∇κ∇B_c_BBL/(1/3),∇κ∇B_f_BBL/(2/3)]
    # labels = ["Total","Canyon","Flanks"]
    # bar1=axs[2].bar(labels, data2, edgecolor = "black",color=colors,alpha=0.2)
    # bar2=axs[2].bar(labels, data1, edgecolor = "black",color=colors,alpha=1, width=0.5)
    # axs[2].set_title("Transport in BBL (mSv)")
    # axs[2].set_ylim(0,220)
    # axs[2].legend([bar1[1],bar2[1]],[L"\mathcal{E}_{BBL}^{diff}\times A_R/A",L"\mathcal{E}_{BBL}^{diff}"],loc="upper left")
    

    tight_layout()
    gcf()   
    savefig(string("output/",simname,"/water_mass_transformation_diffusive_flux",".png"),dpi=250)


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
