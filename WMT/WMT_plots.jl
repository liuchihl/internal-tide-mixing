## this script plots water mass transformation results computed in water_mass_transformation.jl
     using NCDatasets
     # load data
     slope = "tilt"
     timerange = "80-120" 
     θ=3.6e-3
     ds_read = Dataset(string("output/",slope,"/WMT_120_added_diffusivity",".nc"),"r")
     ∇κ∇B_t = ds_read["∇κ∇B_t"][:,:,1]
     div_uB_t = ds_read["div_uB_t"][:,:,1]
     hab = ds_read["bin_center1"][:]
     b_bin = ds_read["bin_center2"][:]
     

## plot diffusive and advective terms together with respect to HAB
    using PyPlot 
    using Statistics
    close(gcf())
    # Create a figure and an array of subplots
    fig, axs = subplots(1, 1, figsize=(3.5, 3.5))  # 1 row, 2 columns

    line1,=axs.plot(dropdims(mean(∇κ∇B_t,dims=2),dims=(2))*1e-3, hab, label="", color=[71,147,175]./255, marker=".",markersize=6)
    line2,=axs.plot(dropdims(mean(div_uB_t,dims=2),dims=(2))*1e-3, hab, label="", color="red", marker=".",markersize=6)
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
    savefig(string("output/",slope,"/water_mass_transformation_terms_120_together_added_diffusivity",".png"),dpi=200)

    
## 
    close(gcf())
    fig, axs = subplots(1, 2, figsize=(8, 5))  # 1 row, 2 columns

    cmap = PyPlot.get_cmap("coolwarm") 
    pc1 = axs[1].pcolor(b_bin,hab,∇κ∇B_z,cmap=cmap)
    pc1.set_clim(-1000,1000)
    axs[1].set_ylim(0,250)
    axs[1].set_xlabel("B")
    axs[1].set_ylabel("HAB (m)")
    axs[1].set_title(L"\frac{\partial}{\partial B}\iiint_V ∇⋅κ∇B ~dV")
        # colorbar(pc1, ax=axs[1])  # Add colorbar to first subplot

    cmap = PyPlot.get_cmap("coolwarm") 
    pc2 = axs[2].pcolor(b_bin,hab,div_uB_z,cmap=cmap)
    pc2.set_clim(-1500,1500)
    axs[2].set_ylim(0,250)
    colorbar(pc2, ax=axs[2])  # Add colorbar to first subplot
    axs[2].set_xlabel("B")
    axs[2].set_title(L"\frac{\partial}{\partial B}\iiint_V ∇⋅(\mathbf{u}B) ~dV")
    gcf()
    savefig(string("output/",slope,"/water_mass_transformation_terms_pcolor",".png"),dpi=100)
    
    
    
## plot the heatmap of ∇κ∇B 
    using NCDatasets
    slope = "tilt"
    timerange = "100-120"
    θ=3.6e-3
    filename_3D = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD.nc")
    ds_3D = Dataset(filename_3D,"r")
    filename_field = string("output/", slope, "/internal_tide_theta=",θ,"_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
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
