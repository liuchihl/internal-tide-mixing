
## plot the volume integrated quantities of the buoyancy budget 
# <∂B/∂t> = <-ũ⋅∇B> - <∇⋅(-κ∇B)>

using NCDatasets
using CairoMakie
using Statistics
using NaNStatistics
include("functions/bins.jl")
include("functions/mmderiv.jl")
include("functions/mmintgrl.jl")
    slope = "tilt"
    θ=0.0036

# Decompose advection term
    # <∇⋅(uB)> = <u⋅∇B> = <u>⋅∇<B> + <u'⋅∇B'>
    file = "output/tilt/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_80-120_threeD_timeavg_Bbudget.nc"
    ds_Bbudget = Dataset(file,"r")
    file = "output/tilt/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_80-120_threeD_timeavg.nc"
    ds_timeavg = Dataset(file,"r")
    filename_3D = "output/tilt/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_80-120_threeD.nc"
    ds_3D = Dataset(filename_3D,"r")
    xC = ds_3D["xC"][:]; dx = diff(xC)[1]
    yC = ds_3D["yC"][:]; dy = diff(yC)[1]
    zC = ds_3D["zC"][:]
    # b = ds["b_avg"][:,:]
    # what_avg = ds["what_avg"][:,:]
  
    t = ds_timeavg["time"][:]#/(2*pi/1.4e-4)
    θ = 0.0036
    N = 1e-3
    Lx = 15e3
    Ly = 30e3
    Lz = 2250
    n = 30
    k = 250#argmin(abs.(zC .- 2000))
        ∇κ∇B = nanmean(ds_Bbudget["∇κ∇B"][:,:,1:k,n:n+10],dim=4)
        div_uB = nanmean(ds_Bbudget["div_uB"][:,:,1:k,n:n+10],dim=4)
        uhat = nanmean(ds_timeavg["uhat"][:,:,1:k,n:n+10],dim=4)
        what = nanmean(ds_timeavg["what"][:,:,1:k+1,n:n+10],dim=4);    # true w
        what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
        b = ds_timeavg["b"][:,:,1:k,n];    # b
        B = nanmean(ds_timeavg["B"][:,:,1:k,n:n+10],dim=4)

        # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
        wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
        u = uhat[:,:,:]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
        w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity

        # mask
        u[u.==0].=NaN
        ∇κ∇B[b.==0].=NaN
        div_uB[b.==0].=NaN
        B[b.==0].=NaN
        w[w.==0].=NaN
        # compute all terms
                # mean flow advective flux (east + west)
        meanflow_transport = N^2*Lx*sin(θ)*mmintgrl(zC[1:k],nansum(u[1,:,:].*dx,dim=1))
        wB_int_total = mmintgrl(zC[1:k],nansum(div_uB*dx*dy,dim=(1,2))) .- meanflow_transport
        w_avgB_avg_int = mmintgrl(zC[1:k],nansum(w.*B*dx*dy,dim=(1,2)))
        w_primeB_prime_int = wB_int_total .- w_avgB_avg_int
        ∇κ∇B_int = mmintgrl(zC[1:k],nansum(∇κ∇B*dx*dy,dim=(1,2)))
        t_diff = diff(t)
        B2 = ds_3D["B"][:,:,1:k,[n-1,n+10]];          # total buoyancy with two timesteps
        b = ds_timeavg["b"][:,:,1:k,n-1:n];          # total buoyancy with two timesteps
        B2[b.==0].=NaN
        # dBdt_int = mmintgrl(zC[1:k],nansum(((B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ t_diff[n-1])*dx*dy ,dim=(1,2) ))
        dBdt_int = mmintgrl(zC[1:k],nansum(((B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ (t[n+10]-t[n-1]))*dx*dy ,dim=(1,2) ))
        rhs = -meanflow_transport - wB_int_total + ∇κ∇B_int
        factor = -N^2*Lx*sin(θ)

    
    using PyPlot
    close("all")
    fig = figure(figsize=(6,5))
    # Set font size
    plt.rcParams["font.size"] = 20

    # Plot each time series
    plt.plot(dBdt_int./factor/1e3, zC[1:k], 
            label = L"\frac{\int_0^{z′} \frac{d\langle \bar{B}\rangle}{dt} dz′}{-N^2L_x sin(θ)}", color = "gray", linewidth=2.5)
    plt.plot(rhs./factor/1e3, zC[1:k], 
            label = L"\frac{RHS}{-N^2 L_x sin(θ)}", color = "black", linestyle="--",linewidth=2.5)
    plt.plot(-w_avgB_avg_int./factor/1e3, zC[1:k], 
            label = L"-\frac{\langle \bar{w} \bar{B} \rangle}{-N^2 L_x sin(θ)}", color = [150,148,255]/255, linestyle="-",linewidth=2.5)
    plt.plot(-w_primeB_prime_int./factor/1e3, zC[1:k], 
            label = L"-\frac{\langle \overline{w′B′} \rangle}{-N^2 L_x sin(θ)}", color = [136,194,115]/255, linestyle="-",linewidth=2.5)
    plt.plot(-wB_int_total./factor/1e3, zC[1:k], 
            label = L"\frac{-\langle wB \rangle}{-N^2L_x sin(θ)}", color = "red", linewidth=2.5)
    plt.plot(∇κ∇B_int./factor/1e3, zC[1:k], 
            label = L"\frac{\langle κ\frac{∂B}{∂z}\rangle}{-N^2L_x sin(θ)} ", color = "orange", linestyle=":", linewidth=2.5)
    plt.plot(-meanflow_transport./factor/1e3, zC[1:k], 
            label = L"∬_{A(x;z′≤ z)} u dydz′", color = "yellow", linewidth=2.5)
    legend(loc="upper center", bbox_to_anchor=(.5, .99), frameon=true, ncol=3, fontsize=12,
    handlelength=1.4,columnspacing=.9, framealpha=.93)


            # ax.plot([0,0],[0,200], color = "black", linewidth=1.5)
    grid("on")
    xlabel("Transport (mSv)")
    ylabel("Height above mean slope (m)")
    ylim(0,2500)
    # Set the y-limits and labels
    # set_ylim(0, 200)
    # plt.set_ylabel("Height Above  (m)")
    title("Integrated buoyancy budget")
    plt.xscale("symlog", linthresh=1)

    plt.minorticks_on()

    savefig("output/tilt/buoyancy_budget_volume_integral.png")


## why is w_avgB_avg_int so large?
# test w̄, B̄, b̄, w̄B̄, w̄b̄
        k=67
        uhat = nanmean(ds_timeavg["uhat"][:,:,1:k,n:n+10],dim=4)
        what = nanmean(ds_timeavg["what"][:,:,1:k+1,n:n+10],dim=4);    # true w
        what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
        b = ds_timeavg["b"][:,:,1:k,n];    # b
        B = nanmean(ds_timeavg["B"][:,:,1:k,n:n+10],dim=4)

        # piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
        wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
        u = uhat[:,:,:]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
        w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity

        # mask
        u[u.==0].=NaN
        ∇κ∇B[b.==0].=NaN
        div_uB[b.==0].=NaN
        B[b.==0].=NaN
        b[b.==0].=NaN
        w[w.==0].=NaN

        
        using PyPlot

# Create a figure with 5 subplots in a single row
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

# Variable names
var_names = [L"$\overline{w}$", L"$\overline{B}$", L"$\overline{b}$", L"$\overline{w}\overline{B}$", L"$\overline{w}\overline{b}$"]

# Variables to plot (replace these with your actual variables)
variables = [w, B, b, w.*B, w.*b]
clim_ranges = [
        (-0.01,0.01),
        (0,0.001),
        (0,0.0005),
        (-1e-5,1e-5),
        (-4e-6,4e-6)
]
# Plotting loop
for (i, (var, name, clims)) in enumerate(zip(variables, var_names, clim_ranges))
    # Select a specific z-level (e.g., 40)
    im = axs[i].pcolor(xC[:], yC[:], var[:,:,67]', 
        cmap="gnuplot2",  # You can change the colormap
        shading="auto",   # Adjust shading as needed
        vmin=clims[1],  # Set minimum of colorbar
        vmax=clims[2]   # Set maximum of colorbar
    )
    # Add colorbar to each subplot    
    fig.colorbar(im, ax=axs[i])
    axs[i].set_title(name)
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")
end

# Adjust layout and display
plt.tight_layout()
plt.show()
savefig("output/tilt/buoyancyflux_w_B_b.png")





## this section tests time series of idividual points
        k=150
        ∇κ∇B = nanmean(ds_Bbudget["∇κ∇B"][:,:,k,:],dim=(1,2))
        div_uB = nanmean(ds_Bbudget["div_uB"][:,:,k,:],dim=(1,2))
        
        # mask
        uhat[uhat.==0].=NaN
        ∇κ∇B[b.==0].=NaN
        div_uB[b.==0].=NaN
        # compute all terms
        
        t_diff = diff(t)
        dBdt = zeros(length(t_diff))
        for n in 1:length(t)-1
        B2 = nanmean(ds_3D["B"][:,:,k,n:n+1],dim=(1,2));          # total buoyancy with two timesteps
        dBdt[n] = (B2[2] .- B2[1]) ./ t_diff[n]
        @info n
        end



        using PyPlot
    close(gcf())
    fig = figure(figsize=(8,5))
    # Set font size
    plt.rcParams["font.size"] = 20

    # Plot each time series
    plt.plot(t[2:end], dBdt[1:end], color = "black", linestyle="-", linewidth=2.5, 
            label = L"d\langle B\rangle/dt")
    plt.plot(t[2:end], -div_uB[2:end]+∇κ∇B[2:end], 
            label = "RHS", color = "red", linestyle="--",linewidth=2.5)
    savefig("output/tilt/buoyancy_budget_pointwise_test.png")
