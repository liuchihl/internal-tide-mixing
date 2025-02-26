# this script plots time averaged quantities from the 3D output 
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using CairoMakie
using NCDatasets
using Statistics


tᶠ = 250
θ = 3.6e-3
if θ==3.6e-3
    simname = "tilt"
else 
    simname = "flat"
end

if  tᶠ ≤ 10
    output_mode = "verification"
    Nt = 10      # number of time averages
    endtime = tᶠ
elseif tᶠ ≤ 1010
    output_mode = "spinup"
    Nt = 4
    if tᶠ == 1010
        endtime = ["$i" for i in 50:40:1010]
    else
        endtime = tᶠ
    end
else
    output_mode = "analysis"
    Nt = 20
    endtime = tᶠ
end

filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",endtime, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

    # grids
    zC = ds_field["zC"][:]; zF = ds_field["zF"][:];
    Nz=length(zC[:]); 

    xC = ds_field["xC"][:]; xF = ds_field["xF"][:]; 
    Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

    yC = ds_field["yC"][:]; yF = ds_field["yF"][:]
    Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
k=2
uhat = ds_field["uhat"][:,:,:,k:k];    # true u
what = ds_field["what"][:,:,:,k:k];    # true w
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center
# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ) # cross-slope velocity
w = -uhat[:,:,:,1]*sin(θ) .+ wtemp*cos(θ)# slope-normal velocity
time = Int.(round.(ds_field["time"][k]/(2*pi/1.4e-4)))
# plot an animation of the time-averaged uhat field
u[uhat[:,:,:,1].==0] .= NaN
what_cen[what_cen.==0] .= NaN
n = Observable(1)
uₙ = @lift(u[:,$n,:])
whatₙ = @lift(what_cen[:,$n,:])


fig = CairoMakie.Figure(resolution = (800, 750), size=(800,750))
axis_kwargs1 = (xlabel = "x (m)",
                  ylabel = "z (m)",
                  limits = ((0, xC[end]), (0, zF[end]) )
                  )

ax_u = Axis(fig[1,1]; title = @lift(string((time-10),"-",time," tidal average cross-slope velocity, y = ", round(yC[$n]/1e3, digits=1), " km")), axis_kwargs1...)
ax_what = Axis(fig[2,1]; title = @lift(string((time-10),"-",time," tidal average true vertical velocity, y = ", round(yC[$n]/1e3, digits=1), " km")), axis_kwargs1...)

using ColorSchemes

hm_u = CairoMakie.heatmap!(ax_u, xC[:], zC[:], uₙ,
    colorrange = (-0.06,0.06), colormap=:diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
Colorbar(fig[1,2], hm_u; label = "m/s", vertical=true)

hm_what = CairoMakie.heatmap!(ax_what, xC[:], zC[:], whatₙ,
    colorrange = (-0.02,0.02), colormap=:diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
Colorbar(fig[2,2], hm_what; label = "m/s", vertical=true)

frames = 1:10:length(yC)

CairoMakie.record(fig, string("output/", simname, "/timeavg_velocities_",simname,"_", (time-10),"-",time, ".mp4"), frames, framerate = 10) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end




## check the average quantities
uhat = ds_field["uhat"][:,500:500,:,4:4]
uhat[uhat .== 0] .= NaN
f = Figure()
ax = Axis(f[1,1])
hm = heatmap!(ax,xC,zC, uhat[:,1,:,1],  colorrange = (-0.05, 0.05),
            colormap = :diverging_bwr_20_95_c54_n256,
            lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
            highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
Colorbar(f[1,2], hm)
display(f)
save("output/tilt/uhat_50.png", f)

file = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=50_slices_xz.nc"
ds = Dataset(file)

# uhat = ds["uhat"][:,1,:,:]
xC = ds["xC"][:]
zC = ds["zC"][:]
t = ds["time"][:]./(2*pi/1.4e-4)
f = Figure()
ax = Axis(f[1,1])
framecount = size(ds["uhat"], 4)
record(f, "output/tilt/uhat_animation.mp4", 1:framecount; framerate=10) do i
    empty!(ax)
    uhat = ds["uhat"][:,1,:,i]
    hm_u=heatmap!(ax, xC, zC, uhat, colorrange=(-0.05, 0.05),colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
    ax.title = "Time: $(round(t[i], digits=1)) TP"
    Colorbar(f[1,2], hm_u)
end