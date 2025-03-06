using Printf
using CairoMakie
using NCDatasets
using Statistics
using NaNStatistics
function deriv(z,y)
    dydz =  diff(y[:,:,:,:],dims=3)./reshape(diff(z[:]),1,1,length(z)-1)
    return dydz
 end
# this script plots terrain following horizontal averaged quantities

simname = "tilt"
tᶠ = 450
if  tᶠ ≤ 10
    output_mode = "verification"
elseif tᶠ ≤ 450
    output_mode = "spinup"
    time = 10:40:tᶠ
else
    output_mode = "analysis"
end

# extracting data
# Initialize arrays
ds_init = Dataset(string("output/",simname,"/TF_avg_tᶠ=",time[1],"_", "verification",".nc"),"r")
Bz_avg_acc = ds_init["Bz_avg"][:,:]
u_avg_acc = ds_init["u_avg"][:,:]
what_avg_acc = ds_init["what_avg"][:,:]
t_acc = ds_init["t"][:]./(2*pi/1.4e-4)
z = ds_init["bin_center"][:]
close(ds_init)

if output_mode == "spinup"

    for i in eachindex(time)[2:end]
        output_mode = "spinup"
        file = string("output/",simname,"/TF_avg_tᶠ=",time[i],"_",output_mode,".nc")
        ds = Dataset(file,"r")
        
        # Concatenate subsequent timesteps horizontally
        Bz_avg_acc = hcat(Bz_avg_acc, ds["Bz_avg"][:,:])
        u_avg_acc = hcat(u_avg_acc, ds["u_avg"][:,:]) 
        what_avg_acc = hcat(what_avg_acc, ds["what_avg"][:,:])
        t_acc = round.(vcat(t_acc, ds["t"][:]./(2*pi/1.4e-4)))
        close(ds)
        @info i
    end
end
## plot Bz, u (cross-slope), what (true)
# dt = diff(t_acc)
# t_cen =  t_acc .- dt/2

fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(1000,1200),
            fontsize=20)

axis_kwargs_hm = (xlabel = "time (tidal cycle)",
                  ylabel = "hab (m)",
                  yminorticksvisible = true,
                  limits = ((t_acc[1], t_acc[end]), (0, z[end]))
                  )
axis_kwargs_ln_Bz = (ylabel = "hab (m)",
                yminorticksvisible = true,
                limits = ((0.1,1.8),(0, 500))) 
axis_kwargs_ln_u = (ylabel = "hab (m)",
                yminorticksvisible = true,
                limits = ((-0.003,0.003),(0, 500))) 
axis_kwargs_ln_w = (ylabel = "hab (m)",
                yminorticksvisible = true,
                limits = ((-0.0025,0.0005),(0, 500))) 

# First row - Bz
ax_Bz = Axis(fig[1, 1]; title = "dB/dz (Total buoyancy gradient)", 
             yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_Bz_ln = Axis(fig[1, 3]; title = "10⁻⁶ x dB/dz", 
                yminorticks = IntervalsBetween(5), axis_kwargs_ln_Bz...)

# Second row - u
ax_u = Axis(fig[2, 1]; title = "Cross-slope velocity", 
            yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_u_ln = Axis(fig[2, 3]; title = "Cross-slope velocity", 
               yminorticks = IntervalsBetween(5), axis_kwargs_ln_u...)

# Third row - what
ax_what = Axis(fig[3, 1]; title = "True vertical velocity", 
               yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_what_ln = Axis(fig[3, 3]; title = "True vertical velocity", 
                  yminorticks = IntervalsBetween(5), axis_kwargs_ln_w...)

# Heatmaps with colorbars
hm_Bz = heatmap!(ax_Bz, t_acc[1:end-1], z[:], Bz_avg_acc[:,1:end-1]',
    colormap = :GnBu_9, colorrange=(5.e-7,10.e-7),
    lowclip=cgrad(:GnBu_9)[1], highclip=cgrad(:GnBu_9)[end])
contour!(ax_Bz, t_acc[:], z[:], Bz_avg_acc', 
    levels=5e-7:1e-7:10e-7, color=:black, linewidth=1)
Colorbar(fig[1,2], hm_Bz)

hm_u = heatmap!(ax_u, t_acc[1:end-1], z[:], u_avg_acc[:,1:end-1]',
    colorrange = (-0.03, 0.03), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
contour!(ax_u, t_acc[:], z[:], u_avg_acc',
    color=:black, linewidth=1, levels=5)
Colorbar(fig[2,2], hm_u)

hm_what = heatmap!(ax_what, t_acc[1:end-1], z[:], what_avg_acc[:,1:end-1]',
    colorrange = (-0.001, 0.001), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
contour!(ax_what, t_acc[:], z[:], what_avg_acc',
    color=:black, linewidth=1,levels=5)
Colorbar(fig[3,2], hm_what)

# Line plots
mid_time_idx = 10#Int(floor(length(t_cen)/2))

# Bz profiles
ln1_Bz = lines!(ax_Bz_ln, 1e6*Bz_avg_acc[:,mid_time_idx], z[:], linewidth=3, color=:black)
ln2_Bz = lines!(ax_Bz_ln, 1e6*Bz_avg_acc[:,end-1], z[:], linewidth=3, color=:red)

# u profiles
ln1_u = lines!(ax_u_ln, u_avg_acc[:,mid_time_idx], z[:], linewidth=3, color=:black)
ln2_u = lines!(ax_u_ln, u_avg_acc[:,end-1], z[:], linewidth=3, color=:red)
lines!(ax_u_ln, [0,0], [0,z[end]], color=:black, linewidth=1) # Zero line

# what profiles
ln1_what = lines!(ax_what_ln, what_avg_acc[:,mid_time_idx], z[:], linewidth=3, color=:black)
ln2_what = lines!(ax_what_ln, what_avg_acc[:,end-1], z[:], linewidth=3, color=:red)
lines!(ax_what_ln, [0,0], [0,z[end]], color=:black, linewidth=1) # Zero line

# Add legend to each profile plot
for (ax, ln1, ln2) in [(ax_Bz_ln, ln1_Bz, ln2_Bz), 
                       (ax_u_ln, ln1_u, ln2_u), 
                       (ax_what_ln, ln1_what, ln2_what)]
    axislegend(ax, 
               [ln1, ln2],
               ["t = $(t_acc[mid_time_idx]) TP", "t = $(t_acc[end-1]) TP"],
               position = :rt)
end

display(fig)
save(string("output/",simname,"/hab_combined_tᶠ=",tᶠ,".png"), fig)


## plot only specfic time steps

simname = "tilt"
tᶠ = 210
if  tᶠ ≤ 10
    output_mode = "verification"
elseif tᶠ ≤ 1010
    output_mode = "spinup"
    time = 10:40:tᶠ
else
    output_mode = "analysis"
end

# extracting data
# Initialize arrays
ds_init = Dataset(string("output/",simname,"/TF_avg_tᶠ=",tᶠ,"_", output_mode,".nc"),"r")
Bz_avg = ds_init["Bz_avg"][:,:]
u_avg = ds_init["u_avg"][:,:]
what_avg = ds_init["what_avg"][:,:]

filename_field = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")
t = ds_field["time"][1:2]/(2*pi/1.4e-4);
z = ds_init["bin_center"][:]
close(ds_init)

fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(1000,1200),
            fontsize=20)

axis_kwargs_hm = (xlabel = "time (tidal cycle)",
                  ylabel = "hab (m)",
                  yminorticksvisible = true,
                  limits = ((t[1], t[2]), (0, z[end]))
                  )
axis_kwargs_Bz_ln = (ylabel = "hab (m)",
                yminorticksvisible = true,
                limits = ((0.45,1.55),(0, 500)))
axis_kwargs_ln = (ylabel = "hab (m)",
                yminorticksvisible = true,
                limits = (nothing,(0, 500)))

# First row - Bz
ax_Bz = Axis(fig[1, 1]; title = "dB/dz (Total buoyancy gradient)", 
             yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_Bz_ln = Axis(fig[1, 3]; title = "10⁻⁶ x dB/dz", 
                yminorticks = IntervalsBetween(5), axis_kwargs_Bz_ln...)

# Second row - u
ax_u = Axis(fig[2, 1]; title = "Cross-slope velocity", 
            yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_u_ln = Axis(fig[2, 3]; title = "Cross-slope velocity", 
               yminorticks = IntervalsBetween(5), axis_kwargs_ln...)

# Third row - what
ax_what = Axis(fig[3, 1]; title = "True vertical velocity", 
               yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_what_ln = Axis(fig[3, 3]; title = "True vertical velocity", 
                  yminorticks = IntervalsBetween(5), axis_kwargs_ln...)

# Heatmaps with colorbars
hm_Bz = heatmap!(ax_Bz, t[:], z[:], Bz_avg[:,1:2]',
    colormap = :GnBu_9, colorrange=(5.e-7,10.e-7),
    lowclip=cgrad(:GnBu_9)[1], highclip=cgrad(:GnBu_9)[end])
contour!(ax_Bz, t[1:2], z[:], Bz_avg[:,1:2]', 
    levels=5e-7:1e-7:10e-7, color=:black, linewidth=1)
Colorbar(fig[1,2], hm_Bz)

hm_u = heatmap!(ax_u, t[:], z[:], u_avg[:,1:2]',
    colorrange = (-0.03, 0.03), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
contour!(ax_u, t[:], z[:], u_avg[:,1:2]',
    color=:black, linewidth=1, levels=5)
Colorbar(fig[2,2], hm_u)

hm_what = heatmap!(ax_what, t[:], z[:], what_avg[:,1:2]',
    colorrange = (-0.001, 0.001), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], 
    highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
contour!(ax_what, t[:], z[:], what_avg[:,1:2]',
    color=:black, linewidth=1,levels=5)
Colorbar(fig[3,2], hm_what)

# Line plots

# Bz profiles
ln1_Bz = lines!(ax_Bz_ln, 1e6*Bz_avg[:,1], z[:], linewidth=3, color=:black)
ln2_Bz = lines!(ax_Bz_ln, 1e6*Bz_avg[:,2], z[:], linewidth=3, color=:red)

# u profiles
ln1_u = lines!(ax_u_ln, u_avg[:,1], z[:], linewidth=3, color=:black)
ln2_u = lines!(ax_u_ln, u_avg[:,2], z[:], linewidth=3, color=:red)
lines!(ax_u_ln, [0,0], [0,z[end]], color=:black, linewidth=1) # Zero line

# what profiles
ln1_what = lines!(ax_what_ln, what_avg[:,1], z[:], linewidth=3, color=:black)
ln2_what = lines!(ax_what_ln, what_avg[:,2], z[:], linewidth=3, color=:red)
lines!(ax_what_ln, [0,0], [0,z[end]], color=:black, linewidth=1) # Zero line

# Add legend to each profile plot
for (ax, ln1, ln2) in [(ax_Bz_ln, ln1_Bz, ln2_Bz), 
                       (ax_u_ln, ln1_u, ln2_u), 
                       (ax_what_ln, ln1_what, ln2_what)]
    axislegend(ax, 
               [ln1, ln2],
               ["t = $(round(t[1])) TP", "t = $(round(t[2])) TP"],
               position = :rt)
end

display(fig)
save(string("output/",simname,"/hab_combined_tᶠ=","200-210",".png"), fig)






########### compare the incorrect plot at 200-210 TP with the correct one at 210 TP
ds_wrong = Dataset(string("output/tilt/TF_avg_tᶠ=210_spinup_old.nc"),"r")
ds_right = Dataset(string("output/tilt/TF_avg_tᶠ=210_spinup.nc"),"r")
Bz_wrong = ds_wrong["Bz_avg"][:,4]
u_wrong = ds_wrong["u_avg"][:,4]
what_wrong = ds_wrong["what_avg"][:,4]
Bz_right = ds_right["Bz_avg"][:,2]
u_right = ds_right["u_avg"][:,2]
what_right = ds_right["what_avg"][:,2]
fig = Figure(resolution = (1000, 400), figure_padding=(10, 40, 10, 10), size=(1200,400),
            fontsize=20)

axis_kwargs_ln = (ylabel = "hab (m)",
                yminorticksvisible = true,
                limits = (nothing,(0, 1500))) 

ax_Bz = Axis(fig[1, 1]; title = "10⁻⁶ x dB/dz", axis_kwargs_ln...)
ax_u = Axis(fig[1, 2]; title = "Cross-slope velocity", axis_kwargs_ln...)
ax_what = Axis(fig[1, 3]; title = "True vertical velocity", axis_kwargs_ln...)

ln1_Bz = lines!(ax_Bz, 1e6*Bz_wrong, z[:], linewidth=3, color=:black, label="wrong")
ln2_Bz = lines!(ax_Bz, 1e6*Bz_right, z[:], linewidth=3, color=:red, label="right", linestyle=:dash)

ln1_u = lines!(ax_u, u_wrong, z[:], linewidth=3, color=:black)
ln2_u = lines!(ax_u, u_right, z[:], linewidth=3, color=:red, linestyle=:dash)

ln1_what = lines!(ax_what, what_wrong, z[:], linewidth=3, color=:black)
ln2_what = lines!(ax_what, what_right, z[:], linewidth=3, color=:red, linestyle=:dash)

axislegend(ax_what,[ln1_what,ln2_what], ["wrong", "right"], position = :rt)

display(fig)
save("output/tilt/comparison_wrong_right.png", fig)







## plot mixing (epsilon, chi, Gamma)
file = string("output/",slope,"/TF_avg_",timerange,"bin.nc")
ds = Dataset(file,"r")

t = ds["t"][:]/(2*pi/1.4e-4)
z = ds["bin_center"][:]
# epsilon_avg = ds["epsilon_avg"][:,:]
Bz_avg = ds["Bz_avg"][:,:]
chi_avg = ds["chi_avg"][:,:]./Bz_avg    # -κ|∇b|²/Bz
# Gamma_avg = chi_avg./epsilon_avg
# eta_avg = chi_avg./(epsilon_avg.+chi_avg)
fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,300))
axis_kwargs_hm = (xlabel = "time (tidal cycle)",
                  ylabel = "hab (m)",
                  limits = ((t[1], t[end]), (0, z[end]))
                  )
axis_kwargs_ln = (ylabel = "hab (m)",limits = (nothing,(0, 500)) ) 
axis_kwargs_ln_Gamma = (ylabel = "hab (m)",limits = ((0,3),(0, 500)) ) 

ax_epsilon = Axis(fig[1, 1]; title = "log ε", axis_kwargs_hm...)
ax_chi = Axis(fig[2, 1]; title = "log χ", axis_kwargs_hm...)
ax_gamma = Axis(fig[3, 1]; title = "Γ mixing coefficient", axis_kwargs_hm...)
ax_epsilon_ln = Axis(fig[1, 3]; title = "log ε", axis_kwargs_ln...)
ax_chi_ln = Axis(fig[2, 3]; title = "log χ", axis_kwargs_ln...)
ax_gamma_ln = Axis(fig[3, 3]; title = "Γ mixing coefficient", axis_kwargs_ln_Gamma...)
χ_hm = chi_avg[:,:]
χ_hm[χ_hm.<0] .= NaN
using ColorSchemes
U₀ = 0.025
hm_epsilon = heatmap!(ax_epsilon, t[:], z[:], log10.(epsilon_avg)',
    colorrange = (-10,-7), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[1,2], hm_epsilon )
hm_chi = heatmap!(ax_chi, t[:], z[:], log10.(χ_hm)', colorrange=(-10,-7),
    colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:blue )
    Colorbar(fig[2,2], hm_chi )
hm_gamma = heatmap!(ax_gamma, t[:], z[:], Gamma_avg',
    colorrange = (0,2), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[3,2], hm_gamma )

lines!(ax_epsilon_ln, log10.(epsilon_avg[:,end]), z[:], linewidth=3)
lines!(ax_chi_ln, log10.(chi_avg[:,end]), z[:], linewidth=3)
lines!(ax_gamma_ln, Gamma_avg[:,end], z[:], linewidth=3)
lines!(ax_gamma_ln, 0.2*ones(size(z)), z[:], linewidth=3, color=:black)

save(string("output/",slope,"/hab_mixing_",timerange,".png"),fig)
display(fig)
close(ds)




## plot x-z slices in terms of velocities

slope = "tilt"
timerange = "40-80"
filename_field = string("output/", slope, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
t = ds_field["time"][:];
n=40;
# b = ds_field["b"][:,:,:,n:n];
# Bz = ds_field["Bz"][:,:,:,n:n];
# epsilon = ds_field["ε"][:,:,:,n:n];
uhat = ds_field["uhat"][:,:,:,n:n];    # true u
what = ds_field["what"][:,:,:,n:n];    # true w
chi = ds_field["χ"][:,:,:,n:n];        # -κ|∇b|²
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
θ = 0.0036
u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ)


# plot x-z slices with varying y
n = Observable(1)
uₙ = @lift(u[:,$n,:])
whatₙ = @lift(what_cen[:,$n,:])


    fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(1000,800))
    axis_kwargs_hm = (xlabel = "x (km)",
                      ylabel = "z (m)"
                      )
title = @lift @sprintf("y=%1.0f (m)", yC[$n])
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
                      
ax_u = Axis(fig[2, 1]; title = "Cross-slope velocity", axis_kwargs_hm...)
ax_what = Axis(fig[3, 1]; title = "True vertical velocity", axis_kwargs_hm...)
# ax_Bz = Axis(fig[3, 1]; title = "Buoyancy Gradient", axis_kwargs_hm...)

hm_u = heatmap!(ax_u, xC[:], zC[:], uₙ,
    colorrange = (-0.04, 0.04), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
    Colorbar(fig[2,2], hm_u)
hm_what = heatmap!(ax_what, xC[:], zC[:], whatₙ,
    colorrange = (-0.005, 0.005), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
    Colorbar(fig[3,2], hm_what)

    frames =  (1:5:length(yC))

    filename_field = string("output/", slope, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
    filename = join(split(filename_field, ".")[1:end-1], ".")
    
    record(fig, string(filename,".mp4"), frames, framerate=13) do i
        @info "Plotting frame $i of $(frames[end])..."
        n[] = i
    end
    


## plot x-z slices in terms of buoyancy

slope = "tilt"
timerange = "0-40"
filename_field = string("output/", slope, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
t = ds_field["time"][:];
n=40;
b = ds_field["b"][:,:,:,n:n];
B = ds_field["B"][:,:,:,n:n];

N = 1.e-3
Bz =  deriv(zC,B);
Bz_bc = 0*ones(Nx,Ny,1,1);
Bz = cat(Bz, Bz_bc, dims=3);

Bz[b.==0] .= NaN
b[b.==0] .= NaN
# plot x-z slices with varying y
n = Observable(1)
bₙ = @lift(b[:,$n,:])
Bzₙ = @lift(Bz[:,$n,:])

    fig = Figure(resolution = (1000, 800), figure_padding=(10, 40, 10, 10), size=(1000,800))
    axis_kwargs_hm = (xlabel = "x (km)",
                      ylabel = "z (m)"
                      )
title = @lift @sprintf("y=%1.0f (m)", yC[$n])
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
                      
ax_b = Axis(fig[2, 1]; title = "Buoyancy perturbation", axis_kwargs_hm...)
ax_Bz = Axis(fig[3, 1]; title = "dB/dz (N²)", axis_kwargs_hm...)

hm_b = heatmap!(ax_b, xC[:], zC[:], bₙ,
    colorrange = (-0.0001,0.0001), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color=:gray )
    Colorbar(fig[2,2], hm_b )

hm_Bz = heatmap!(ax_Bz, xC[:], zC[:], Bzₙ,
    colorrange = (0,2e-6), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
    Colorbar(fig[3,2], hm_Bz)

    frames =  (1:5:length(yC))

    filename_field = string("output/", slope, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg_buoyancy.nc")
    filename = join(split(filename_field, ".")[1:end-1], ".")
    
    record(fig, string(filename,".mp4"), frames, framerate=13) do i
        @info "Plotting frame $i of $(frames[end])..."
        n[] = i
    end






## plot x-z slices in terms of velocities

slope = "tilt"
timerange = "40-80"
filename_field = string("output/", slope, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
t = ds_field["time"][:];
n=40;
# b = ds_field["b"][:,:,:,n:n];
# Bz = ds_field["Bz"][:,:,:,n:n];
# epsilon = ds_field["ε"][:,:,:,n:n];
uhat = ds_field["uhat"][:,:,:,n:n];    # true u
what = ds_field["what"][:,:,:,n:n];    # true w
chi = ds_field["χ"][:,:,:,n:n];        # -κ|∇b|²
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
θ = 0.0036
u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ)


# plot x-z slices with varying y
n = Observable(1)
uₙ = @lift(u[:,$n,:])
whatₙ = @lift(what_cen[:,$n,:])


    fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(1000,800))
    axis_kwargs_hm = (xlabel = "x (km)",
                      ylabel = "z (m)"
                      )
title = @lift @sprintf("y=%1.0f (m)", yC[$n])
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
                      
ax_u = Axis(fig[2, 1]; title = "Cross-slope velocity", axis_kwargs_hm...)
ax_what = Axis(fig[3, 1]; title = "True vertical velocity", axis_kwargs_hm...)
# ax_Bz = Axis(fig[3, 1]; title = "Buoyancy Gradient", axis_kwargs_hm...)

hm_u = heatmap!(ax_u, xC[:], zC[:], uₙ,
    colorrange = (-0.04, 0.04), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
    Colorbar(fig[2,2], hm_u)
hm_what = heatmap!(ax_what, xC[:], zC[:], whatₙ,
    colorrange = (-0.005, 0.005), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
    nan_color = :gray)
    Colorbar(fig[3,2], hm_what)

    frames =  (1:5:length(yC))

    filename_field = string("output/", slope, "/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_", timerange, "_threeD_timeavg.nc")
    filename = join(split(filename_field, ".")[1:end-1], ".")
    
    record(fig, string(filename,".mp4"), frames, framerate=13) do i
        @info "Plotting frame $i of $(frames[end])..."
        n[] = i
    end
    
    
## plot buoyancy flux divergence with both tilt and flat cases
    slope = "tilt"
    timerange = "40-80"
    file_tilt = string("output/",slope,"/TF_avg_",timerange,"bin.nc")
    ds_tilt = Dataset(file_tilt,"r")
    t = ds_tilt["t"][:]/(2*pi/1.4e-4)
    z = ds_tilt["bin_center"][:]
    ∇κ∇B_avg_tilt = ds_tilt["∇κ∇B_avg"][:,:]
    ∇κ∇B_avg_30_39_tilt = dropdims(mean(∇κ∇B_avg_tilt[:,30:40],dims=2),dims=2)
    
    # slope = "notilt"
    # timerange = "0-40"
    # file_notilt = string("output/",slope,"/TF_avg_",timerange,".nc")
    # ds_notilt = Dataset(file_notilt,"r")
    # t_notilt = ds_notilt["t"][:]/(2*pi/1.4e-4)
    # z = ds_notilt["new_height"][:]
    # ∇κ∇B_avg_notilt = ds_notilt["∇κ∇B_avg"][:,:]
    # ∇κ∇B_avg_30_38_notilt = dropdims(mean(∇κ∇B_avg_notilt[:,31:38],dims=2),dims=2)
## 
    fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800),
                fontsize=20)
    axis_kwargs_hm = (xlabel = "time (tidal cycle)",
                    ylabel = "hab (m)",
                    yminorticksvisible = true,
                    limits = ((t[1], t[end]), (0, 200))
                    )

    axis_kwargs_line_∇κ∇B = (ylabel = "hab (m)",yminorticksvisible = true,limits = (nothing,(0, 200)) ) 

    ax_∇κ∇B_tilt = Axis(fig[1, 1]; title = "∇⋅κ∇B (tilt)", yminorticks = IntervalsBetween(5),axis_kwargs_hm...)
    # ax_∇κ∇B_notilt = Axis(fig[2, 1]; title = "∇⋅κ∇B (flat)", yminorticks = IntervalsBetween(5),axis_kwargs_hm...)

    ax_∇κ∇B_tilt_ln = Axis(fig[1, 3]; title = "∇⋅κ∇B (tilt)", yminorticks = IntervalsBetween(5),axis_kwargs_line_∇κ∇B...)
    # ax_∇κ∇B_notilt_ln = Axis(fig[2, 3]; title = "∇⋅κ∇B (flat)", yminorticks = IntervalsBetween(5),axis_kwargs_line_∇κ∇B...)
    

    using ColorSchemes
    U₀ = 0.025
    hm_∇κ∇B_tilt = heatmap!(ax_∇κ∇B_tilt, t[:], z[:], ∇κ∇B_avg_tilt',
        colorrange = (-maximum(filter(!isnan,∇κ∇B_avg_tilt))*0.2,maximum(filter(!isnan,∇κ∇B_avg_tilt))*0.2), colormap = :diverging_bwr_20_95_c54_n256,
        lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
        
    # hm_∇κ∇B_notilt = heatmap!(ax_∇κ∇B_notilt, t[:], z[:], ∇κ∇B_avg_notilt',
    #     colorrange = (-maximum(filter(!isnan,∇κ∇B_avg_notilt))*0.1,maximum(filter(!isnan,∇κ∇B_avg_notilt))*0.1), colormap = :diverging_bwr_20_95_c54_n256,
    #     lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
        Colorbar(fig[1,2], hm_∇κ∇B_tilt)
    
    
    ind = argmin(abs.(t .- 20))  
    ln1=lines!(ax_∇κ∇B_tilt_ln, ∇κ∇B_avg_tilt[:,ind], z[:], linewidth=3, color=:black)
    ln2=lines!(ax_∇κ∇B_tilt_ln, ∇κ∇B_avg_tilt[:,end], z[:], linewidth=3, color=:red)
    ln3=lines!(ax_∇κ∇B_tilt_ln, ∇κ∇B_avg_30_39_tilt[:], z[:], linewidth=3, color=:blue)
    
    # ln1=lines!(ax_∇κ∇B_notilt_ln, ∇κ∇B_avg_notilt[:,ind], z[:], linewidth=3, color=:black)
    # ln2=lines!(ax_∇κ∇B_notilt_ln, ∇κ∇B_avg_notilt[:,end], z[:], linewidth=3, color=:red)
    # ln3=lines!(ax_∇κ∇B_notilt_ln, ∇κ∇B_avg_30_39_notilt[:], z[:], linewidth=3, color=:blue)
    
    axislegend(ax_∇κ∇B_tilt_ln, [ln1,ln2,ln3],["60 tidal period","80 tidal period","70-80 tidal average"], position = :rt)
    display(fig)
    slope = "tilt"
    save(string("output/",slope,"/buoayncy flux divergence_tilt",timerange,".png"),fig)


## plot difference between flat and tilted case
# slope = "tilt"
# timerange = "0-40"
# file = string("output/",slope,"/TF_avg_",timerange,".nc")
# ds = Dataset(file,"r")
# t = ds["t"][:]/(2*pi/1.4e-4)
# z = ds["new_height"][:]
# # average between 30-40 tidal period
# u_avg_tilt = dropdims(mean(ds["u_avg"][:,30:39],dims=2),dims=2)
# what_avg_tilt = dropdims(mean(ds["what_avg"][:,30:39],dims=2),dims=2)
# Bz_avg_tilt = dropdims(mean(ds["Bz_avg"][:,30:39],dims=2),dims=2)
# b_avg_tilt = dropdims(mean(ds["b_avg"][:,30:39],dims=2),dims=2)


# slope = "notilt"
# timerange = "0-40"
# file = string("output/",slope,"/TF_avg_",timerange,".nc")
# ds = Dataset(file,"r")
# t = ds["t"][:]/(2*pi/1.4e-4)
# z = ds["new_height"][:]
# # average between 30-40 tidal period
# u_avg_notilt = dropdims(mean(ds["u_avg"][:,30:39],dims=2),dims=2)
# what_avg_notilt = dropdims(mean(ds["what_avg"][:,30:39],dims=2),dims=2)
# Bz_avg_notilt = dropdims(mean(ds["Bz_avg"][:,30:39],dims=2),dims=2)
# b_avg_notilt = dropdims(mean(ds["b_avg"][:,30:39],dims=2),dims=2)

# diff_u_avg = u_avg_tilt.-u_avg_notilt
# diff_what_avg = what_avg_tilt.-what_avg_notilt


# fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,300))

# axis_kwargs_ln = (ylabel = "hab (m)",limits = (nothing,(0, 1500)) ) 

# ax_u_ln = Axis(fig[1, 1]; title = "u", axis_kwargs_ln...)
# ax_u_diff_ln = Axis(fig[1, 2]; title = "uₛ-uₙ", axis_kwargs_ln...)
# ax_what_ln = Axis(fig[2, 1]; title = "ŵ", axis_kwargs_ln...)
# ax_what_diff_ln = Axis(fig[2, 2]; title = "ŵₛ-ŵₙ", axis_kwargs_ln...)
# ax_b_ln = Axis(fig[1, 3]; title = "b", axis_kwargs_ln...)
# ax_Bz_ln = Axis(fig[2, 3]; title = "Bz", axis_kwargs_ln...)

# ln1=lines!(ax_u_ln, u_avg_tilt, z[:], linewidth=3)
# ln2=lines!(ax_u_ln, u_avg_notilt, z[:], linewidth=3)
# lines!(ax_u_diff_ln, diff_u_avg, z[:], linewidth=3)
# lines!(ax_what_ln, what_avg_tilt, z[:], linewidth=3)
# lines!(ax_what_ln, what_avg_notilt, z[:], linewidth=3)
# lines!(ax_what_diff_ln, diff_what_avg, z[:], linewidth=3)
# lines!(ax_b_ln, b_avg_tilt, z[:], linewidth=3)
# lines!(ax_b_ln, b_avg_notilt, z[:], linewidth=3)
# lines!(ax_Bz_ln, Bz_avg_tilt, z[:], linewidth=3)
# lines!(ax_Bz_ln, Bz_avg_notilt, z[:], linewidth=3)


# axislegend(ax_u_ln, [ln1,ln2],["slope","no slope"], position = :rt)

# save(string("output/",slope,"/hab_velocity_diff_",timerange,".png"),fig)




## plot tilted case from 0-40 tidal cycle and 40-80 together

slope = "tilt"
timerange = "0-40"
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t1 = ds["t"][:]/(2*pi/1.4e-4)
z = ds["new_height"][:]
u_avg1 = ds["u_avg"][:,:]
what_avg1 = ds["what_avg"][:,:]
Bz_avg1 = ds["Bz_avg"][:,:]
b_avg1 = ds["b_avg"][:,:]

slope = "tilt"
timerange = "40-80"
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t2 = ds["t"][:]/(2*pi/1.4e-4)
z = ds["new_height"][:]
u_avg2 = ds["u_avg"][:,:]
what_avg2 = ds["what_avg"][:,:]
Bz_avg2 = ds["Bz_avg"][:,:]
b_avg2 = ds["b_avg"][:,:]


fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800))
axis_kwargs_hm = (xlabel = "time (tidal cycle)",
                  ylabel = "hab (m)",
                  yminorticksvisible = true,
                  limits = ((t1[1], t2[end]), (0, z[end]))
                  )

ax_u = Axis(fig[1, 1]; title = "Cross-slope velocity", yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_what = Axis(fig[2, 1]; title = "True vertical velocity", axis_kwargs_hm...)

ax_b = Axis(fig[1, 3]; title = "b (buoyancy perturbation)", yminorticks = IntervalsBetween(5), axis_kwargs_hm...)
ax_Bz = Axis(fig[2, 3]; title = "dB/dz (Total buoyancy gradient)", yminorticks = IntervalsBetween(5), axis_kwargs_hm...)

using ColorSchemes

hm_u = heatmap!(ax_u, [t1;t2], z[:], [u_avg1 u_avg2]',
    colorrange = (-0.02,0.02), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[1,2], hm_u )
hm_what = heatmap!(ax_what, [t1;t2], z[:], [what_avg1  what_avg2]',
    colormap = :diverging_bwr_20_95_c54_n256, colorrange=(-0.001, 0.001),
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
    Colorbar(fig[2,2], hm_what)
hm_b = heatmap!(ax_b, [t1;t2], z[:], [b_avg1  b_avg2]',
    colorrange = (minimum(filter(!isnan,b_avg2)),maximum(filter(!isnan,b_avg2))*0.8), colormap = :diverging_bwr_20_95_c54_n256,
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end] )
    Colorbar(fig[1,4], hm_b )
hm_Bz = heatmap!(ax_Bz, [t1;t2], z[:], [Bz_avg1  Bz_avg2]',
    colormap = :diverging_bwr_20_95_c54_n256, colorrange=(8.e-7,10.e-7),
    lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end])
    Colorbar(fig[2,4], hm_Bz)
fig
save(string("output/",slope,"/hab_tilt_0-80.png"),fig)




## 
slope = "tilt"
timerange = "0-40"
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t1 = ds["t"][:]/(2*pi/1.4e-4)
z = ds["new_height"][:]
b_avg1 = ds["b_avg"][:,:]

slope = "tilt"
timerange = "40-80"
file = string("output/",slope,"/TF_avg_",timerange,".nc")
ds = Dataset(file,"r")
t2 = ds["t"][:]/(2*pi/1.4e-4)
z = ds["new_height"][:]
b_avg2 = ds["b_avg"][:,:]

fig = Figure(size=(600,300))
axis_kwargs_hm = (xlabel = "time (tidal cycle)",
                  limits = ((t1[1], t2[end]),(0,0.00015))
                  )

ax_b = Axis(fig[1, 1]; title = "b (buoyancy perturbation) 0~100 m average", 
yticks = (0:0.00005:0.0001, ["0", "5x10⁻⁵", "10⁻⁴"]), axis_kwargs_hm...)

ln_b = lines!(ax_b, [t1;t2], vec(mean([b_avg1  b_avg2][1:21,:],dims=1)'),linewidth=3)

fig
save(string("output/",slope,"/hab_tilt_0-80_curve.png"),fig)





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


# epsilon[epsilon.==0].=NaN
# ln=lines(log10.(epsilon[1,1,:,1]),zC[:])
# for i=1:500:Nx
#     for j=1:500:Ny
# lines!(log10.(epsilon[i,j,:,1]),zC[:])
#     end
# end
# save("output/higher_top_resolution/profile_all_epsilon.png",ln)













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
# lines!(ax_2, b_avg[:,1], new_height,linewidth=2.5)
# lines!(ax_2, b_avg[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# lines!(ax_3, bz_avg_no[:,1], new_height,linewidth=2.5)
# lines!(ax_3, bz_avg_no[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# a = lines!(ax_4, bz_avg[:,1], new_height,linewidth=2.5)
# b = lines!(ax_4, bz_avg[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)

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
# lines!(ax_1, b_avg[:,1]*1e6, new_height,linewidth=2.5,color=:red)
# # lines!(ax_1, b_avg[:,2]*1e6, new_height,linewidth=2.5,linestyle=:dot,color=:red)
# # lines!(ax_1, b_avg[:,3]*1e6, new_height,linewidth=2.5,linestyle=:dash,color=:red)
# a=lines!(ax_2, bz_avg_no_0_20[:,1], new_height,linewidth=2.5,color=:blue)
# b=lines!(ax_2, bz_avg_no_0_20[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:blue)
# c=lines!(ax_2, bz_avg_no_20_30[:,1], new_height,linewidth=2.5,linestyle=:dash,color=:blue)
# d = lines!(ax_2, bz_avg[:,1], new_height,linewidth=2.5,color=:red)
# e = lines!(ax_2, bz_avg[:,2], new_height,linewidth=2.5,linestyle=:dot,color=:red)
# f = lines!(ax_2, bz_avg[:,3], new_height,linewidth=2.5,linestyle=:dash,color=:red)
# # legend(fig[3,2],[a,b],["ε", "ε_oceanostics"])
# # axislegend("Titled Legend", position = :lb)
# axislegend(ax_1, [a,b,c,d,e,f],["Day 10, θ=0","Day 20, θ=0","Day 30, θ=0",
#  "Day 10, θ=3.6e-3","Day 20, θ=3.6e-3","Day 30, θ=3.6e-3"], position = :rt,
#     orientation = :vertical)

# save("buoyancy_pert_together_3times.png",fig)


