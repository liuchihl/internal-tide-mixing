# This script reads in output, makes some plot, and saves an animation

using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Makie
using JLD2
using Plots
using BSplineKit
using LinearAlgebra


function deriv(x,y)
spl = interpolate(x, y, BSplineOrder(6))
D1f = diff(spl, Derivative(1))   # change 
    return D1f.(x)
end


function nan_solid(x, z, u, mask)
    Nx, Nz = size(u)
    x2 = reshape(x, (Nx, 1))
    z2 = reshape(z, (1, Nz))
    u[mask] .= NaN
    return u
end


# filename = "internal_tide_20days-theta=0.002"
filename = "internal_tide_3days-theta=0.0036_realtopo3D_Nx150_slice"
filename = "internal_tide_3days-theta=0.0036_realtopo3D_Nx150_slice_nochi"
# Read in the first iteration.  We do this to load the grid
# filename * ".jld2" concatenates the extension to the end of the filename
u_ic = FieldTimeSeries(string("output/", filename * ".jld2"), "uhat", iterations = 0)
# v_ic = FieldTimeSeries(string("output/", filename * ".jld2"), "v", iterations = 0)
# w_ic = FieldTimeSeries(string("output/", filename * ".jld2"), "w", iterations = 0)
# b_ic = FieldTimeSeries(string("output/", filename * ".jld2"), "b", iterations = 0)

# u = GPUAdaptedFieldTimeSeries(string("output/", filename * ".jld2"), "u", iterations = 0)
# ## Load in coordinate arrays
# ## We do this separately for each variable since Oceananigans uses a staggered grid
# xu, yu, zu = nodes(u_ic)
# xv, yv, zv = nodes(v_ic)
# xw, yw, zw = nodes(w_ic)
# xb, yb, zb = nodes(b_ic)

file = jldopen(string("output/", filename * ".jld2"))

ig = file["serialized/grid"]
ug = ig.underlying_grid
ĝ = file["serialized/buoyancy"].gravity_unit_vector
zu = ug.zᵃᵃᶜ[3:end-2]
xu = ug.xᶜᵃᵃ[3:end-2]
zb = ug.zᵃᵃᶜ[3:end-2]
xb = ug.xᶜᵃᵃ[3:end-2]

## Extract a vector of iterations
iterations = parse.(Int, keys(file["timeseries/t"]))

# t_save = zeros(length(iterations))
# B_150 = zeros(length(b_ic[:, 1, 1, 1]), length(iterations))


fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10))
axis_kwargs = (xlabel = "zonal distance (x)",
                  ylabel = "elevation (z)",
                  limits = ((0, ug.Lx), (0, ug.Lz*0.8)),
                  )
ax_u = Axis(fig[1, 1]; title = "u and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_uz = Axis(fig[1, 2]; title = "du/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_bz = Axis(fig[2, 1]; title = "db/dz and equally-spaced buoyancy contours (B)", axis_kwargs...)
ax_ε = Axis(fig[2, 2]; title = "TKE dissipation rate (ε) and equally-spaced buoyancy contours (B)", axis_kwargs...)

# fig[1, :] = Label(fig, "t = 1", fontsize=20, tellwidth=false)
û_xz = zeros(size(file["timeseries/uhat/0"][:,1,:])...,length(iterations));
B_xz = zeros(size(file["timeseries/B/0"][:,1,:])...,length(iterations));
ε_xz = zeros(size(file["timeseries/ε/0"][:,1,:])...,length(iterations));
t = zeros(length(iterations));


anim = @animate for (i, iter) in enumerate(iterations)
    û_xz[:,:,i] = file["timeseries/uhat/$iter"][:, 1, :]; 
    # ŵ_xz = file["timeseries/what/$iter"][:, 1, :];
    B_xz[:,:,i] = file["timeseries/B/$iter"][:, 1, :];
    ε_xz[:,:,i] = file["timeseries/ε/$iter"][:, 1, :];
    dBdz_xz = zeros(size(B_xz)); dûdz_xz = zeros(size(û_xz)); 
   
    for j in 1:length(xb)
        dBdz_xz[j,:,i] = deriv(zb,B_xz[j,:,i])
        dûdz_xz[j,:,i] = deriv(zb,û_xz[j,:,i]) 
    end
    t[i] = file["timeseries/t/$iter"];

    # Save some variables to plot at the end
    # B_150[:,i] = B_xz[1:end, 12]; # This is the buouyancy along the bottom wall
    # t_save[i] = t # save the time
    U₀ = 0.025
        û_xz_plot = Makie.heatmap!(ax_u, xu, zu, û_xz[:,:,i]; color = :balance, xlabel = "x", ylabel = "z",
              aspect_ratio = :auto,ylim=(0,3000), colorrange = (-3U₀, 3U₀), colormap = :balance,clim = (-3U₀,3U₀));  
        ct_û = Makie.contour!(ax_u, xb, zb, B_xz[:,:,i],
              levels=0.:0.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
              
        dûdz_xz_plot = Makie.heatmap!(ax_uz, xu, zu, dûdz_xz[:,:,i]; color = :balance, xlabel = "x", ylabel = "z",
              aspect_ratio = :auto, ylim=(0,3000), clim=(-maximum(dûdz_xz)*0.5,maximum(dûdz_xz)*0.5)); 
                ct_ûz = Makie.contour!(ax_uz, xb, zb, B_xz[:,:,i],
              levels=0.:0.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
             
        dBdz_xz_plot = Makie.heatmap!(ax_bz, xb, zb, dBdz_xz[:,:,i]; color = :balance, xlabel = "x", ylabel = "z", 
              aspect_ratio = :auto,ylim=(0,3000), clim=(-maximum(dBdz_xz)*0.5,maximum(dBdz_xz)*0.5)); 
              ct_Bz = Makie.contour!(ax_bz, xb, zb, B_xz[:,:,i],
              levels=0.:0.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)
             
        ε_xz_plot = Makie.heatmap!(ax_ε, xu, zu, log10.(ε_xz)[:,:,i]; color = :thermal, xlabel = "x", ylabel = "z",
              aspect_ratio = :auto, ylim=(0,3000), clim=(-12,-6)); 
             ct_ε = Makie.contour!(ax_ε, xb, zb, B_xz[:,:,i],
              levels=0.:0.5e-4:4.e-3, linewidth=0.6, color=:black, alpha=0.5)

    # û_title = @sprintf("û, t = %s", round(t));
    # # v_title = @sprintf("v̂, t = %s", round(t));
    # ûz_title = @sprintf("dû/dz, t = %s", round(t));
    # Bz_title = @sprintf("dB/dz, t = %s", round(t));
    # ε_title = @sprintf("ε, t = %s", round(t));

    iter == iterations[end] && close(file)
end
# Save the animation to a file
mp4(anim, "realtopo_slices.mp4", fps = 20) # hide



# Save the animation to a file
# mp4(anim, string("plots/",filename,".mp4"), fps = 20) # hide
# frames = (1:12)
# record(fig, string(split(filename, "1.")[1], ".mp4"), frames, framerate=16) do i
#     @info "Plotting frame $i of $(frames[end])..."
# end
# # Now, make a plot of our saved variables
# # In this case, plot the buoyancy at the bottom of the domain as a function of x and t
# # You can (and should) change this to interrogate other quantities
# heatmap(xb, t_save, b_150', xlabel="x", ylabel="t", title="buoyancy at z=150m")