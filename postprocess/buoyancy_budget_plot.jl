# this script computes the buoyancy budget equation: 
# ∂B/∂t = -ũ⋅∇B - ∇⋅(-κ∇B)

using NCDatasets
using CairoMakie
using Statistics
using NaNStatistics
include("functions/bins.jl")
include("functions/mmderiv.jl")
    slope = "tilt"
    timerange = "40-80"
    θ=0.0036

## Decompose advection term
    # <∇⋅(uB)> = <u⋅∇B> = <u>⋅∇<B> + <u'⋅∇B'>
    file = string("output/",slope,"/TF_avg_",timerange,"bin.nc")
    ds = Dataset(file,"r")
    filename_3D = "output/tilt/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_40-80_threeD.nc"
    
    ds_3D = Dataset(filename_3D,"r")
    xC = ds_3D["xC"][:]
    zC = ds_3D["zC"][:]
    b = ds["b_avg"][:,:]
    # what_avg = ds["what_avg"][:,:]
  
     t = ds["t"][:]/(2*pi/1.4e-4)
    z = ds["bin_center"][:]
    ind1 = findfirst(x -> isapprox(x, 50; atol=5), z)
    ind2 = findfirst(x -> isapprox(x, 100; atol=5), z)
    div_uB_avg = ds["div_uB_avg"][:,:]
    u_bar_∇B_bar_avg = ds["u_bar_∇B_bar_avg"][:,:]
    u_prime∇B_prime_avg = ds["u_prime∇B_prime_avg"][:,:]
    dBdt_avg = ds["dBdt_avg"][:,:]
    ∇κ∇B_avg = ds["∇κ∇B_avg"][:,:]
    div_uB_avg_z = mean(div_uB_avg[ind1:ind2,:],dims=1)
    u_bar_∇B_bar_avg_z = mean(u_bar_∇B_bar_avg[ind1:ind2,:],dims=1)
    u_prime∇B_prime_avg_z = mean(u_prime∇B_prime_avg[ind1:ind2,:],dims=1) 
    dBdt_avg_z = mean(dBdt_avg[ind1:ind2,:],dims=1)
    ∇κ∇B_avg_z = mean(∇κ∇B_avg[ind1:ind2,:], dims=1)
    # residual = dBdt_avg[:] - (-div_uB_avg[1,2:end] .+ ∇κ∇B_avg[1,2:end])

## Create a new figure
fig = CairoMakie.Figure(resolution = (1000, 500), fontsize=20, size=(1000,500))

# Create an axis
ax = Axis(fig[1, 1], xlabel = "Time (tidal cycle)", title = "Buoyancy Budget", limits=((40,80),nothing))

# Plot each time series
lines!(ax, t[2:end], dBdt_avg_z[:], label = "<dB/dt>", color = :red, linewidth=2.5)
lines!(ax, t[2:end], u_bar_∇B_bar_avg_z[2:end], label = "<ṵ>⋅<∇B>", linestyle=:dash, color = :dodgerblue, linewidth=2.5)
lines!(ax, t[2:end], div_uB_avg_z[2:end], label = "<ṵ⋅∇B>", color = :dodgerblue, linewidth=2.5)
lines!(ax, t[2:end], u_prime∇B_prime_avg_z[2:end], label = "<ṵ'⋅∇B'>", linestyle=:dot, color = :dodgerblue, linewidth=2.5)
lines!(ax, t[2:end], ∇κ∇B_avg_z[2:end], label = "<∇⋅(κ∇B)>", color = :green, linewidth=2.5)
lines!(ax, t[2:end], -div_uB_avg_z[2:end].+∇κ∇B_avg_z[2:end], label = "-<ṵ⋅∇B> + <∇⋅(κ∇B)>", color = :black, linewidth=2.5)
# lines!(ax, t[2:end], residual, label = "Residual", color = :black, linewidth=2.0, linestyle=:dash)
lines!(ax, [40,80], [0,0],color=:black)
# Add a legend
# axislegend(ax, position = :rc, framevisible = false, ncolumns=2)
legend = Legend(fig, ax, framevisible=false, nbanks=1)
# fig[1, 1] = ax  # Ensure the axis fills the grid position
fig[1, 2] = legend  # Place the legend in the same grid position as the axis

# Display the figure
display(fig)
save("output/tilt/Buoyancy_budget_50-100m_decompose.png",fig)


## plot quantities vs hab
fig = CairoMakie.Figure(resolution = (500, 500), fontsize=20, size=(500,500))

# Create an axis
ax = Axis(fig[1, 1], ylabel = "HAB (m)", title = "Buoyancy Budget", limits=((-5e-10,5e-10),(0,200)))
log_symm(v) = sign.(v).*real(log10.(Complex.(v)))
# Plot each time series
lines!(ax, dropdims(mean(dBdt_avg[:,:],dims=2),dims=2), z, label = "<dB/dt>", color = :red, linewidth=2.5)
lines!(ax, dropdims(mean(u_bar_∇B_bar_avg[:,:],dims=2),dims=2), z, label = "<ṵ>⋅<∇B>", linestyle=:dash, color = :dodgerblue, linewidth=2.5)
lines!(ax, dropdims(mean(div_uB_avg[:,:],dims=2),dims=2), z, label = "<ṵ⋅∇B>", color = :dodgerblue, linewidth=2.5)
lines!(ax, dropdims(mean(u_prime∇B_prime_avg[:,:],dims=2),dims=2), z, label = "<ṵ'⋅∇B'>",linestyle=:dot, color = :dodgerblue, linewidth=2.5)
lines!(ax, dropdims(mean(∇κ∇B_avg[:,:],dims=2),dims=2), z, label = "<∇⋅(κ∇B)>", color = :orange, linestyle=:dash,linewidth=2.5)
lines!(ax, dropdims(mean(-div_uB_avg[:,:].+∇κ∇B_avg[:,:],dims=2),dims=2), z, label = "-<ṵ⋅∇B> + <∇⋅(κ∇B)>", color = :black, linewidth=2.5)
# scatter!(ax, log_symm(dBdt_avg[:,1]), z, label="<dB/dt>", color=:red, markersize=5)
# scatter!(ax, log_symm(u_bar_∇B_bar_avg[:,1]), z, label="<ṵ>⋅<∇B>", color=:dodgerblue4, markersize=5)
# scatter!(ax, log_symm(div_uB_avg[:,1]),z, label="<ṵ⋅∇B>", color=:dodgerblue, markersize=5)
# scatter!(ax, log_symm(u_prime∇B_prime_avg[:,1]),z, label="<ṵ'⋅∇B'>", color=:blue, markersize=5)
# scatter!(ax, log_symm(∇κ∇B_avg[:,1]),z, label="<∇⋅(κ∇B)>", color=:green, markersize=5)
# scatter!(ax, log_symm(-div_uB_avg[:,1] .+ ∇κ∇B_avg_z[:,1]), z,label="-<ṵ⋅∇B> + <∇⋅(κ∇B)>", color=:black, markersize=5)# lines!(ax, t[2:end], residual, label = "Residual", color = :black, linewidth=2.0, linestyle=:dash)
# lines!(ax, t[2:end], residual, label = "Residual", color = :black, linewidth=2.0, linestyle=:dash)
# lines!(ax, [40,80], [0,0],color=:black)
# Add a legend
axislegend(ax, position = :rt, framevisible = false, nbanks=1)
# legend = Legend(fig, ax, framevisible=false, nbanks=1)
# fig[1, 1] = ax  # Ensure the axis fills the grid position
# fig[1, 2] = legend  # Place the legend in the same grid position as the axis

# Display the figure
display(fig)
# save("output/tilt/Buoyancy_budget_hab_t=40-80_decompose.png",fig)



## test a few time steps and recalculate the terrain following averages
# load data

filename_field_budget = "output/tilt/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_40-80_threeD_timeavg_Bbudget.nc"
ds_budget = Dataset(filename_field_budget,"r")
filename_3D = "output/tilt/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_40-80_threeD.nc"
ds_3D = Dataset(filename_3D,"r")
θ = 0.0036
zC = ds_budget["zC"][:]; zF = ds_budget["zF"][:];
Nz=length(zC[:]); 

xC = ds_budget["xC"][:]; xF = ds_budget["xF"][:]; 
Nx=length(xC[:]);       dx = xF[end]-xF[end-1];

yC = ds_budget["yC"][:]; yF = ds_budget["yF"][:]
Ny=length(yC[:]);       dy = yF[end]-yF[end-1];
z_face = xF * sin(θ) .+ zF' * cos(θ)     # Cartesian coordinate
t = ds_budget["time"][:];

bin_edge = 0:5:1500#0
bin_center = (bin_edge[1:end-1] .+ bin_edge[2:end]) ./ 2

filename_hab = "output/hab.nc"
ds_hab = Dataset(filename_hab,"r")
hab = ds_hab["hab"][:,:,:];
bin_mask = hab

∇κ∇B_avg = zeros(length(bin_edge)-1,length(t))
div_uB_avg = zeros(length(bin_edge)-1,length(t))
dBdt_avg = zeros(length(bin_edge)-1,length(t)-1)
t_diff = diff(t)  # Precompute time differences
for n in 1:2
    # b = ds_budget["b"][:,:,:,n:n];          # buoyancy perturbation
    ∇κ∇B = ds_budget["∇κ∇B"][:,:,:,n:n];    # ∇⋅κ∇B: buoyancy flux divergence
    div_uB = ds_budget["div_uB"][:,:,:,n:n];# ∇⋅uB: buoyancy flux divergence        
    # terrain following quantities:
    @time ∇κ∇B_avg[:,n], _ = bins(∇κ∇B,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    @time div_uB_avg[:,n], _ = bins(div_uB,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    
    if n < length(t)
        B2 = ds_3D["B"][:,:,:,n:n+1];          # total buoyancy with two timesteps
        dBdt = (B2[:, :, :, 2] .- B2[:, :, :, 1]) ./ t_diff[n]
        @time dBdt_avg[:,n], _ = bins(dBdt,bin_edge,bin_mask,dx=dx,dy=dy,z_face=z_face,normalize=true)
    end 
end













## test to see if the budget is closed


using CairoMakie
fig = CairoMakie.Figure()
ax = Axis(fig[1, 1],title = "Buoyancy budget")
# hm = heatmap!(x,z,dBdt[:,1,:,end])
ln1=lines!(ax,t[2:end],dBdt[1:end],linewidth=3,color=:black)
ln2=lines!(ax,t[2:end],-div_uB[2:end])
ln3=lines!(ax,t[2:end], ∇κ∇B[2:end])
ln4=lines!(ax,t[2:end],-div_uB[2:end] .+ ∇κ∇B[2:end],linewidth=3,color=:red,linestyle=:dot)
axislegend(ax,[ln1,ln2,ln3,ln4],["dB/dt","-∇⋅(u⃗B)","∇⋅(κ∇B)","-∇⋅(u⃗B)+∇⋅(κ∇B)"],position=:lb)

display(fig)

# save("output/test_buoyancy_budget/budget_25x4x25.png",fig)