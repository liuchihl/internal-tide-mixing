function smooth_triangle(x; h=1000, w=3000, edge=400, tip_width=4000, tip_smoothness=1)
    tri = h * (1 .- abs.(x) ./ w)
    tri = max.(tri, 0)
    left = exp.(-((x .+ w) ./ edge) .^ 2)
    right = exp.(-((x .- w) ./ edge) .^ 2)
    
    # Tanh-based tip smoothing
    # The tanh function creates a smooth transition near x=0
    # tip_smoothness controls how sharp/smooth the tip is (lower = smoother)
    tip_smooth = h * (1 .- tanh.((abs.(x) ./ tip_width) .^ tip_smoothness))
    
    # Blend between the triangular shape and the smooth tip
    # Use tanh to create a smooth mask
    blend_factor = 0.5 * (1 .+ tanh.((tip_width .- abs.(x)) ./ (tip_width * 0.2)))
    
    # Combine: use smooth tip near center, transition to triangle further out
    tri_smooth = blend_factor .* tip_smooth .+ (1 .- blend_factor) .* tri
    
    # Apply edge smoothing as before
    topo = tri_smooth .* (1 .- left) .* (1 .- right) .+ h .* left .* right

    return topo
end

# load the 2D topography
using MAT
using Interpolations
using NaNStatistics
# load topography 
file = matopen("topo.mat")
z_topo = read(file, "z_noslope_periodic")
z_topo = nanmean(z_topo[:, 28:36], dim=2)   # take the middle column
x_topo = read(file, "x_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1], x_topo[end], size(z_topo, 1))
close(file)
# high-resolution grids
Nx = 500
x_interp = range(x_topo[1], x_topo[end], length=Nx)

# Interpolation object (caches coefficients and such)
itp = LinearInterpolation(x_topo_lin, z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x) for x in x_interp]
z_interp = z_interp .- minimum(z_interp)

## 
x = minimum(x_interp):1:maximum(x_interp)
x_sym = minimum(x_interp)-maximum(x_interp)/2:1:maximum(x_interp)-maximum(x_interp)/2
y = smooth_triangle(x_sym; h=850, w=3600, edge=1600, tip_width=3600, tip_smoothness=4)

using Plots
Plots.plot(x, y, lw=3, color=:black, xlabel="x", ylabel="height", legend=false)
Plots.plot!(x_interp, z_interp, lw=3, color=:blue, xlabel="x", ylabel="height", legend=false)
Plots.ylims!(0, 2000)
display(Plots.current())