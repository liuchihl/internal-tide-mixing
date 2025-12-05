using Oceananigans.Units
H = 3kilometers
Lx = 15kilometers
Nx = 2000
Nz = 500
θ = 0.02

# Grid setup
kwarp(k, N) = (N + 1 - k) / N
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))
z_faces(k) = -H * (ζ(k, Nz, 1.2) * Σ(k, Nz, 10) - 1)


# Improved asymmetric triangle function with adaptive smoothing
function asymmetric_triangle_slope(x; h=1000, w_left=3000, w_right=3000, 
                                    edge_fraction=0.4, tip_fraction=0.8, 
                                    tip_smoothness=1.4)
    """
    Create an asymmetric triangle in slope coordinates with adaptive smoothing
    w_left: width on left side (upstream)
    w_right: width on right side (downstream)
    edge_fraction: edge smoothing width as fraction of the respective side width
    tip_fraction: tip smoothing width as fraction of average width
    tip_smoothness: controls sharpness of tip (higher = sharper)
    """
    x_mutable = collect(x)  # Convert x to a mutable array
    
    # Basic triangular shape
    # Left side (negative x)
    tri_left = h * (1 .+ x_mutable ./ w_left)
    tri_left[x_mutable .> 0] .= 0
    
    # Right side (positive x)
    tri_right = h * (1 .- x_mutable ./ w_right)
    tri_right[x_mutable .< 0] .= 0
    
    # Combine
    tri = tri_left .+ tri_right
    tri = max.(tri, 0)
    
    # Adaptive edge smoothing - scale with the actual width on each side
    edge_left = edge_fraction * w_left
    edge_right = edge_fraction * w_right
    
    # Left edge: smooth transition starting at -w_left
    left_smooth = exp.(-((x .+ w_left) ./ edge_left) .^ 2)
    
    # Right edge: smooth transition starting at +w_right
    right_smooth = exp.(-((x .- w_right) ./ edge_right) .^ 2)
    
    # Adaptive tip smoothing - use average width as reference
    w_avg = (w_left + w_right) / 2
    tip_width = tip_fraction * w_avg
    
    # Tip smoothing using tanh
    tip_smooth = h * (1 .- tanh.((abs.(x) ./ tip_width) .^ tip_smoothness))
    
    # Blend between triangular shape and smooth tip
    # Create smooth transition zone around x=0
    blend_width = tip_width * 0.3
    blend_factor = 0.5 * (1 .+ tanh.((tip_width .- abs.(x)) ./ blend_width))
    
    # Apply tip smoothing
    tri_smooth = blend_factor .* tip_smooth .+ (1 .- blend_factor) .* tri
    
    # Apply edge smoothing
    # At the edges, transition smoothly to h (the baseline value)
    topo = tri_smooth .* (1 .- left_smooth) .* (1 .- right_smooth) .+ 
           h .* left_smooth .* right_smooth
    
    return topo
end

# Coordinates
x_centered = range(-Lx / 2, stop=Lx / 2, length=Nx)
x_coord = range(0, stop=Lx, length=Nx)

h = 850
width = 3600
ϕ = atan(h/width)

# Calculate asymmetric widths based on desired slopes
α_left = ϕ - θ
α_right = ϕ + θ


w_left = h / tan(α_left)    # Wider = gentler slope
w_right = h / tan(α_right)  # Narrower = steeper slope
# Generate topography with adaptive smoothing
ẑ_asymmetric = asymmetric_triangle_slope(x_centered; 
                                          h=h, 
                                          w_left=w_left, 
                                          w_right=w_right,
                                          edge_fraction=0.35,  # Adjust these for smoothness
                                          tip_fraction=0.7,
                                          tip_smoothness=1.4)

######################################################
using PyPlot

# Main topography plot
figure(figsize=(16, 10))
plot(x_coord ./ 1000, ẑ_asymmetric, linewidth=2, color="black", label="Asymmetric (θ=$(θ)")
xlabel("x (km)")
ylabel("ẑ (m)")
title("Asymmetric Topography in Slope Coordinates")
legend(loc="upper left")
grid(true)
ylim(0,1000)
# Save the figure
savefig("output/topography_asymmetric_improved.png")
