using NCDatasets
using PyPlot

fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=460_threeD.nc"
ds = Dataset(fname)
t = ds["time"][:]/(2*pi/1.4e-4)
c = ds["c"][:,:,:,1]
c[c.==0] .= NaN


figure(figsize=(10,5))
# Create a subplot layout for x-y and x-z planes
subplot(2, 1, 1)

# Extract the dimensions
x = ds["xC"][:]
y = ds["yC"][:]
z = ds["zC"][:]

# Get middle index for slicing
ind_z = 133
mid_y = 17
# Plot x-y plane at middle z
pcolor(x, y, log10.(abs.(c[:,:,ind_z]')), shading="auto")
clim(-5,0)
colorbar(label="Concentration")
title("X-Y Plane (z = $(z[ind_z]))")
xlabel("X")
ylabel("Y")

# Plot x-z plane at middle y
subplot(2, 1, 2)
pcolor(x, z, log10.(abs.(c[:,mid_y,:]')), shading="auto")
clim(-5,0)
colorbar(label="Concentration")
title("X-Z Plane (y = $(y[mid_y]))")
xlabel("X")
ylabel("Z")

tight_layout()
show()
savefig("output/tilt/test_c.png")


x_center = 0
y_center = Ny÷2
z_center = 1000
σ_x = 1000  # in meters
σ_y = 1000  
σ_z = 50    
# Set standard deviations

σ_x = (x[end] - x[1]) / 10
σ_y = (y[end] - y[1]) / 10
σ_z = (z[end] - z[1]) / 10
C = 1.0  # Amplitude of the tracer concentration at the center
cᵢ(x, y, z) = C * exp(-((x - x_center)^2 / (2σ_x^2) + (y - y_center)^2 / (2σ_y^2) + (z - z_center)^2 / (2σ_z^2)))

# Create a new figure for comparison
figure(figsize=(20, 10))

# Original numerical solution
subplot(2, 2, 1)
pcolormesh(x, y, permutedims(c[:,:,ind_z]), shading="auto")
colorbar(label="Concentration")
title("Numerical: X-Y Plane (z = $(z[ind_z]))")
xlabel("X")
ylabel("Y")

subplot(2, 2, 3)
pcolormesh(x, z, permutedims(c[:,mid_y,:]), shading="auto")
colorbar(label="Concentration")
title("Numerical: X-Z Plane (y = $(y[mid_y]))")
xlabel("X")
ylabel("Z")

# Analytical solution
subplot(2, 2, 2)
c_analytical_xy = [cᵢ(x[i], y[j], z[ind_z]) for j in 1:length(y), i in 1:length(x)]
pcolormesh(x, y, c_analytical_xy, shading="auto")
colorbar(label="Concentration")
title("Analytical: X-Y Plane (z = $(z[ind_z]))")
xlabel("X")
ylabel("Y")

subplot(2, 2, 4)
c_analytical_xz = [cᵢ(x[i], y[mid_y], z[k]) for k in 1:length(z), i in 1:length(x)]
pcolormesh(x, z, c_analytical_xz, shading="auto")
colorbar(label="Concentration")
title("Analytical: X-Z Plane (y = $(y[mid_y]))")
xlabel("X")
ylabel("Z")

tight_layout()
show()
savefig("output/tilt/test_c_comparison.png")


fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=450_slices_xz.nc"
ds = Dataset(fname)
t = ds["time"][:]/(2*pi/1.4e-4)
