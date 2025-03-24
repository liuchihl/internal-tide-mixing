using NCDatasets
using CairoMakie

include("functions/fft_psd.jl")

simname = "tilt"    # this is the only thing that needs to be changed
θ = simname == "tilt" ? 0.0036 : 0

tᶠ = 10:40:450
z_ind = 160    # z = 1200.87 m
dx = 30


# Initialize arrays to store results
all_PSDs = []
all_kjs = []
all_times = []

for i in 1:length(tᶠ)
    # Load data
    filename = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ[i], "_slices_xz.nc")
    ds = Dataset(filename,"r")
    uhat = ds["uhat"][:,1,z_ind,:]; 
    time = ds["time"][:]
    close(ds)
    
    # Store PSDs for each time slice
    time_PSDs = []
    time_kjs = []
    times = []
    
    for j in 1:size(uhat, 2)
        PSD, kj = fft_psd(uhat[:,j]; dt=dx, win="rec")
        push!(time_PSDs, PSD)
        push!(times, time)
    end
    
    push!(all_PSDs, time_PSDs)
    push!(all_times, times)
end

# Create the plot
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], 
    xscale=log10, yscale=log10,
    xlabel="Wavenumber", ylabel="Power Spectral Density",
    title="PSD for simulation: $simname")

# Create a color array with distinct colors
num_times = length(tᶠ)
colors = cgrad(:viridis, num_times; categorical=true)

# Plot each time slice with a different color from the array
for i in [1,5,9,12]
    j = size(all_PSDs[i], 1)
    # Calculate normalized time
    time_value = all_times[i][j][end]
    normalized_time = time_value/(2*pi/1.4e-4)
    rounded_time = round(normalized_time, digits=1)
    
    # Use the index i to select a color from the gradient
    lines!(ax, 2π*all_kjs[i][j], all_PSDs[i][j], 
           label="tᶠ=$(rounded_time)",
           color=colors[i],  # Assign a specific color based on index
           alpha=0.8)
end

# Improve the legend
axislegend(ax, position=:lb, nbanks=3, framevisible=true, backgroundcolor=(:white, 0.8))

# Save the figure
save(string("output/",simname,"/psd_spectrum_$simname.png"), fig)

# Show the figure
fig


## Create a matrix of PSD versus wavenumber and time
times = []
for j in 1:length(all_times)
    times = vcat(times, all_times[j][1]/(2*pi/1.4e-4) )
end
PSD_matrix = zeros(length(times),length(kj))  # Pre-allocate
ind = 0
for i in 1:length(all_times)
    for j in 1:length(all_times[i])
        ind += 1
        PSD_matrix[ind,:] = all_PSDs[i][j]
        @info i, j
    end
end

# Create heatmap figure
fig_heatmap = Figure(size=(900, 600))
ax_heatmap = Axis(fig_heatmap[1, 1], 
    xscale=log10, 
    xlabel="Wavenumber", ylabel="Time (periods)",
    title="PSD for simulation: $simname")

# Apply log10 to PSD values, handling zeros
log_psd = copy(PSD_matrix)
log_psd[log_psd .<= 0] .= minimum(log_psd[log_psd .> 0]) / 10  # Replace zeros with small value
log_psd = log10.(log_psd)

hm = heatmap!(ax_heatmap, 2*pi*kj, times, log_psd',
    colormap=:linear_bmy_10_95_c71_n256, colorrange=(-8, 0))	

Colorbar(fig_heatmap[1, 2], hm, label="log10(PSD)")

# Save the heatmap
save(string("output/",simname,"/psd_heatmap_$simname.png"), fig_heatmap)

# Show the heatmap
fig_heatmap
