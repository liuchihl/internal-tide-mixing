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
all_times = []

for i in eachindex(tᶠ)
    # Load data
    filename = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ[i], "_slices_xz.nc")
    ds = Dataset(filename,"r")
    uhat = ds["uhat"][:,1,z_ind:z_ind+40,:]; 
    time = ds["time"][:]
    # Call fft_psd on a single slice to get wavenumbers
    _, kj = fft_psd(uhat[:,1,1]; dt=dx, win="rec")

    close(ds)
    # Store PSDs for each time slice
    time_PSDs = []
    times = []
    counts = 0
    PSD_cum = zeros(length(kj))  # Will be resized with first valid data
    for k in axes(uhat, 3)
        for j in axes(uhat, 2) 
            PSD, _ = fft_psd(uhat[:,j,k]; dt=dx, win="rec")
            PSD_cum += PSD
            counts += 1
            @info counts
        end
        PSD_cum ./= counts
        push!(time_PSDs, PSD_cum)
        push!(times, time)
    end
    push!(all_PSDs, time_PSDs)
    push!(all_times, times)
    # Store the wavenumber for this iteration
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
    lines!(ax, 2π*kj, all_PSDs[i][j], 
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



##### plot using 1D high resolution profile
# calculate the Ozmidov time scale


simname = "tilt"
fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=460_point_center.nc"
ds = Dataset(fname,"r") 
zC = ds["zC"]; Nz=length(zC)
xC = ds["xC"]; Nx=length(xC)
uhat = ds["uhat"][1,1,:,:];

# eliminate topography values
ind = findfirst(uhat[:,1].!=0)
uhat = uhat[ind:end, :]

dt = ds["time"][2]-ds["time"][1]
# Create arrays to store PSDs and frequency
avg_PSDs = []
_, fj = fft_psd(uhat[1,:]; dt=dt, win="hann")
PSD_cum = zeros(0)  # Will be properly sized on first valid calculation
# Loop through each time step
valid_points = 0
for zz in 1:size(uhat, 1)
    # Initialize for accumulating PSDs
    # Process each depth point
            local_PSD, _ = fft_psd(uhat[zz,:]; dt=dt, win="hann")
            # Initialize the cumulative PSD array on first valid calculation
            if isempty(PSD_cum)
                PSD_cum = zeros(length(local_PSD))
            end
            # Add to cumulative PSD
            PSD_cum .+= local_PSD
            valid_points += 1
end
# Calculate average PSD for this time step
PSD_avg = PSD_cum./valid_points
using CairoMakie

# Plot the result
ω = 1.4e-4
f = 5.3e-5
N = 1e-3
fig = CairoMakie.Figure(size=(850, 500), fontsize=18)  # Increase base fontsize
ax = Axis(fig[1, 1], 
    xscale=log10, yscale=log10,
    xlabel="Frequency [1/hr]", ylabel="Power Spectral Density [(m/s)²/hr⁻¹]",
    # title="Averaged PSD for simulation: $simname", 
    limits=((.04,40), (1e-6, 1e3)),
    xticks = [1e-2, 1e-1, 1e0, 1e1],
    yticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    xtickformat = values -> [L"10^{%$e}" for e in Int.(log10.(values))],
    ytickformat = values -> [L"10^{%$e}" for e in Int.(log10.(values))],
    titlesize=20,  # Larger title
    xlabelsize=20, ylabelsize=20,  # Larger axis labels
    xticklabelsize=20, yticklabelsize=20  # Larger tick labels
    )

ax.xminorticksvisible = true
ax.yminorticksvisible = true
ax.xminorticks = IntervalsBetween(9)  # 9 minor ticks between major ticks
ax.yminorticks = IntervalsBetween(9)  # 9 minor ticks between major ticks
    
lines!(ax, 2*pi*fj*3600, PSD_avg, linewidth=2.8, color=:royalblue)
lines!(ax, [f,f]*3600, [1e-6,1e3],color=:black, linestyle=:dash )
lines!(ax, [ω-f,ω-f]*3600, [1e-6,1e3],color=:black,Linlinestyleestyle=:dash )
lines!(ax, [ω,ω]*3600, [1e-6,1e3],color=:black,linestyle=:dash )
lines!(ax, [2ω,2ω]*3600, [1e-6,1e3],color=:black,linestyle=:dash )
lines!(ax, [3ω,3ω]*3600, [1e-6,1e3],color=:black,linestyle=:dash )
lines!(ax, [N,N]*3600, [1e-6,1e3],color=:black, linestyle=:dash )
# Add a -2 slope reference line
x_range = [1e-1, 1e3]  # Choose a good x-range for the slope line
y_start = 100  # Starting point on y-axis
slope_line = [y_start * (x/x_range[1])^(-2) for x in x_range]  # Calculate -2 slope
lines!(ax, x_range, slope_line, color=:black, linestyle=:solid, linewidth=2, 
    label=L"\propto f^{-2}")
x_range = [5, 1e3]  # Choose a good x-range for the slope line
y_start = 0.015  # Starting point on y-axis
slope_line = [y_start * (x/x_range[1])^(-4) for x in x_range]  # Calculate -2 slope
lines!(ax, x_range, slope_line, color=:black, linestyle=:dashdot, linewidth=2, 
    label=L"\propto f^{-3}")

fig
save(string("output/",simname,"/psd_spectrum_averaged_1Dprofile_$simname.png"), fig)










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





## plot PSD from time series 
# Load data 
tᶠ = 460
θ = simname == "tilt" ? 0.0036 : 0
simname = "tilt"    # this is the only thing that needs to be changed
filename = string("output/",simname,"/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_slices_xz_u-v-w-B.nc")
ds_tilt = Dataset(filename,"r")

zC = ds_tilt["zC"]; Nz=length(zC)
xC = ds_tilt["xC"]; Nx=length(xC)
t = ds_tilt["time"];
dt = t[2]-t[1]
# Initialize variables to store results
PSD_cum = zeros(0)  # Will be resized with first valid data
fj = nothing
count = 0
z_ind = 160    # z = 1200.87 m

# Accumulate PSD over all grid points
for i in 1:Nx
        uhat = ds_tilt["uhat"][i,1,z_ind,:]
        if any(uhat .!= 0)  # Check if any value is non-zero
            local_PSD, local_fj = fft_psd(uhat; dt=dt, win="rec")
            
            # Initialize PSD_cum with correct size on first valid data
            if isempty(PSD_cum)
                PSD_cum = zeros(length(local_PSD))
                fj = local_fj
            end
            
            PSD_cum .+= local_PSD
            count += 1
        end
        
end

# Average the accumulated PSD
if count > 0
    PSD_cum ./= count
end

# Plot the result
ω = 1.4e-4
f = 5.3e-5
N = 1e-3
fig = Figure(size=(850, 500), fontsize=18)  # Increase base fontsize
ax = Axis(fig[1, 1], 
    xscale=log10, yscale=log10,
    xlabel="Frequency [rad/hr]", ylabel="Power Spectral Density [(m/s)²/hr⁻¹]",
    # title="Averaged PSD for simulation: $simname", 
    limits=(nothing, (5e-5, 1)),
    xticks = [1e-2, 1e-1, 1e0, 1e1],
    yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
    xtickformat = values -> [L"10^{%$e}" for e in Int.(log10.(values))],
    ytickformat = values -> [L"10^{%$e}" for e in Int.(log10.(values))],
    titlesize=20,  # Larger title
    xlabelsize=20, ylabelsize=20,  # Larger axis labels
    xticklabelsize=20, yticklabelsize=20  # Larger tick labels
    )

ax.xminorticksvisible = true
ax.yminorticksvisible = true
ax.xminorticks = IntervalsBetween(9)  # 9 minor ticks between major ticks
ax.yminorticks = IntervalsBetween(9)  # 9 minor ticks between major ticks
    
lines!(ax, 2*pi*fj*3600, PSD_cum, linewidth=2.8, color=:royalblue)
lines!(ax, [f,f]*3600, [1e-5,1],color=:black )
lines!(ax, [ω-f,ω-f]*3600, [1e-5,1],color=:black )
lines!(ax, [ω,ω]*3600, [1e-5,1],color=:black )
lines!(ax, [2ω,2ω]*3600, [1e-5,1],color=:black )
lines!(ax, [3ω,3ω]*3600, [1e-5,1],color=:black )
lines!(ax, [N,N]*3600, [1e-5,1],color=:black )
# Add a -2 slope reference line
x_range = [1e-1, 1e1]  # Choose a good x-range for the slope line
y_start = 5e-1  # Starting point on y-axis
slope_line = [y_start * (x/x_range[1])^(-2) for x in x_range]  # Calculate -2 slope
lines!(ax, x_range, slope_line, color=:black, linestyle=:dash, linewidth=2, 
    label=L"\propto f^{-2}")
fig
save(string("output/",simname,"/psd_spectrum_averaged_$simname.png"), fig)


