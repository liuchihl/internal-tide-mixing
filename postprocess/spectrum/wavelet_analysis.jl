using ContinuousWavelets, Wavelets, FFTW
using PyPlot

n = 2047;
x = range(0.0001,100,length=n)
ω = 2*pi./x
f = sin.(0.1 * x) + 0.5 * sin.(2 * x)  # Tide + small-scale feature
# Create a figure with two subplots with specific height ratios
close("all")
fig, (ax1, ax2) = subplots(2, 1, figsize=(10, 8))

# Plot the Doppler signal on the first subplot
ax1.plot(x,f)
ax1.set_title("Doppler")
ax1.set_xlim(0, x[end])

# Compute the wavelet transform
c = wavelet(Morlet(π), averagingType=NoAve(), β=2);
res = cwt(f, c)

# Create a heatmap on the second subplot
pc = ax2.pcolor(x, 1:30, log10.(abs.(res')), cmap="viridis")
ax2.set_xlabel("time index")
ax2.set_ylabel("frequency index")

# Add colorbar
fig.colorbar(pc, ax=ax2)

# Adjust layout
tight_layout()
savefig("output/wavelet_analysis.png")