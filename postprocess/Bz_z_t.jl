using NCDatasets
using PyPlot
using NaNStatistics

base = "output/tilt/"
times = [50, 90, 130, 170, 210, 250, 290, 330, 370, 410, 450]

global Bz_all = Float64[]
zC = nothing
t_all = Float64[]

for i in 1:length(times)
    ds = Dataset(string(base, "internal_tide_theta=", 0.0036, "_Nx=500_Nz=250_tᶠ=", times[i], "_threeD_timeavg.nc"), "r")
    zC = ds["zC"][:]
    t = ds["time"][:]
    Nz = length(zC)
    for n in 1:length(t)
        Bz = zeros(Nz-1)
        for k in 1:Nz-1
            # Read only one z slice at a time
            Bslice1 = ds["B"][:, :, k, n]
            Bslice2 = ds["B"][:, :, k+1, n]
            Bmean1 = nanmean(Bslice1)
            Bmean2 = nanmean(Bslice2)
            Bz[k] = (Bmean2 - Bmean1) / (zC[k+1] - zC[k])
        end
        # Store
        if i == 1 && n == 1
            global Bz_all = Bz
            global t_all = [t[n]]
        else
            global Bz_all = hcat(Bz_all, Bz)
            push!(t_all, t[n])
        end
    end
    close(ds)
    println("Processed time = ", times[i])
end

# Convert to array for plotting
Bz_mat = Array(Bz_all)  # (z-1, ntotal)
t_vec = t_all
ds = Dataset(string(base, "internal_tide_theta=", 0.0036, "_Nx=500_Nz=250_tᶠ=", times[1], "_threeD_timeavg.nc"), "r")
zF = ds["zF"][2:end-1]

# Plot Hovmöller diagram
figure(figsize=(8, 6))
pcolormesh(t_vec, zF, Bz_mat, shading="auto", cmap="viridis")
colorbar(label="∂B/∂z [s⁻²]")
xlabel("Time [s]")
ylabel("z [m]")
title("Hovmöller diagram of horizontal mean ∂B/∂z")
tight_layout()
savefig("output/tilt/Bz_hovmoller.png", dpi=150)