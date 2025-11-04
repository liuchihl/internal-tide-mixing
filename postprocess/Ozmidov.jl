using NCDatasets
using Statistics
using PyPlot
using NaNStatistics

fname = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=461.0_analysis_round=all_threeD.nc"
N0 = 1e-3
ds = Dataset(fname, "r")
z = ds["z_aac"][1:250]
zf = ds["z_aaf"][1:251]
nt = size(ds["ε"], 4)
LO_flat = Float64[]
LO_all = Float64[]
Δz_min = 7.5

avg_ε = Float64[]  # Array to store average epsilon for each time step
avg_N = Float64[]

for t in nt
    ε = ds["ε"][:, :, :, t]
    ε[ε.<1e-10] .= NaN
    B = ds["B"][:, :, :, t]
    N2 = similar(B)
    for k in 2:size(B, 3)-1
        N2[:, :, k] = (B[:, :, k+1] - B[:, :, k-1]) / (z[k+1] - z[k-1])
    end
    N2[:, :, 1] = (B[:, :, 2] - B[:, :, 1]) / (z[2] - z[1])
    N2[:, :, 249] = (B[:, :, 250] - B[:, :, 249]) / (z[250] - z[249])
    N2[N2.<=1e-10] .= NaN
    N = sqrt.(N2)
    N[isnan.(ε)] .= NaN
    LO = 2 * π * (ε ./ N .^ 3) .^ (1 / 2)
    append!(LO_flat, LO[.!isnan.(LO).&&LO.<105])
    append!(LO_all, LO[.!isnan.(LO)])

    # Compute dz for each vertical level
    dz = diff(zf)

    # Broadcast dz to the full 3D shape
    volumes = ones(size(ε))  # shape (Nx, Ny, Nz)
    for k in 1:length(dz)
        volumes[:, :, k] .= dz[k]
    end

    # Mask volumes where ε or N is NaN
    volumes_ε = volumes .* .!isnan.(ε)
    volumes_N = volumes .* .!isnan.(N)

    # Volume-weighted average for ε and N
    avg_ε_vol = nansum(ε .* volumes_ε) / sum(volumes_ε)
    avg_N_vol = nansum(N .* volumes_N) / sum(volumes_N)

    push!(avg_ε, avg_ε_vol)
    push!(avg_N, avg_N_vol)

    println("Processed time step ", t, " / ", nt)
end

# LO_Khani = 2*pi*(nanmean(avg_ε)/N0^3)^(1/2)
LO_Khani = 2 * pi * (nanmean(avg_ε) / nanmean(avg_N)^3)^(1 / 2)
Δz_avg = mean(diff(zf))
# Plot
close("all")
plt.figure(figsize=(8, 6))
plt.hist(LO_flat, bins=5000, density=true, color="black", alpha=0.4)
plt.xlabel(L"$L_O$ [m]", fontsize=18)
plt.ylabel("PDF", fontsize=18)
plt.axvline(Δz_min, color="black", linestyle="solid", linewidth=2.0, label=L"$Δz_{min}$")
plt.axvline(Δz_avg, color="black", linestyle="dashed", linewidth=2.0, label=L"$\overline{Δz}$")
plt.axvline(mean(LO_all), color="red", linestyle="solid", linewidth=2.5, label=L"$\overline{L_O(x,y,z)}^{V}$")
plt.axvline(LO_Khani, color="blue", linestyle="dashed", linewidth=2.5, label="Khani 2018")
plt.minorticks_on()
plt.tick_params(axis="x", labelsize=16, top=true, which="both")
plt.tick_params(axis="y", labelsize=16)
plt.xlim(0, 80)
plt.legend(fontsize=16)
plt.savefig("output/tilt/Ozmidov_Lengthscale.pdf.png")