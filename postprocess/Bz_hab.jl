using CairoMakie
using NCDatasets

# --- Settings ---
times = [10, 50, 90, 130, 170, 210, 250, 290, 330, 370, 410, 450]
simnames = ["tilt", "flat"]
output_modes = ["verification", "spinup", "spinup", "spinup", "spinup", "spinup", "spinup", "spinup", "spinup", "spinup", "spinup", "spinup"]

function load_Bz_and_time(simname, times, output_modes)
    Bz_acc = nothing
    t_acc = Float64[]
    for (i, t) in enumerate(times)
        file = "output/$simname/TF_avg_tᶠ=$(t)_$(output_modes[i]).nc"
        ds = Dataset(file, "r")
        Bz = ds["Bz_avg"][:, :]  # (z, nt)
        t_local = ds["t"][:]
        if Bz_acc === nothing
            Bz_acc = Bz
        else
            Bz_acc = hcat(Bz_acc, Bz)
        end
        append!(t_acc, t_local)
        close(ds)
    end
    return Bz_acc, t_acc
end

# Load data
Bz_tilt, t_tilt = load_Bz_and_time("tilt", times, output_modes)
Bz_tilt[:, end:end] = Dataset("output/tilt/TF_avg_tᶠ=450_spinup_440_450.nc", "r")["Bz_avg"][:, 2]
Bz_flat, t_flat = load_Bz_and_time("flat", times, output_modes)
T_tide = 2 * pi / 1.4e-4
# Load z for axis
ds = Dataset("output/tilt/TF_avg_tᶠ=$(times[1])_$(output_modes[1]).nc", "r")
z = ds["bin_center"][:]
close(ds)

# --- Plot ---
set_theme!(fontsize=24)  # Increase overall font size

fig = Figure(resolution=(1000, 700))

ax1 = Axis(fig[1, 1], ylabel="HAB [m]",
    xticklabelsize=22, yticklabelsize=22, limits=((0, 450), nothing))
hm1 = heatmap!(ax1, t_tilt / T_tide, z, Bz_tilt', colormap=reverse(cgrad(:batlow, 100)), colorrange=(0, 1e-6))
contour!(ax1, t_tilt / T_tide, z, Bz_tilt', levels=0:1e-7:1e-6, color=:black, linewidth=1.5)
text!(ax1, 0.01, 0.98, text="(a) tilt", align=(:left, :top), fontsize=32, space=:relative, color=:white)


ax2 = Axis(fig[2, 1], ylabel="HAB [m]", xlabel="Time [Tidal Period]",
    xticklabelsize=22, yticklabelsize=22, limits=((0, 450), nothing))
hm2 = heatmap!(ax2, t_flat / T_tide, z, Bz_flat', colormap=reverse(cgrad(:batlow, 100)), colorrange=(0, 1e-6))
contour!(ax2, t_flat / T_tide, z, Bz_flat', levels=0:1e-7:1e-6, color=:black, linewidth=1.5)
text!(ax2, 0.01, 0.98, text="(b) flat", align=(:left, :top), fontsize=32, space=:relative, color=:white)

Colorbar(fig[2, 2], hm1, label="∂B/∂z [s⁻²]", width=30, ticklabelsize=22, labelsize=26)

# Add minor ticks to x-axis
ax1.xticksmirrored = true
ax2.xticksmirrored = true
ax1.xminorgridvisible = true
ax2.xminorgridvisible = true
ax1.xminorticksvisible = true
ax2.xminorticksvisible = true

fig
save("output/Bz_compare_timeseries.png", fig)