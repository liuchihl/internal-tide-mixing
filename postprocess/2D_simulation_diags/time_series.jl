using Printf
using PyPlot
using PyCall
using NCDatasets
using Statistics
using NaNStatistics
using Interpolations

fn1 = "output/2D_idealized_tilt_0/internal_tide_theta=0_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
fn2 = "output/2D_idealized_tilt_0.008/internal_tide_theta=0.008_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"
fn3 = "output/2D_idealized_tilt_0.02/internal_tide_theta=0.02_Nx=2000_Nz=500_tᶠ=200.0_snapshot.nc"

ds1 = Dataset(fn1, "r")
ds2 = Dataset(fn2, "r")
ds3 = Dataset(fn3, "r")

zC = ds1["z_aac"][:];
zF = ds1["z_aaf"][:];
Nz = length(zC);
xC = ds1["x_caa"][:];
Nx = length(xC);
t = ds1["time"][:];
ind_begin = findfirst(t / (2 * pi / 1.4e-4) .>= 195)
ind_end = findfirst(t / (2 * pi / 1.4e-4) .>= 199.99)
x_begin = findfirst(xC .>= 6000.0)

# time_range = length(t)-1:length(t)
time_range = ind_begin:ind_end
# time_range = 721:length(t)
b1 = ds1["b"][x_begin, :, 1];
z_begin1 = findfirst(b1[:] .> 0.0)
b2 = ds2["b"][x_begin, :, 1];
z_begin2 = findfirst(b2[:] .> 0.0)
b3 = ds3["b"][x_begin, :, 1];
z_begin3 = findfirst(b3[:] .> 0.0)

u1 = ds1["uhat"][x_begin, z_begin1:end, time_range]
u2 = ds2["uhat"][x_begin, z_begin2:end, time_range]
u3 = ds3["uhat"][x_begin, z_begin3:end, time_range]
# u4 = nanmean(ds4["uhat"][:, :, time_range], dim=3)

w1 = ds1["what"][x_begin, z_begin1:end, time_range]
w2 = ds2["what"][x_begin, z_begin2:end, time_range]
w3 = ds3["what"][x_begin, z_begin3:end, time_range]
# w4 = nanmean(ds4["what"][:, :, time_range], dim=3)

# B1 = ds1["B"][:, :, time_range]
# B2 = ds2["B"][:, :, time_range]
# B3 = ds3["B"][:, :, time_range]

Bz1 = ds1["Bz"][x_begin, z_begin1:end, time_range]
Bz2 = ds2["Bz"][x_begin, z_begin2:end, time_range]
Bz3 = ds3["Bz"][x_begin, z_begin3:end, time_range]

∇κ∇B1 = ds1["∇κ∇B"][x_begin, z_begin1:end, time_range]
∇κ∇B2 = ds2["∇κ∇B"][x_begin, z_begin2:end, time_range]
∇κ∇B3 = ds3["∇κ∇B"][x_begin, z_begin3:end, time_range]

ε1 = ds1["ε"][x_begin, z_begin1:end, time_range]
ε2 = ds2["ε"][x_begin, z_begin2:end, time_range]
ε3 = ds3["ε"][x_begin, z_begin3:end, time_range]

χ1 = ds1["χ"][x_begin, z_begin1:end, time_range]#./Bz1
χ2 = ds2["χ"][x_begin, z_begin2:end, time_range]#./Bz2
χ3 = ds3["χ"][x_begin, z_begin3:end, time_range]#./Bz3

Ri1 = ds1["Rig"][x_begin, z_begin1:end, time_range]
Ri2 = ds2["Rig"][x_begin, z_begin2:end, time_range]
Ri3 = ds3["Rig"][x_begin, z_begin3:end, time_range]

z1 = zC[z_begin1:end].-zC[z_begin1]
zF1 = zF[z_begin1:end].-zF[z_begin1]
z2 = zC[z_begin2:end].-zC[z_begin2]
zF2 = zF[z_begin2:end].-zF[z_begin2]
z3 = zC[z_begin3:end].-zC[z_begin3]
zF3 = zF[z_begin3:end].-zF[z_begin3]
u1[u1.==0] .= NaN
u2[u2.==0] .= NaN
u3[u3.==0] .= NaN
w1[w1.==0] .= NaN
w2[w2.==0] .= NaN
w3[w3.==0] .= NaN
Bz1[Bz1.==0] .= NaN
Bz2[Bz2.==0] .= NaN
Bz3[Bz3.==0] .= NaN
∇κ∇B1[Bz1.==0] .= NaN
∇κ∇B2[Bz2.==0] .= NaN
∇κ∇B3[Bz3.==0] .= NaN
ε1[ε1.==0] .= NaN
ε2[ε2.==0] .= NaN
ε3[ε3.==0] .= NaN
χ1[χ1.==0] .= NaN
χ2[χ2.==0] .= NaN
χ3[χ3.==0] .= NaN
Ri1[Ri1.==0] .= NaN
Ri2[Ri2.==0] .= NaN
Ri3[Ri3.==0] .= NaN

# plot time series at selected depths
# Time axis (convert to tidal periods)
t_plot = t[time_range] ./ (2π / 1.4e-4)

row_labels = ["θ=0", "θ=0.008", "θ=0.02"]
z_axes = [z1, z2, z3]
zF_axes = [zF1, zF2, zF3]
# ========== FIGURE 1: u, w, Bz ==========
fig1, axes1 = subplots(3, 3, figsize=(18, 10))

col_titles1 = ["u (m/s)", "w (m/s)", "Bz (1/s²)"]
data_case1_fig1 = [u1, w1, Bz1]
data_case2_fig1 = [u2, w2, Bz2]
data_case3_fig1 = [u3, w3, Bz3]
all_data_fig1 = [data_case1_fig1, data_case2_fig1, data_case3_fig1]

for row in 1:3
    for col in 1:3
        ax = axes1[row, col]
        
        data = all_data_fig1[row][col]
        if col == 2  # w is defined on zF
            z_ax = zF_axes[row]
        else
            z_ax = z_axes[row]
        end
        
        # Create contourf (time vs z)
        if col == 1
            levels = range(-0.15, 0.15, length=21)
            pcm = ax.contourf(t_plot, z_ax, data, levels=levels, cmap="RdBu_r", extend="both")
        elseif col == 2
            levels = range(-0.025, 0.025, length=21)
            pcm = ax.contourf(t_plot, z_ax, data, levels=levels, cmap="RdBu_r", extend="both")
        else  # Bz
            levels = range(0, 2.5e-6, length=21)
            pcm = ax.contourf(t_plot, z_ax, data, levels=levels, cmap="RdBu_r", extend="both")
        end
        
        # Add colorbar
        colorbar(pcm, ax=ax)
        
        # Labels
        if row == 3
            ax.set_xlabel("Time (tidal periods)")
        else
            ax.set_xticklabels([])
        end
        
        if col == 1
            ax.set_ylabel("hab (m)")
            ax.text(-0.3, 0.5, row_labels[row], transform=ax.transAxes, 
                   fontsize=12, va="center", rotation=90, fontweight="bold")
        end
        
        if row == 1
            ax.set_title(col_titles1[col])
        end
        ax.set_xlim(0,10)
        # ax.set_ylim(0,2000)
        ax.grid(true, alpha=0.3)
    end
end

tight_layout()
savefig("output/2D_idealized_tilt_0/time_series_3cases_uwBz_beginning.png", dpi=150, bbox_inches="tight")
println("Saved figure to output/2D_idealized_tilt_0/time_series_3cases_uwBz_beginning.png")

# ========== FIGURE 2: ∇κ∇B, ε, χ ==========
fig2, axes2 = subplots(3, 3, figsize=(18, 10))

col_titles2 = ["∇κ∇B (m/s³)", "ε (m²/s³)", "χ (m²/s³)"]
data_case1_fig2 = [∇κ∇B1, ε1, χ1]
data_case2_fig2 = [∇κ∇B2, ε2, χ2]
data_case3_fig2 = [∇κ∇B3, ε3, χ3]
all_data_fig2 = [data_case1_fig2, data_case2_fig2, data_case3_fig2]

for row in 1:3
    for col in 1:3
        ax = axes2[row, col]
        
        data = all_data_fig2[row][col]
        z_ax = z_axes[row]
        
        # Create contourf (time vs z)
        if col == 1  # ∇κ∇B - symmetric colormap
            vmax = 1e-10
            levels = range(-vmax, vmax, length=21)
            pcm = ax.contourf(t_plot, z_ax, data, levels=levels, cmap="RdBu_r", extend="both")
        elseif col ==2  # ε and χ - log scale
            data[data.<0] .= NaN
            if length(data) > 0
                vmin = -9
                vmax = -7
                levels = range(vmin, vmax, length=21)
                pcm = ax.contourf(t_plot, z_ax, log10.(data), levels=levels, cmap="CMRmap", extend="both")
                
            end
        else
            data[data.<0] .= NaN
            if length(data) > 0
                vmin = -15
                vmax = -13
                levels = range(vmin, vmax, length=21)
                pcm = ax.contourf(t_plot, z_ax, log10.(data), levels=levels, cmap="CMRmap", extend="both")
                
            end
        end
        
        # Add colorbar
        colorbar(pcm, ax=ax)
        
        # Labels
        if row == 3
            ax.set_xlabel("Time (tidal periods)")
        else
            ax.set_xticklabels([])
        end
        
        if col == 1
            ax.set_ylabel("hab (m)")
            ax.text(-0.3, 0.5, row_labels[row], transform=ax.transAxes, 
                   fontsize=12, va="center", rotation=90, fontweight="bold")
        end
        
        if row == 1
            ax.set_title(col_titles2[col])
        end
        ax.set_xlim(195,200)
        # ax.set_ylim(0,500)
        ax.grid(true, alpha=0.3)
    end
end

tight_layout()
savefig("output/2D_idealized_tilt_0/_time_series_3cases_diffusion_dissipation.png", dpi=150, bbox_inches="tight")
println("Saved figure to output/2D_idealized_tilt_0/_time_series_3cases_diffusion_dissipation.png")