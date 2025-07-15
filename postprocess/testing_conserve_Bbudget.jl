using NCDatasets
θ=0.0036

simname = "tilt"
tᶠ = 453.0
filename_3D_452 = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ-1, "_analysis_round=all_threeD.nc")
filename_3D_4525 = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ-0.5, "_analysis_round=all_threeD.nc")
filename_3D_453 = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_analysis_round=all_threeD.nc")
filename_avg_453 = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ, "_analysis_round=all_threeD_timeavg.nc")
filename_avg_4525 = string("output/", simname, "/internal_tide_theta=",θ,"_Nx=500_Nz=250_tᶠ=",tᶠ-0.5, "_analysis_round=all_threeD_timeavg.nc")

ds_3D_452 = Dataset(filename_3D_452)
ds_3D_4525 = Dataset(filename_3D_4525)
ds_3D_453 = Dataset(filename_3D_453)
ds_avg_4525 = Dataset(filename_avg_4525)
ds_avg_453 = Dataset(filename_avg_453)

dBdt = (ds_3D_453["B"][500,500,:,end] .- ds_3D_452["B"][500,500,:,end]) ./ (ds_3D_453["time"][end].-ds_3D_452["time"][end])

# ∇κ∇B_4525 = ds_avg_4525["∇κ∇B"][500,500,:,1]
# div_uB_4525 = ds_avg_4525["div_uB"][500,500,:,1]
# ∇κ∇B_453 = ds_avg_453["∇κ∇B"][500,500,:,1]
# div_uB_453 = ds_avg_453["div_uB"][500,500,:,1]
# ∇κ∇B = (∇κ∇B_4525 .+ ∇κ∇B_453) ./ 2
# div_uB = (div_uB_4525 .+ div_uB_453) ./2
# rhs = ∇κ∇B .- div_uB
zC = ds_3D["z_aac"][:]
∇κ∇B = mean(vcat(ds_3D_452["∇κ∇B"][500,500,:,end]',ds_3D_4525["∇κ∇B"][500,500,:,:]',ds_3D_453["∇κ∇B"][500,500,:,:]')',dims=2)
div_uB = mean(vcat(ds_3D_452["div_uB"][500,500,:,end]',ds_3D_4525["div_uB"][500,500,:,:]',ds_3D_453["div_uB"][500,500,:,:]')',dims=2)

rhs = ∇κ∇B .- div_uB
using PyPlot
close("all")
fig=PyPlot.figure(figsize=(10, 6))
plt.plot(rhs,zC, label="rhs: ∇κ∇B - div_uB")
plt.plot(dBdt, zC, label="lhs: dB/dt",linestyle="--")
legend()
ylabel("z")
xlim(-6e-9,3e-9)
savefig("output/tilt/budget_check.png")



using Statistics
filename_3D_460 = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=460_threeD_B-c.nc"
filename_avg_460 = "output/tilt/internal_tide_theta=0.0036_Nx=500_Nz=250_tᶠ=460_threeD_timeavg_const_dt_Bbudget-wb-eps-chi.nc"
ds_3D_460 = Dataset(filename_3D_460)
ds_avg_460 = Dataset(filename_avg_460)

dBdt_460 = (ds_3D_460["B"][500,500,:,36] .- ds_3D_460["B"][500,500,:,24]) ./ (ds_3D_460["time"][36].-ds_3D_460["time"][24])
∇κ∇B_460 = mean(ds_avg_460["∇κ∇B"][500,500,:,3], dims=2)
div_uB_460 = mean(ds_avg_460["div_uB"][500,500,:,3], dims=2)

zC = ds_3D_460["zC"][:]
rhs = ∇κ∇B_460 .- div_uB_460
using PyPlot
close("all")
fig=PyPlot.figure(figsize=(10, 6))
plt.plot(rhs,zC, label="rhs: ∇κ∇B - div_uB")
plt.plot(dBdt_460, zC, label="lhs: dB/dt",linestyle="--")
legend()
ylabel("z")
xlim(-6e-9,3e-9)
savefig("output/tilt/budget_check_460.png")

