using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra
using MAT
using Statistics
using Oceanostics
using Oceanostics.TKEBudgetTerms: BuoyancyProductionTerm

suffix = "3days"

## Simulation parameters
 Nx = 20  #150 #250 500 1000
 Ny = 40 #300 #500 1000 2000
 Nz = 50 #250

 tᶠ = 0.5days # simulation run time
 Δtᵒ = 30minutes # interval for saving output

 H = 3.5kilometers # 4.926e3, 6.e3 # vertical extent
 Lx = 15kilometers
 Ly = 30kilometers

print("dx=",Lx/Nx,", dy=",Ly/Ny)

## Create grid
# Creates a vertical grid with near-constant spacing `refinement * Lz / Nz` near the bottom:
# "Warped" coordinate
kwarp(k, N) = (N + 1 - k) / N
# Linear near-surface generator
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
# Bottom-intensified stretching function
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))

# Generating function
# large refinement gets better resolution at the bottom but coarser at the surface
# large stretching: resolution difference between the bottom and surface increases
# z_faces(k) = - H * (ζ(k, Nz, 1.8) * Σ(k, Nz, 10) - 1)
z_faces(k) = - H * (ζ(k, Nz, 1.5) * Σ(k, Nz, 15) - 1)


grid = RectilinearGrid(size=(Nx, Ny, Nz), 
        x = (0, Lx),
        y = (0, Ly), 
        z = z_faces,
        halo = (4,4,4),
        topology = (Periodic, Periodic, Bounded)
)

using GLMakie

lines(zspacings(grid, Center()), znodes(grid, Center()),
     axis = (ylabel = "Depth (m)",
             xlabel = "Vertical spacing (m)"))

scatter!(zspacings(grid, Center()), znodes(grid, Center()))

# yᶜ = ynodes(grid, Center())
# Δyᶜ = yspacings(grid, Center())

# load topography 
file = matopen("/Users/chihlunliu/Library/CloudStorage/OneDrive-SharedLibraries-UCIrvine/Chih-Lun - Documents/UC Irvine/research/topo.mat")
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
#Ny=2Nx
y_interp = range(y_topo[1],y_topo[end], length=Ny)

using Interpolations
# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
# itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
z_interp = z_interp.-minimum(z_interp)

# grids
zC = znodes(grid, Center())
zF = znodes(grid, Face()); 
xC = xnodes(grid, Center())
yC = ynodes(grid, Center())
# find the grid that is above z_interp at x-y plane
inx = zeros(Nx,Ny)  # Preallocate inx array to store the indices
# create an array of indices that captures the frist element above the topography
# for i in 1:Nx
#    for j in 1:Ny
# inx[i,j] = findfirst(x -> x > z_interp[i,j], zC)
#    end
# end

# using LoopVectorization

# @allowscalar for i in 1:Nx
#     for j in 1:Ny
#         inx[i,j] = findfirst(x -> x > z_interp[i,j], zC)
#     end
# end


# using Plots, StatsBase

# function terrain_follow_average(ϕ)

# ## creating terrain-aligned horizontal average

# # Find the tallest point and use histogram to bin the vertical grids first
# binsize = ceil(maximum(diff(zF)));  # smallest binsize has to be larger than the maximun Δz
# row, col = findmax(z_interp)[2][1],findmax(z_interp)[2][2]
# h = fit(Histogram, zC[Int(inx[row,col]):end],[zC[Int(inx[row,col])]:binsize:maximum(zC);])
# bins = diff(h.edges[1])/2 .+ h.edges[1][1:end-1]   # central bin
# # preallocation
# temp = zeros(Nx,Ny,length(h.weights));
# u_TFM=zeros(1,length(h.weights))

# # loop over the entire x-y plane to get the depth histogram from the topography to the surface
# Threads.@threads for k in 1:length(h.weights)
#     for i in 1:Nx
#         for j in 1:Ny
#             h = fit(Histogram, zC[Int(inx[i,j]):end],[zC[Int(inx[i,j])]:binsize:maximum(zC);])
#             window = 0; 
#             # lg and ug are the lowest and upmost grids within the bin.
#             # For example, if zF = 0, 0.5, 1, 1.5, 2, 2.5, 3, and bins = 0.1, 2.1, 4.1. Within the first bin, lg=0 and ug=2
#             lg = Int(inx[i,j])+window # lowest grid in the bin
#             ug = Int(inx[i,j])+window+h.weights[k]-1 #upmost grid in the bin
#             # temp[i,j,k] = sum(ϕ[i,j,lg:ug].*diff(zF)[lg:ug]) ./ (zF[lg]-zF[ug])
#             temp[i,j,k] = sum(ϕ[i,j,lg:ug].*diff(zF)[lg:ug]) ./ (zF[lg]-zF[ug])
#             window = window + h.weights[k]
#         end
#     end
# end


# calculate without a loop
# h = fit(Histogram, zC[Int(inx[i,j]):end],[zC[Int(inx[i,j])]:binsize:maximum(zC);],weights = )
# bins = 0:binsize:maximum(z_interp)

# h = fit.(Histogram, eachrow(z_interp), Ref(bins), weights = grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ*grid.Δzᵃᵃᶜ[1:end-4])
# hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]
# hh = hab .* (hab .> 0)    # mask hab



# u.*reshape(grid.Δxᶜᵃᵃ*grid.Δyᵃᶜᵃ*grid.Δzᵃᵃᶜ[1:end-4],1,1,Nz)



# ϕ̄ = vec(mean(temp,dims=(1,2))) 
# # uu = vec(mean(temp,dims=(1,2))) 
# # shift the bins to 0
# return ϕ̄, bins.-minimum(bins)    
# end
# heatmap(x_interp, y_interp, z_interp'; color = :balance, xlabel = "x", ylabel = "z", aspect_ratio = :equal)

# Environmental parameters
N = 1.e-3 # Brunt-Väisälä buoyancy frequency
f₀ = 0.53e-4 # Coriolis frequency
θ = 3.6e-3 # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates


# Create immersed boundary grid
	# GridFittedBottom: real topography. GridFittedBoundary: need a mask (logical)
grid_real = ImmersedBoundaryGrid(grid, GridFittedBottom(z_interp)) 

# setting quadratic drag BC at domain bottom and top of the immersed boundary
z₀ = 0.1 # m (roughness length)
κ_von = 0.4  # von Karman constant

z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
cᴰ = (κ_von / log(z₁ / z₀))^2 # Drag coefficient

@inline drag_u(x, y, z, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, y, z, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ,))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ,))

u_immerse = ImmersedBoundaryCondition(top=drag_bc_u)
v_immerse = ImmersedBoundaryCondition(top=drag_bc_v)

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u, immersed=u_immerse)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v, immersed=v_immerse)
w_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0))      # ImpenetrableBoundaryCondition is used as default at other non-immersed boundary points

# no-flux boundary condition
normal = -N^2*cos(θ)    # normal slope 
cross = -N^2*sin(θ)     # cross slope
B_immerse = ImmersedBoundaryCondition(top=GradientBoundaryCondition(normal),
                                west = GradientBoundaryCondition(-cross), east = GradientBoundaryCondition(cross))
B_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(normal),immersed=B_immerse);


# # with background field, gradient boundary condition needs to be specified to make buoyancy flux at the boundary =0
# T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(),
#                                 bottom = GradientBoundaryCondition(0.01));

# Rotate gravity vector
buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -[ĝ...])
coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)

# Linear background stratification (in ẑ)
@inline ẑ(x, z, ĝ) = x*ĝ[1] .+ z*ĝ[3]
@inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

# Tidal forcing 
U₀ = 0.025
ω₀ = 1.4e-4
u_tidal_forcing(x, y, z, t) = U₀*ω₀*sin(ω₀*t)

#  Background buoyancy diffusion (set as a forcing)
# eddy_diffusivity = CenterField(grid)
# b_forcing_func(x, y, z, t, κ) = N^2*cos(θ)*eddy_diffusivity      #N^2*cos(θ)*Field(∂z(κ)*∂ẑ(z))
# b_forcing = Forcing(b_forcing_func, field_dependencies=:b)


# A term that damps the local velocity field in the presence of stratification
# using Oceananigans.Operators: ∂zᶠᶜᶠ, ℑxzᶠᵃᶜ

# function u_forcing_func(i, j, k, grid, clock, model_fields, ε)
#     # The vertical derivative of buoyancy, interpolated to the u-velocity location:
#     N² = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, model_fields.b)

#     # Set to zero in unstable stratification where N² < 0:
#     N² = max(N², zero(typeof(N²)))

#     return @inbounds - ε * sqrt(N²) * model_fields.u[i, j, k]
# end

# u_forcing = Forcing(u_forcing_func, discrete_form=true, parameters=1e-3)


s = sqrt((ω₀^2-f₀^2)/(N^2-ω₀^2))
#γ = h*π/(s*6kilometers)
#print("Steepness parameter of γ=",round(γ, digits=3))

model = NonhydrostaticModel(
    grid = grid_real,
    advection = WENO(),
    buoyancy = buoyancy,
    coriolis = coriolis,
    boundary_conditions=(u=u_bcs, v=v_bcs, w=w_bcs,  b = B_bcs,),
    forcing = (u = u_tidal_forcing,),
    closure = ScalarDiffusivity(; ν=1e-4, κ=1e-4),
    tracers = :b,
    timestepper = :RungeKutta3,
    background_fields = (; b=B̄_field),
)


# IC such that flow is in phase with predicted linear response, but otherwise quiescent
Uᵣ = U₀ * ω₀^2/(ω₀^2 - f₀^2 - (N*sin(θ))^2) # quasi-resonant linear barotropic response
uᵢ(x, y, z) = -Uᵣ
vᵢ(x, y, z) = 0.
bᵢ(x, y, z) = 1e-9*rand() # seed infinitesimal perturbations in buoyancy field

set!(model, b=bᵢ, u=uᵢ, v=vᵢ)

## Configure simulation
#Δt = (1/N)*0.03
Δt = 0.5 * minimum_zspacing(grid) / Uᵣ
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

# # The `TimeStepWizard` manages the time-step adaptively, keeping the Courant-Freidrichs-Lewy
# # (CFL) number close to `0.5` while ensuring the time-step does not increase beyond the
# # maximum allowable value for numerical stability.

wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.2)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


## Diagnostics
b = model.tracers.b
B̄ = model.background_fields.tracers.b
B = B̄ + b # total buoyancy field

u, v, w = model.velocities
û = @at (Face, Center, Center) u*ĝ[3] - w*ĝ[1] # true zonal velocity
ŵ = @at (Center, Center, Face) w*ĝ[3] + u*ĝ[1] # true vertical velocity
ζᶻ = ∂x(v) - ∂y(û)

ν = model.closure.ν
κ = model.closure.κ

# Oceanostics
KE = KineticEnergy(model)
ε = KineticEnergyDissipationRate(model)
χ = TracerVarianceDissipationRate(model, :b)
wb = BuoyancyProductionTerm(model)

# # Terrain-following horizontal averages
# ε_avg, bin = terrain_follow_average(ε) # includes the surface grid to the grid right above the topography
# χ_avg = terrain_follow_average(χ)[1]
# KE_avg = terrain_follow_average(KE)[1]
# wb_avg = terrain_follow_average(wb)[1]

# # Write custom vectors and arrays to disk
# ε_avg_disk(model) = ε_avg
# χ_avg_disk(model) = χ_avg
# KE_avg_disk(model) = KE_avg
# wb_avg_disk(model) = wb_avg
# bathy(model) = z_interp
            

Oceanostics_diags = (; KE, ε, wb, χ)
custom_diags = (; uhat=û, what=ŵ,B=B)
state_diags = merge(model.velocities, model.tracers, 
            model.background_fields.tracers,custom_diags) 


fname = string("internal_tide_", suffix,"-theta=",string(θ),"_realtopo3D_Nx",Nx)

## NetCDF output
# output_attributes = Dict(
#         "B"  => Dict("longname" => "total buoyancy field", "units" => "m/s²"),
#         "uhat" => Dict("longname" => "true zonal velocity", "units" => "m/s"),
#         "what"   => Dict("longname" => "true vertical velocity", "units" => "m/s")
#     )

# simulation.output_writers[:fields_nc] = NetCDFOutputWriter(model, state_diags,
#                                        schedule = TimeInterval(Δtᵒ),
#                                        filename = string("output/", fname, "_field.nc"),
#                                        verbose=true,
#                                        overwrite_existing = true)      
# output 2D slices
 #1) xz
simulation.output_writers[:slice_xz_nc] = NetCDFOutputWriter(model, state_diags,
                                       schedule = TimeInterval(Δtᵒ),
                                       indices = (:,Ny÷2,:), # center of the domain (on the canyon)
                                       #max_filesize = 500MiB, #needs to be uncommented when running large simulation
                                       verbose=true,
                                       filename = string("output/", fname, "_slices.nc"),
			                  		   overwrite_existing = true)


# Terrian-following horizontal avg.
# outputs = Dict("ε_avg" => ε_avg_disk, "χ_avg" => χ_avg_disk, 
#         "KE_avg" => KE_avg_disk, "wb_avg" => wb_avg_disk)

# dims = Dict("ε_avg" => ("zC",), "χ_avg" => ("zC",), "KE_avg" => ("zC",), 
#         "wb_avg" => ("zC",))
# output_attributes = Dict(
#         "ε_avg"  => Dict("long_name" => "Terrain-following horizontal average KE dissipation rate", "units" => "m²/s³"),
#         "χ_avg" => Dict("long_name" => "Terrain-following horizontal average buoyancy variance dissipation rate", "units" => "m²/s³"),
#         "wb_avg"   => Dict("long_name" => "Terrain-following horizontal average buoyancy flux", "units" => "m²/s³"),
#         "KE_avg"   => Dict("long_name" => "Terrain-following horizontal average KE", "units" => "m²/s²")
#         )

# simulation.output_writers[:TFH] = NetCDFOutputWriter(model, outputs,
#                                        schedule = TimeInterval(Δtᵒ), dimensions=dims,
#                                        filename = string("output/", fname, "_TF_horizontal_average.nc"),
#                                        output_attributes=output_attributes,
#                                        overwrite_existing = true)

# bathymetry = Dict("bathy" => bathy)
# dims = Dict("bathy" => ("xC","yC")) 
# output_attributes = Dict(
#         "bathy"   => Dict("long_name" => "Bathymetry", "units" => "m"))
# simulation.output_writers[:bathy] = NetCDFOutputWriter(model, bathymetry,
#                                        schedule = TimeInterval(0), dimensions=dims,
#                                        filename = "output/bathymetry.nc",
#                                        verbose=true,
#                                        output_attributes=output_attributes,
#                                        overwrite_existing = true)      
                                  
## Progress messages

progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, 
                            advective CFL: %.2e, diffusive CFL: %.2e\n",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w),
                            AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))

## Running the simulation!
run!(simulation)


@info """
    Simulation complete.
"""



# hab = ones(Nx,Ny).*reshape(zC,1,1,Nz) .- z_interp  # height above bottom [Nx,Ny,Nz]

# hh = hab .* (hab .> 0)    # mask hab
# hh[hh.==0] .= NaN
# hab_interp = 1:Nz  #range(0,maximum(hh),100)
# itp = interpolate(u, BSpline(Linear()))
# hh_interp = [itp[xC, yC, zi] for xi in xC, yi in yC, zi in hab_interp]


# x_interp = range(x_topo[1],x_topo[end], length=Nx)
# #Ny=2Nx
# y_interp = range(y_topo[1],y_topo[end], length=Ny)




# using Interpolations
# # Interpolation object (caches coefficients and such)
# itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# # Interpolate z_topo onto a higher-resolution grid
# # itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# z_interp = [itp(x_topo_lin, y_topo_lin) for x_topo_lin in x_interp, y_topo_lin in y_interp]
# z_interp = z_interp.-minimum(z_interp)