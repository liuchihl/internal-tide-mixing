## this script is a test of the bottom boundary condition as well as the added diffusive 
# flux forcing term by comparing the numerical result with the steady state analytical solution
# from Callies 2018 in terms of stratification. The setup is a gentle slope without topographic 
# features and the diffusivity is set to vary in the vertical direction.

using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using LinearAlgebra
using MAT
using Statistics
using Oceanostics
using Oceanostics.TKEBudgetTerms: BuoyancyProductionTerm
using Interpolations
using GLMakie
using NCDatasets
# using Oceanostics.PotentialEnergyEquationTerms: PotentialEnergy

suffix = "500days"

## Simulation parameters
 Nx = 250  #150 #250 500 1000
 Ny = 1 #300 #500 1000 2000
 Nz = 200 #250

 tᶠ = 500days # simulation run time
 Δtᵒ = 60minutes # interval for saving output

 H = 2kilometers
 Lx = 10kilometers#500meters#15kilometers
 Ly = 70meters#500meters

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
        topology = (Oceananigans.Periodic, Oceananigans.Periodic, Bounded)
)

# Environmental parameters
N = 1.3e-3              # Brunt-Väisälä buoyancy frequency        
f₀ = -5.5e-5            # Coriolis frequency
θ = 2e-3                # tilting of domain in (x,z) plane, in radians [for small slopes tan(θ)~θ]
ĝ = (sin(θ), 0, cos(θ)) # vertical (gravity-oriented) unit vector in rotated coordinates
κ₀ = 5.2e-5             # Far-Field diffusivity
κ₁ = 1.8e-3             # Bottom enhancement of diffusivity
h = 230meter            # decay scale of diffusivity
σ = 1                   # Prandtl number
ν₀ = κ₀
ν₁ = κ₁

# diffusivity
@inline κ(x,y,z,t) = κ₀ + κ₁*exp(-z / h)
closure = ScalarDiffusivity(;κ=κ, ν=κ)

# setting quadratic drag BC at domain bottom and top of the immersed boundary
z₀ = 0.1 # m (roughness length)
κ_von = 0.4  # von Karman constant

z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
cᴰ = (κ_von / log(z₁ / z₀))^2 # Drag coefficient
# non-immersed and immersed boundary conditions
@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v
@inline immersed_drag_u(x, y, z, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
@inline immersed_drag_v(x, y, z, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))
immersed_drag_bc_u = FluxBoundaryCondition(immersed_drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
immersed_drag_bc_v = FluxBoundaryCondition(immersed_drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))

u_immerse = ImmersedBoundaryCondition(top=immersed_drag_bc_u)
v_immerse = ImmersedBoundaryCondition(top=immersed_drag_bc_v)

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u, immersed=u_immerse)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v, immersed=v_immerse)
# ImpenetrableBoundaryCondition is used as default at other non-immersed boundary points
w_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0.0))   

# no-flux boundary condition
normal = -N^2*cos(θ)    # normal slope 
cross = -N^2*sin(θ)     # cross slope
B_immerse = ImmersedBoundaryCondition(bottom=GradientBoundaryCondition(normal),
                    west = GradientBoundaryCondition(cross), east = GradientBoundaryCondition(-cross))
B_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(normal),immersed=B_immerse);


# Rotate gravity vector
buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -[ĝ...])
coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)

# Linear background stratification (in ẑ)
@inline ẑ(x, z, ĝ) = x*ĝ[1] .+ z*ĝ[3]
@inline constant_stratification(x, y, z, t, p) = p.N² * ẑ(x, z, p.ĝ)
B̄_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N^2))

using Oceananigans.Operators: ∂zᶠᶜᶠ, ℑxzᶠᵃᶜ, ∂zᶜᶜᶠ, ℑzᵃᵃᶜ, ℑxzᶠᵃᶜ, Δzᶜᶜᶠ, Δzᶜᶜᶜ, Δzᶜᶜᶠ

# Tidal forcing 
U₀ = 0#0.025
ω₀ = 1.4e-4
u_tidal_forcing(x, y, z, t) = U₀*ω₀*sin(ω₀*t)

model = NonhydrostaticModel(
    grid = grid,
    advection = WENO(),
    buoyancy = buoyancy,
    coriolis = coriolis,
    boundary_conditions=(u=u_bcs, v=v_bcs, w=w_bcs,  b = B_bcs,),
    forcing = (u = u_tidal_forcing,),
    closure = closure,
    tracers = :b,
    timestepper = :RungeKutta3,
    background_fields = (; b=B̄_field),
)

    # discrete forcing
    import Oceananigans.TurbulenceClosures
    # κzᶠᶜᶜ = Oceananigans.TurbulenceClosures.κzᶠᶜᶜ
    κzᶜᶜᶠ = Oceananigans.TurbulenceClosures.κzᶜᶜᶠ
    
    using Oceananigans.Operators: ∂zᶠᶜᶠ, ℑxzᶠᵃᶜ, ∂zᶜᶜᶜ, ℑzᵃᵃᶜ, ℑxzᶜᵃᶠ, Δzᶜᶜᶠ
    
    function b_forcing_func(i, j, k, grid, clock, model_fields)
        
        # [κN²cosθ](z+Δz/2) - [κN²cosθ](z-Δz/2)     
    diffusive_flux = @inbounds ((κzᶜᶜᶠ(i, j, k+1, grid, model.closure, model.diffusivity_fields, Val(:b), clock, model_fields) *
                        ℑxzᶜᵃᶠ(i, j, k+1, grid, ∂zᶜᶜᶜ, model.background_fields.tracers.b) * cos(θ)) -
                        (κzᶜᶜᶠ(i, j, k, grid, model.closure, model.diffusivity_fields, Val(:b), clock, model_fields) *
                        ℑxzᶜᵃᶠ(i, j, k, grid, ∂zᶜᶜᶜ, model.background_fields.tracers.b) * cos(θ)) ) / Δzᶜᶜᶠ(i, j, k, grid)   
        # C,C,C ➡ C,C,C ➡ C,C,F
        return diffusive_flux
    end
    
    b_forcing = Forcing(b_forcing_func, discrete_form=true)


    model = NonhydrostaticModel(
        grid = grid,
        advection = WENO(),
        buoyancy = buoyancy,
        coriolis = coriolis,
        boundary_conditions=(u=u_bcs, v=v_bcs, w=w_bcs,  b = B_bcs,),
        forcing = (u = u_tidal_forcing, b = b_forcing),
        closure = closure,
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
Δt = (1/N)*0.03
# Δt = 0.5 * minimum_zspacing(grid) / Uᵣ
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)
wizard = TimeStepWizard(cfl=0.5, diffusive_cfl=0.2)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

## Diagnostics
b = model.tracers.b
B̄ = model.background_fields.tracers.b
B = B̄ + b # total buoyancy field

u, v, w = model.velocities
û = @at (Face, Center, Center) u*ĝ[3] - w*ĝ[1] # true zonal velocity
Bz = @at (Center, Center, Center) ∂z(B)            

fname = string("nonconstantdiffusivity", suffix,"-theta=",string(θ),"_forcing")


simulation.output_writers[:checkpointer] = Checkpointer(
                                        model,
                                        schedule=TimeInterval(200days),
                                        dir=fname,
                                        prefix=string(fname, "_checkpoint"),
                                        cleanup=true)

 #1) xz
simulation.output_writers[:slice_xz_nc] = NetCDFOutputWriter(model, (;B=B, Bz=Bz),
                                       schedule = TimeInterval(Δtᵒ),
                                       indices = (:,1,:), # center of the domain (on the canyon)
                                       #max_filesize = 500MiB, #needs to be uncommented when running large simulation
                                       verbose=true,
                                       filename = string("output/", fname, "_slices.nc"),
			                  		   overwrite_existing = true)

## Progress messages

progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, 
                            advective CFL: %.2e, diffusive CFL: %.2e\n",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w),
                            AdvectiveCFL(s.Δt)(s.model), DiffusiveCFL(s.Δt)(s.model))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δtᵒ))

## Running the simulation!
run!(simulation; pickup=false)



filename = "output/nonconstantdiffusivity500days-theta=0.002_noforcing_slices.nc"
ds_noforce = Dataset(filename,"r")
# grids
zC = ds_noforce["zC"]; Nz=length(zC)
xC = ds_noforce["xC"]; Nx=length(xC)
yC = ds_noforce["yC"]; Ny=length(yC)
t = ds_noforce["time"];

# load all data
B = ds_noforce["B"].var;       B = B[:,:,:,:];
Bz_noforce = ds_noforce["Bz"].var;       Bz_noforce = Bz_noforce[:,:,:,:];
Bz_noforce_mean = dropdims(mean(Bz_noforce,dims=1),dims=(1,2))

filename = "output/nonconstantdiffusivity50days-theta=0.002_forcing_slices.nc"
ds_force = Dataset(filename,"r")
# grids
zC = ds_force["zC"]; Nz=length(zC)
xC = ds_force["xC"]; Nx=length(xC)
yC = ds_force["yC"]; Ny=length(yC)
t = ds_force["time"];

# load all data
B = ds_force["B"].var;       B = B[:,:,:,:];
Bz_force = ds_force["Bz"].var;       Bz_force = Bz_force[:,:,:,:];
Bz_force_mean = dropdims(mean(Bz_force,dims=1),dims=(1,2))


## analytical steady solution from Callies 2018
z = znodes(grid, Center())
z = zC[:]
S = N^2*tan(θ)^2/f₀^2
q = (f₀^2*cos(θ)^2*(1+S*σ) / (4*(ν₀+ν₁)^2))^(1/4)
Bz_analytical = @. N^2*cos(θ)*(κ₀./(κ₀+κ₁*exp(-z/h)) + 
       κ₁*exp(-z/h)/(κ₀+κ₁*exp(-z/h))*S*σ/(1+S*σ) -
       (κ₀/(κ₀+κ₁)+κ₁/(κ₀+κ₁)*S*σ/(1+S*σ))*exp(-q*z)*(cos(q*z)+sin(q*z))
     )

#      fig = Figure();
#      axis_kwargs = (xlabel = "zonal distance (x)",
#                        ylabel = "elevation (z)",
#                        limits = ((0, ds_noforce["xF"][end]), (0, 100)),
#                        )
#      ax_B = Axis(fig[1, 1]; title = "B", axis_kwargs...)
#      ax_Bz = Axis(fig[2, 1]; title = "Bz", axis_kwargs...)
     
#      heatmap!(ax_B,xC[:],zC[:],B[:,1,:,end],
#      colormap = :matter,lowclip=cgrad(:matter)[1], highclip=cgrad(:matter)[end], 
#      colorrange=(minimum(B),1e-4))
#      heatmap!(ax_Bz,xC[:],zC[:],Bz[:,1,:,end])
# fig

fig = Figure();
axis_kwargs = (xlabel = "Bz",
                  ylabel = "elevation (z)",
                  limits = ((0, 2e-6), (0, 2000)),
                  )
ax_B = Axis(fig[1, 1]; axis_kwargs...)


# inx = argmin(abs.(t/86400 .- 100))
# lines!(ax_B,Bz_noforce_mean[:,inx],z,linestyle=:solid)
# inx = argmin(abs.(t/86400 .- 200))
# lines!(ax_B,Bz_noforce_mean[:,inx],z,linestyle=:solid)
# inx = argmin(abs.(t/86400 .- 300))
# lines!(ax_B,Bz_noforce_mean[:,inx],z,linestyle=:solid)
# inx = argmin(abs.(t/86400 .- 400))
# lines!(ax_B,Bz_noforce_mean[:,inx],z,linestyle=:solid)
# # lines!(ax_B,Bz_noforce_mean[:,end],z)
# lines!(ax_B,Bz_noforce_mean[:,end],z,linestyle=:solid)



lines!(ax_B,Bz_force_mean[:,end],z,linestyle=:solid)
lines!(ax_B,Bz_force_mean[:,end-1000],z,linestyle=:solid)

lines!(ax_B,Bz_analytical,z,color=:red)
# save("Bz_force.png",fig)