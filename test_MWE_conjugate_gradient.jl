# using Oceananigans
# using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver

# N = 2
# x = y = (0, 1)
# z = [0, 0.2, 1]
# grid = RectilinearGrid(size=(N, N, N); x, y, z, halo=(2, 2, 2), topology=(Bounded, Periodic, Bounded))
# mount(x, y=0) = 1 * exp(-(x^2 + y^2) / 2^2)
# bottom(x, y=0) = -1 + mount(x, y)
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

# preconditioner = fft_poisson_solver(grid.underlying_grid)
# pressure_solver = ConjugateGradientPoissonSolver(grid; preconditioner, maxiter=20)

# model = NonhydrostaticModel(; grid, pressure_solver)
# simulation = Simulation(model; Δt=0.1, stop_iteration=5)
# run!(simulation)



# using Printf
# using Statistics
# using Oceananigans
# using Oceananigans.Grids: with_number_type
# using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition
# using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver
# using Oceananigans.Utils: prettytime

# N = 16
# h, w = 50, 20
# H, L = 100, 100
# x = y = (-L/2, L/2)
# z = [0,2,4,6]
# # z = (-H, 0)

# grid = RectilinearGrid(size=(N, N, 3); x, y, z, halo=(2, 2, 2), topology=(Bounded, Periodic, Bounded))

# prescribed_flow = OpenBoundaryCondition(0.01)
# extrapolation_bc = FlatExtrapolationOpenBoundaryCondition()
# u_bcs = FieldBoundaryConditions(west = prescribed_flow,
#                                 east = extrapolation_bc)
#                                 #east = prescribed_flow)

# boundary_conditions = (; u=u_bcs)

# # # setting quadratic drag BC at domain bottom and top of the immersed boundary
# # z₀ = 0.1 # m (roughness length)
# # z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
# # cᴰ = (κ_von / log(z₁ / z₀))^2 # Drag coefficient
# # # non-immersed and immersed boundary conditions
# # @inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
# # @inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v
# # drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
# # drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))

# # u_bcs = FieldBoundaryConditions(bottom = drag_bc_u, top = FluxBoundaryCondition(nothing))
# # v_bcs = FieldBoundaryConditions(bottom = drag_bc_v, top = FluxBoundaryCondition(nothing))

# boundary_conditions = (; u=u_bcs)
# preconditioner = fft_poisson_solver(grid)
# pressure_solver = ConjugateGradientPoissonSolver(grid; preconditioner, maxiter=20)

# model = NonhydrostaticModel(; grid, boundary_conditions,pressure_solver)
# simulation = Simulation(model; Δt=0.1, stop_iteration=10)


# run!(simulation)






using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver
using MAT
using Statistics
using Oceanostics
using Interpolations: LinearInterpolation



    Nx = 16
    Ny = 4
    Nz = 4        
    ω₀ = 1.4e-4     # tidal freq.
    tᶠ = 0.1*2π/ω₀    # endtime of the simulation: 40 tidal cycle
    U₀ = 0.025      # tidal amplitude
    topo_file = "topo.mat"


## Simulation parameters
    Lx = 15kilometers # along-canyon extent
    Ly = 30kilometers # cross-canyon extent

underlying_grid = RectilinearGrid(CPU(),size=(Nx, Ny, Nz), 
       x = (0, Lx),
       y = (0, Ly), 
       z = [0,200,400,600,800]meters,
       halo = (4,4,4),
       topology = (Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded)
)

# load topography 
file = matopen(topo_file)
z_topo = read(file, "z_noslope_periodic") 
x_topo = read(file, "x_domain")
y_topo = read(file, "y_domain")
# grids has to be evenly spaced
x_topo_lin = range(x_topo[1],x_topo[end],size(z_topo,1))
y_topo_lin = range(y_topo[1],y_topo[end],size(z_topo,2))
close(file)
# high-resolution grids
x_interp = range(x_topo[1],x_topo[end], length=Nx)
y_interp = range(y_topo[1],y_topo[end], length=Ny)


# Interpolation object (caches coefficients and such)
itp = LinearInterpolation((x_topo_lin, y_topo_lin), z_topo)
# Interpolate z_topo onto a higher-resolution grid
z_interp = [itp(x, y) for x in x_interp, y in y_interp]
z_interp = z_interp.-minimum(z_interp)
bot = rand(16,4)
# mount(x, y) = 1 * exp(-(x^2 + y^2) / 2^2)
# bottom(x, y) = -1 + mount(x, y)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bot))

# Create immersed boundary grid
# grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(z_interp))

uᵢ(x, y, z) = 0.1

   model = NonhydrostaticModel(;
       grid=grid,
       pressure_solver = ConjugateGradientPoissonSolver(
               grid; preconditioner = fft_poisson_solver(underlying_grid)),
       advection = WENO(),
       tracers = :b,
       timestepper = :RungeKutta3,
   )
    
set!(model, u=uᵢ)

# ## Configure simulation
# Δt = (1/1e-3)*0.03
# simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

# run!(simulation)