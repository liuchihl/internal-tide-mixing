using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver

underlying_grid = RectilinearGrid(size=(4, 4, 4), 
       x = (0, 1),
       y = (0, 1), 
       z = [0, 0.2,0.4,0.6, 2],
       halo = (2,2,2),
       topology = (Oceananigans.Periodic, Oceananigans.Periodic, Oceananigans.Bounded)
)
    bottom =   [0.0  0.0  0.0   0.0;
                0.0  0.0  0.0   0.0;
                0.0  0.0  0.2   0.0;
                0.0  0.0  0.0   0.0]

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

    uᵢ(x, y, z) = 0.1

    model = NonhydrostaticModel(;
        grid=grid,
        pressure_solver = ConjugateGradientPoissonSolver(
                        grid; preconditioner = fft_poisson_solver(underlying_grid)),
        advection = WENO(),
        timestepper = :RungeKutta3,
    )
    set!(model, u=uᵢ)