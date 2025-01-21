using MPI
MPI.Init()
println("CUDA-aware MPI: ", MPI.has_cuda())
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!


partition = Partition(1,2,1)
comm = MPI.COMM_WORLD
arch = Distributed(GPU(); partition, communicator=comm)
x = y = z = (0, 1)
grid = RectilinearGrid(arch; size=(16, 16, 16), x, y, z, topology=(Periodic, Periodic, Bounded))
c = CenterField(grid)
fill_halo_regions!(c)