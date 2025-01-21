# using MPI
# using CUDA

# function main()
#     MPI.Init()

#     comm = MPI.COMM_WORLD
#     rank = MPI.Comm_rank(comm)
#     size = MPI.Comm_size(comm)

#     data_size = 64
#     d_data = CUDA.zeros(Float32, data_size)
#     CUDA.@sync d_data .= Float32.(1:data_size)

#     if rank == 0
#         println("CUDA-aware MPI Test Results:")
        
#         # Attempt a CUDA-aware MPI operation
#         try
#             MPI.Send(d_data, 0, 0, comm)
#             println("✓ Successfully performed CUDA-aware MPI operation")
#             println("✓ This MPI implementation is likely CUDA-aware")
#         catch e
#             println("✗ Failed to perform CUDA-aware MPI operation")
#             println("✗ This MPI implementation might not be CUDA-aware")
#             println("Error: ", e)
#         end

#         # Check MPI version
#         version = MPI.Get_version()
#         println("MPI Version: $(version.major).$(version.minor)")

#         # Check if running OpenMPI
#         if MPI.identify_implementation() == :OpenMPI
#             println("✓ Confirmed running OpenMPI")
#             openmpi_version = MPI.Get_library_version()
#             println("OpenMPI Version: $openmpi_version")
#         end
#     end

#     MPI.Finalize()
# end

# main()



using MPI
using CUDA
MPI.Init()

function sendrecv_works(grid)
    arch = architecture(grid)
    comm = arch.communicator
    rank = arch.local_rank
    size = MPI.Comm_size(comm)
    dst = mod(rank+1, size)
    src = mod(rank-1, size)
    N = 4
    FT = eltype(grid)
    send_mesg = CuArray{FT}(undef, N)
    recv_mesg = CuArray{FT}(undef, N)
    fill!(send_mesg, FT(rank))
    CUDA.synchronize()
    try
        MPI.Sendrecv!(send_mesg, dst, 0, recv_mesg, src, 0, comm)
        return true
    catch err
        @warn "MPI.Sendrecv test failed." exception=(err, catch_backtrace())
        return false
    end
end