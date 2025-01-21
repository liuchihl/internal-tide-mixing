#!/bin/bash
#SBATCH --job-name="multiGPU"
#SBATCH --output="output_message/internal-tide-realtopo.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest
#SBATCH --account=bcpi-delta-gpu
#SBATCH --exclusive
#SBATCH -t 24:00:00

module purge
module load gcc/11.4.0
module load cuda/12.4.0
module load openmpi/4.1.6
# module load openmpi/4.1.5+cuda
module load julia/1.11.1


# Export OPENMPI_HOME
export OPENMPI_HOME=$(dirname $(dirname $(which mpirun)))
# Add environment variables for CUDA-aware MPI
export LD_LIBRARY_PATH=$OPENMPI_HOME/lib:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_HAS_CUDA=true
export JULIA_CUDA_MEMORY_POOL=none

export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_MPI_BINARY=system
# steps to fix MPIError(18): MPI_ERR_IN_STATUS: error code in status
# export FI_CXI_RX_MATCH_MODE=software 
# unset OMPI_MCA_mtl_ofi_provider_include
# export OMPI_MCA_mtl_ofi_provider_exclude=cxi

# Create launch script for GPU binding
cat > launch.sh << 'EOF'
#!/bin/bash

export MPICH_GPU_SUPPORT_ENABLED=1
export LOCAL_RANK=${SLURM_LOCALID}
export GLOBAL_RANK=${SLURM_PROCID}
export CUDA_VISIBLE_DEVICES=$(expr ${LOCAL_RANK} % 2)  # Changed to 2 since there are 2 GPUs per node

echo "Global Rank ${GLOBAL_RANK} / Local Rank ${LOCAL_RANK} / CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} / $(hostname)"

exec $*
EOF

chmod +x launch.sh

# Clean up existing MPI preferences
# julia --project -e 'using Pkg; Pkg.rm("MPI"); Pkg.rm("MPIPreferences")'
# rm -f ~/.julia/compiled/v1.10/MPI/
# rm -f ~/.julia/compiled/v1.10/MPIPreferences/

# Configure Julia environment
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.add("MPI"); Pkg.add("MPIPreferences"); Pkg.add("CUDA")'
# Configure system MPI
# julia --project -e "ENV[\"OPENMPI_HOME\"] = \"$OPENMPI_HOME\"; using MPIPreferences; MPIPreferences.use_system_binary()"
julia --project -e "ENV[\"OPENMPI_HOME\"] = \"$OPENMPI_HOME\"; using MPIPreferences; MPIPreferences.use_system_binary(library_names=[\"libmpi\"], extra_paths=[\"$OPENMPI_HOME/lib\"])"
# Check MPI library path
julia --project -e 'using MPI; println("MPI library path: ", MPI.libmpi)'
# Precompile
julia --project -e 'using MPI; using CUDA; CUDA.precompile_runtime()'
# Run your simulation
srun -n 2 ./launch.sh julia --project test_multiGPU.jl
# srun -n 2 julia --project test_multiGPU.jl
