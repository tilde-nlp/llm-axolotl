#!/bin/bash
#SBATCH --job-name=llama30b-ft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1            # 1 launcher process per node
#SBATCH --gpus-per-node=16             # 8 MI250 cards (16 GPU devices) per node&#8203;:contentReference[oaicite:7]{index=7}
#SBATCH --cpus-per-task=64             # CPU cores per task (adjust as needed per node)
#SBATCH --partition=large-g            # Example partition name (adjust to LUMI's GPU partition)
#SBATCH --time=02:00:00
#SBATCH --account=<YOUR_PROJECT_ID>

module load rocm/6.2.4            # Load ROCm environment (if applicable on LUMI)
# Activate your Python environment with Axolotl, PyTorch 2.6.0, etc.
source activate my-axolotl-env    # (Or use your env setup method)

# ROCm/HPC environment tweaks for LUMI (Cray Slingshot networking and MIOpen caching):
export NCCL_SOCKET_IFNAME=hsn               # Use HPE Slingshot high-speed network interface&#8203;:contentReference[oaicite:8]{index=8}
export NCCL_NET_GDR_LEVEL=3                 # Enable GPU Direct RDMA for RCCL (optimal setting on Slingshot)&#8203;:contentReference[oaicite:9]{index=9}
export CXI_FORK_SAFE=1                      # Required for Cray CXI (Slingshot) libfabric provider&#8203;:contentReference[oaicite:10]{index=10}
export CXI_FORK_SAFE_HP=1                   # Ensure fork safety (for dataloader forks, etc.)
export FI_CXI_DISABLE_CQ_HUGETLB=1          # Workaround for certain libfabric hugepage issues
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}   # Use node-local RAM disk for MIOpen (kernel cache)&#8203;:contentReference[oaicite:11]{index=11}

# Set PyTorch distributed variables for multi-node run
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500   # Port for c10d rendezvous (choose an open port)
export WORLD_SIZE=$(( $SLURM_NNODES * 16 ))   # total number of processes (GPUs)
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=16

# Launch distributed training via torchrun (PyTorch distributed launcher)
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE \
             --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
             axolotl train /shared/path/to/your_config.yaml
