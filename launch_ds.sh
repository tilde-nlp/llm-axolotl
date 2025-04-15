#!/bin/bash

#SBATCH --account project_465001281
#SBATCH --partition dev-g
#SBATCH --exclusive=user
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=mi250:8
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=ds_llama30b_finetune

set -euo pipefail

module purge
module load CrayEnv
module load cray-python/3.9.13.1
module load gcc/12.2.0
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=0
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=PHB

# figure out master address/port
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=1337

CONTAINER_PATH=/scratch/project_465001281/containers/finetune/rocm624torch26

# again, set model parallel environment if needed:
export MODEL_PARALLEL_SIZE=8

echo "Launching DeepSpeed on 16 nodes, 1 rank/node, 8 GPUs each..."

# We'll run srun with 1 task per node. That 1 task calls deepspeed.
# We'll pass --num_nodes=16 --num_gpus=8 so that the DS launcher can spawn local ranks for each GPU.
srun --cpu-bind=mask_cpu:0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000 singularity exec "$CONTAINER_PATH" \
  deepspeed --num_nodes 16 --num_gpus 8 \
    -m axolotl.cli.train fft_TildeLM.yml \
    --deepspeed fft_TildeLM.yml
