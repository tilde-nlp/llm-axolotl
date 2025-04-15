#!/bin/bash

#SBATCH --account project_465001281
#SBATCH --partition dev-g  # or dev-g if you want test
#SBATCH --exclusive=user
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1        # 1 data-parallel process per node
#SBATCH --cpus-per-task=56         # for example, let it have 32 cpus
#SBATCH --gpus-per-node=mi250:8
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=llama30b_finetune

set -euo pipefail

module purge
module load CrayEnv
module load cray-python/3.9.13.1
module load gcc/12.2.0
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems

# recommended environment variables for NCCL/RCCL on LUMI:
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=0
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=PHB
# etc. as in your original script

# Master address / port
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=1337

# Possibly set your model parallel env if your code picks it up:
export MODEL_PARALLEL_SIZE=8   # or something Axolotl might read
# If your code doesn't read this, check how axolotl sets model parallel.
# Some frameworks need a param like --tensor-model-parallel-size 8.

CONTAINER_PATH=/scratch/project_465001281/containers/finetune/rocm624torch26

# optional: we can do a run directory

echo "Starting Torchrun job with 16 nodes, 1 rank per node, 8 GPUs per rank..."

srun --cpu-bind=mask_cpu:0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000 singularity exec "$CONTAINER_PATH" \
  torchrun --nnodes=16 --nproc_per_node=8 \
           --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
           --rdzv_id=llama30b_sft_run \
           -m axolotl.cli.train -m axolotl.cli.train fft_TildeLM.yml


# Potential snippet to measure performance:
# One approach is to patch Axolotl's train loop to record:
# (1) total tokens processed
# (2) total samples
# (3) measure time across steps
# Then at the end, tokens/sec = total_tokens / total_time
# samples/sec = total_samples / total_time
# TFLOPs = (FMA_ops_per_token * total_tokens) / (time_seconds * 1e12)
# For a GPT-like model, FMA_ops_per_token ~ 2 * hidden_dim * n_layers * seq_len etc. (approx).
