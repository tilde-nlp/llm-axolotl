#!/bin/bash

#SBATCH --account project_465001281
#SBATCH --partition dev-g  # or dev-g if you want test
#SBATCH --exclusive=user
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # 1 data-parallel process per node
#SBATCH --cpus-per-task=7        # for example, let it have 32 cpus
#SBATCH --gpus-per-node=mi250:8
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=llama30b_finetune

set -euo pipefail

export CC=gcc-12
export CXX=g++-12

export MEMORY_OPT_ALLREDUCE_SIZE=100000000

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
export WORLD_SIZE=8
# Set PyTorch distributed variables for multi-node run
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=8

# Possibly set your model parallel env if your code picks it up:
export MODEL_PARALLEL_SIZE=8   # or something Axolotl might read
# If your code doesn't read this, check how axolotl sets model parallel.
# Some frameworks need a param like --tensor-model-parallel-size 8.

CONTAINER_PATH=/scratch/project_465001281/containers/finetune/rocm624torch26_backup

# optional: we can do a run directory
mkdir -p workdir
wd=$(realpath workdir)
if [ ! -d "$wd"/cray-deps ] ; then
  rm -rf "$wd"/cray-deps
  mkdir "$wd"/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

# meme stuff
GPUS_PER_NODE=8
mkdir -p ./hostfiles
hostfile=./hostfiles/hosts_$SLURM_JOBID
# loop over the node names
for i in `scontrol show hostnames $SLURM_NODELIST`
do
    # add a line to the hostfile
    echo $i slots=$GPUS_PER_NODE >>$hostfile
done
export DLTS_HOSTFILE=./hostfiles/hosts_$SLURM_JOBID


# FIXME: dangerous iads
export RCCL_DISABLE_RSMI=1
echo "Starting Torchrun job with 1 nodes, 8 rank per node, 1 GPUs per rank..."

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 singularity exec "$CONTAINER_PATH" \
    torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE \
           --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
           --rdzv_id=llama30b_sft_run420204 \
           -m axolotl.cli.train examples/fft_TildeLM_fsdp2.yml


# Potential snippet to measure performance:
# One approach is to patch Axolotl's train loop to record:
# (1) total tokens processed
# (2) total samples
# (3) measure time across steps
# Then at the end, tokens/sec = total_tokens / total_time
# samples/sec = total_samples / total_time
# TFLOPs = (FMA_ops_per_token * total_tokens) / (time_seconds * 1e12)
# For a GPT-like model, FMA_ops_per_token ~ 2 * hidden_dim * n_layers * seq_len etc. (approx).