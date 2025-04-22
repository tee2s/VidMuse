#!/bin/sh

#SBATCH --job-name=two_tasks
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00

module load cuda

srun --cpu-bind=cores bash -c '
  echo "Rank $SLURM_PROCID on GPU $CUDA_VISIBLE_DEVICES"
  nvidia-smi -L
'