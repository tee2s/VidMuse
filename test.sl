#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=audiocraft_31a205cf
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --open-mode=append
#SBATCH --output=/work/users/t/i/tis/VidMuse/output/VidMuse/xps/31a205cf/submitit/%j_0_log.out
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --signal=USR2@90
#SBATCH --time=600
#SBATCH --wckey=submitit

# setup
source ~/.bashrc
echo "FFMPEG located at $(which ffmpeg)"
echo "$(ffmpeg -version)"
cd /work/users/t/i/tis/VidMuse
source .venv/bin/activate

sleep 60
scontrol show job $SLURM_JOB_ID
