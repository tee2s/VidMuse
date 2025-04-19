#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --exclude=''
#SBATCH --gres=gpu:1
#SBATCH --job-name=audiocraft_31a205cf
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/work/users/t/i/tis/VidMuse/output/VidMuse/xps/31a205cf/submitit/%j_0_log.out
#SBATCH --partition=a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --signal=USR2@90
#SBATCH --time=600
#SBATCH --wckey=submitit

# setup
source ~/.bashrc
echo "FFMPEG located at $(which ffmpeg)"
echo "$(ffmpeg -version)"
cd /work/users/t/i/tis/demucs
source .venv/bin/activate

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /work/users/t/i/tis/VidMuse/output/VidMuse/xps/31a205cf/submitit/%j_%t_log.out /work/users/t/i/tis/VidMuse/.venv/bin/python3 -u -m submitit.core._submit /work/users/t/i/tis/VidMuse/output/VidMuse/xps/31a205cf/submitit
