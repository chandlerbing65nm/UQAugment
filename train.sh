#!/bin/bash

#SBATCH --job-name=affia3k         # Job name (optional)
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPUs per task
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --mem=16G                  # Total memory allocation
#SBATCH --partition=gpu            # Partition to submit to
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --time=48:00:00            # Time limit (hh:mm:ss)
#SBATCH --output=slurm/affia3k/*PANNS-CNN6(dropout=0.5)+DSTFT.out

# Load necessary modules (if required)
module purge
module load Python/3.10
conda init
conda activate ffia
cd Repositories/FFIA

# Run your job
# python train.py \
#     --model_name "ast" \
#     --sample_rate 16000 \
#     --window_size 256 \
#     --hop_size 128 \
#     --mel_bins 128 \
#     --fmax 8000 \
#     --wand_mode "online" \
#     --wand_project "affia-3k" \

python train.py \
    --model_name "panns_cnn6" \
    --frontend "dstft" \
    --wandb_mode "offline" \