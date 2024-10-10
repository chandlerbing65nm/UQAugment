#!/bin/bash

#SBATCH --job-name=affia3k         
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=4         
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1                                  
#SBATCH --partition=standard-g            
#SBATCH --time=24:00:00           
#SBATCH --account=project_465001389
#SBATCH --output=/users/doloriel/work/slurm/affia3k/*AST+DiffRes(trial).out

# Load necessary modules (if required)
conda init
conda activate uwac
cd Repo/UWAC

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
    --model_name "ast" \
    --frontend "diffres" \
    --batch_size 200 \
    --wandb_mode "offline" \