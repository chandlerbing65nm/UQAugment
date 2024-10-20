#!/bin/bash

#SBATCH --job-name=affia3k         
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=4         
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1                                  
#SBATCH --partition=small-g            
#SBATCH --time=24:00:00           
#SBATCH --account=project_465001389
#SBATCH --output=/users/doloriel/work/slurm/affia3k/ast-diffres(dtw3)-lumi.out

# Load necessary modules (if required)
conda init
conda activate uwac
cd /users/doloriel/work/Repo/UWAC

# First run
python train.py \
    --model_name "ast" \
    --frontend "ours" \
    --batch_size 200 \
    --wandb_mode "offline"

# # Second run (manually handle output redirection)
# python train.py \
#     --model_name "panns_cnn6" \
#     --frontend "ours" \
#     --batch_size 200 \
#     --wandb_mode "offline"

# # Third run (manually handle output redirection)
# python train.py \
#     --model_name "panns_cnn6" \
#     --frontend "ours" \
#     --batch_size 200 \
#     --wandb_mode "offline"
