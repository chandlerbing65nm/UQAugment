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
#SBATCH --output=/users/doloriel/work/slurm/affia3k/cnn6-nafa(gaussian_0.2temp2)-lumi.out

# Load necessary modules (if required)
conda init
conda activate uwac
cd /users/doloriel/work/Repo/UWAC


srun python train.py \
    --model_name "panns_cnn6" \
    --spec_aug "nafa" \
    --batch_size 200 \
    --wandb_mode "offline"
