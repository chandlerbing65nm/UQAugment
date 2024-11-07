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
#SBATCH --output=/users/doloriel/work/slurm/affia3k/ast-specmix2-lumi.out

# Load necessary modules (if required)
conda init
conda activate uwac
cd /users/doloriel/work/Repo/UWAC


python train.py \
    --model_name "ast" \
    --spec_aug "specmix" \
    --batch_size 200 \
    --wandb_mode "offline"

# python train.py \
#     --batch_size 200 \
#     --max_epoch 400 \
#     --wandb_mode "offline" \
#     --dataset uffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/uffia \
#     --num_classes 4 \
#     --sample_rate 64000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --model_name "ast" \
#     --spec_aug "specaugment" \
#     --fmin 1 \
#     --fmax 128000 \
