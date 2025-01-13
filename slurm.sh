#!/bin/bash

#SBATCH --job-name=audio_classification         
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=4         
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1                                  
#SBATCH --partition=small-g            
#SBATCH --time=24:00:00           
#SBATCH --account=project_465001389
#SBATCH --output=/users/doloriel/work/Repo/FrameMixer/logs/affia3k/panns_cnn6/none.out


# fma
# diffres
# specaugment
# specmix
# none

# panns_mobilenetv1
# panns_mobilenetv2
# panns_cnn6
# ast

# Load necessary modules (if required)
conda init
conda activate framemixer
cd /users/doloriel/work/Repo/FrameMixer

############################ AFFIA3K ############################
python train.py \
    --batch_size 200 \
    --max_epoch 500 \
    --wandb_mode "offline" \
    --dataset affia3k \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/affia3k \
    --model_name "panns_cnn6" \
    --spec_aug "none" \
    --num_classes 4 \
    --sample_rate 128000 \
    --window_size 2048 \
    --hop_size 1024 \
    --mel_bins 64 \
    --fmin 50 \
    --target_duration 2

############################ MRS-FFIA ############################
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset mrsffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
#     --model_name "ast" \
#     --spec_aug "none" \
#     --num_classes 4 \
#     --sample_rate 22050 \
#     --window_size 1024 \
#     --hop_size 512 \
#     --mel_bins 64 \
#     --fmin 1 \
#     --fmax 14000 \
#     --target_duration 3 \


############################ UFFIA ############################
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset uffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/uffia \
#     --model_name "panns_cnn6" \
#     --spec_aug "specmix" \
#     --num_classes 4 \
#     --sample_rate 64000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --fmin 1 \
#     --fmax 128000 \
#     --target_duration 2

