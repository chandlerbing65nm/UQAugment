#!/bin/bash

# ===================== SLURM Configuration =====================
#SBATCH --job-name=uqfishaugment         
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=4         
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1                                  
#SBATCH --partition=small-g            
#SBATCH --time=24:00:00           
#SBATCH --account=project_465001897
#SBATCH --output=/users/doloriel/work/Repo/UQFishAugment/logs/ablation/mrsffia/panns_cnn6/time_stretch-3.out

# ===================== Environment Setup =====================
conda init
conda activate framemixer
cd /users/doloriel/work/Repo/UQFishAugment

# ===================== Augmentation Parameters Reference =====================
# Time Mask Parameters:
#   1. 0.01,0.05,0.3
#   2. 0.05,0.15,0.5 (default)
#   3. 0.1,0.3,0.8
#
# Band Stop Filter Parameters:
#   1. 200,1000,0.3
#   2. 500,3000,0.5
#   3. 1000,6000,0.8 (default)
#
# Gaussian Noise Parameters:
#   1. 0.005,0.02,0.3
#   2. 0.01,0.05,0.5 (default)
#   3. 0.02,0.1,0.8
#
# Pitch Shift Parameters:
#   1. -1,1,0.3 (default)
#   2. -4,4,0.5
#   3. -6,6,0.8
#
# Time Stretch Parameters:
#   1. 0.95,1.05,0.3
#   2. 0.8,1.25,0.5
#   3. 0.7,1.5,0.8 (default)

# ===================== Training Configuration =====================
# Available Models: panns_cnn6, panns_mobilenetv2, ast
# Available Augmentations: time_mask, band_stop_filter, gaussian_noise, pitch_shift, time_stretch

# ===================== MRS-FFIA =====================
python train.py \
    --batch_size 200 \
    --max_epoch 500 \
    --wandb_mode "offline" \
    --dataset mrsffia \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
    --model_name "panns_cnn6" \
    --num_classes 4 \
    --sample_rate 22050 \
    --window_size 1024 \
    --hop_size 512 \
    --mel_bins 64 \
    --fmin 1 \
    --fmax 14000 \
    --target_duration 3 \
    --audiomentations time_stretch \
    --ablation \
    --time_stretch_params=0.7,1.5,0.8

# ===================== AVFFIA Training =====================
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset uffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/uffia \
#     --model_name "panns_mobilenetv1" \
#     --num_classes 4 \
#     --sample_rate 64000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --fmin 1 \
#     --fmax 128000 \
#     --target_duration 2 \
#     --audiomentations band_stop_filter

# ===================== AFFIA3K Training =====================
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset affia3k \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/affia3k \
#     --model_name "ast" \
#     --num_classes 4 \
#     --sample_rate 128000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --fmin 50 \
#     --target_duration 2 \
#     --audiomentations band_stop_filter
#     # --ablation \
#     # --time_mask_params 0.05,0.15,0.5 \
#     # --band_stop_filter_params 1000,6000,0.8 \
#     # --gaussian_noise_params 0.01,0.05,0.5 \
#     # --pitch_shift_params -1,1,0.3 \
#     # --time_stretch_params 0.7,1.5,0.8
