#!/bin/bash

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

############################ MRS-FFIA ############################
python test.py \
    --batch_size 200 \
    --max_epoch 500 \
    --wandb_mode "offline" \
    --dataset mrsffia \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
    --checkpoint /scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints/mrsffia_logmel_panns_mobilenetv2_none_best_map_dur-3.0_sr-22050_win-1024_hop-512_mel-64_fmin-1_fmax-14000_cls-4_seed-20_bs-200_epoch-500_loss-ce.pth \
    --model_name "panns_mobilenetv2" \
    --spec_aug "none" \
    --num_classes 4 \
    --sample_rate 22050 \
    --window_size 1024 \
    --hop_size 512 \
    --mel_bins 64 \
    --fmin 1 \
    --fmax 14000 \
    --target_duration 3 \

############################ AFFIA3K ############################
# python test.py \
#     --batch_size 200 \
#     --dataset affia3k \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/affia3k \
#     --checkpoint /scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints/affia3k_logmel_ast_none_best_map_dur-2.0_sr-128000_win-2048_hop-1024_mel-64_fmin-50_fmax-none_cls-4_seed-20_bs-200_epoch-500_loss-ce.pth \
#     --model_name "ast" \
#     --spec_aug "none" \
#     --num_classes 4 \
#     --sample_rate 128000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --fmin 50 \
#     --target_duration 2


############################ UFFIA ############################
# python test.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset uffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/uffia \
#     --num_classes 4 \
#     --sample_rate 64000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --model_name "panns_cnn6" \
#     --spec_aug "specmix" \
#     --fmin 1 \
#     --fmax 128000 \
#     --target_duration 2
