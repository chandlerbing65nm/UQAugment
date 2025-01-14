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

# --specaugment_params '64,2,8,2' \ # 32,1,4,1 # 64,2,8,2 # 128,4,16,4
# --diffres_params '0.60,False' \ # 0.10,False # 0.60,False # 0.90,False
# --specmix_params '0.5,8,16,2,2' \ # 0.3,4,8,1,1 # 0.5,8,16,2,2 # 0.7,16,32,4,4

# Load necessary modules (if required)
conda init
conda activate framemixer
cd /users/doloriel/work/Repo/FrameMixer

############################ UFFIA ############################
python test.py \
    --batch_size 200 \
    --max_epoch 500 \
    --wandb_mode "offline" \
    --dataset uffia \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/uffia \
    --checkpoint /scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints/uffia_logmel_panns_mobilenetv2_specaugment_best_map_dur-2.0_sr-64000_win-2048_hop-1024_mel-64_fmin-1_fmax-128000_cls-4_seed-20_bs-200_epoch-500_loss-ce.pth \
    --model_name "panns_mobilenetv2" \
    --spec_aug "specaugment" \
    --num_classes 4 \
    --sample_rate 64000 \
    --window_size 2048 \
    --hop_size 1024 \
    --mel_bins 64 \
    --fmin 1 \
    --fmax 128000 \
    --target_duration 2

############################ AFFIA3K ############################
# python test.py \
#     --batch_size 200 \
#     --dataset affia3k \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/affia3k \
#     --checkpoint /scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints/ablation/specmix/affia3k_logmel_panns_cnn6_specmix_best_map_dur-2.0_sr-128000_win-2048_hop-1024_mel-64_fmin-50_fmax-none_cls-4_seed-20_bs-200_epoch-500_loss-ce_abl-specmix_0.7,16,32,4,4.pth \
#     --model_name "panns_cnn6" \
#     --spec_aug "specmix" \
#     --num_classes 4 \
#     --sample_rate 128000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --fmin 50 \
#     --target_duration 2 \
#     # --ablation \
#     # --specmix_params '0.7,16,32,4,4' \

############################ MRS-FFIA ############################
# python test.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset mrsffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
#     --checkpoint /scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints/mrsffia_logmel_panns_mobilenetv2_none_best_map_dur-3.0_sr-22050_win-1024_hop-512_mel-64_fmin-1_fmax-14000_cls-4_seed-20_bs-200_epoch-500_loss-ce.pth \
#     --model_name "panns_mobilenetv2" \
#     --spec_aug "none" \
#     --num_classes 4 \
#     --sample_rate 22050 \
#     --window_size 1024 \
#     --hop_size 512 \
#     --mel_bins 64 \
#     --fmin 1 \
#     --fmax 14000 \
#     --target_duration 3 \



