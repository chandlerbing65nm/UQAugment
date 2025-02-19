#!/bin/bash

# fma
# diffres
# specaugment
# specmix

# panns_mobilenetv1
# panns_mobilenetv2
# panns_cnn6
# panns_resnet22
# ast


# --specaugment_params '64,2,8,2' \ # 32,1,4,1 # 64,2,8,2
# --diffres_params '0.60,False' \ # 0.10,False # 0.60,False # 0.90,False
# --specmix_params '0.5,8,16,2,2' \ # 0.3,4,8,1,1 # 0.5,8,16,2,2 # 0.7,16,32,4,4

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
    --model_name "ast" \
    --spec_aug "fma" \
    --num_classes 4 \
    --sample_rate 128000 \
    --window_size 2048 \
    --hop_size 2048 \
    --mel_bins 64 \
    --fmin 50 \
    --target_duration 2 \
    --ablation \
    # --noise \
    # --frontend 'mfcc' \
    # --audiomentation 'gaussian_noise'
    # --diffres_params '0.10,False' \

############################ MRS-FFIA ############################
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset mrsffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
#     --model_name "ast" \
#     --spec_aug "fma" \
#     --num_classes 4 \
#     --sample_rate 22050 \
#     --window_size 1024 \
#     --hop_size 512 \
#     --mel_bins 64 \
#     --fmin 1 \
#     --fmax 14000 \
#     --target_duration 3 \
#     --ablation \
#     --noise \

############################ UFFIA ############################
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset uffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/uffia \
#     --model_name "panns_cnn6" \
#     --spec_aug "fma" \
#     --num_classes 4 \
#     --sample_rate 64000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --fmin 1 \
#     --fmax 128000 \
#     --target_duration 2
