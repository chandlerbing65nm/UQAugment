#!/bin/bash

# fma
# diffres
# specaugment
# specmix

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
    --spec_aug "diffres" \
    --num_classes 4 \
    --sample_rate 128000 \
    --window_size 2048 \
    --hop_size 1024 \
    --mel_bins 64 \
    --fmin 50 \
    --target_duration 2


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

############################ MRS-FFIA ############################
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset mrsffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
#     --model_name "panns_mobilenetv2" \
#     --spec_aug "specmix" \
#     --num_classes 4 \
#     --sample_rate 22050 \
#     --window_size 1024 \
#     --hop_size 512 \
#     --mel_bins 64 \
#     --fmin 1 \
#     --fmax 14000 \
#     --target_duration 3 \
