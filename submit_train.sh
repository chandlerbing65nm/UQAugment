#!/bin/bash

#SBATCH --job-name=uqaugment         
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=4         
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1                                  
#SBATCH --partition=standard-g            
#SBATCH --time=24:00:00           
#SBATCH --account=project_465001389
#SBATCH --output=/users/doloriel/work/Repo/FrameMixer/logs/uqaugment/mrsffia/panns_cnn6/time_stretch.out

# Load necessary modules (if required)
conda init
conda activate framemixer
cd /users/doloriel/work/Repo/FrameMixer

# specaugment
# specmix
# none
# gaussian_noise
# pitch_shift
# time_stretch

# panns_cnn6
# panns_mobilenetv2
# ast

# --specaugment_params      \ #1: 32,1,4,1                  #2: 64,2,8,2                    #3: 128,4,16,4 (default)
# --specmix_params          \ #1: 0.3,4,8,1,1 (default)     #2: 0.5,8,16,2,2                #3: 0.7,16,32,4,4
# --gaussian_noise_params   \ #1: 0.005,0.02,0.3            #2: 0.01,0.05,0.5 (default)     #3: 0.02,0.1,0.8
# --pitch_shift_params      \ #1: -1,1,0.3 (default)        #2: 4,4,0.5                     #3: -6,6,0.8
# --time_stretch_params     \ #1: 0.95,1.05,0.3             #2: 0.8,1.25,0.5                #3: 0.7,1.5,0.8 (default)


############################ MRS-FFIA ############################
python train.py \
    --batch_size 200 \
    --max_epoch 500 \
    --wandb_mode "offline" \
    --dataset mrsffia \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
    --model_name "panns_cnn6" \
    --spec_aug "none" \
    --num_classes 4 \
    --sample_rate 22050 \
    --window_size 1024 \
    --hop_size 512 \
    --mel_bins 64 \
    --fmin 1 \
    --fmax 14000 \
    --target_duration 3 \
    --audiomentations time_stretch \
    # --ablation \
    # --noise \


# ############################ AFFIA3K ############################
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset affia3k \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/affia3k \
#     --model_name "ast" \
#     --spec_aug "none" \
#     --num_classes 4 \
#     --sample_rate 128000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --fmin 50 \
#     --target_duration 2 \
#     --audiomentations time_stretch \
#     # --ablation \
#     # --time_stretch_params=0.7,1.5,0.8 \
#     # --specmix_params 0.7,16,32,4,4
#     # --specaugment_params 128,4,16,4
#     # --noise \
#     # --frontend 'mfcc' \
#     # --audiomentation 'gaussian_noise'
#     # --diffres_params '0.10,False' \


