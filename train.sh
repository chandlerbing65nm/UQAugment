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
#SBATCH --output=/users/doloriel/work/slurm/fsdnoisy18k/cnn6-specaugment2-lumi.out


# fma(temp=0.2)
# diffres
# specaugment

# Load necessary modules (if required)
conda init
conda activate uwac
cd /users/doloriel/work/Repo/UWAC

############################ AFFIA3K ############################
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset affia3k \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/affia3k \
#     --num_classes 4 \
#     --sample_rate 128000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --model_name "panns_cnn6" \
#     --spec_aug "fma" \
#     --fmin 50 \
#     --fmax None

############################ UFFIA ############################
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
#     --model_name "panns_cnn6" \
#     --spec_aug "fma" \
#     --fmin 1 \
#     --fmax 128000 \

############################ DEBUG DATASETS ############################
python train.py \
    --batch_size 64 \
    --max_epoch 200 \
    --wandb_mode "offline" \
    --dataset fsdnoisy18k \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/fsdnoisy18k \
    --num_classes 20 \
    --sample_rate 16000 \
    --window_size 512 \
    --hop_size 128 \
    --mel_bins 64 \
    --model_name "panns_cnn6" \
    --spec_aug "specaugment" \
    --fmin 1 \
    --fmax 8000 \
    --target_duration 2

