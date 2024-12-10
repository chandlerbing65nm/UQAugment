#!/bin/bash

#SBATCH --job-name=audio_classification         
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=4         
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1                                  
#SBATCH --partition=standard-g            
#SBATCH --time=24:00:00           
#SBATCH --account=project_465001389
#SBATCH --output=/users/doloriel/work/slurm/uffia/ast-specmix2-lumi.out


# fma
# diffres
# specaugment
# specmix

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
#     --spec_aug "diffres" \
#     --fmin 50 \
#     --target_duration 2


############################ UFFIA ############################
python train.py \
    --batch_size 200 \
    --max_epoch 500 \
    --wandb_mode "offline" \
    --dataset uffia \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/uffia \
    --num_classes 4 \
    --sample_rate 64000 \
    --window_size 2048 \
    --hop_size 1024 \
    --mel_bins 64 \
    --model_name "ast" \
    --spec_aug "specmix" \
    --fmin 1 \
    --fmax 128000 \
    --target_duration 2

############################ MRS-FFIA ############################
# python train.py \
#     --batch_size 200 \
#     --max_epoch 500 \
#     --wandb_mode "offline" \
#     --dataset mrsffia \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
#     --num_classes 4 \
#     --sample_rate 22050 \
#     --window_size 1024 \
#     --hop_size 512 \
#     --mel_bins 64 \
#     --model_name "panns_mobilenetv1" \
#     --spec_aug "mixup" \
#     --fmin 1 \
#     --fmax 14000 \
#     --target_duration 3 \
