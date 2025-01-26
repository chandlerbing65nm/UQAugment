# #!/bin/bash

# #SBATCH --job-name=uq         
# #SBATCH --ntasks=1                 
# #SBATCH --cpus-per-task=4         
# #SBATCH --ntasks-per-node=1
# #SBATCH --mem-per-cpu=8G
# #SBATCH --gpus-per-node=1
# #SBATCH --nodes=1                                  
# #SBATCH --partition=standard-g            
# #SBATCH --time=24:00:00           
# #SBATCH --account=project_465001389
# #SBATCH --output=/users/doloriel/work/Repo/FrameMixer/logs/test/mcdropout20.out

# # Load necessary modules (if required)
# conda init
# conda activate framemixer
# cd /users/doloriel/work/Repo/FrameMixer

python uq.py \
    --batch_size 200 \
    --dataset affia3k \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/affia3k \
    --checkpoint  /scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints/ablation/none/affia3k_logmel_panns_cnn6_none_best_map_dur-2.0_sr-128000_win-2048_hop-1024_mel-64_fmin-50_fmax-none_cls-4_seed-20_bs-200_epoch-500_loss-ce_abl-none_unknown_withnoise_seg-0.3.pth \
    --model_name "panns_cnn6" \
    --spec_aug "none" \
    --num_classes 4 \
    --sample_rate 128000 \
    --window_size 2048 \
    --hop_size 1024 \
    --mel_bins 64 \
    --fmin 50 \
    --target_duration 2 \
    --ablation \
    --noise \
    --noise_segment_ratio 0.1 \