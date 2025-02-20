# #!/bin/bash

# panns_cnn6
# panns_mobilenetv2
# ast

# specaugment
# specmix
# none
# gaussian_noise
# pitch_shift
# time_stretch

############################ MRS-FFIA ############################
python uq.py \
    --batch_size 200 \
    --dataset mrsffia \
    --data_path /scratch/project_465001389/chandler_scratch/Datasets/mrsffia \
    --checkpoint /scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints_uq/mrsffia_logmel_ast_specmix_best_map_dur-3.0_sr-22050_win-1024_hop-512_mel-64_fmin-1_fmax-14000_cls-4_seed-20_bs-200_epoch-500_loss-ce.pth \
    --model_name "panns_cnn6" \
    --num_classes 4 \
    --sample_rate 22050 \
    --window_size 1024 \
    --hop_size 512 \
    --mel_bins 64 \
    --fmin 1 \
    --fmax 14000 \
    --target_duration 3 \
    --tta \
    --spec_aug "pitch_shift" \
    # --audiomentations time_stretch \

############################ AFFIA3K ############################
# python uq.py \
#     --batch_size 200 \
#     --dataset affia3k \
#     --data_path /scratch/project_465001389/chandler_scratch/Datasets/affia3k \
#     --checkpoint  /scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints_uq/affia3k_logmel_ast_none_best_map_dur-2.0_sr-128000_win-2048_hop-1024_mel-64_fmin-50_fmax-none_cls-4_seed-20_bs-200_epoch-500_loss-ce_audioment-pitch_shift.pth \
#     --model_name "panns_cnn6" \
#     --num_classes 4 \
#     --sample_rate 128000 \
#     --window_size 2048 \
#     --hop_size 1024 \
#     --mel_bins 64 \
#     --fmin 50 \
#     --target_duration 2 \
#     --tta \
#     --spec_aug "none" \
#     --audiomentations pitch_shift \
