import os
import torch

def save_checkpoint(model, args, best_val_map, best_val_acc, current_val_map, current_val_acc):
    ckpt_dir = f'/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    # Construct a formatted string for additional arguments to include in the filename
    params_str = (
        f"dur-{args.target_duration}_sr-{args.sample_rate}_win-{args.window_size}_hop-{args.hop_size}_"
        f"mel-{args.mel_bins}_fmin-{args.fmin}_fmax-{args.fmax or 'none'}_"
        f"cls-{args.num_classes}_seed-{args.seed}_bs-{args.batch_size}_"
        f"epoch-{args.max_epoch}_loss-{args.loss}"
    )

    # Save the best model based on validation mAP
    if current_val_map > best_val_map:
        best_val_map = current_val_map
        print(f"Best validation map updated to {best_val_map:.4f}")
        torch.save(
            model.state_dict(),
            f'{ckpt_dir}/{args.dataset}_{args.frontend}_{args.model_name}_{args.spec_aug}_best_map_{params_str}.pth'
        )

    # Save the best model based on validation accuracy
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        print(f"Best validation acc updated to {best_val_acc:.4f}")
        torch.save(
            model.state_dict(),
            f'{ckpt_dir}/{args.dataset}_{args.frontend}_{args.model_name}_{args.spec_aug}_best_acc_{params_str}.pth'
        )

    return best_val_map, best_val_acc
