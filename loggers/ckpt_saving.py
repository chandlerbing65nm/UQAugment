# File: checkpoints/ckpt_saving.py

import os
import torch

def save_checkpoint(model, args, best_val_map, best_val_acc, current_val_map, current_val_acc):
    ckpt_dir = f'/scratch/project_465001389/chandler_scratch/Projects/UWAC/checkpoints/{args.frontend}/{args.loss}'
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the best model based on validation accuracy
    if current_val_map > best_val_map:
        best_val_map = current_val_map
        print(f"Best validation map updated to {best_val_map:.4f}")
        torch.save(model.state_dict(), f'{ckpt_dir}/{args.model_name}_{args.freq_band.lower()}_band_model_best_map.pth')

    # Save the best model based on validation accuracy
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        print(f"Best validation acc updated to {best_val_acc:.4f}")
        torch.save(model.state_dict(), f'{ckpt_dir}/{args.model_name}_{args.freq_band.lower()}_band_model_best_acc.pth')

    return best_val_map, best_val_acc
