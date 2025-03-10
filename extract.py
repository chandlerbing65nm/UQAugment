import os
import ssl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pprint import pprint
import ipdb

from config.config import parse_args
from methods.model_selection import get_model
from transforms.audio_transforms import get_transforms
from datasets.dataset_selection import get_dataloaders

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
from datasets.noise import get_dataloader as noise_loader

warnings.filterwarnings("ignore", category=UserWarning)

def load_checkpoint(model, checkpoint_path):
    """Load model weights from the checkpoint."""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)  # No 'model_state_dict' key
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

def save_probabilities(args, all_test_targets, all_mc_preds):
    """Save all_test_targets and all_mc_preds to a file in the 'probs/' folder."""

    # Construct a formatted string for additional arguments to include in the filename
    params_str = (
        f"dur-{args.target_duration}_sr-{args.sample_rate}_win-{args.window_size}_hop-{args.hop_size}_"
        f"mel-{args.mel_bins}_fmin-{args.fmin}_fmax-{args.fmax or 'none'}_"
        f"cls-{args.num_classes}_seed-{args.seed}_bs-{args.batch_size}_"
        f"epoch-{args.max_epoch}_loss-{args.loss}"
    )

    # Add ablation parameters if applicable
    if args.ablation:
        if args.spec_aug == 'specaugment':
            ablation_params = args.specaugment_params
        elif args.spec_aug == 'diffres':
            ablation_params = args.diffres_params
        elif args.spec_aug == 'specmix':
            ablation_params = args.specmix_params
        else:
            ablation_params = "unknown"
        params_str += f"_abl-{args.spec_aug}_{ablation_params}"

    # Add audiomentations parameters if applicable
    if args.audiomentations:
        audiomentations_str = "-".join(args.audiomentations)
        params_str += f"_audioment-{audiomentations_str}"
        
        if args.ablation:
            # Append additional parameters if specific augmentations are chosen
            if 'gaussian_noise' in args.audiomentations:
                params_str += f"_gaussian_noise_params-{args.gaussian_noise_params}"
            if 'pitch_shift' in args.audiomentations:
                params_str += f"_pitch_shift_params-{args.pitch_shift_params}"
            if 'time_stretch' in args.audiomentations:
                params_str += f"_time_stretch_params-{args.time_stretch_params}"

    # Add noise toggle note if both ablation and noise are True
    if args.ablation and args.noise:
        params_str += f"_withnoise_seg-{args.noise_segment_ratio}"

    # File path for saving probabilities
    probs_path = f"probs/{args.dataset}/{args.model_name}/{args.frontend}_{args.spec_aug}_probs_{params_str}.npz"

    # Ensure the folder exists
    os.makedirs(f"probs/{args.dataset}/{args.model_name}", exist_ok=True)

    # Save probabilities
    np.savez(probs_path, all_test_targets=all_test_targets, all_mc_preds=all_mc_preds)
    print(f"Probabilities saved to {probs_path}")

def add_poisson_noise_in_segment(inputs, noise_batch, alpha=0.1, lam=100, segment_ratio=0.1):
    """
    Adds noise to `inputs` in a localized segment for each sample in the batch.
    """
    B, L = inputs.shape
    poisson_offsets = torch.poisson(torch.full((B,), lam, dtype=torch.float, device=inputs.device))
    poisson_offsets = torch.clamp(poisson_offsets, max=L-1)

    seg_length = int(segment_ratio * L)
    if seg_length < 1:
        seg_length = 1  # at least 1 sample

    for i in range(B):
        start = int(poisson_offsets[i].item())
        end = int(min(start + seg_length, L))
        inputs[i, start:end] += alpha * noise_batch[i, start:end]

    return inputs

def main():
    args = parse_args()

    # You can customize this to an argument if you'd like
    # how many MC Dropout runs for UQ. Let's pick 20 by default.
    # If num_mc_runs = 1, it means TTA is used not MCDropout
    num_mc_runs = 1

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print arguments
    print("Arguments:")
    pprint(vars(args))
    print(f"MC Dropout runs for UQ = {num_mc_runs} (can be adjusted)")
    print(f"If num_mc_runs = 1, it means TTA is used not MCDropout")

    # Initialize model
    model = get_model(args).to(device)

    # Load trained weights
    load_checkpoint(model, args.checkpoint)

    # Get transforms
    transform = get_transforms(args)

    # Initialize test data loader
    _, _, test_dataset, test_loader = get_dataloaders(args, transform)

    # Noise data loaders
    _, noisetest_loader = noise_loader(split='val',
                                       batch_size=args.batch_size//10,
                                       sample_rate=args.sample_rate,
                                       shuffle=False, seed=args.seed,
                                       drop_last=True, transform=None, args=args)
    noise_iter = iter(noisetest_loader)

    # We'll store predictions across the entire test set
    all_test_targets = []
    all_mc_preds = []  # each entry will be shape (B, C, num_mc_runs)

    # Evaluate in a no_grad block, but we will do .train() for dropout
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)
            all_test_targets.append(targets.argmax(dim=-1).cpu().numpy())

            # Add noise if ablation & noise
            if args.ablation and args.noise:
                try:
                    noise_batch = next(noise_iter)['waveform'].to(device)
                except StopIteration:
                    noise_iter = iter(noisetest_loader)
                    noise_batch = next(noise_iter)['waveform'].to(device)

                repeat_factor = (inputs.size(0) + noise_batch.size(0) - 1) // noise_batch.size(0)
                noise_batch = noise_batch.repeat(repeat_factor, 1)[:inputs.size(0)]
                lam = int(args.target_duration * args.sample_rate) // 2
                segment_ratio = args.noise_segment_ratio
                inputs = add_poisson_noise_in_segment(inputs, noise_batch, lam=lam, segment_ratio=segment_ratio)

            # Perform MC Dropout by multiple forward passes in training mode
            mc_preds = []
            model.train()  # Enable dropout for MC runs
            for _ in range(num_mc_runs):
                outputs_run = model(inputs)['clipwise_output']
                outputs_run = torch.softmax(outputs_run, dim=-1)
                mc_preds.append(outputs_run.unsqueeze(-1))  # shape (B, C, 1)

            # Concatenate along last dimension => shape (B, C, num_mc_runs)
            mc_preds = torch.cat(mc_preds, dim=-1)
            all_mc_preds.append(mc_preds.cpu().numpy())

    # Concatenate all targets
    all_test_targets = np.concatenate(all_test_targets, axis=0)
    # Concatenate MC preds => shape (TotalSamples, C, num_mc_runs)
    all_mc_preds = np.concatenate(all_mc_preds, axis=0)

    # Save probabilities
    save_probabilities(args, all_test_targets, all_mc_preds)

if __name__ == '__main__':
    main()