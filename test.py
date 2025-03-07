import os
import ssl
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from pprint import pprint

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

def save_results(args, test_acc, test_map, test_f1):
    """Save test results to a file."""
    # Base results directory
    results_dir = 'results/'

    # If ablation is enabled, create a subdirectory for ablations
    if args.ablation:
        save_dir = os.path.join(results_dir, f'ablation/{args.spec_aug}')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = results_dir

    os.makedirs(results_dir, exist_ok=True)

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
    if hasattr(args, 'audiomentations') and args.audiomentations:
        audiomentations_str = "-".join(args.audiomentations)
        params_str += f"_audioment-{audiomentations_str}"

    # Add noise toggle note if both ablation and noise are True
    if args.ablation and args.noise:
        params_str += f"_withnoise_seg-{args.noise_segment_ratio}"

    # Define the results file path
    results_path = f"{save_dir}/{args.dataset}_{args.frontend}_{args.model_name}_{args.spec_aug}_results_{params_str}.txt"
    
    # Save results to the file
    with open(results_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test mAP: {test_map:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")

    print(f"Results saved to {results_path}")

def add_poisson_noise_in_segment(inputs, noise_batch, alpha=0.1, lam=100, segment_ratio=0.1):
    """
    Adds noise to `inputs` in a localized segment for each sample in the batch.

    Args:
        inputs (torch.Tensor): Shape [B, L], your clean audio batch.
        noise_batch (torch.Tensor): Shape [B, L], the noise batch (already repeated to match B).
        alpha (float): Scaling factor for how loud the noise is.
        lam (float): Lambda for the Poisson distribution controlling start offsets.
        segment_ratio (float): Fraction of total length for the noise segment size.
    Returns:
        torch.Tensor: The inputs with localized noise added.
    """
    B, L = inputs.shape

    # Sample B offsets from a Poisson distribution, each offset indicating where noise starts.
    # (Shape = [B], each entry is a random int >= 0)
    poisson_offsets = torch.poisson(torch.full((B,), lam, dtype=torch.float, device=inputs.device))

    # Make sure offsets don't exceed the audio length.
    # You might experiment with a smaller or bigger clamp depending on lam, etc.
    poisson_offsets = torch.clamp(poisson_offsets, max=L-1)

    # Decide how large the noise window is. E.g., 10% of total length
    seg_length = int(segment_ratio * L)
    if seg_length < 1:
        seg_length = 1  # at least 1 sample

    for i in range(B):
        start = int(poisson_offsets[i].item())  # Cast to int
        end = int(min(start + seg_length, L))  # Cast to int
        # Mix noise into inputs only in [start : end]
        inputs[i, start:end] += alpha * noise_batch[i, start:end]

    return inputs

def main():
    args = parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print arguments
    print("Arguments:")
    pprint(vars(args))

    # Initialize model
    model = get_model(args).to(device)

    # Load trained weights
    load_checkpoint(model, args.checkpoint)

    # Get transforms
    transform = get_transforms(args)

    # Initialize test data loader
    _, _, test_dataset, test_loader = get_dataloaders(args, transform)

    # Noise data loaders
    _, noisetest_loader = noise_loader(split='val', batch_size=args.batch_size//10, sample_rate=args.sample_rate, shuffle=False, seed=args.seed, drop_last=True, transform=None, args=args)

    # Model evaluation
    model.eval()
    all_test_targets = []
    all_test_outputs = []

    noise_iter = iter(noisetest_loader)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            # Add noise if args.ablation and args.noise are True
            if args.ablation and args.noise:
                try:
                    noise_batch = next(noise_iter)['waveform'].to(device)
                except StopIteration:
                    # Restart noise loader iterator when exhausted
                    noise_iter = iter(noisetest_loader)
                    noise_batch = next(noise_iter)['waveform'].to(device)

                repeat_factor = (inputs.size(0) + noise_batch.size(0) - 1) // noise_batch.size(0)
                noise_batch = noise_batch.repeat(repeat_factor, 1)[:inputs.size(0)]
                
                lam = int(args.target_duration * args.sample_rate) // 2
                segment_ratio = args.noise_segment_ratio
                inputs = add_poisson_noise_in_segment(inputs, noise_batch, lam=lam, segment_ratio=segment_ratio)

            # Forward pass
            if any(keyword in args.model_name for keyword in ('panns', 'ast')):
                outputs = model(inputs)['clipwise_output']
            else:
                outputs = model(inputs)

            outputs = torch.softmax(outputs, dim=-1)  # for classification metrics

            # Store predictions and targets
            all_test_targets.append(targets.argmax(dim=-1).cpu().numpy())
            all_test_outputs.append(outputs.detach().cpu().numpy())

    # Compute test metrics
    all_test_targets = np.concatenate(all_test_targets, axis=0)
    all_test_targets_one_hot = label_binarize(all_test_targets, classes=np.arange(args.num_classes))
    all_test_outputs = np.concatenate(all_test_outputs, axis=0)

    test_acc = accuracy_score(all_test_targets, all_test_outputs.argmax(axis=-1))
    test_map = average_precision_score(all_test_targets_one_hot, all_test_outputs, average='weighted')
    test_f1 = f1_score(all_test_targets, all_test_outputs.argmax(axis=-1), average='weighted')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test mAP: {test_map:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    # Save results
    save_results(args, test_acc, test_map, test_f1)

if __name__ == '__main__':
    main()
