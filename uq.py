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

def compute_ece(probs, labels, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE).
    Args:
        probs (ndarray): shape (N, C), predicted probabilities for each sample.
        labels (ndarray): shape (N,), true class indices for each sample.
        n_bins (int): Number of bins to use for ECE.
    Returns:
        float: The ECE value.
    """
    # For each sample, the predicted confidence is max(prob)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Bin boundaries
        bin_lower, bin_upper = bins[i], bins[i+1]
        # Indices for samples whose confidence falls into this bin
        in_bin = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
        if len(in_bin) > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_prob = len(in_bin) / len(probs)
            ece += np.abs(bin_confidence - bin_accuracy) * bin_prob

    return ece

def compute_nll(probs, labels):
    """
    Compute the Negative Log-Likelihood (NLL).
    Args:
        probs (ndarray): shape (N, C), predicted probabilities for each sample.
        labels (ndarray): shape (N,), true class indices for each sample.
    Returns:
        float: The average NLL across all samples.
    """
    # For each sample i, the predicted probability for the true class is probs[i, labels[i]]
    eps = 1e-12  # for numerical stability
    true_class_probs = probs[np.arange(len(labels)), labels]
    nll = -np.mean(np.log(true_class_probs + eps))
    return nll

def save_results(args, test_acc, test_map, test_f1, ece, nll):
    """Save test results to a file."""
    results_dir = 'results/'
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

    # Results file path
    results_path = f"{save_dir}/{args.dataset}_{args.frontend}_{args.model_name}_{args.spec_aug}_results_{params_str}.txt"

    # Save results
    with open(results_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test mAP: {test_map:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Expected Calibration Error (ECE): {ece:.4f}\n")
        f.write(f"Negative Log-Likelihood (NLL): {nll:.4f}\n")

    print(f"Results saved to {results_path}")

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
    # (4) Suggest how many MC Dropout runs for UQ. Let's pick 20 by default.
    num_mc_runs = 20

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print arguments
    print("Arguments:")
    pprint(vars(args))
    print(f"Recommended MC Dropout runs for UQ = {num_mc_runs} (can be adjusted)")

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
    # Then compute mean & std for uncertainty, plus ECE & NLL
    all_test_targets = []
    # We'll collect MC predictions so we can compute std dev
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

            # (5) Perform MC Dropout by multiple forward passes in training mode
            mc_preds = []
            for _ in range(num_mc_runs):
                model.train()  # ensure dropout is active
                # (2) Check if self.training is True
                if model.training:
                    print("Dropout is active...")
                else:
                    raise ValueError('self.training==False')

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

    # Mean & STD over the MC dimension
    # shape (TotalSamples, C)
    mean_preds = all_mc_preds.mean(axis=-1)
    std_preds = all_mc_preds.std(axis=-1)  # for your uncertainty measure (item 7)

    # One-hot for mAP
    all_test_targets_one_hot = label_binarize(all_test_targets, classes=np.arange(args.num_classes))

    # Standard classification metrics from average predictions
    test_acc = accuracy_score(all_test_targets, mean_preds.argmax(axis=-1))
    test_map = average_precision_score(all_test_targets_one_hot, mean_preds, average='weighted')
    test_f1 = f1_score(all_test_targets, mean_preds.argmax(axis=-1), average='weighted')

    # (8,9) ECE from the average predictions
    ece = compute_ece(mean_preds, all_test_targets, n_bins=10)

    # (10) NLL from the average predictions
    nll = compute_nll(mean_preds, all_test_targets)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test mAP: {test_map:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Negative Log-Likelihood (NLL): {nll:.4f}")

    # Save results
    save_results(args, test_acc, test_map, test_f1, ece, nll)

if __name__ == '__main__':
    main()