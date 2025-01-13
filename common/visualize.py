import os
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config.config import parse_args
from datasets.dataset_selection import get_dataloaders
from transforms.audio_transforms import get_transforms
from methods.model_selection import get_model
from argparse import Namespace

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    import numpy as np
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_spectrogram_figure(spectrogram, save_path):
    """
    Save a spectrogram as a figure.
    
    Args:
        spectrogram (torch.Tensor): Spectrogram tensor of shape [1, frames, frequency].
        save_path (str): Path to save the figure.
    """
    spectrogram_np = spectrogram.squeeze(0).cpu().numpy()  # Remove batch dimension
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_np.T, aspect='auto', origin='lower', cmap='viridis')  # Transpose for correct axis
    plt.colorbar(label="Intensity")
    plt.xlabel("Frames")
    plt.ylabel("Frequency")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args, sample_index):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Get transforms and dataloaders
    transform = get_transforms(args)
    _, train_loader, _, _ = get_dataloaders(args, transform)

    # Initialize augmented and non-augmented models
    model_augm = get_model(args).to(device)
    model_augm.train()

    non_aug_args = Namespace(**vars(args))
    non_aug_args.spec_aug = None
    model_orig = get_model(non_aug_args).to(device)
    model_orig.train()

    # Create output directory
    os.makedirs("figures", exist_ok=True)

    random_idx = None  # Variable to store the random index for consistency

    for batch in train_loader:
        inputs = batch['waveform'].to(device)

        # Generate spectrograms
        with torch.no_grad():
            x_augm = model_augm(inputs)['augmented']  # Shape: [batch, 1, frames, frequency]
            x_orig = model_orig(inputs)['augmented']  # Shape: [batch, 1, frames, frequency]

        # Validate the input index
        if sample_index < 0 or sample_index >= x_augm.size(0):
            raise ValueError(f"Invalid sample index: {sample_index}. Must be between 0 and {x_augm.size(0) - 1}.")

        spectrogram_augm = x_augm[sample_index]
        spectrogram_orig = x_orig[sample_index]

        # Save figures
        save_path_augm = f"figures/{args.dataset}_{args.spec_aug}_augm_{sample_index}idx.png"
        save_path_orig = f"figures/{args.dataset}_orig_{sample_index}idx.png"

        save_spectrogram_figure(spectrogram_augm, save_path_augm)
        save_spectrogram_figure(spectrogram_orig, save_path_orig)

        print(f"Figures saved: {save_path_augm}, {save_path_orig}")
        break  # Only process the first batch

if __name__ == "__main__":
    # Parse default arguments
    args = parse_args()

    # Override some arguments for the script
    override_args = Namespace(
        batch_size=200,
        dataset="affia3k",
        data_path="/scratch/project_465001389/chandler_scratch/Datasets/affia3k",
        spec_aug="specaugment", # fma, diffres, specaugment, specmix
        num_classes=4,
        sample_rate=128000,
        window_size=2048,
        hop_size=1024,
        mel_bins=64,
        fmin=50,
        target_duration=2,
        seed=42,
    )

    # Update the default arguments with the hardcoded overrides
    for key, value in vars(override_args).items():
        setattr(args, key, value)

    # Input sample index
    sample_index = 167  # Replace with your desired index

    # Run the main function
    main(args, sample_index)
