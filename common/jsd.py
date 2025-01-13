import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets.dataset_selection import get_dataloaders
from config.config import parse_args
from transforms.audio_transforms import get_transforms
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from methods.model_selection import get_model
from pprint import pprint
from argparse import Namespace
from tqdm import tqdm

from methods.panns.pytorch_utils import *
from methods.panns.models import *

from specaug.diffres.frontend import DiffRes
from specaug.fma.frontend import FMA
from specaug.specmix.frontend import SpecMix

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def compute_jsd(x_input, x_aug):
    """
    Compute the Jensen-Shannon Divergence between x_input and x_aug.
    First take the mean over dimension 2 (time steps), then flatten.
    """
    batch_size = x_input.size(0)

    # Take the mean over the time dimension (dim=2)
    x_input_mean = x_input.mean(dim=2)  # Shape: (batch_size, 1, feat_dim)
    x_aug_mean = x_aug.mean(dim=2)      # Shape: (batch_size, 1, feat_dim)

    # Flatten the tensors
    x_input_flat = x_input_mean.view(batch_size, -1)
    x_aug_flat = x_aug_mean.view(batch_size, -1)

    # Apply softmax to get probability distributions
    p = torch.nn.functional.softmax(x_input_flat, dim=1)
    q = torch.nn.functional.softmax(x_aug_flat, dim=1)

    # Compute the pointwise mean
    m = 0.5 * (p + q)

    # Compute KL divergence between p and m, and q and m
    kl_pm = torch.sum(p * (torch.log(p + 1e-10) - torch.log(m + 1e-10)), dim=1)
    kl_qm = torch.sum(q * (torch.log(q + 1e-10) - torch.log(m + 1e-10)), dim=1)

    # Compute JSD per sample
    jsd_per_sample = 0.5 * (kl_pm + kl_qm)

    # Average over the batch
    jsd_mean = torch.mean(jsd_per_sample)
    return jsd_mean

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_checkpoint(model, checkpoint_path):
    """Load model weights from the checkpoint."""
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)  # No 'model_state_dict' key
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducibility
    set_seed(args.seed)

    transform = get_transforms(args)

    # Initialize augmented model
    model_augm = get_model(args).to(device)
    # load_checkpoint(model_augm, args.checkpoint)
    model_augm.train()

    # Create a copy of the arguments for the non-augmented model
    non_aug_args = Namespace(**vars(args))
    non_aug_args.spec_aug = None  # Disable spec augmentation for the non-augmented model

    # Initialize non-augmented model
    model_orig = get_model(non_aug_args).to(device)
    # load_checkpoint(model_orig, args.checkpoint)
    model_orig.train()

    total_jsd = 0.0
    num_batches = 0

    # Get dataloaders
    _, train_loader, _, val_loader = get_dataloaders(args, transform)

    # Process all batches
    for batch in tqdm(train_loader):
        inputs = batch['waveform'].to(device)  # Ensure data is on the same device

        # Apply spectrogram and log-mel extraction
        with torch.no_grad():
            x_augm = model_augm(inputs)['augmented']
            x_orig = model_orig(inputs)['augmented']  # Non-augmented model output

            import ipdb; ipdb.set_trace() 
            print(x_augm.shape)
            print(x_orig.shape)

        # Compute the Jensen-Shannon Divergence
        with torch.no_grad():
            jsd = compute_jsd(x_orig, x_augm)

        total_jsd += jsd.item()
        num_batches += 1

    # Compute the average JSD over all batches
    if num_batches > 0:
        average_jsd = total_jsd / num_batches
        print(f"Average Jensen-Shannon Divergence over {num_batches} batches: {average_jsd}")
    else:
        print("No batches were processed.")


if __name__ == "__main__":
    # Parse default arguments
    args = parse_args()

    # AFFIA3K
    override_args = Namespace(
        batch_size=200,
        dataset="affia3k",
        data_path="/scratch/project_465001389/chandler_scratch/Datasets/affia3k",
        spec_aug="diffres", # fma, diffres, specaugment, specmix
        num_classes=4,
        sample_rate=128000,
        window_size=2048,
        hop_size=1024,
        mel_bins=64,
        fmin=50,
        target_duration=2,
        seed=42 # Seed for reproducibility
    )

    # # MRS-FFIA
    # override_args = Namespace(
    #     batch_size=200,
    #     dataset='mrsffia',
    #     data_path='/scratch/project_465001389/chandler_scratch/Datasets/mrsffia',
    #     spec_aug="fma", # fma, diffres, specaugment, specmix
    #     num_classes=4,
    #     sample_rate=22050,
    #     window_size=1024,
    #     hop_size=512,
    #     mel_bins=64,
    #     fmin=1,
    #     fmax=14000,
    #     target_duration=3,
    #     seed=42 # Seed for reproducibility
    # )

    # Update the default arguments with the hardcoded overrides
    for key, value in vars(override_args).items():
        setattr(args, key, value)

    # Run the main function
    main(args)