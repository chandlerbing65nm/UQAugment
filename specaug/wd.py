import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets.dataset_selection import get_dataloaders
from config.config import parse_args
from transforms.audio_transforms import get_transforms
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from methods.panns.pytorch_utils import *
from methods.panns.models import *

from specaug.diffres.frontend import DiffRes
from specaug.fma.frontend import FMA
from specaug.specmix.frontend import SpecMix

class SpecAugmenter(nn.Module):
    def __init__(self, sample_rate, hop_size, duration, mel_bins, args):
        super(SpecAugmenter, self).__init__()

        self.args = args
        self.training = True  # Will be updated based on the model's training state

        # SpecAugment
        self.specaugment = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2
        )

        # DiffRes
        self.diffres = DiffRes(
            in_t_dim=int((sample_rate / hop_size) * duration) + 1,
            in_f_dim=mel_bins,
            dimension_reduction_rate=0.60,
            learn_pos_emb=False
        )

        # SpecMix
        self.specmix = SpecMix(
            prob=0.5,
            min_band_size=mel_bins // 8,
            max_band_size=mel_bins // 2,
            max_frequency_bands=2,
            max_time_bands=2
        )

        # FMA
        self.fma = FMA(
            in_t_dim=int((sample_rate / hop_size) * duration) + 1,
            in_f_dim=mel_bins
        )

    def forward(self, x):
        spec_aug = self.args.spec_aug
        output_dict = {}

        if spec_aug == 'diffres':
            x = x.squeeze(1)  # Shape: [batch_size, seq_len, feat_dim]
            ret = self.diffres(x)
            guide_loss = ret["guide_loss"]
            x_aug = ret["features"].unsqueeze(1)  # Shape: [batch_size, 1, seq_len_aug, feat_dim]
            output_dict['diffres_loss'] = guide_loss

        elif spec_aug == 'fma':
            x = x.squeeze(1)
            x_aug = self.fma(x)
            x_aug = x_aug.unsqueeze(1)

        elif spec_aug == 'specaugment':
            x_aug = self.specaugment(x)

        elif spec_aug == 'specmix':
            x_aug, rn_indices, lam = self.specmix(x)
            output_dict['rn_indices'] = rn_indices
            output_dict['mixup_lambda'] = lam

        else:
            x_aug = x  # If no augmentation is specified, x_aug is the same as x

        return x_aug, output_dict

def compute_wasserstein_distance(x_input, x_aug):
    """
    Compute the Wasserstein-1 distance between x_input and x_aug.
    1. Take the mean over dimension 2 (time steps).
    2. Flatten the result.
    3. Convert to probability distributions using softmax.
    4. Compute the Wasserstein-1 distance via cumulative distribution differences.
    """
    batch_size = x_input.size(0)

    # Take the mean over the time dimension (dim=2)
    x_input_mean = x_input.mean(dim=2)  # Shape: (batch_size, 1, feat_dim)
    x_aug_mean = x_aug.mean(dim=2)      # Shape: (batch_size, 1, feat_dim)

    # Flatten the tensors
    x_input_flat = x_input_mean.view(batch_size, -1)  # (batch_size, feat_dim)
    x_aug_flat = x_aug_mean.view(batch_size, -1)      # (batch_size, feat_dim)

    # Apply softmax to get probability distributions
    p = F.softmax(x_input_flat, dim=1)
    q = F.softmax(x_aug_flat, dim=1)

    # Compute the cumulative distribution functions (CDFs)
    p_cdf = torch.cumsum(p, dim=1)  # (batch_size, feat_dim)
    q_cdf = torch.cumsum(q, dim=1)  # (batch_size, feat_dim)

    # Compute the Wasserstein distance per sample
    # W_1(p, q) = sum(|P_cdf(i) - Q_cdf(i)|) over i
    wasserstein_per_sample = torch.sum(torch.abs(p_cdf - q_cdf), dim=1)

    # Average over the batch
    wasserstein_mean = torch.mean(wasserstein_per_sample)
    return wasserstein_mean

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

if __name__ == "__main__":
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse arguments and get transforms
    args = parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    transform = get_transforms(args)

    # Set arguments
    args.data_path = '/scratch/project_465001389/chandler_scratch/Datasets/mrsffia'
    args.dataset = 'mrsffia' 
    args.batch_size = 200
    args.sample_rate = 22050
    args.target_duration = 3
    args.spec_aug = 'diffres'  # Choose the augmentation method ('fma', 'diffres', etc.)

    # Initialize data loaders
    train_dataset, train_loader, val_dataset, val_loader = get_dataloaders(args, transform)

    # Initialize the spectrogram and log-mel extractor
    window_size = 1024
    hop_size = 512
    sample_rate = args.sample_rate
    mel_bins = 64

    fmin = 1
    fmax = sample_rate // 2
    amin = 1e-10
    ref = 1.0
    top_db = None
    window = 'hann'
    center = True
    pad_mode = 'reflect'

    spectrogram_extractor = Spectrogram(
        n_fft=window_size, hop_length=hop_size, win_length=window_size,
        window=window, center=center, pad_mode=pad_mode, freeze_parameters=True
    ).to(device)

    logmel_extractor = LogmelFilterBank(
        sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
        fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True
    ).to(device)

    bn = nn.BatchNorm2d(mel_bins).to(device)

    # Initialize SpecAugmenter
    duration = args.target_duration
    spec_augmenter = SpecAugmenter(sample_rate, hop_size, duration, mel_bins, args).to(device)
    spec_augmenter.train()  # Set to evaluation mode

    total_wd = 0.0
    num_batches = 0

    # Process all batches
    for batch in val_loader:
        inputs = batch['waveform'].to(device)  # Ensure data is on the same device

        # Apply spectrogram and log-mel extraction
        with torch.no_grad():
            x = spectrogram_extractor(inputs)  # Shape: (batch, 1, time_steps, freq_bins)
            x = logmel_extractor(x)  # Shape: (batch, 1, time_steps, mel_bins)

            # Apply batch normalization
            x = x.transpose(1, 3)  # Shape: (batch, mel_bins, time_steps, 1)
            x = bn(x)   # Apply batch normalization
            x = x.transpose(1, 3)  # Shape: (batch, 1, time_steps, mel_bins)

        # Clone x before augmentation
        x_clone = x.clone()

        # Apply the augmentation
        with torch.no_grad():
            x_aug, output_dict = spec_augmenter(x_clone)

            # Compute the Wassertein distance
            wd = compute_wasserstein_distance(x, x_aug)

        total_wd += wd.item()
        num_batches += 1

    # Compute the average WD over all batches
    if num_batches > 0:
        average_wd = total_wd / num_batches
        print(f"Wassertein distance over {num_batches} batches using {args.spec_aug} augmentation: {average_wd}")
    else:
        print("No batches were processed.")
