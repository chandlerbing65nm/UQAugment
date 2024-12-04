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

def plot_feature_map(feature, title, filename, sample_rate, mel_bins):
    """
    Plot the feature map with the frames on the x-axis and feature dimensions starting from 50 Hz.
    """
    plt.figure(figsize=(12, 8))

    # Transpose the feature map to have frames on the x-axis and frequency on the y-axis
    feature = feature.T  # Now shape is [feat_dim, frames]
    
    # Plot with a gradient color scheme
    plt.imshow(feature.cpu().numpy(), aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar()
    
    # Set axis labels
    plt.title(title)
    plt.xlabel('Frames')
    plt.ylabel('Frequency (Hz)')
    
    # Calculate the frequency labels based on mel bins and set them starting from 50 Hz
    freq_bins = np.linspace(50, sample_rate // 2, mel_bins)
    plt.yticks(np.linspace(0, feature.shape[0] - 1, num=6), [f"{int(freq)} Hz" for freq in np.linspace(50, sample_rate // 2, num=6)])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

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
            x = x.squeeze(1)
            ret = self.diffres(x)
            guide_loss = ret["guide_loss"]
            x = ret["features"].unsqueeze(1)
            output_dict['diffres_loss'] = guide_loss

        elif spec_aug == 'fma':
            x = x.squeeze(1)
            x = self.fma(x)
            x = x.unsqueeze(1)

        elif spec_aug == 'specaugment':
            x = self.specaugment(x)

        elif spec_aug == 'mixup':
            bs = x.size(0)
            rn_indices, lam = mixup(bs, 0.4)
            lam = lam.to(x.device)
            x = x * lam.view(bs, 1, 1, 1) + x[rn_indices] * (1. - lam.view(bs, 1, 1, 1))
            output_dict['rn_indices'] = rn_indices
            output_dict['mixup_lambda'] = lam

        elif spec_aug == 'specmix':
            x, rn_indices, lam = self.specmix(x)
            output_dict['rn_indices'] = rn_indices
            output_dict['mixup_lambda'] = lam

        return x, output_dict

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse arguments and get transforms
    args = parse_args()
    transform = get_transforms(args)

    # Set arguments
    args.data_path = '/scratch/project_465001389/chandler_scratch/Datasets/mrsffia'
    args.dataset = 'mrsffia' 
    args.batch_size = 6
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

    # Get a batch of data
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

        # Compute duration
        batch_size, _, seq_len, feat_dim = x.size()
        duration = args.target_duration

        # Initialize SpecAugmenter
        spec_augmenter = SpecAugmenter(sample_rate, hop_size, duration, mel_bins, args).to(device)
        spec_augmenter.train()  # Set to correct mode

        # Apply the augmentation
        with torch.no_grad():
            x_input = x.clone()
            x_aug, output_dict = spec_augmenter(x_input)

        # Select a random sample from the batch
        random_sample_idx = np.random.randint(0, x.size(0))
        input_sample = x[random_sample_idx].squeeze(0).cpu()  # Shape: (time_steps, mel_bins)
        augmented_sample = x_aug[random_sample_idx].squeeze(0).cpu()  # Shape: (time_steps, mel_bins)

        # Plot and save the original input feature map
        plot_feature_map(input_sample, "Original Feature Map", "original_feature_map.png", sample_rate, mel_bins)

        # Plot and save the augmented feature map
        plot_feature_map(augmented_sample, "Augmented Feature Map", "augmented_feature_map.png", sample_rate, mel_bins)

        print("Figures saved: original_feature_map.png and augmented_feature_map.png")

        break # only use one batch
        
