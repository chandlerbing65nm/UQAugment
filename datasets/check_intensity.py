import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets.dataset_selection import get_dataloaders
from config.config import parse_args
from transforms.audio_transforms import get_transforms
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from tqdm import tqdm

class EmptinessIntensityChangeDetector(nn.Module):
    def __init__(self):
        super(EmptinessIntensityChangeDetector, self).__init__()
        # No parameters or layers needed for this module

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, frames, mel_bins]
        
        Returns:
            dict: A dictionary containing measures of emptiness and intensity change
        """
        # Measure of emptiness: compute the mean magnitude across frames and mel bins
        mean_magnitude = x.mean(dim=(1, 2))  # Shape: [batch]

        # Measure of intensity change: compute the mean absolute difference between consecutive frames
        delta_x = x[:, 1:, :] - x[:, :-1, :]       # Shape: [batch, frames - 1, mel_bins]
        abs_delta_x = torch.abs(delta_x)
        mean_abs_delta = abs_delta_x.mean(dim=(1, 2))  # Shape: [batch]

        # Return the measures as a dictionary
        return {
            'mean_magnitude': mean_magnitude,
            'mean_abs_delta': mean_abs_delta
        }

if __name__ == "__main__":
    # Parse arguments and get transforms
    args = parse_args()
    transform = get_transforms(args)

    # Initialize data loaders using get_dataloaders
    # args.data_path = '/scratch/project_465001389/chandler_scratch/Datasets/watkins'
    # args.dataset = 'watkins' 
    args.data_path = '/scratch/project_465001389/chandler_scratch/Datasets/uffia'
    args.dataset = 'uffia' 
    args.batch_size = 200
    args.sample_rate = 64000
    args.target_duration=2
    train_dataset, train_loader, val_dataset, val_loader = get_dataloaders(args, transform)

    # Initialize the spectrogram and log-mel extractor
    window_size = 2048
    hop_size = 1024
    sample_rate = args.sample_rate
    mel_bins = 64
    fmin = 1
    fmax = 128000
    amin = 1e-10
    ref = 1.0
    top_db = None
    window = 'hann'
    center = True
    pad_mode = 'reflect'

    spectrogram_extractor = Spectrogram(
        n_fft=window_size, hop_length=hop_size, win_length=window_size,
        window=window, center=center, pad_mode=pad_mode, freeze_parameters=True
    )

    logmel_extractor = LogmelFilterBank(
        sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
        fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True
    )

    bn = nn.BatchNorm2d(mel_bins)

    # Instantiate the detector
    detector = EmptinessIntensityChangeDetector()

    # Lists to store results
    all_mean_magnitudes = []
    all_mean_abs_deltas = []

    # Process all batches in the train loader
    for batch in tqdm(train_loader):
        inputs = batch['waveform']  # Assuming the batch contains waveform data

        # Apply spectrogram and log-mel extraction
        with torch.no_grad():
            x = spectrogram_extractor(inputs)  # Shape: (batch, 1, time_steps, freq_bins)
            x = logmel_extractor(x)  # Shape: (batch, 1, time_steps, mel_bins)

            # Apply batch normalization
            x = x.transpose(1, 3)  # Shape: (batch, mel_bins, time_steps, 1)
            x = bn(x)              # BatchNorm2d over mel_bins
            x = x.transpose(1, 3)  # Shape back to (batch, 1, time_steps, mel_bins)

        # Prepare input features for the detector
        x = x.squeeze(1)  # Shape: (batch, time_steps, mel_bins)

        # Pass through the detector
        measures = detector(x)

        # Collect measures
        all_mean_magnitudes.extend(measures['mean_magnitude'].cpu().numpy())
        all_mean_abs_deltas.extend(measures['mean_abs_delta'].cpu().numpy())

    # Convert lists to NumPy arrays for further processing
    all_mean_magnitudes = np.array(all_mean_magnitudes)
    all_mean_abs_deltas = np.array(all_mean_abs_deltas)

    # Print or process the collected measures
    print("Mean Magnitude (Emptiness) for all samples:", all_mean_magnitudes)
    print("Mean Absolute Delta (Intensity Change) for all samples:", all_mean_abs_deltas)

    # Compute overall statistics
    overall_mean_magnitude = all_mean_magnitudes.mean()
    overall_std_magnitude = all_mean_magnitudes.std()

    overall_mean_abs_delta = all_mean_abs_deltas.mean()
    overall_std_abs_delta = all_mean_abs_deltas.std()

    print(f"Overall Mean Magnitude (Emptiness): {overall_mean_magnitude:.4f} ± {overall_std_magnitude:.4f}")
    print(f"Overall Mean Absolute Delta (Intensity Change): {overall_mean_abs_delta:.4f} ± {overall_std_abs_delta:.4f}")