import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.uffia import get_dataloader
from torchlibrosa.stft import Spectrogram
import torch.nn as nn

def compute_avg_psd_per_class(spectrograms, targets, args=None):
    class_psd = {}
    for i in range(spectrograms.shape[0]):
        class_id = targets[i].argmax(dim=0).item()
        if class_id not in class_psd:
            class_psd[class_id] = {'low': [], 'mid': [], 'high': []}
        
        psd = np.mean(np.abs(np.fft.rfft(spectrograms[i], axis=0))**2, axis=0)
        freqs = np.fft.rfftfreq(args.window_size, 1 / args.sample_rate)
        low_band = (freqs >= 0) & (freqs < 200)
        mid_band = (freqs >= 200) & (freqs < 2000)
        high_band = (freqs >= 2000) & (freqs < args.sample_rate / 2)

        class_psd[class_id]['low'].append(np.mean(psd[low_band]))
        class_psd[class_id]['mid'].append(np.mean(psd[mid_band]))
        class_psd[class_id]['high'].append(np.mean(psd[high_band]))

    avg_psd_per_class = {}
    for class_id, bands in class_psd.items():
        avg_psd_per_class[class_id] = {
            'low': np.mean(bands['low']),
            'mid': np.mean(bands['mid']),
            'high': np.mean(bands['high'])
        }
    
    return avg_psd_per_class

def inject_noise_per_batch(spectrograms, targets, avg_psd_per_class, args=None):
    target_psd = {
        'low': np.mean([psd['low'] for psd in avg_psd_per_class.values()]),
        'mid': np.mean([psd['mid'] for psd in avg_psd_per_class.values()]),
        'high': np.mean([psd['high'] for psd in avg_psd_per_class.values()])
    }

    noise_injected_spectrograms = []
    for i in range(spectrograms.shape[0]):
        class_id = targets[i].argmax(dim=0).item()
        current_psd = avg_psd_per_class[class_id]
        noise = np.zeros_like(spectrograms[i])

        freqs = np.fft.rfftfreq(args.window_size, 1 / args.sample_rate)
        low_band = (freqs >= 0) & (freqs < 200)
        mid_band = (freqs >= 200) & (freqs < 2000)
        high_band = (freqs >= 2000) & (freqs < args.sample_rate / 2)

        current_low_psd = current_psd['low']
        current_mid_psd = current_psd['mid']
        current_high_psd = current_psd['high']

        low_noise_std = np.sqrt(target_psd['low'] - current_low_psd) if target_psd['low'] > current_low_psd else 0
        mid_noise_std = np.sqrt(target_psd['mid'] - current_mid_psd) if target_psd['mid'] > current_mid_psd else 0
        high_noise_std = np.sqrt(target_psd['high'] - current_high_psd) if target_psd['high'] > current_high_psd else 0

        noise[:, low_band] = low_noise_std * np.random.randn(spectrograms[i][:, low_band].shape[1])
        noise[:, mid_band] = mid_noise_std * np.random.randn(spectrograms[i][:, mid_band].shape[1])
        noise[:, high_band] = high_noise_std * np.random.randn(spectrograms[i][:, high_band].shape[1])

        noise_injected_spectrograms.append(spectrograms[i] + noise)

    return np.stack(noise_injected_spectrograms)

def spectrogram_masking(spectrogram, sample_rate, num_masks=1, mask_ratio=0.9):
    """
    Apply frequency masking to specified frequencies of a spectrogram for all batch samples on GPU.

    :param spectrogram: The input spectrogram (4D torch tensor) with shape (batch_size, 1, time_steps, freq_bins).
    :param num_masks: The number of frequency masks to apply.
    :return: The masked spectrogram.
    """
    device = spectrogram.device  # Get the device of the spectrogram
    batch_size, _, time_steps, freq_bins = spectrogram.shape

    # Calculate frequency bins based on the shape of the spectrogram
    window_size = (freq_bins - 1) * 2  # Inverse of np.fft.rfftfreq to get window size
    freqs = torch.fft.rfftfreq(window_size, 1 / sample_rate).to(device)
    
    # Define the band based on the freqs of the input spectrogram
    band = (freqs >= 100) & (freqs < sample_rate/2)

    # Find indices of the band
    mask_indices = torch.where(band)[0]

    # Calculate mask size as 90% of the total number of bins in the band
    mask_size = int(mask_ratio * mask_indices.shape[0])

    if mask_indices.shape[0] <= mask_size:
        # Print the number of bins in this band and the mask size for debugging
        print(f"Total number of bins in this band: {mask_indices.shape[0]}")
        print(f"Mask size: {mask_size}")
        raise ValueError("Mask size is larger than or equal to the number of bins available.")

    for _ in range(num_masks):
        for i in range(batch_size):
            # Randomly choose a starting point for the mask within the band indices
            start_idx = torch.randint(0, mask_indices.shape[0] - mask_size, (1,), device=device).item()
            end_idx = start_idx + mask_size
            selected_indices = mask_indices[start_idx:end_idx]
            
            # Apply the mask
            spectrogram[i, 0, :, selected_indices] = 0

    return spectrogram


def emphasize_high_freq(spectrogram, sample_rate, amplification_factor=1.5):
    device = spectrogram.device
    batch_size, _, time_steps, freq_bins = spectrogram.shape

    window_size = (freq_bins - 1) * 2
    freqs = torch.fft.rfftfreq(window_size, 1 / sample_rate).to(device)
    
    high_band = (freqs >= 2000) & (freqs < sample_rate / 2)
    high_indices = torch.where(high_band)[0]

    spectrogram[:, 0, :, high_indices] *= amplification_factor
    
    return spectrogram
