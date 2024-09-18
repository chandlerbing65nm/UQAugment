import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.uffia import get_dataloader as uffia_loader
from datasets.affia3k import get_dataloader as affia3k_loader
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d
from helpers.masks import spectrogram_masking
from models.AudioModel import Audio_Frontend

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=200, help='Batch size for dataloader')
parser.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
parser.add_argument('--window_size', type=int, default=2048, help='Window size for audio feature extraction')
parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/affia3k/')
parser.add_argument('--seed', type=int, default=20, help='Random seed')
parser.add_argument('--band', type=str, choices=['low', 'mid', 'high', 'all'], default='all', help='Frequency band to use for PSD computation')
parser.add_argument('--mode', type=str, choices=['class', 'band'], default='class', help='Mode to compute JS divergence: class or band')
parser.add_argument('--fmax', type=int, default=128000, help='Maximum frequency for mel bins')
parser.add_argument('--fmin', type=int, default=1, help='Minimum frequency for mel bins')
parser.add_argument('--mel_bins', type=int, default=128, help='Number of mel bins for audio feature extraction')

args = parser.parse_args()

_, loader = affia3k_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_frontend = Audio_Frontend(
    args.sample_rate, 
    args.window_size, 
    args.hop_size, 
    args.mel_bins, 
    args.fmin, 
    args.fmax, 
    pooling=False).to(device)

class_labels = {0: 'none', 1: 'strong', 2: 'medium', 3: 'weak'}

from torchlibrosa.stft import Spectrogram

# Function to compute PSD using STFT directly on the waveform
def compute_psd_from_waveform(loader, sample_rate, window_size, hop_size):
    class_psd = {}
    class_counts = {}

    # Define the Spectrogram object (STFT computation)
    spectrogram = Spectrogram(n_fft=window_size, hop_length=hop_size, power=2.0,
                                                win_length=window_size, window='hann', center=True,
                                                pad_mode='reflect',
                                                freeze_parameters=True).to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing PSD"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            # Compute the spectrogram (Power Spectral Density)
            psd = spectrogram(inputs)  # (batch_size, 1, freq_bins, time_steps)
            psd = psd.squeeze(1).cpu().numpy()  # (batch_size, freq_bins, time_steps)

            for i in range(psd.shape[0]):
                class_id = targets[i, :].argmax(dim=0).item()
                if class_id not in class_psd:
                    class_psd[class_id] = {'low': [], 'mid': [], 'high': []}
                    class_counts[class_id] = 0

                num_freq_bins = psd.shape[1]
                low_end = int(num_freq_bins * (1 / 3))  # low: first 33% of freq bins
                mid_start = low_end
                mid_end = int(num_freq_bins * (2 / 3))  # mid: next 33% of freq bins
                high_start = mid_end  # high: last 33% of freq bins

                # Mean over time for each band
                low_band_psd = np.mean(psd[i, :low_end, :], axis=1)  # Mean over time for low band
                mid_band_psd = np.mean(psd[i, mid_start:mid_end, :], axis=1)  # Mean over time for mid band
                high_band_psd = np.mean(psd[i, high_start:, :], axis=1)  # Mean over time for high band

                # Store the mean across frequency bins for each band
                class_psd[class_id]['low'].append(np.mean(low_band_psd))
                class_psd[class_id]['mid'].append(np.mean(mid_band_psd))
                class_psd[class_id]['high'].append(np.mean(high_band_psd))

                class_counts[class_id] += 1

    return class_psd, class_counts

def plot_psd(class_psd, class_counts, class_labels):
    # Compute the average PSD per class and per band
    avg_psd_per_class = {class_id: {} for class_id in class_psd.keys()}
    
    for class_id, bands in class_psd.items():
        avg_psd_per_class[class_id]['low'] = np.mean(bands['low'])
        avg_psd_per_class[class_id]['mid'] = np.mean(bands['mid'])
        avg_psd_per_class[class_id]['high'] = np.mean(bands['high'])
    
    # Plotting
    bands = ['low', 'mid', 'high']
    n_classes = len(class_psd)
    bar_width = 0.25
    index = np.arange(n_classes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    low_band_energy = [avg_psd_per_class[class_id]['low'] for class_id in class_psd.keys()]
    mid_band_energy = [avg_psd_per_class[class_id]['mid'] for class_id in class_psd.keys()]
    high_band_energy = [avg_psd_per_class[class_id]['high'] for class_id in class_psd.keys()]
    
    ax.bar(index, low_band_energy, bar_width, label='Low Band')
    ax.bar(index + bar_width, mid_band_energy, bar_width, label='Mid Band')
    ax.bar(index + 2 * bar_width, high_band_energy, bar_width, label='High Band')
    
    # Add labels
    ax.set_xlabel('Class')
    ax.set_ylabel('Average PSD')
    ax.set_title('Average Energy per Class and Band')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels([class_labels[class_id] for class_id in class_psd.keys()])
    ax.legend()
    
    # Save the figure
    plt.savefig('helpers/class_band_energy.png')
    plt.close()


# Call the compute_psd function to get PSD data
class_psd, class_counts = compute_psd_from_waveform(loader, args.sample_rate, args.window_size, args.hop_size)

# Plot and save the PSD
plot_psd(class_psd, class_counts, class_labels)

