import argparse
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from datasets.affia3k import get_dataloader as affia3k_loader
from datasets.affia3k_merge import get_dataloader as affia3k_merge_loader
import pywt
from audiomentations import Compose, HighPassFilter, LowPassFilter

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for dataloader')
parser.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/affia3k/')
parser.add_argument('--seed', type=int, default=25, help='Random seed')
parser.add_argument('--num_wavelet_scales', type=int, default=128, help='Number of scales for wavelet transform')
parser.add_argument('--wavelet', type=str, default='morl', help='Type of wavelet to use (e.g., morl, cmor, mexh)')
args = parser.parse_args()

# transform = Compose([
#     HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=args.sample_rate/2, p=0.99)
# ])

_, loader = affia3k_merge_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path, transform=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_wavelet_transform(waveform, wavelet_name, num_scales):
    # Define scales based on the number of scales desired
    scales = np.arange(1, num_scales + 1)
    
    # Compute the Continuous Wavelet Transform (CWT)
    coeffs, _ = pywt.cwt(waveform, scales, wavelet_name, sampling_period=1.0 / args.sample_rate)
    
    # Convert the absolute value of coefficients to decibels
    wavelet_spectrogram = 20 * np.log10(np.abs(coeffs) + 1e-10)
    
    return wavelet_spectrogram

def plot_wavelet_spectrograms(loader, num_samples_per_class=3):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    class_sample_count = {}
    
    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(loader, desc="Plotting Wavelet Spectrograms"):
            inputs = batch['waveform'].to(device).cpu().numpy()  # Move to CPU for wavelet computation
            targets = batch['target'].to(device)
            
            for i in range(inputs.shape[0]):
                class_id = targets[i, :].argmax(dim=0).item()
                if class_id not in class_sample_count:
                    class_sample_count[class_id] = 0
                
                if class_sample_count[class_id] < num_samples_per_class:
                    class_sample_count[class_id] += 1
                    
                    # Compute the wavelet spectrogram for the current waveform
                    wavelet_spectrogram = compute_wavelet_transform(inputs[i], args.wavelet, args.num_wavelet_scales)
                    
                    # Plot and save the wavelet spectrogram
                    plt.figure(figsize=(10, 4))
                    plt.imshow(wavelet_spectrogram, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
                    plt.title(f'Class {class_id} Sample {class_sample_count[class_id]}')
                    plt.colorbar(format='%+2.0f dB')
                    plt.xlabel('Time')
                    plt.ylabel('Scales')
                    
                    # Save the plot
                    os.makedirs(f'plots/wavelet/class_{class_id}', exist_ok=True)
                    plot_filename = f'plots/wavelet/class_{class_id}/sample_{class_sample_count[class_id]}.png'
                    plt.savefig(plot_filename)
                    plt.close()

plot_wavelet_spectrograms(loader)
