import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from datasets.uffia import get_dataloader as uffia_loader
from datasets.affia3k import get_dataloader as affia3k_loader
from models.Audio_model import Audio_Frontend
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from librosa.feature import spectral_contrast

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=300, help='Batch size for dataloader')
parser.add_argument('--sample_rate', type=int, default=64000, help='Sample rate for audio')
parser.add_argument('--window_size', type=int, default=1024, help='Window size for audio feature extraction')
parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
parser.add_argument('--mel_bins', type=int, default=32, help='Number of mel bins for audio feature extraction')
parser.add_argument('--fmin', type=int, default=1, help='Minimum frequency for mel bins')
parser.add_argument('--fmax', type=int, default=32000, help='Maximum frequency for mel bins')
parser.add_argument('--pooling', action='store_true', help='If using a pooling operation')
parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/u-ffia/audio_dataset/')
parser.add_argument('--seed', type=int, default=25, help='Random seed')

args = parser.parse_args()

_, loader = uffia_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize spectrogram extractor
spectrogram_extractor = Spectrogram(
    n_fft=args.window_size, hop_length=args.hop_size,
    win_length=args.window_size, window='hann', center=True,
    pad_mode='reflect',
    freeze_parameters=True).to(device)

# Logmel feature extractor
logmel_extractor = LogmelFilterBank(
    sr=args.sample_rate, n_fft=args.window_size,
    n_mels=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
    ref=1.0, amin=1e-10, top_db=None,
    freeze_parameters=True).to(device)


def plot_spectrograms(loader, num_samples_per_class=3):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    class_sample_count = {}
    
    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(loader, desc="Plotting Spectrograms"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)
            
            x = spectrogram_extractor(inputs)  # (batch_size, 1, time_steps, freq_bins) 
            spectrograms = logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            
            for i in range(spectrograms.size(0)):
                class_id = targets[i, :].argmax(dim=0).item()
                if class_id not in class_sample_count:
                    class_sample_count[class_id] = 0
                
                if class_sample_count[class_id] < num_samples_per_class:
                    class_sample_count[class_id] += 1
                    spectrogram = spectrograms[i].squeeze().cpu().numpy()

                    # Plot and save the spectrogram
                    plt.figure(figsize=(10, 4))
                    plt.imshow(spectrogram.T, aspect='auto', origin='lower', interpolation='nearest')
                    plt.title(f'Class {class_id} Sample {class_sample_count[class_id]}')
                    plt.colorbar(format='%+2.0f dB')
                    plt.xlabel('Time')
                    plt.ylabel('Mel Frequency Bins')
                    
                    # Save the plot
                    os.makedirs(f'plots/class_{class_id}', exist_ok=True)
                    plot_filename = f'plots/class_{class_id}/_sample_{class_sample_count[class_id]}.png'
                    plt.savefig(plot_filename)
                    plt.close()

plot_spectrograms(loader)
