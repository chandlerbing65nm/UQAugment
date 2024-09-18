import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.uffia import get_dataloader as uffia_loader
from datasets.affia3k import get_dataloader as affia3k_loader
from models.AudioModel import Audio_Frontend
from sklearn.decomposition import PCA
from scipy.signal import stft

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

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
parser.add_argument("--selected_classes", type=int, nargs='+', default=[0, 1, 2, 3], 
                    help="List of classes to use (e.g., --selected_classes 0 3)")
parser.add_argument('--transform', type=str, choices=['DFT', 'STFT', 'Wavelet'], default='STFT', help='Transform to apply: DFT or STFT')
parser.add_argument('--return_part', type=str, choices=['real', 'imag'], default='real', help='Return real or imaginary part')

args = parser.parse_args()

_, loader = affia3k_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path, selected_classes=args.selected_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_labels = {0: 'none', 1: 'strong', 2: 'medium', 3: 'weak'}

import numpy as np
from scipy.signal import stft
import pywt  # Import PyWavelets for wavelet transform

# Function to apply DFT, STFT, or Wavelet Transform
def apply_transform(waveform, transform='DFT', n_fft=512, hop_length=None, wavelet='morl', scales=None):
    batch_size, wav_len = waveform.shape

    if transform == 'DFT':
        # Apply DFT using numpy's FFT
        dft = np.fft.fft(waveform, axis=-1)
        real_part = np.real(dft)
        imag_part = np.imag(dft)

    elif transform == 'STFT':
        real_part_list = []
        imag_part_list = []
        if hop_length is None:
            hop_length = n_fft // 4  # default hop length
        
        # Process each waveform in the batch
        for i in range(batch_size):
            _, _, Zxx = stft(waveform[i], nperseg=n_fft, noverlap=n_fft - hop_length)
            real_part_list.append(np.real(Zxx))
            imag_part_list.append(np.imag(Zxx))
        
        real_part = np.stack(real_part_list, axis=0)
        imag_part = np.stack(imag_part_list, axis=0)

    elif transform == 'Wavelet':
        # Initialize lists to store real and imaginary parts of wavelet coefficients
        real_part_list = []
        imag_part_list = []
        
        if scales is None:
            scales = np.arange(1, 128)  # Default scales (you can adjust this)

        # Process each waveform in the batch
        for i in range(batch_size):
            # Perform the Continuous Wavelet Transform (CWT)
            coeffs, freqs = pywt.cwt(waveform[i], scales, wavelet)
            
            # Extract real and imaginary parts of the coefficients
            real_part_list.append(np.real(coeffs))
            imag_part_list.append(np.imag(coeffs))
        
        real_part = np.stack(real_part_list, axis=0)
        imag_part = np.stack(imag_part_list, axis=0)
        
    return real_part, imag_part


# Function to compute PSD using the selected transform (DFT or STFT) directly on the waveform
def compute_features_from_waveform(loader, sample_rate, window_size, hop_size, transform, return_part):
    class_feat = {}
    class_counts = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing PSD"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            # Move inputs to CPU and numpy for transformation
            inputs_np = inputs.cpu().numpy()

            # Apply the selected transform
            real_part, imag_part = apply_transform(inputs_np, transform=transform, n_fft=window_size, hop_length=hop_size)

            # Choose which part to return (real or imaginary)
            if return_part == 'real':
                feat = real_part
            else:
                feat = imag_part

            # Continue processing as before
            for i in range(feat.shape[0]):
                class_id = targets[i, :].argmax(dim=0).item()
                if class_id not in class_feat:
                    class_feat[class_id] = {'low': [], 'mid': [], 'high': []}
                    class_counts[class_id] = 0
                    
                feat_size = feat.shape[-1]
                low_end = int(feat_size * (1 / 3))
                mid_start = low_end
                mid_end = int(feat_size * (2 / 3))
                high_start = mid_end

                if transform == 'STFT':
                    low_band = feat[i, :, :low_end]  # Keep full PSD for low band
                    mid_band = feat[i, :, mid_start:mid_end]  # Keep full PSD for mid band
                    high_band = feat[i, :, high_start:]  # Keep full PSD for high band

                    class_feat[class_id]['low'].append(low_band)
                    class_feat[class_id]['mid'].append(mid_band)
                    class_feat[class_id]['high'].append(high_band)

                elif transform == 'DFT':
                    low_band = feat[i, :low_end]  # Keep full PSD for low band
                    mid_band = feat[i, mid_start:mid_end]  # Keep full PSD for mid band
                    high_band = feat[i, high_start:]  # Keep full PSD for high band

                    class_feat[class_id]['low'].append(low_band)
                    class_feat[class_id]['mid'].append(mid_band)
                    class_feat[class_id]['high'].append(high_band)


                class_counts[class_id] += 1

    return class_feat, class_counts

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def plot_pca(features, labels, title, save_path):
    """Apply PCA to the features and plot the 2D projection."""
    # Normalize the data before PCA
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Apply PCA after normalization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_normalized)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Plot the PCA result
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette='deep', s=60)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close()

def prepare_data_for_pca(class_feat, band, class_counts):
    """Prepare data for PCA by flattening features and collecting labels."""
    features = []
    labels = []
    
    for class_id, data in class_feat.items():
        # Flatten each band's features and collect them
        band_features = np.vstack(data[band])
        features.append(band_features)
        labels.extend([class_id] * band_features.shape[0])
    
    features = np.vstack(features)  # Combine all features
    labels = np.array(labels)       # Combine all labels
    
    return features, labels

def plot_features_pca(class_feat, class_counts, args=None):
    """Generate PCA plots for each frequency band."""
    bands = ['low', 'mid', 'high']
    
    for band in bands:
        print(f"Plotting {band} band...")
        
        # Prepare data for PCA
        features, labels = prepare_data_for_pca(class_feat, band, class_counts)
        
        # Plot and save the PCA figure
        plot_pca(features, labels, f'{band.capitalize()} Band PCA', f'helpers/pca_{band}_band_{args.transform}_{args.return_part}_part.png')




class_feat, class_counts = compute_features_from_waveform(
    loader, 
    args.sample_rate, 
    args.window_size, 
    args.hop_size, 
    args.transform, 
    args.return_part
    )

# Plot PCA for low, mid, and high bands
plot_features_pca(class_feat, class_counts, args=args)

