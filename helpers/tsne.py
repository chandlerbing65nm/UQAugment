import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from datasets.uffia import get_dataloader
from torchlibrosa.stft import Spectrogram

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=300, help='Batch size for dataloader')
parser.add_argument('--sample_rate', type=int, default=64000, help='Sample rate for audio')
parser.add_argument('--window_size', type=int, default=2048, help='Window size for audio feature extraction')
parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/u-ffia/audio_dataset/')
parser.add_argument('--seed', type=int, default=25, help='Random seed')
parser.add_argument('--band', type=str, choices=['low', 'mid', 'high', 'all'], default='all', help='Frequency band for PSD calculation')

args = parser.parse_args()

_, loader = get_dataloader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize spectrogram extractor
spectrogram_extractor = Spectrogram(
    n_fft=args.window_size, hop_length=args.hop_size,
    win_length=args.window_size, window='hann', center=True,
    pad_mode='reflect',
    freeze_parameters=True).to(device)

def compute_psd(loader, band):
    psd_list = []
    class_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(loader, desc="Computing PSD"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)
            
            x = spectrogram_extractor(inputs)  # (batch_size, 1, time_steps, freq_bins)
            x = x.squeeze(1).cpu().numpy()  # (batch_size, time_steps, freq_bins)
            
            for i in range(x.shape[0]):
                class_id = targets[i, :].argmax(dim=0).item()
                freqs = np.fft.rfftfreq(args.window_size, 1 / args.sample_rate)
                psd = np.mean(np.abs(np.fft.rfft(x[i], axis=0))**2, axis=0)
                
                # Define frequency bands (low: 0-200 Hz, mid: 200-2000 Hz, high: 2000-32000 Hz)
                low_band = (freqs >= 0) & (freqs < 200)
                mid_band = (freqs >= 200) & (freqs < 2000)
                high_band = (freqs >= 2000) & (freqs < args.sample_rate / 2)
                
                if band == 'low':
                    psd_list.append(psd[low_band])
                elif band == 'mid':
                    psd_list.append(psd[mid_band])
                elif band == 'high':
                    psd_list.append(psd[high_band])
                else:  # 'all'
                    psd_list.append(np.concatenate([psd[low_band], psd[mid_band], psd[high_band]]))
                
                class_labels.append(class_id)

    return np.array(psd_list), np.array(class_labels)

def plot_cluster_distribution(psd_list, class_labels):
    # Normalize the PSD features
    scaler = StandardScaler()
    psd_list_normalized = scaler.fit_transform(psd_list)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=25, learning_rate=200, random_state=args.seed)
    psd_2d = tsne.fit_transform(psd_list_normalized)
    
    # Mapping class IDs to class names
    class_names = {0: 'none', 1: 'strong', 2: 'medium', 3: 'weak'}
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(psd_2d[:, 0], psd_2d[:, 1], c=class_labels, cmap='viridis', alpha=0.7)
    
    # Produce a legend with the unique colors from the scatter
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    
    # Extract integers from LaTeX formatted labels
    cleaned_labels = [int(re.search(r'\d+', label).group()) for label in labels]
    legend_labels = [class_names[label] for label in cleaned_labels]
    
    plt.legend(handles, legend_labels)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Class Cluster Distribution in 2D Space Based on PSD')
    
    # Save the plot
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/psd[{args.band}]_cluster_distribution.png')
    plt.close()

psd_list, class_labels = compute_psd(loader, args.band)
plot_cluster_distribution(psd_list, class_labels)
