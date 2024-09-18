import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.uffia import get_dataloader as uffia_loader
from datasets.affia3k import get_dataloader as affia3k_loader
from models.AudioModel import Audio_Frontend
from torchlibrosa.stft import Spectrogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering  # Import DBSCAN

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
parser.add_argument('--fmax', type=int, default=None, help='Maximum frequency for mel bins')
parser.add_argument('--fmin', type=int, default=50, help='Minimum frequency for mel bins')
parser.add_argument('--mel_bins', type=int, default=128, help='Number of mel bins for audio feature extraction')
parser.add_argument('--selected_classes', type=int, nargs='+', default=[0, 1, 2, 3], 
                    help="List of classes to use (e.g., --selected_classes 0 3)")
parser.add_argument('--cluster', type=str, default='none', 
                    help='Clustering algorithm to use: kmeans or dbscan')  # New argument for clustering

args = parser.parse_args()

# Setup dataloader
_, loader = affia3k_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path, selected_classes=args.selected_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Audio frontend
audio_frontend = Audio_Frontend(
    args.sample_rate, 
    args.window_size, 
    args.hop_size, 
    args.mel_bins, 
    args.fmin, 
    args.fmax, 
    pooling=False).to(device)

# Function to compute PSD and apply clustering
def compute_and_cluster_psd(loader, bands=('low', 'mid', 'high')):
    band_indices = {
        'low': (0, 4),    # Low-frequency 50-160 Hz
        'mid': (5, 59),   # Mid-frequency: 190-5150 Hz
        'high': (60, 128)  # High frequency: 5360-64000 Hz
    }
    
    psd_data = {band: [] for band in bands}
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing PSD and clustering"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            psd = audio_frontend(inputs, training=False).squeeze(1).cpu().numpy()  # (batch_size, time_steps, mel_bins)
            targets = targets.cpu().numpy()

            for i in range(psd.shape[0]):
                # Get class label
                class_label = np.argmax(targets[i])
                labels.append(class_label)

                # Extract and store PSD data for each band
                for band in bands:
                    start_idx, end_idx = band_indices[band]
                    band_psd = psd[i, :, start_idx:end_idx]
                    psd_data[band].append(band_psd)

    return psd_data, labels

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # Import t-SNE
import seaborn as sns

def cluster_and_plot(psd_data, labels, band, save_dir, clustering_algo='kmeans', reduction_algo='pca'):
    psd_data_band = np.array(psd_data[band])
    psd_data_band= psd_data_band.reshape(psd_data_band.shape[0], -1)

    # Perform dimensionality reduction (either PCA or t-SNE)
    if reduction_algo == 'pca':
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(psd_data_band)
    elif reduction_algo == 'tsne':
        reducer = TSNE(n_components=2, random_state=args.seed, perplexity=30, n_iter=300)
        reduced_data = reducer.fit_transform(psd_data_band)

    # Perform clustering based on the selected algorithm
    if clustering_algo == 'kmeans':
        kmeans = KMeans(n_clusters=4, random_state=args.seed)
        targets = kmeans.fit_predict(reduced_data)
    elif clustering_algo == 'dbscan':
        dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps and min_samples as needed
        targets = dbscan.fit_predict(reduced_data)
    elif clustering_algo == 'spectral':
        spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', random_state=args.seed)
        targets = spectral.fit_predict(reduced_data)
    else:
        targets = np.array(labels)

    # Map targets to class names
    class_mapping = {0: 'none', 1: 'strong', 2: 'weak', 3: 'medium'}
    mapped_targets = [class_mapping.get(target, 'none') for target in targets]  # Default to 'none' if not found

    # Plot the results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=mapped_targets, palette='deep', s=60)
    plt.xlabel(f'{reduction_algo.upper()} Component 1')
    plt.ylabel(f'{reduction_algo.upper()} Component 2')

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'clustering_{band}_band_{clustering_algo}_{reduction_algo}.png')
    plt.savefig(save_path)
    print(f'Saved plot to {save_path}')
    plt.show()

# Main execution
psd_data, labels = compute_and_cluster_psd(loader, bands=['low', 'mid', 'high'])
save_dir = './helpers/'  # Set your desired save directory

for band in ['low', 'mid', 'high']:
    cluster_and_plot(psd_data, labels, band, save_dir, clustering_algo=args.cluster)
