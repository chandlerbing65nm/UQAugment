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

args = parser.parse_args()

_, loader = affia3k_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path, selected_classes=args.selected_classes)
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


# Function to compute PSD using STFT directly on the waveform
def compute_psd_from_waveform(loader, sample_rate, window_size, hop_size, feature='logmel'):
    class_feat = {}
    class_counts = {}

    # spectrogram = Spectrogram(n_fft=window_size, hop_length=hop_size, power=2.0,
    #                           win_length=window_size, window='hann', center=True,
    #                           pad_mode='reflect', freeze_parameters=True).to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing PSD"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            # Compute the spectrogram (Power Spectral Density)
            # psd = spectrogram(inputs) # (batch_size, 1, time_steps, freq_bins)
            if feature == 'logmel':
                feat = audio_frontend(inputs, training=True) # (batch_size, 1, time_steps, mel_bins)
                feat = feat.squeeze(1).cpu().numpy()  # (batch_size, time_steps, mel_bins)

            for i in range(feat.shape[0]):
                class_id = targets[i, :].argmax(dim=0).item()
                if class_id not in class_feat:
                    class_feat[class_id] = {'low': [], 'mid': [], 'high': []}
                    class_counts[class_id] = 0

                feat_size = feat.shape[1]
                low_end = int(feat_size * (1 / 3))
                mid_start = low_end
                mid_end = int(feat_size * (2 / 3))
                high_start = mid_end

                # # # Store the full PSD for each band (no averaging)
                # low_band = feat[i, :low_end, :5]  # Keep full PSD for low band
                # mid_band = feat[i, mid_start:mid_end, :5]  # Keep full PSD for mid band
                # high_band = feat[i, high_start:, :5] # Keep full PSD for high band

                low_band = feat[i, :, :4]  # Keep full PSD for low band
                mid_band = feat[i, :, 5:59]  # Keep full PSD for mid band
                high_band = feat[i, :, 60:]  # Keep full PSD for high band

                class_feat[class_id]['low'].append(low_band)
                class_feat[class_id]['mid'].append(mid_band)
                class_feat[class_id]['high'].append(high_band)

                class_counts[class_id] += 1

    return class_feat, class_counts

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from scipy.interpolate import interp1d

def resize_to_fixed_dim(feature, target_dim=128):
    """
    Resize feature to the fixed dimensionality using interpolation.
    :param feature: Original feature vector (variable length)
    :param target_dim: The target dimensionality (e.g., 1024)
    :return: Resized feature vector (target_dim,)
    """
    original_dim = feature.shape[0]
    # Define the interpolation function
    interpolation = interp1d(np.arange(original_dim), feature, kind='linear')
    # Generate the new resized feature vector
    resized_feature = interpolation(np.linspace(0, original_dim - 1, target_dim))
    return resized_feature

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import jensenshannon

def normalize_distribution(vec):
    """
    Normalize the vector to be a valid probability distribution.
    Ensures the values are non-negative and sum to 1.
    :param vec: Input vector.
    :return: Normalized vector.
    """
    # Ensure the values are non-negative
    vec = np.maximum(vec, 0)
    # Normalize the vector to sum to 1
    norm_vec = vec / np.sum(vec) if np.sum(vec) != 0 else np.ones_like(vec) / len(vec)
    return norm_vec

def compute_similarity(class_psd, class_labels, metric='euclidean'):
    """
    Compute the similarity between classes for low, mid, and high frequency bands.
    :param class_psd: Dictionary containing PSD values per class and band.
    :param class_labels: Dictionary of class labels.
    :param metric: The similarity metric to use: 'cosine', 'euclidean', or 'jsd' (Jensen-Shannon Divergence).
    :return: Similarity matrices for low, mid, and high bands.
    """
    n_classes = len(class_psd)
    classes = list(class_psd.keys())

    # Prepare matrices to store similarity values
    sim_low = np.zeros((n_classes, n_classes))
    sim_mid = np.zeros((n_classes, n_classes))
    sim_high = np.zeros((n_classes, n_classes))

    # Compute similarity between each pair of classes for low, mid, and high bands
    for i in range(n_classes):
        for j in range(i, n_classes):
            # Flatten the PSD values across time and frequency bins
            p_low = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[i]]['low']], axis=0).flatten()
            q_low = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[j]]['low']], axis=0).flatten()
            p_mid = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[i]]['mid']], axis=0).flatten()
            q_mid = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[j]]['mid']], axis=0).flatten()
            p_high = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[i]]['high']], axis=0).flatten()
            q_high = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[j]]['high']], axis=0).flatten()

            # Project the PSD features to (1024,)
            p_low_proj = resize_to_fixed_dim(p_low)
            q_low_proj = resize_to_fixed_dim(q_low)
            p_mid_proj = resize_to_fixed_dim(p_mid)
            q_mid_proj = resize_to_fixed_dim(q_mid)
            p_high_proj = resize_to_fixed_dim(p_high)
            q_high_proj = resize_to_fixed_dim(q_high)

            # Compute the similarity based on the selected metric
            if metric == 'cosine':
                sim_low[i, j] = sim_low[j, i] = cosine_similarity([p_low_proj], [q_low_proj])[0, 0]
                sim_mid[i, j] = sim_mid[j, i] = cosine_similarity([p_mid_proj], [q_mid_proj])[0, 0]
                sim_high[i, j] = sim_high[j, i] = cosine_similarity([p_high_proj], [q_high_proj])[0, 0]
            elif metric == 'euclidean':
                sim_low[i, j] = sim_low[j, i] = euclidean(p_low_proj, q_low_proj)  # Negative for consistency (larger is better)
                sim_mid[i, j] = sim_mid[j, i] = euclidean(p_mid_proj, q_mid_proj)
                sim_high[i, j] = sim_high[j, i] = euclidean(p_high_proj, q_high_proj)
            # Normalize for Jensen-Shannon divergence (if needed)
            elif metric == 'jsd':
                p_low_proj = normalize_distribution(p_low_proj)
                q_low_proj = normalize_distribution(q_low_proj)
                p_mid_proj = normalize_distribution(p_mid_proj)
                q_mid_proj = normalize_distribution(q_mid_proj)
                p_high_proj = normalize_distribution(p_high_proj)
                q_high_proj = normalize_distribution(q_high_proj)

                sim_low[i, j] = sim_low[j, i] = jensenshannon(p_low_proj, q_low_proj)  # 1 - JSD to have higher values as better
                sim_mid[i, j] = sim_mid[j, i] = jensenshannon(p_mid_proj, q_mid_proj)
                sim_high[i, j] = sim_high[j, i] = jensenshannon(p_high_proj, q_high_proj)

    return sim_low, sim_mid, sim_high

# def compute_cosine_similarity(class_psd, class_labels):
#     """
#     Compute the cosine similarity between classes for low, mid, and high frequency bands.
#     :param class_psd: Dictionary containing PSD values per class and band.
#     :param class_labels: Dictionary of class labels.
#     :return: Cosine similarity matrices for low, mid, and high bands.
#     """
#     n_classes = len(class_psd)
#     classes = list(class_psd.keys())

#     # Prepare matrices to store cosine similarity values
#     cos_sim_low = np.zeros((n_classes, n_classes))
#     cos_sim_mid = np.zeros((n_classes, n_classes))
#     cos_sim_high = np.zeros((n_classes, n_classes))

#     # Compute cosine similarity between each pair of classes for low, mid, and high bands
#     for i in range(n_classes):
#         for j in range(i, n_classes):
#             # # Flatten the PSD values across time and frequency bins
#             # p_low = np.concatenate(class_psd[classes[i]]['low']).flatten()
#             # q_low = np.concatenate(class_psd[classes[j]]['low']).flatten()
#             # p_mid = np.concatenate(class_psd[classes[i]]['mid']).flatten()
#             # q_mid = np.concatenate(class_psd[classes[j]]['mid']).flatten()
#             # p_high = np.concatenate(class_psd[classes[i]]['high']).flatten()
#             # q_high = np.concatenate(class_psd[classes[j]]['high']).flatten()

#             # # Compute the cosine similarity for each band
#             # cos_sim_low[i, j] = cos_sim_low[j, i] = cosine_similarity([p_low], [q_low])[0, 0]
#             # cos_sim_mid[i, j] = cos_sim_mid[j, i] = cosine_similarity([p_mid], [q_mid])[0, 0]
#             # cos_sim_high[i, j] = cos_sim_high[j, i] = cosine_similarity([p_high], [q_high])[0, 0]

#             p_low = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[i]]['low']], axis=0).flatten()
#             q_low = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[j]]['low']], axis=0).flatten()
#             p_mid = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[i]]['mid']], axis=0).flatten()
#             q_mid = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[j]]['mid']], axis=0).flatten()
#             p_high = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[i]]['high']], axis=0).flatten()
#             q_high = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[classes[j]]['high']], axis=0).flatten()

#             # Project the PSD features to (1024,)
#             p_low_proj = resize_to_fixed_dim(p_low)
#             q_low_proj = resize_to_fixed_dim(q_low)
#             p_mid_proj = resize_to_fixed_dim(p_mid)
#             q_mid_proj = resize_to_fixed_dim(q_mid)
#             p_high_proj = resize_to_fixed_dim(p_high)
#             q_high_proj = resize_to_fixed_dim(q_high)

#             import ipdb; ipdb.set_trace() 
#             print(p_low_proj.shape)
#             # Now use p_low_proj, q_low_proj, etc. for cosine similarity
#             cos_sim_low[i, j] = cos_sim_low[j, i] = cosine_similarity([p_low_proj], [q_low_proj])[0, 0]
#             cos_sim_mid[i, j] = cos_sim_mid[j, i] = cosine_similarity([p_mid_proj], [q_mid_proj])[0, 0]
#             cos_sim_high[i, j] = cos_sim_high[j, i] = cosine_similarity([p_high_proj], [q_high_proj])[0, 0]

#     return cos_sim_low, cos_sim_mid, cos_sim_high

def plot_similarity(cos_sim_low, cos_sim_mid, cos_sim_high, class_labels):
    """
    Plot and save the cosine similarity heatmaps for low, mid, and high frequency bands with consistent color scale.
    :param cos_sim_low: Cosine similarity matrix for low band.
    :param cos_sim_mid: Cosine similarity matrix for mid band.
    :param cos_sim_high: Cosine similarity matrix for high band.
    :param class_labels: Dictionary of class labels.
    """
    labels = [class_labels[i] for i in range(len(class_labels))]

    # Find the global min and max across all cosine similarity matrices to have a consistent color scale
    min_val = min(cos_sim_low.min(), cos_sim_mid.min(), cos_sim_high.min())
    max_val = max(cos_sim_low.max(), cos_sim_mid.max(), cos_sim_high.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Low band heatmap
    sns.heatmap(cos_sim_low, annot=True, cmap="crest", xticklabels=labels, yticklabels=labels, ax=axes[0], vmin=min_val, vmax=max_val)
    axes[0].set_title('Low Band Similarity')

    # Mid band heatmap
    sns.heatmap(cos_sim_mid, annot=True, cmap="crest", xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=min_val, vmax=max_val)
    axes[1].set_title('Mid Band Similarity')

    # High band heatmap
    sns.heatmap(cos_sim_high, annot=True, cmap="crest", xticklabels=labels, yticklabels=labels, ax=axes[2], vmin=min_val, vmax=max_val)
    axes[2].set_title('High Band Similarity')

    # Save figure
    plt.tight_layout()
    plt.savefig('helpers/similarity_bands.png')
    plt.close()


# Call the compute_psd function to get PSD data
class_psd, class_counts = compute_psd_from_waveform(loader, args.sample_rate, args.window_size, args.hop_size)

sim_low, sim_mid, sim_high = compute_similarity(class_psd, class_labels)
plot_similarity(sim_low, sim_mid, sim_high, class_labels)


