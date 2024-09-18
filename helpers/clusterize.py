import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.affia3k import get_dataloader as affia3k_loader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
# from umap import UMAP

from models.AudioModel import Audio_Frontend

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=200, help='Batch size for dataloader')
parser.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
parser.add_argument('--window_size', type=int, default=2048, help='Window size for audio feature extraction')
parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/affia3k/')
parser.add_argument('--seed', type=int, default=20, help='Random seed')
parser.add_argument('--fmax', type=int, default=128000, help='Maximum frequency for mel bins')
parser.add_argument('--fmin', type=int, default=1, help='Minimum frequency for mel bins')
parser.add_argument('--mel_bins', type=int, default=128, help='Number of mel bins for audio feature extraction')
parser.add_argument("--selected_classes", type=int, nargs='+', default=[0, 1, 2, 3], 
                    help="List of classes to use (e.g., --selected_classes 0 3)")

args = parser.parse_args()


# Set random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Load data
_, train_loader = affia3k_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path)
_, val_loader = affia3k_loader(split='val', batch_size=1, sample_rate=args.sample_rate, shuffle=False, seed=args.seed, drop_last=True, data_path=args.data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming `Audio_Frontend` is your model for feature extraction
audio_frontend = Audio_Frontend(
    args.sample_rate, 
    args.window_size, 
    args.hop_size, 
    args.mel_bins, 
    args.fmin, 
    args.fmax, 
    pooling=False).to(device)

# Function to compute PSD using STFT directly on the waveform
def compute_psd_from_waveform(loader, sample_rate, window_size, hop_size):
    class_psd = {}
    class_counts = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing PSD"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            # Compute the spectrogram (Power Spectral Density)
            psd = audio_frontend(inputs)  # (batch_size, 1, time_steps, mel_bins)
            psd = psd.squeeze(1).cpu().numpy()  # (batch_size, time_steps, mel_bins)

            for i in range(psd.shape[0]):
                class_id = targets[i, :].argmax(dim=0).item()
                if class_id not in class_psd:
                    class_psd[class_id] = {'low': [], 'mid': [], 'high': []}
                    class_counts[class_id] = 0

                low_band_psd = psd[i, :, :5]  # Low band
                mid_band_psd = psd[i, :, 6:59]  # Mid band
                high_band_psd = psd[i, :, 60:]  # High band

                class_psd[class_id]['low'].append(low_band_psd)
                class_psd[class_id]['mid'].append(mid_band_psd)
                class_psd[class_id]['high'].append(high_band_psd)

                class_counts[class_id] += 1

    return class_psd, class_counts

def compute_psd_from_single_waveform(waveform, label):
    """
    Compute the Power Spectral Density (PSD) for a single waveform using STFT.

    :param waveform: A single waveform input (shape: [1, time_steps] or [time_steps])
    :param label: The class label of the input waveform.
    :return: PSD split into low, mid, and high frequency bands, organized in a dictionary by class.
    """
    # class_psd = {label: {'low': [], 'mid': [], 'high': []}}  # Initialize with empty lists for bands
    class_psd = {}

    with torch.no_grad():
        # Compute the spectrogram (Power Spectral Density) for the single waveform
        psd = audio_frontend(waveform)  # (1, 1, time_steps, mel_bins)
        psd = psd.squeeze(1).cpu().numpy()  # Remove the extra channel dimension: (1, time_steps, mel_bins)

        for i in range(psd.shape[0]):
            class_id = label[i, :].argmax(dim=0).item()
            if class_id not in class_psd:
                class_psd[class_id] = {'low': [], 'mid': [], 'high': []}

        # Split PSD into low, mid, and high frequency bands
        low_band_psd = psd[:, :, :5]  # Low band
        mid_band_psd = psd[:, :, 6:59]  # Mid band
        high_band_psd = psd[:, :, 60:]  # High band

        # Append the PSD values to the corresponding class bands
        class_psd[class_id]['low'].append(low_band_psd)
        class_psd[class_id]['mid'].append(mid_band_psd)
        class_psd[class_id]['high'].append(high_band_psd)
        
    return class_psd

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

def extract_flattened_band_features_single(class_psd, band):
    """
    Extract and flatten PSD features for a specific band (low, mid, or high) for a single image.

    :param class_psd: Dictionary containing PSD values for the single image, split into 'low', 'mid', and 'high' bands.
    :param band: 'low', 'mid', or 'high' specifying the band to extract.
    :return: Flattened PSD features for the specified band and the class label as a scalar.
    """
    all_features = []
    all_labels = []

    # Compute cosine similarity between each pair of classes for low, mid, and high bands
    for class_id, psd_dict in class_psd.items():
        # Flatten the PSD values across time and frequency bins
        # band_features = np.array(class_psd[class_id][band]).flatten()
        band_features = np.array([np.expand_dims(x, axis=0) for x in class_psd[class_id][band]]).flatten()
        feat_proj = resize_to_fixed_dim(band_features)

        all_features.append(feat_proj)
        all_labels.append(class_id)

    return np.array(all_features), np.array(all_labels)  # class_id remains as a scalar

# Function to extract and flatten PSD features for each band
def extract_flattened_band_features(class_psd, band):
    """
    Extract and flatten PSD features for a specific band (low, mid, or high).
    :param class_psd: Dictionary containing PSD values per class and band.
    :param band: 'low', 'mid', or 'high' specifying the band to extract.
    :return: Flattened PSD features and corresponding class labels.
    """
    all_features = []
    all_labels = []

    n_classes = len(class_psd)
    classes = list(class_psd.keys())

    # Compute cosine similarity between each pair of classes for low, mid, and high bands
    for class_id, psd_dict in class_psd.items():
        # Flatten the PSD values across time and frequency bins
        # band_features = np.concatenate(class_psd[class_id][band]).flatten()
        band_features = np.concatenate([np.expand_dims(x, axis=0) for x in class_psd[class_id][band]], axis=0).flatten()
        feat_proj = resize_to_fixed_dim(band_features)

        all_features.append(feat_proj)
        all_labels.append(class_id)

    # for class_id, psd_dict in class_psd.items():
    #     band_features = np.concatenate(psd_dict[band], axis=0).flatten()
    #     all_features.append(band_features)
    #     all_labels.append(class_id)

    return np.array(all_features), np.array(all_labels)


# Compute PSDs for all classes
train_psd, train_class_counts = compute_psd_from_waveform(train_loader, args.sample_rate, args.window_size, args.hop_size)
train_features, train_labels = extract_flattened_band_features(train_psd, 'low')


def evaluation(val_loader, train_features, train_labels):
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            val_psd = compute_psd_from_single_waveform(inputs, targets)
            val_features, val_labels = extract_flattened_band_features_single(val_psd, 'low')

            # Predict label using cosine similarity
            predicted_label, distances = predict_using_cosine_similarity(train_features, train_labels, val_features)

            # Compare the predicted label with the true label
            if predicted_label == val_labels[0]:  # Assuming val_labels is a list or array with one element
                correct_predictions += 1
            
            total_samples += 1  # Increase the total number of samples processed

    # Compute the accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")


from sklearn.metrics.pairwise import cosine_similarity

def predict_using_cosine_similarity(train_features, train_labels, val_features):
    # Compute cosine similarity between val_features and each sample in train_features
    cos_sim = cosine_similarity(val_features, train_features)  # Shape will be (1, num_train_samples)
    
    # Find the index of the highest similarity
    most_similar_index = np.argmax(cos_sim)  # This gives the index of the highest similarity
    
    # Use the index to find the corresponding label from train_labels
    predicted_label = train_labels[most_similar_index]
    
    return predicted_label, cos_sim

from scipy.spatial.distance import mahalanobis

evaluation(val_loader, train_features, train_labels)
