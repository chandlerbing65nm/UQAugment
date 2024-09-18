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

_, loader = affia3k_loader(split='test', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path, selected_classes=args.selected_classes)
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

# Function to visualize two samples per class
def visualize_samples(loader, frontend, classes, save_dir="helpers"):
    samples_per_class = {class_idx: [] for class_idx in classes}
    
    # Collect two samples per class
    with torch.no_grad():
        for batch in loader:
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)  # Assuming this is one-hot encoded
            feat = frontend(inputs, training=False)  # (batch_size, 1, time_steps, mel_bins)
            feat = feat.squeeze(1).cpu().numpy()  # (batch_size, time_steps, mel_bins)
            targets = targets.cpu().numpy()
            
            for i, target in enumerate(targets):
                class_idx = np.argmax(target)
                if class_idx in samples_per_class and len(samples_per_class[class_idx]) < 2:
                    samples_per_class[class_idx].append(feat[i])
            
            # Check if we have collected enough samples
            if all(len(samples) == 12 for samples in samples_per_class.values()):
                break
    
    # Create the plot
    fig, axs = plt.subplots(len(classes), 2, figsize=(10, len(classes) * 5))
    
    for i, class_idx in enumerate(classes):
        for j, sample in enumerate(samples_per_class[class_idx]):
            axs[i, j].imshow(sample.T, aspect='auto', origin='lower')
            axs[i, j].set_title(f"Class: {class_labels[class_idx]} - Sample {j + 1}")
            axs[i, j].set_xlabel('Time Steps')
            axs[i, j].set_ylabel('Mel Bins')
    
    plt.tight_layout()
    
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'dataset_samples_visualization.png'))
    plt.close()

# Visualize and save the samples
visualize_samples(loader, audio_frontend, args.selected_classes)
