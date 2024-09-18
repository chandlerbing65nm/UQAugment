from sklearn.neighbors import KNeighborsClassifier

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from datasets.uffia import get_dataloader as uffia_loader
from datasets.affia3k import get_dataloader as affia3k_loader

from methods.panns.models import *

from tqdm import tqdm
from pprint import pprint
import wandb  # Import wandb
import os
import random
from sklearn.metrics import average_precision_score

from transformers import AutoFeatureExtractor, ASTForAudioClassification
import transformers
import torchvision
import torchaudio
import audiomentations

import torch.nn.functional as F
import imagehash

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def parse_args():
    parser = argparse.ArgumentParser(description='Train Audio Model with Learning Rate Scheduler')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--max_epoch', type=int, default=1, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for audio feature extraction')
    parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
    parser.add_argument('--mel_bins', type=int, default=64, help='Number of mel bins for audio feature extraction')
    parser.add_argument('--fmin', type=int, default=50, help='Minimum frequency for mel bins')
    parser.add_argument('--fmax', type=int, default=None, help='Maximum frequency for mel bins')
    parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/affia3k/')
    parser.add_argument('--extractor', type=str, default='none')
    return parser.parse_args()


# Define a custom DTW distance function
def dtw_distance(x, y):
    # x and y are reshaped to [time_steps, mel_bins] to compare them on the time dimension
    x = x.reshape(-1, x.shape[-1])  # Reshape to [time_steps, mel_bins]
    y = y.reshape(-1, y.shape[-1])
    
    # Use fastdtw with Euclidean distance for comparison along the time axis
    distance, _ = fastdtw(x, y, dist=euclidean)
    return distance

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Arguments:")
    pprint(vars(args))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # # Initialize the audio frontend
    # audio_frontend = Audio_Frontend(
    #     args.sample_rate, args.window_size, args.hop_size, 
    #     args.mel_bins, args.fmin, args.fmax, pooling=args.pooling
    # ).to(device)

    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None

    spectrogram_extractor = Spectrogram(n_fft=args.window_size, hop_length=args.hop_size, 
        win_length=args.window_size, window=window, center=center, pad_mode=pad_mode, 
        freeze_parameters=True).to(device)

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=args.sample_rate, n_fft=args.window_size, 
        n_mels=args.mel_bins, fmin=args.fmin, fmax=args.fmax, ref=ref, amin=amin, top_db=top_db, 
        freeze_parameters=True).to(device)

    # Spectral Centroid extractor if enabled
    if args.extractor == 'spec_centroid':
        spectral_centroid_extractor = torchaudio.transforms.SpectralCentroid(
            sample_rate=args.sample_rate, n_fft=args.window_size, hop_length=args.hop_size
        ).to(device)

    # Initialize KNN model with cosine similarity
    # knn = KNeighborsClassifier(n_neighbors=7, metric='cosine')
    knn = KNeighborsClassifier(n_neighbors=7, metric=dtw_distance)
    
    transform = None

    # Initialize data loaders
    train_dataset, train_loader = affia3k_loader(
        split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, 
        shuffle=True, seed=args.seed, class_num=args.num_classes, drop_last=True, 
        data_path=args.data_path, 
        transform=transform
    )
    
    val_dataset, val_loader = affia3k_loader(
        split='test', batch_size=args.batch_size, sample_rate=args.sample_rate, 
        shuffle=False, seed=args.seed, class_num=args.num_classes, drop_last=True, 
        data_path=args.data_path,
        transform=None
    )

    # Training loop
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')

        # Extract features and labels from training data
        train_features = []
        train_labels = []

        for batch in tqdm(train_loader, "Training:"):
            train_inputs = batch['waveform'].to(device)
            train_targets = batch['target'].to(device)
            
            # Optionally, add spectral centroid features if enabled
            if args.extractor == 'spec_centroid':
                features = spectral_centroid_extractor(train_inputs) 
            else:
                # Extract features from audio frontend
                features = spectrogram_extractor(train_inputs)   # (batch_size, 1, time_steps, freq_bins)
                features = logmel_extractor(features).mean(dim-1)    # (batch_size, 1, time_steps, mel_bins)
                features = features.reshape(features.size(0), -1)
            
            # import ipdb; ipdb.set_trace() 
            # print(features.shape)
            
            train_features.append(features.detach().cpu().numpy())
            train_labels.append(train_targets.argmax(dim=-1).cpu().numpy())

        # Train KNN
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        knn.fit(train_features, train_labels)
        print('done')
        # Validation with KNN
        val_features = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                val_inputs = batch['waveform'].to(device)
                val_targets = batch['target'].to(device)
                
                # Optionally, add spectral centroid features if enabled
                if args.extractor == 'spec_centroid':
                    features = spectral_centroid_extractor(val_inputs) 
                else:
                    # Extract features from audio frontend
                    features = spectrogram_extractor(val_inputs)   # (batch_size, 1, time_steps, freq_bins)
                    features = logmel_extractor(features).mean(dim-1)    # (batch_size, 1, time_steps, mel_bins)
                    features = features.reshape(features.size(0), -1)
            
                val_features.append(features.detach().cpu().numpy())
                val_labels.append(val_targets.argmax(dim=-1).cpu().numpy())
        
        val_features = np.concatenate(val_features, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        
        # Get predicted probabilities instead of hard labels for mAP calculation
        predictions_proba = knn.predict_proba(val_features)
        
        # mAP calculation
        mAP = average_precision_score(val_labels, predictions_proba, average="macro")
        print(f'Mean Average Precision (mAP): {mAP:.4f}')
        
        # Evaluate the KNN model
        predictions = knn.predict(val_features)
        accuracy = np.mean(predictions == val_labels)  # Convert val_labels back to class labels for accuracy
        print(f'Validation accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()