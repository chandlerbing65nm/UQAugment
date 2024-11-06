import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class FrameMixup(nn.Module):
    def __init__(
        self, 
        seq_len, 
        feat_dim, 
        temperature=0.2, 
        frame_reduction_ratio=None, 
        frame_augmentation_ratio=1.0,  # Ratio of frames to augment
        device='cuda'
        ):
        super(FrameMixup, self).__init__()
        self.seq_len = seq_len
        self.frame_reduction_ratio = frame_reduction_ratio
        if frame_reduction_ratio is not None:
            assert 0 < frame_reduction_ratio <= 1, "frame_reduction_ratio must be between 0 and 1"
            self.reduced_len = max(1, int(seq_len * (1 - frame_reduction_ratio)))
        else:
            self.reduced_len = self.seq_len
        assert 0 <= frame_augmentation_ratio <= 1, "frame_augmentation_ratio must be between 0 and 1"
        self.num_augmented_frames = max(1, int(self.reduced_len * frame_augmentation_ratio))
        self.noise_template = torch.randn(1, self.reduced_len, seq_len).to(device=device)  # [1, reduced_len, seq_len]
        self.temperature = temperature

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()  # feature: [batch_size, seq_len, feat_dim]
        noise_template = self.noise_template.expand(batch_size, -1, -1)  # [batch_size, reduced_len, seq_len]
        augmenting_path = self.compute_augmenting_path(noise_template)  # [batch_size, reduced_len, seq_len]
        if self.num_augmented_frames < self.reduced_len:
            augmenting_path = self.randomly_apply_augmentation(augmenting_path)  # [batch_size, reduced_len, seq_len]
        augmented_feature = self.apply_augmenting(feature, augmenting_path)  # [batch_size, reduced_len, feat_dim]
        return augmented_feature  # [batch_size, reduced_len, feat_dim]

    def compute_augmenting_path(self, noise_template):
        mu, sigma = 0, 1  # mean and standard deviation for Gaussian
        gaussian_noise = torch.normal(mu, sigma, size=noise_template.size(), device=noise_template.device)  # [batch_size, reduced_len, seq_len]
        return F.softmax(gaussian_noise / self.temperature, dim=-1)  # [batch_size, reduced_len, seq_len]

    def randomly_apply_augmentation(self, augmenting_path):
        batch_size, reduced_len, seq_len = augmenting_path.size()
        mask = torch.zeros(batch_size, reduced_len, 1, device=augmenting_path.device)  # [batch_size, reduced_len, 1]
        start_index = torch.randint(0, reduced_len - self.num_augmented_frames + 1, (1,)).item()
        mask[:, start_index:start_index + self.num_augmented_frames, :] = 1  # mask with ones in selected frames
        augmenting_path = augmenting_path * mask  # [batch_size, reduced_len, seq_len]
        return augmenting_path  # [batch_size, reduced_len, seq_len]

    def apply_augmenting(self, feature, augmenting_path):
        # Ensure feature is on the same device as augmenting_path
        feature = feature.to(augmenting_path.device)
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)  # [batch_size, reduced_len, feat_dim]
        return augmented_feature  # [batch_size, reduced_len, feat_dim]



class NAFA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim
        
        self.frame_augment = FrameMixup(
            seq_len=self.input_seq_length, 
            feat_dim=self.input_f_dim,
            temperature=0.2, 
            frame_reduction_ratio=None,
            frame_augmentation_ratio=1.0,
            device='cuda'
        )

    def forward(self, x):
        ret = {}

        augment_frame = self.frame_augment(x.exp())
        augment_frame = torch.log(augment_frame + EPS)

        return augment_frame

import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets.dataset_selection import get_dataloaders
from frontends.nafa.modules.debug import FrameMixup, NAFA  # Replace with your actual module paths if necessary
from config.config import parse_args
from transforms.audio_transforms import get_transforms
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

def plot_feature_map(feature, title, filename, sample_rate, mel_bins):
    """
    Plot the feature map with the frames on the x-axis and feature dimensions starting from 50 Hz.
    """
    plt.figure(figsize=(12, 8))
    # Transpose the feature map to have frames on the x-axis and frequency on the y-axis
    feature = feature.T  # Now shape is [feat_dim, frames]
    
    # Plot with a gradient color scheme
    plt.imshow(feature.cpu().numpy(), aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar()
    
    # Set axis labels
    plt.title(title)
    plt.xlabel('Frames')
    plt.ylabel('Frequency (Hz)')
    
    # Calculate the frequency labels based on mel bins and set them starting from 200 Hz
    freq_bins = np.linspace(50, sample_rate // 2, mel_bins)
    plt.yticks(np.linspace(0, feature.shape[0] - 1, num=6), [f"{int(freq)} Hz" for freq in np.linspace(50, sample_rate // 2, num=6)])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Parse arguments and get transforms
    args = parse_args()
    transform = get_transforms(args)

    # Initialize data loaders using get_dataloaders
    args.data_path='/scratch/project_465001389/chandler_scratch/Datasets/uffia'
    args.dataset='uffia' 
    train_dataset, train_loader, val_dataset, val_loader = get_dataloaders(args, transform)

    # Initialize the spectrogram and log-mel extractor
    window_size = args.window_size
    hop_size = args.hop_size
    sample_rate = 64000
    mel_bins = args.mel_bins
    fmin = 50
    fmax = sample_rate // 2
    amin = 1e-10
    ref = 1.0
    top_db = None
    window = 'hann'
    center = True
    pad_mode = 'reflect'

    spectrogram_extractor = Spectrogram(
        n_fft=window_size, hop_length=hop_size, win_length=window_size,
        window=window, center=center, pad_mode=pad_mode, freeze_parameters=True
    )

    logmel_extractor = LogmelFilterBank(
        sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
        fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True
    )

    bn = nn.BatchNorm2d(mel_bins)

    # Get a batch of data
    for batch in val_loader:
        inputs = batch['waveform']  # Assuming the batch contains waveform data
        break  # We only need one batch for visualization

    # Apply spectrogram and log-mel extraction
    with torch.no_grad():
        x = spectrogram_extractor(inputs)  # Shape: (batch, 1, time_steps, freq_bins)
        x = logmel_extractor(x)  # Shape: (batch, 1, time_steps, mel_bins)

        # Pass the precomputed features (MFCC or LogMel) into the base model conv blocks
        x = x.transpose(1, 3)  # Align dimensions for the base model
        x = bn(x)   # Apply the batch normalization from base
        x = x.transpose(1, 3)

    # Prepare input features for the NAFA model
    x = x.squeeze(1)  # Remove the singleton dimension, shape: (batch, time_steps, mel_bins)

    # Check the shape of input features
    batch_size, seq_len, feat_dim = x.size()

    # Initialize the NAFA model with appropriate dimensions
    nafa_model = NAFA(in_t_dim=seq_len, in_f_dim=feat_dim)
    nafa_model.eval()  # Set the model to evaluation mode

    # Select a random sample from the batch
    random_sample_idx = np.random.randint(0, batch_size)
    input_sample = x[random_sample_idx].detach()

    # Plot and save the original input feature map
    plot_feature_map(input_sample, "Original Feature Map", "original_feature_map.png", sample_rate, mel_bins)

    # Apply the augmentation
    with torch.no_grad():
        augmented_sample = nafa_model(x)[random_sample_idx].detach()

    # Plot and save the augmented feature map
    plot_feature_map(augmented_sample, "Augmented Feature Map", "augmented_feature_map.png", sample_rate, mel_bins)

    print("Figures saved: original_feature_map.png and augmented_feature_map.png")


