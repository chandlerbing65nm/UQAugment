import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torchaudio
import librosa

from methods.ast.utils import mixup

from specaug.fma.frontend import FMA
from specaug.specmix.frontend import SpecMix

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from methods.ast.models import ASTModel


# Helper function to initialize weights
def init_layer(layer):
    if isinstance(layer, nn.Linear):
        trunc_normal_(layer.weight, std=.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Conv2d):
        trunc_normal_(layer.weight, std=.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class SpecAugmenter(nn.Module):
    def __init__(self, sample_rate, hop_size, mel_bins, duration, args):
        super(SpecAugmenter, self).__init__()
        self.args = args
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.duration = duration

        # Initialize augmentation modules
        self.specaugment = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2
        )

        self.specmix = SpecMix(
            prob=0.5,
            min_band_size=mel_bins // 8,
            max_band_size=mel_bins // 2,
            max_frequency_bands=2,
            max_time_bands=2,
        )


    def forward(self, x, training=True):
        """
        Apply selected spectrogram augmentation based on self.args.spec_aug.

        :param x: Input tensor of shape (batch, 1, time_steps, mel_bins)
        :param training: bool indicating whether the model is in training mode
        :return: A dictionary containing:
            - 'x': The augmented features
            - 'rn_indices': If mixup or specmix is used
            - 'mixup_lambda': If mixup or specmix is used
        """

        output_dict = {}
        spec_aug = self.args.spec_aug

        if spec_aug == 'specaugment':
            # Standard SpecAugmentation
            if training:
                x = self.specaugment(x)
            elif self.args.tta:
                x = self.specaugment(x)
            x = x.squeeze(1)
        elif spec_aug == 'specmix':
            # SpecMix augmentation
            if training:
                x, rn_indices, lam = self.specmix(x)
                output_dict['rn_indices'] = rn_indices
                output_dict['mixup_lambda'] = lam
            elif self.args.tta:
                x, rn_indices, lam = self.specmix(x)
                output_dict['rn_indices'] = rn_indices
                output_dict['mixup_lambda'] = lam
            x = x.squeeze(1)
        else:
            x = x.squeeze(1)
            
        # Always return the transformed features
        output_dict['x'] = x
        return output_dict

class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='dstft', batch_size=200,
                 freeze_base=False, device=None,
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384',
                 verbose=True, args=None):
        """
        Classifier using ASTModel as the backbone with selectable frontend feature extractors.
        
        :param sample_rate: Sampling rate of the audio.
        :param window_size: Window size for STFT.
        :param hop_size: Hop size for STFT.
        :param mel_bins: Number of Mel bins.
        :param fmin: Minimum frequency.
        :param fmax: Maximum frequency.
        :param num_classes: Number of target classes.
        :param frontend: Frontend type ('logmel', 'mfcc', 'chroma', 'ensemble', etc.).
        :param batch_size: Batch size for certain frontend extractors.
        :param freeze_base: Whether to freeze the ASTModel backbone.
        :param device: Device to run the model on.
        :param imagenet_pretrain: Use ImageNet pretraining for ASTModel.
        :param audioset_pretrain: Use AudioSet pretraining for ASTModel.
        :param model_size: Size of the ASTModel ('tiny224', 'small224', 'base224', 'base384').
        :param verbose: Whether to print model summaries.
        """
        super(AudioSpectrogramTransformer, self).__init__()
        self.frontend = frontend
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize frontend feature extractors
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.args = args
        self.duration = args.target_duration

        # Initialize frontends
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
                                                win_length=window_size, window=window, center=center, 
                                                pad_mode=pad_mode, freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
                                                 amin=amin, top_db=top_db, freeze_parameters=True)


        # Initialize SpecAugmenter
        self.spec_augmenter = SpecAugmenter(
            sample_rate=sample_rate,
            hop_size=hop_size,
            mel_bins=mel_bins,
            duration=args.target_duration,
            args=args
        )

        ast_fdim = mel_bins
        ast_tdim = int((sample_rate / hop_size) * self.duration) + 1

        self.backbone = ASTModel(
            label_dim=self.num_classes,  # Initial label_dim; will adjust later
            fstride=10,
            tstride=10,
            input_fdim=ast_fdim,
            input_tdim=ast_tdim,  # Adjust based on input duration
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
        )

        self.bn = nn.BatchNorm2d(mel_bins)
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn)

    # @autocast()
    def forward(self, input): 
        
        # 1. Compute spectrogram and log-mel features
        x = self.spectrogram_extractor(input)  # Shape: (B, T, F)
        x = self.logmel_extractor(x)            # Shape: (B, T, mel_bins)

        # 2. Apply batch normalization
        # Transpose to (B, C=1, F, T) -> BN on F dimension -> transpose back
        x = x.transpose(1, 3)
        x = self.bn(x)
        x = x.transpose(1, 3)  # Now shape is (B, 1, T, mel_bins)

        # 3. Apply spectral augmentations
        # The SpecAugmenter returns a dictionary containing:
        # 'x': augmented features (B, T, mel_bins) or (B, 1, T, mel_bins)
        # 'rn_indices', 'mixup_lambda' (if specmix is used)
        aug_output = self.spec_augmenter(x, training=self.training)
        x = aug_output['x']

        # Extract specmix parameters if present
        rn_indices = aug_output.get('rn_indices', None)
        lam = aug_output.get('mixup_lambda', None)

        # 4. Pass through AST backbone to get classification logits
        logits = self.backbone(x)

        # 5. Prepare output dictionary
        output_dict = {
            'clipwise_output': logits
        }

        # Add specmix parameters if used
        if self.training and rn_indices is not None and lam is not None:
            output_dict.update({
                'rn_indices': rn_indices,
                'mixup_lambda': lam
            })

        return output_dict


if __name__ == '__main__':
    # Example usage and testing
    sample_rate = 128000
    window_size = 2048
    hop_size = 1024
    mel_bins = 128
    fmin = 50
    fmax = sample_rate//2
    num_classes = 4 
    batch_size = 40
    frontend = 'leaf'

    # Initialize the model
    model = AudioSpectrogramTransformer(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax,
        num_classes=num_classes,
        frontend=frontend,  # Change as needed
        batch_size=batch_size,
        freeze_base=False,
        device=None,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size='base384',
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a dummy input batch
    audio_len = sample_rate*2
    test_input = torch.rand([batch_size, audio_len], device=device)  # (batch_size, data_length)

    # Forward pass
    test_output = model(test_input)
    print(f"Output shape: {test_output['clipwise_output'].shape}")  # Expected: [10, num_classes]