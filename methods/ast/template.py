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

from frontends.leaf.frontend import Leaf
from frontends.diffres.frontend import DiffRes
from frontends.dmel.frontend import DMel
from frontends.dstft.frontend import DSTFT
from frontends.sincnet.frontend import SincNet
from frontends.nafa.frontend import NAFA

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

class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='dstft', batch_size=200,
                 freeze_base=False, device=None,
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384',
                 verbose=True):
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

        # Initialize frontends
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
                                                win_length=window_size, window=window, center=center, 
                                                pad_mode=pad_mode, freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
                                                 amin=amin, top_db=top_db, freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                                               freq_drop_width=8, freq_stripes_num=2)

        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=mel_bins,  # Number of MFCC coefficients
            melkwargs={
                "n_fft": window_size,
                "n_mels": mel_bins,
                "hop_length": hop_size,
                "f_min": fmin,
                "f_max": fmax,
                "center": center,
                "pad_mode": pad_mode
            }
        )

        self.leaf_extractor = Leaf(
            n_filters=40,
            sample_rate=sample_rate,
            window_len=16,
            window_stride=8,
            init_min_freq=50.0,
            init_max_freq=sample_rate // 2,
        )

        self.diffres_extractor = DiffRes(
            in_t_dim=int((sample_rate / hop_size) * 2) + 1,  # Adjust based on input dimensions
            in_f_dim=mel_bins,
            dimension_reduction_rate=0.60,
            learn_pos_emb=False
        )

        self.nafa_extractor = NAFA(
            in_t_dim=int((sample_rate / hop_size) * 2) + 1,  # Adjust based on input dimensions
            in_f_dim=mel_bins,
        )


        self.dmel_extractor = DMel(
            init_lambd=5.0, 
            n_fft=window_size, 
            win_length=window_size, 
            hop_length=hop_size
        )

        self.dstft_extractor = DSTFT(
            x=torch.randn(batch_size, sample_rate*2).to(self.device),
            win_length=window_size,
            support=window_size,
            stride=hop_size,
            pow=2,
            win_pow=2,
            win_requires_grad=True,
            stride_requires_grad=True,
            pow_requires_grad=False,    
            win_p="t",
            win_min=window_size//2,              
            win_max=window_size,            
            stride_min=hop_size//2,          
            stride_max=hop_size,           
            sr=sample_rate,
        )

        self.sincnet_extractor = SincNet(
            out_channels=mel_bins, 
            sample_rate=sample_rate, 
            kernel_size=hop_size, 
            window_size=window_size, 
            hop_size=hop_size, 
        )

        # Initialize ASTModel backbone
        if self.frontend == 'diffres':
            ast_fdim = mel_bins
            ast_tdim = int((int((sample_rate / hop_size) * 2) + 1) * (1 - 0.60))
        elif self.frontend == 'leaf':
            ast_fdim = 40
            ast_tdim = int((sample_rate / hop_size) * 2)
        else:
            ast_fdim = mel_bins
            ast_tdim = int((sample_rate / hop_size) * 2) + 1

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
        """
        Forward pass of the model.
        
        :param input: Tensor of shape (batch_size, data_length)
        :return: Dictionary with outputs
        """

        if self.frontend == 'leaf':
            x = self.leaf_extractor(input.unsqueeze(1))  # (batch_size, channels, time_steps, freq_bins)
            x = x.transpose(1, 2)
            x = x.unsqueeze(1)

            if self.training:
                x = self.spec_augmenter(x)

        elif self.frontend == 'diffres':
            x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)          # (batch_size, 1, time_steps, mel_bins)

            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.bn(x)   # Apply the batch normalization from base
            x = x.transpose(1, 3)

            # if self.training:
            #     x = self.spec_augmenter(x)

            x = x.squeeze(1)                      # (batch_size, time_steps, mel_bins)
            ret = self.diffres_extractor(x)
            guide_loss = ret["guide_loss"]
            x = ret["features"]

        elif self.frontend == 'nafa':
            x = self.spectrogram_extractor(input)  # (batch, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch, time_steps, mel_bins)

            # Pass the precomputed features (MFCC or LogMel) into the base model conv blocks
            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.bn(x)  # Apply the batch normalization from base
            x = x.transpose(1, 3)

            # if self.training:
            #     x = x = self.spec_augmenter(x)

            x = x.squeeze(1)
            ret = self.nafa_extractor(x)

            # Access the outputs
            aux_loss = ret["total_loss"]
            x = ret["features"].unsqueeze(1)

        elif self.frontend == 'dmel':
            x = self.dmel_extractor(input) 
            x = x.transpose(1, 2)
            x = x.unsqueeze(1)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.bn(x)   # Apply the batch normalization from base
            x = x.transpose(1, 3)

            if self.training:
                x = self.spec_augmenter(x)

        elif self.frontend == 'dstft':
            x, _ = self.dstft_extractor(input)
            x = x.transpose(1, 2)
            x = x.unsqueeze(1)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.bn(x)   # Apply the batch normalization from base
            x = x.transpose(1, 3)

            if self.training:
                x = self.spec_augmenter(x)

        elif self.frontend == 'sincnet':
            x = self.sincnet_extractor(input.unsqueeze(1)) 
            x = x.transpose(1, 2)
            x = x.unsqueeze(1)

            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.bn(x)   # Apply the batch normalization from base
            x = x.transpose(1, 3)

            if self.training:
                x = self.spec_augmenter(x)

        else: # for log-mel spectrogram
            x = self.spectrogram_extractor(input)  # (batch, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch, 1, time_steps, mel_bins)
            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.bn(x)   # Apply the batch normalization from base
            x = x.transpose(1, 3)

            if self.training:
                x = self.spec_augmenter(x)

        # Pass the preprocessed features to the ASTModel backbone
        if self.frontend not in ['diffres']:
            x = x.squeeze(1)  # (batch_size, time_steps, mel_bins)

        # import ipdb; ipdb.set_trace() 
        # print(x.shape)
        # has_nan = torch.isnan(x).any()
        # print("Contains NaN:", has_nan.item())
        
        # Get logits from ASTModel
        logits = self.backbone(x)  # Shape: (batch_size, num_classes)

        # If using 'diffres', include guide_loss
        if self.training and self.frontend == 'diffres':
            return {'clipwise_output': logits, 'diffres_loss': guide_loss}
        elif self.training and self.frontend == 'ours':
            return {'clipwise_output': logits, 'aux_loss': aux_loss}
        else:
            return {'clipwise_output': logits}

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
