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

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torchaudio
import librosa

from methods.panns.pytorch_utils import *
from methods.panns.models import *
from frontends.leaf.frontend import Leaf
from frontends.dmel.frontend import DMel
from frontends.dstft.frontend import DSTFT
from frontends.sincnet.frontend import SincNet
from frontends.diffres.frontend import DiffRes
from frontends.nafa.frontend import NAFA

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

class PANNS_CNN6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='logmel', batch_size=200,
                 freeze_base=False, device=None, args=None,
                 ):
        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_CNN6, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.frontend = frontend
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.num_classes = num_classes
        self.args = args

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = Cnn6(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Step 2: Optionally store the custom modules (but do not apply them yet)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.specaugment = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                                               freq_drop_width=8, freq_stripes_num=2)

        self.diffres = DiffRes(
            in_t_dim=int(int((sample_rate / hop_size) * 2) + 1),
            in_f_dim=mel_bins,
            dimension_reduction_rate=0.60,
            learn_pos_emb=False
        )

        self.nafa = NAFA(
            in_t_dim=int(int((sample_rate / hop_size) * 2) + 1),
            in_f_dim=mel_bins,
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, num_classes, bias=True)  # Assuming 512 is embedding size
        
        self.bn = nn.BatchNorm2d(mel_bins)
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)

        self.base.spectrogram_extractor = self.spectrogram_extractor
        self.base.logmel_extractor = self.logmel_extractor
        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch, time_steps, mel_bins)

        # Pass the precomputed features (MFCC or LogMel) into the base model conv blocks
        x = x.transpose(1, 3)  # Align dimensions for the base model
        x = self.bn(x)   # Apply the batch normalization from base
        x = x.transpose(1, 3)

        if self.args.spec_aug == 'diffres':
            x = x.squeeze(1)
            ret = self.diffres(x)
            guide_loss = ret["guide_loss"]
            x = ret["features"].unsqueeze(1)

        elif self.args.spec_aug == 'nafa':
            x = x.squeeze(1)
            x = self.nafa(x)
            x = x.unsqueeze(1)

        elif self.args.spec_aug == 'specaugment':
            if self.training:
                x = self.specaugment(x)

        elif self.args.spec_aug == 'mixup':
            if self.training:
                bs = x.size(0)
                rn_indices, lam = mixup(bs, 0.4)
                lam = lam.to(x.device)
                x = x * lam.reshape(bs, 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))

        # else:
        #     output_dict = self.base(input, mixup_lambda)
        #     embedding = output_dict['embedding']
        #     clipwise_output = self.fc_transfer(embedding)

        #     return output_dict

        # import ipdb; ipdb.set_trace() 
        # print(x.shape)

        x = self.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_transfer(embedding)

        if self.training and self.args.spec_aug == 'mixup':
            output_dict = {'rn_indices':rn_indices, 'mixup_lambda': lam, 'clipwise_output': clipwise_output, 'embedding': embedding}
        elif self.training and self.args.spec_aug == 'diffres':
            output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding, 'diffres_loss': guide_loss}
        else:
            output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict


class PANNS_RESNET22(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, freeze_base=False
                 ):
        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_RESNET22, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = ResNet22(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Step 2: Optionally store the custom modules (but do not apply them yet)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, num_classes, bias=True)  # Assuming 512 is embedding size
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)

        self.base.spectrogram_extractor = self.spectrogram_extractor
        self.base.logmel_extractor = self.logmel_extractor
        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = self.fc_transfer(embedding)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict

class PANNS_MOBILENETV1(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, freeze_base=False
                 ):
        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_MOBILENETV1, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = MobileNetV1(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Step 2: Optionally store the custom modules (but do not apply them yet)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(1024, num_classes, bias=True)  # Assuming 512 is embedding size
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)

        self.base.spectrogram_extractor = self.spectrogram_extractor
        self.base.logmel_extractor = self.logmel_extractor
        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)
        
    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = self.fc_transfer(embedding)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict

class PANNS_WAVEGRAM_CNN14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, freeze_base=False
                 ):
        """Classifier for a new task using pretrained Cnn6 as a sub-module."""
        super(PANNS_WAVEGRAM_CNN14, self).__init__()
        audioset_classes_num = 527

        # Step 1: Create base Cnn6 instance (original architecture)
        self.base = Wavegram_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
                         fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, num_classes, bias=True)  # Assuming 512 is embedding size
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights into the base model before applying changes."""
        # Step 3: Load pretrained weights for the base Cnn6 model
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=True)

        # Load the model state dict with strict=False to ignore incompatible layers
        pretrained_dict = checkpoint['model']
        model_dict = self.base.state_dict()

        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the current model's dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.base.load_state_dict(model_dict)


        self.base.fc_audioset = self.fc_transfer

    def load_finetuned_weights(model, checkpoint_path):
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Get the state_dict of the model to be loaded
        pretrained_dict = checkpoint  # Or 'state_dict' depending on your saving convention
        model_dict = model.state_dict()

        # Filter out fc_audioset (or any other mismatched layers) to ignore mismatches
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # Update the model_dict with the filtered pretrained weights
        model_dict.update(pretrained_dict)

        # Load the updated model_dict
        model.load_state_dict(model_dict)
        
    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = self.fc_transfer(embedding)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict


def main():
    # Create a dummy input
    batch_size = 8  # Define the batch size
    seq_length = 32000  # Sequence length (e.g., could be the length of an audio clip)
    
    # Dummy input tensor (batch_size, seq_length)
    dummy_input = torch.randn(batch_size, seq_length)

    # Instantiate the model
    model = PANNS_CNN6(
        sample_rate=128000, 
        window_size=2048, 
        hop_size=1024, 
        mel_bins=64, 
        fmin=50, 
        fmax=None, 
        num_classes=4  # Example number of classes for the classification task
    )

    # Load pretrained weights (if available)
    pretrained_checkpoint_path = "./weights/Cnn6_mAP=0.343.pth"
    
    # Check if the path to the pretrained model exists and load
    if os.path.exists(pretrained_checkpoint_path):
        print(f"Loading pretrained weights from {pretrained_checkpoint_path}")
        model.load_from_pretrain(pretrained_checkpoint_path)
    else:
        print(f"Pretrained model not found at {pretrained_checkpoint_path}. Skipping weight loading.")


    # Set the model to evaluation mode for testing (optional if no training-specific layers like BatchNorm)
    model.eval()

    # Forward pass through the model with the dummy input
    with torch.no_grad():  # No gradient calculation is needed for inference
        output_dict = model(dummy_input)

    # Retrieve and print outputs
    clipwise_output = output_dict['clipwise_output']
    embedding = output_dict['embedding']

    print(f"Clipwise output shape: {clipwise_output.shape}")  # Expected: (batch_size, classes_num)
    print(f"Embedding shape: {embedding.shape}")  # Expected: (batch_size, embedding_size)

if __name__ == "__main__":
    main()