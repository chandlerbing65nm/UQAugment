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

from methods.panns.models import *


class PANNS_CNN6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, num_classes, frontend='logmel',
                 freeze_base=False,
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

        # MFCC feature extractor
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=mel_bins,  # This determines the number of MFCC coefficients
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
        self.bn0 = nn.BatchNorm2d(64*3)
        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, num_classes, bias=True)  # Assuming 512 is embedding size
        
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

    # def forward(self, input, mixup_lambda=None):
    #     """Input: (batch_size, data_length)"""
    #     output_dict = self.base(input, mixup_lambda)
    #     embedding = output_dict['embedding']

    #     clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
    #     output_dict['clipwise_output'] = clipwise_output

    #     return output_dict

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        
        if self.frontend == 'mfcc':
            # If MFCC is chosen, extract MFCC features
            x = self.mfcc_extractor(input)
            x = x.unsqueeze(1).transpose(2, 3) 

            # Pass the precomputed features (MFCC or LogMel) into the base model conv blocks
            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.base.bn0(x)   # Apply the batch normalization from base
            x = x.transpose(1, 3)

        elif self.frontend == 'chroma':
            # Extract chroma features using librosa's chroma_stft function
            input_np = input.cpu().numpy()  # Convert to NumPy for librosa
            chroma_features = []
            for i in range(input_np.shape[0]):  # Loop over batch
                chroma = librosa.feature.chroma_stft(y=input_np[i], sr=self.sample_rate, n_fft=self.window_size, 
                                                     hop_length=self.hop_size, win_length=self.window_size, 
                                                     window='hann', n_chroma=self.mel_bins, tuning=0)
                chroma_features.append(chroma)
            chroma_features = np.stack(chroma_features, axis=0)  # Shape: (batch_size, n_chroma, time_steps)
            x = torch.tensor(chroma_features, dtype=torch.float32, device=input.device)  # Convert back to tensor
            x = x.unsqueeze(1).transpose(2, 3)  # Shape: (batch_size, 1, n_chroma, time_steps)

            # Pass the precomputed features (MFCC or LogMel) into the base model conv blocks
            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.base.bn0(x)   # Apply the batch normalization from base
            x = x.transpose(1, 3)

        elif self.frontend == 'ensemble':
            # If MFCC is chosen, extract MFCC features
            x1 = self.mfcc_extractor(input)
            x1 = x1.unsqueeze(1).transpose(2, 3)  # Shape: (batch_size, 1, time_steps, n_mfcc)

            # Extract chroma features using librosa's chroma_stft function
            input_np = input.cpu().numpy()  # Convert to NumPy for librosa
            chroma_features = []
            for i in range(input_np.shape[0]):  # Loop over batch
                chroma = librosa.feature.chroma_stft(y=input_np[i], sr=self.sample_rate, n_fft=self.window_size, 
                                                    hop_length=self.hop_size, win_length=self.window_size, 
                                                    window='hann', n_chroma=self.mel_bins, tuning=0)
                chroma_features.append(chroma)
            chroma_features = np.stack(chroma_features, axis=0)  # Shape: (batch_size, n_chroma, time_steps)
            x2 = torch.tensor(chroma_features, dtype=torch.float32, device=input.device)  # Convert back to tensor
            x2 = x2.unsqueeze(1).transpose(2, 3)  # Shape: (batch_size, 1, time_steps, n_chroma)

            # Extract LogMel features
            x3 = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
            x3 = self.logmel_extractor(x3)  # (batch_size, 1, time_steps, mel_bins)

            x = torch.cat((x1, x2, x3), dim=3) 

            # Pass the precomputed features (MFCC or LogMel) into the base model conv blocks
            x = x.transpose(1, 3)  # Align dimensions for the base model
            x = self.bn0(x)   # Apply the batch normalization from base
            x = x.transpose(1, 3)

        else:
            output_dict = self.base(input, mixup_lambda)
            embedding = output_dict['embedding']

            clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
            output_dict['clipwise_output'] = clipwise_output

            return output_dict

        # import ipdb; ipdb.set_trace() 
        # print(x.shape)


        if self.training:
            x = self.base.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

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
        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)

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

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
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

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
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

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
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
