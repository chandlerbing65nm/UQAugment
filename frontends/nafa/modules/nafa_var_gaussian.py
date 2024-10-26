import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from frontends.nafa.modules.dilated_convolutions_1d.conv import DilatedConv, DilatedConv_Out_128

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class FrameAugment(nn.Module):
    def __init__(self, seq_len, feat_dim, device='cuda'):
        super(FrameAugment, self).__init__()
        self.seq_len = seq_len
        self.device = device

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()

        # Compute stability-driven noise
        stability_scores = self.compute_stability(feature)
        augmenting_path = self.compute_augmenting_path(batch_size, seq_len, stability_scores)

        # Apply stability-aware noise augmentation
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        
        return augmented_feature

    def compute_stability(self, feature):
        # Compute variance along the frame dimension as stability measure
        variance_scores = torch.var(feature, dim=2, keepdim=True)  # Shape: [batch, seq_len, 1]
        
        # Normalize variance to create stability scores (inverse of variance)
        stability_scores = 1 / (variance_scores + 1e-6)  # Add epsilon to prevent division by zero
        stability_scores = stability_scores / stability_scores.max()  # Normalize to [0, 1]
        
        return stability_scores.squeeze(-1)  # Shape: [batch, seq_len]

    def compute_augmenting_path(self, batch_size, seq_len, stability_scores):
        # Generate Gaussian noise and apply softmax normalization along frame dimension
        gaussian_noise = torch.normal(0, 1, size=(batch_size, seq_len, seq_len), device=self.device)
        
        # Scale noise based on stability scores (higher stability gets more noise)
        noise_scaled = gaussian_noise * stability_scores.unsqueeze(2)
        
        augmenting_path = F.softmax(noise_scaled, dim=-1)
        
        return augmenting_path

    def apply_augmenting(self, feature, augmenting_path):
        # Matrix multiplication of feature with stability-aware noise
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)
        
        return augmented_feature

class NAFA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim
        
        self.frame_augment = FrameAugment(
            seq_len=self.input_seq_length, 
            feat_dim=self.input_f_dim,
            device='cuda'
        )

    def forward(self, x):
        ret = {}

        augment_frame = self.frame_augment(x.exp())
        augment_frame = torch.log(augment_frame + EPS)

        # Final outputs
        ret["x"] = x
        ret["features"] = augment_frame
        ret["dummy"] = torch.tensor(0.0, device=x.device)
        ret["total_loss"] = ret["dummy"]

        return ret