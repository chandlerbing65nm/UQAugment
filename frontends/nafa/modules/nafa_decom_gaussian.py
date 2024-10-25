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
    def __init__(self, seq_len, feat_dim, device=None):
        super(FrameAugment, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define a lower-triangular matrix parameter for Cholesky decomposition
        self.cholesky_lower = nn.Parameter(torch.randn(seq_len, seq_len, device=self.device))
        
        # Initialize the noise matrix for normalization
        self.noise_normalization = nn.Softmax(dim=-1)
        
    def forward(self, features):
        batch_size, seq_len, feat_dim = features.size()
        
        # Move features to device if not already there
        features = features.to(self.device)
        
        # Ensure the correlation matrix is positive semi-definite
        correlation_matrix = torch.matmul(self.cholesky_lower, self.cholesky_lower.T)
        
        # Generate standard Gaussian noise and apply the learned correlation matrix
        raw_noise = torch.randn(batch_size, seq_len, seq_len, device=self.device)
        correlated_noise = torch.matmul(raw_noise, correlation_matrix)
        
        # Normalize the noise
        normalized_noise = self.noise_normalization(correlated_noise)
        
        # Apply the noise to the feature data (along frame dimension only)
        augmented_features = torch.matmul(normalized_noise, features)
        
        return augmented_features


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