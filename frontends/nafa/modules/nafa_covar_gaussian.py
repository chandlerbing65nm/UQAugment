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
    def __init__(self, seq_len, feat_dim, device='cuda', epsilon=1e-5):
        super(FrameAugment, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.epsilon = epsilon
        
        # Initialize a learnable data-dependent prior for covariance estimation
        self.cov_prior = nn.Parameter(torch.eye(seq_len).to(device=device), requires_grad=True)
        
    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()
        
        # Compute incrementally adaptive noise
        augmenting_path = self.compute_adaptive_noise(batch_size, seq_len)
        
        # Apply the adaptive noise via matrix multiplication
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        
        return augmented_feature

    def compute_adaptive_noise(self, batch_size, seq_len):
        noise_matrix = []
        
        for t in range(seq_len):
            # Generate a mean vector for the current timestep
            mean_t = torch.zeros((batch_size, t + 1), device=self.device)
            
            # Apply Tikhonov regularization to ensure positive definiteness
            cov_t = self.cov_prior[:t+1, :t+1] + self.epsilon * torch.eye(t + 1, device=self.device)
            
            # Sample from the multivariate normal with stabilized covariance
            noise_t = torch.distributions.MultivariateNormal(mean_t, covariance_matrix=cov_t).sample()
            
            # Pad noise to match the overall [seq_len, seq_len] shape
            padded_noise = F.pad(noise_t, (0, seq_len - (t + 1)), value=0)
            noise_matrix.append(padded_noise.unsqueeze(0))
        
        noise_matrix = torch.cat(noise_matrix, dim=0)
        noise_matrix = noise_matrix.permute(1, 0, 2).expand(batch_size, -1, -1)
        
        # Softmax normalization along the frame dimension
        return F.softmax(noise_matrix, dim=-1)

    def apply_augmenting(self, feature, augmenting_path):
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