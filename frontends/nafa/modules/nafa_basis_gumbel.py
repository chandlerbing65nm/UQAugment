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
    def __init__(
        self, 
        seq_len, 
        feat_dim, 
        num_bases=10,  # Number of basis matrices for decomposition
        temperature=0.2, 
        device='cuda'
    ):
        super(FrameAugment, self).__init__()
        
        # Store sequence length and feature dimension
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.device = device
        
        # Initialize learnable basis matrices (B) for decomposition
        # Basis matrices have shape [num_bases, seq_len, seq_len]
        self.basis_matrices = nn.Parameter(
            torch.randn(num_bases, seq_len, seq_len).to(device=device)
        )
        
        # Noise template for activation coefficients (non-learnable)
        self.noise_template = torch.randn(1, seq_len, seq_len).to(device=device)
        
    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()
        
        # Generate Gumbel-based activation coefficients
        activation_coeffs = self.compute_activation_coeffs(batch_size)
        
        # Decompose the noise matrix as a product of basis matrices and activation coefficients
        mixing_matrix = self.compose_noise_matrix(activation_coeffs)
        
        # Apply structured augmentation to the input features
        augmented_feature = self.apply_augmenting(feature, mixing_matrix)
        
        return augmented_feature

    def compute_activation_coeffs(self, batch_size):
        # Generate Gumbel noise for activation coefficients
        gumbel_noise = -torch.log(-torch.log(torch.rand(batch_size, self.basis_matrices.size(0)) + EPS) + EPS).to(self.device)
        # Normalize activation coefficients with softmax along basis dimension
        activation_coeffs = F.softmax(gumbel_noise / self.temperature, dim=-1)  # Shape: [batch_size, num_bases]
        return activation_coeffs

    def compose_noise_matrix(self, activation_coeffs):
        # Reshape activation coefficients to broadcast for matrix multiplication
        # Shape: [batch_size, num_bases, 1, 1] to [batch_size, num_bases, seq_len, seq_len]
        activation_coeffs = activation_coeffs.view(-1, self.basis_matrices.size(0), 1, 1)
        
        # Compute structured noise matrix as weighted sum of basis matrices
        # Shape after summing: [batch_size, seq_len, seq_len]
        mixing_matrix = torch.sum(activation_coeffs * self.basis_matrices, dim=1)
        
        # Normalize the final mixing matrix along the last dimension with softmax
        return F.softmax(mixing_matrix, dim=-1)

    def apply_augmenting(self, feature, mixing_matrix):
        # Apply augmentation using matrix multiplication along the frame dimension
        augmented_feature = torch.einsum('bij,bjf->bif', mixing_matrix, feature)  # Shape: [batch_size, seq_len, feat_dim]
        return augmented_feature


class NAFA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim
        
        self.frame_augment = FrameAugment(
            seq_len=self.input_seq_length, 
            feat_dim=self.input_f_dim,
            temperature=0.2, 
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