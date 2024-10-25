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
        temperature=0.2, 
        sinkhorn_iters=10, 
        device='cuda'
    ):
        super(FrameAugment, self).__init__()

        # Initialize sequence length, temperature, and device
        self.seq_len = seq_len
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.noise_template = torch.randn(1, seq_len, seq_len, device=device)

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()

        # Expand noise template to match batch size
        mixing_matrix = self.noise_template.expand(batch_size, -1, -1)
        
        # Apply Gumbel noise and Sinkhorn normalization
        augmenting_path = self.compute_augmenting_path(mixing_matrix)
        
        # Apply augmented path to the feature tensor
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        return augmented_feature

    def compute_augmenting_path(self, mixing_matrix):
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(mixing_matrix) + EPS) + EPS)
        
        # Add Gumbel noise to the mixing matrix
        perturbed_matrix = (mixing_matrix + gumbel_noise) / self.temperature
        
        # Apply the Sinkhorn normalization
        for _ in range(self.sinkhorn_iters):
            perturbed_matrix = F.softmax(perturbed_matrix, dim=-1)
            perturbed_matrix = F.softmax(perturbed_matrix, dim=-2)
        
        return perturbed_matrix

    def apply_augmenting(self, feature, augmenting_path):
        # Apply augmentation via matrix multiplication on the frame dimension
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