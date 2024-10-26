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
        self.noise_template = torch.randn(1, seq_len, seq_len).to(device=device)

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()
        mixing_matrix = self.noise_template.expand(batch_size, -1, -1)
        augmenting_path = self.compute_augmenting_path(mixing_matrix)
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        return augmented_feature

    def compute_augmenting_path(self, mixing_matrix):
        mu, sigma = 0, 1  # mean and standard deviation for Gaussian
        gaussian_noise = torch.normal(mu, sigma, size=mixing_matrix.size(), device=mixing_matrix.device)
        return F.softmax(gaussian_noise, dim=-1)

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