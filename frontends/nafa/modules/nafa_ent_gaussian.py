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
        entropy_level=0.5, 
        device='cuda'
        ):
        super(FrameAugment, self).__init__()
        
        # Define sequence length, entropy level, and noise template on the specified device
        self.seq_len = seq_len
        self.entropy_level = entropy_level
        self.temperature = temperature
        self.device = device
        self.noise_template = torch.randint(low=1, high=10, size=(1, seq_len, seq_len)).float().to(device)

    def forward(self, feature):
        # Move feature to model's device to ensure compatibility
        feature = feature.to(self.device)
        
        batch_size, seq_len, feat_dim = feature.size()
        mixing_matrix = self.noise_template.expand(batch_size, -1, -1)
        augmenting_path = self.compute_augmenting_path(mixing_matrix)
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        return augmented_feature

    def compute_augmenting_path(self, mixing_matrix):
        # Normalize using softmax and then apply entropy constraint
        modulated_noise = F.softmax(mixing_matrix / self.temperature, dim=-1)
        
        # Apply discrete entropy modulation by adjusting values to match entropy constraints
        modulated_noise = self.entropy_constrained_resampling(modulated_noise, self.entropy_level)
        return modulated_noise

    def entropy_constrained_resampling(self, noise, entropy_target):
        # Enforce entropy target on the mixing matrix along the frame dimension
        entropy_values = -torch.sum(noise * torch.log(noise + 1e-8), dim=-1, keepdim=True)
        noise_adjustment = (entropy_target - entropy_values)
        
        # Scale the adjustment back into the noise matrix
        adjusted_noise = noise + noise_adjustment * torch.sign(noise)
        adjusted_noise = F.softmax(adjusted_noise / self.temperature, dim=-1)
        
        return adjusted_noise

    def apply_augmenting(self, feature, augmenting_path):
        # Apply the augmenting path to the input features
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