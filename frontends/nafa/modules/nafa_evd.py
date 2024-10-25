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
    def __init__(self, seq_len, feat_dim, temperature=0.2, frame_reduction_ratio=None, device='cuda'):
        """
        Initialize the FrameAugment module using Extreme Value Distribution.
        """
        super(FrameAugment, self).__init__()

        # Compute reduced sequence length
        self.seq_len = seq_len
        self.frame_reduction_ratio = frame_reduction_ratio

        if frame_reduction_ratio is not None:
            assert 0 < frame_reduction_ratio <= 1, "frame_reduction_ratio must be between 0 and 1"
            self.reduced_len = max(1, int(seq_len * (1 - frame_reduction_ratio)))
        else:
            self.reduced_len = self.seq_len

        # noise templates for augmentation (initialized randomly)
        self.noise_template = torch.randn(1, self.reduced_len, seq_len).to(device=device)

        self.temperature = temperature

    def forward(self, feature):
        """
        Forward function that computes an augmented feature representation.
        """
        batch_size, seq_len, feat_dim = feature.size()

        # Create a mixing matrix from the noise template
        mixing_matrix = self.noise_template.expand(batch_size, -1, -1)

        # Compute the augmenting path using Extreme Value Distribution
        augmenting_path = self.compute_augmenting_path(mixing_matrix)

        # Augment the features based on the augmenting path
        augmented_feature = self.apply_augmenting(feature, augmenting_path)

        return augmented_feature

    def compute_augmenting_path(self, mixing_matrix):
        """
        Compute an augmenting matrix using the Extreme Value Distribution.
        """
        # Transform logits using negative mixing matrix for extreme value sampling
        logits = -mixing_matrix
        return self.extreme_value_sample(logits)

    def extreme_value_sample(self, logits):
        """
        Samples from the Extreme Value Distribution.
        """
        # Parameters for the GEV
        mu, sigma, xi = 0, 1, 0.1  # Adjust these parameters as needed

        # Normalize logits for sampling
        logits = (logits - logits.min()) / (logits.max() - logits.min())

        # Apply inverse GEV to get samples
        samples = mu + sigma * ((-torch.log(torch.rand_like(logits))) ** (-xi) - 1) / xi
        samples = F.softmax(samples / self.temperature, dim=-1)

        return samples

    def apply_augmenting(self, feature, augmenting_path):
        """
        Apply the augmenting to the feature using the augmenting path.
        """
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)  # [batch, reduced_len, feat_dim]
        return augmented_feature


class NAFA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim
        
        self.frame_augment = FrameAugment(
            seq_len=self.input_seq_length, 
            feat_dim=self.input_f_dim,
            frame_reduction_ratio=0.6,
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
