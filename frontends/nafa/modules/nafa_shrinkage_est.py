import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from frontends.nafa.modules.dilated_convolutions_1d.conv import DilatedConv, DilatedConv_Out_128

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class FrameMixup(nn.Module):
    def __init__(
        self, 
        seq_len, 
        feat_dim, 
        temperature=0.2, 
        frame_reduction_ratio=None, 
        frame_augmentation_ratio=1.0,  # Ratio of frames to augment
        shrinkage_coefficient=0.5,     # New parameter for variance reduction
        device='cuda'
        ):
        super(FrameMixup, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.frame_reduction_ratio = frame_reduction_ratio
        if frame_reduction_ratio is not None:
            assert 0 < frame_reduction_ratio <= 1, "frame_reduction_ratio must be between 0 and 1"
            self.reduced_len = max(1, int(seq_len * (1 - frame_reduction_ratio)))
        else:
            self.reduced_len = self.seq_len
        assert 0 <= frame_augmentation_ratio <= 1, "frame_augmentation_ratio must be between 0 and 1"
        self.num_augmented_frames = max(1, int(self.reduced_len * frame_augmentation_ratio))
        self.noise_template = torch.randn(1, self.reduced_len, seq_len).to(device=device)  # [1, reduced_len, seq_len]
        self.temperature = temperature
        self.shrinkage_coefficient = shrinkage_coefficient
        self.device = device

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()  # feature: [batch_size, seq_len, feat_dim]
        # Compute mean feature vector across the batch
        mean_feature = feature.mean(dim=0, keepdim=True)  # [1, seq_len, feat_dim]
        mean_feature = mean_feature.to(self.device)

        noise_template = self.noise_template.expand(batch_size, -1, -1)  # [batch_size, reduced_len, seq_len]
        augmenting_path = self.compute_augmenting_path(noise_template)  # [batch_size, reduced_len, seq_len]
        if self.num_augmented_frames < self.reduced_len:
            augmenting_path = self.randomly_apply_augmentation(augmenting_path)  # [batch_size, reduced_len, seq_len]
        augmented_feature = self.apply_augmenting(feature, augmenting_path)  # [batch_size, reduced_len, feat_dim]

        # Apply shrinkage estimator
        augmented_feature = self.shrinkage_coefficient * augmented_feature + \
                            (1 - self.shrinkage_coefficient) * mean_feature[:, :self.reduced_len, :]

        return augmented_feature  # [batch_size, reduced_len, feat_dim]

    def compute_augmenting_path(self, noise_template):
        mu, sigma = 0, 1  # mean and standard deviation for Gaussian
        gaussian_noise = torch.normal(mu, sigma, size=noise_template.size(), device=noise_template.device)  # [batch_size, reduced_len, seq_len]
        return F.softmax(gaussian_noise / self.temperature, dim=-1)  # [batch_size, reduced_len, seq_len]

    def randomly_apply_augmentation(self, augmenting_path):
        batch_size, reduced_len, seq_len = augmenting_path.size()
        mask = torch.zeros(batch_size, reduced_len, 1, device=augmenting_path.device)  # [batch_size, reduced_len, 1]
        start_index = torch.randint(0, reduced_len - self.num_augmented_frames + 1, (1,)).item()
        mask[:, start_index:start_index + self.num_augmented_frames, :] = 1  # mask with ones in selected frames
        augmenting_path = augmenting_path * mask  # [batch_size, reduced_len, seq_len]
        return augmenting_path  # [batch_size, reduced_len, seq_len]

    def apply_augmenting(self, feature, augmenting_path):
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)  # [batch_size, reduced_len, feat_dim]
        return augmented_feature  # [batch_size, reduced_len, feat_dim]


class NAFA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim
        
        self.frame_augment = FrameMixup(
            seq_len=self.input_seq_length, 
            feat_dim=self.input_f_dim,
            temperature=1.0, 
            frame_reduction_ratio=None,
            frame_augmentation_ratio=1.0,
            shrinkage_coefficient=0.5,
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