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
        frame_reduction_ratio=None, 
        frame_augmentation_ratio=1.0,  # Ratio of frames to augment
        evd_type="gumbel", 
        device='cuda'
        ):
        """
        Initialize the FrameAugment module with an option for different augmenting paths.

        Args:
            seq_len (int): The length of the input sequence.
            feat_dim (int): The dimensionality of each feature.
            temperature (float): The temperature for the Gumbel-Softmax. Lower values make it sharper.
            frame_reduction_ratio (float): Ratio to reduce the sequence length (0 < ratio <= 1).
            frame_augmentation_ratio (float): Ratio of reduced_len frames to augment (0 <= ratio <= 1).
            evd_type (str): Type of extreme value distribution to use.
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

        # Ratio of frames to apply augmentation (converted to count)
        assert 0 <= frame_augmentation_ratio <= 1, "frame_augmentation_ratio must be between 0 and 1"
        self.num_augmented_frames = max(1, int(self.reduced_len * frame_augmentation_ratio))

        # Noise templates for augmentation (initialized randomly)
        self.noise_template = torch.randn(1, self.reduced_len, seq_len).to(device=device)

        self.temperature = temperature
        self.evd_type = evd_type

    def forward(self, feature):
        """
        Forward function that computes an augmented feature representation
        using a differentiable FrameAugment mechanism conditioned on the score tensor.

        Args:
            feature (Tensor): A tensor representing the features.

        Returns:
            out_feature (Tensor): augmented output feature.
        """
        batch_size, seq_len, feat_dim = feature.size()

        # Step 1: Create a mixing matrix from the noise template
        mixing_matrix = self.noise_template.expand(batch_size, -1, -1)

        # Step 2: Apply selected activation to get a differentiable augmenting path
        augmenting_path = self.compute_augmenting_path(mixing_matrix)

        # Step 3: Randomly select frames to augment
        if self.num_augmented_frames < self.reduced_len:
            augmenting_path = self.randomly_apply_augmentation(augmenting_path)

        # Step 4: Augment the features based on the augmenting path
        augmented_feature = self.apply_augmenting(feature, augmenting_path)

        return augmented_feature

    def compute_augmenting_path(self, mixing_matrix):
        """
        Compute an augmenting matrix (augmenting path) using the selected activation function.

        Args:
            mixing_matrix (Tensor): A tensor of shape [batch, seq_len, seq_len].

        Returns:
            augmenting_path (Tensor): An augmenting path matrix.
        """
        if self.evd_type == "gumbel":
            # Gumbel-Softmax path
            logits = mixing_matrix

            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + EPS) + EPS)

            # Add Gumbel noise and apply softmax with temperature scaling
            return F.softmax(gumbel_noise / self.temperature, dim=-1)

        else:
            raise ValueError(f"Unsupported activation type: {self.evd_type}")

    def randomly_apply_augmentation(self, augmenting_path):
        """
        Applies augmentation to a single contiguous block of frames.

        Args:
            augmenting_path (Tensor): A tensor of shape [batch, reduced_len, seq_len].

        Returns:
            augmenting_path (Tensor): A tensor where only a contiguous block of frames is augmented.
        """
        batch_size, reduced_len, seq_len = augmenting_path.size()

        # Create a mask to select a contiguous block of frames for augmentation
        mask = torch.zeros(batch_size, reduced_len, 1, device=augmenting_path.device)

        # Randomly select a starting index for the block
        start_index = torch.randint(0, reduced_len - self.num_augmented_frames + 1, (1,)).item()
        
        # Create a contiguous block from start_index
        mask[:, start_index:start_index + self.num_augmented_frames, :] = 1

        # Apply the mask to the augmenting path (only selected frames will have augmenting applied)
        augmenting_path = augmenting_path * mask

        return augmenting_path

    def apply_augmenting(self, feature, augmenting_path):
        """
        Apply the augmenting to the feature using the augmenting path.

        Args:
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim].
            augmenting_path (Tensor): A tensor of shape [batch, reduced_len, seq_len].

        Returns:
            augmented_feature (Tensor): augmented feature of shape [batch, reduced_len, feat_dim].
        """
        # Use einsum to apply augmenting across the sequence length dimension
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
            temperature=0.2, 
            frame_reduction_ratio=0.6,
            frame_augmentation_ratio=0.9,  # Use ratio instead of fixed number
            evd_type='gumbel', 
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
