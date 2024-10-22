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
    def __init__(self, seq_len, feat_dim, temperature=0.2):
        """
        Initialize the FrameAugment module with a learnable template for augmentment.

        Args:
            seq_len (int): The length of the input sequence.
            feat_dim (int): The dimensionality of each feature.
            temperature (float): The temperature for the Gumbel-Softmax. Lower values make it sharper.
        """
        super(FrameAugment, self).__init__()

        # Learnable template for augmentment (initialized randomly)
        self.learned_template = nn.Parameter(torch.randn(1, seq_len, 1))
        self.score_template = nn.Parameter(torch.randn(1, seq_len, 1))
        self.temperature = temperature

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

        score = self.score_template.expand(batch_size, -1, -1)

        # import ipdb; ipdb.set_trace() 
        # print(score.shape)

        # Step 1: Create a pairwise distance matrix between the learned template and the score
        distance_matrix = self.compute_pairwise_distances(score)

        # Step 2: Apply Gumbel-Softmax to get a differentiable augmentment path (soft augmenting path)
        soft_augmenting_path = self.compute_gumbel_soft_augmenting_path(distance_matrix)

        # Step 3: augment the features based on the soft augmenting path
        augmented_feature = self.apply_augmenting(feature, soft_augmenting_path)

        return augmented_feature, soft_augmenting_path

    def compute_pairwise_distances(self, score):
        """
        Compute pairwise distances between the learned template and the score matrix for augmentment.

        Args:
            score (Tensor): A tensor of shape [batch, seq_len, 1].

        Returns:
            distance_matrix (Tensor): Pairwise distances for augmentment
        """
        batch_size, seq_len, _ = score.size()

        # Expand the learned_template to match batch size
        learned_template_expanded = self.learned_template.expand(batch_size, -1, -1)

        # Compute pairwise distances between learned_template and score
        distance_matrix = torch.cdist(learned_template_expanded, score)

        return distance_matrix

    def compute_gumbel_soft_augmenting_path(self, distance_matrix):
        """
        Compute a soft augmentment matrix (soft augmenting path) using Gumbel-Softmax over the distances.

        Args:
            distance_matrix (Tensor): A tensor of shape [batch, seq_len, seq_len].

        Returns:
            soft_augmenting_path (Tensor): A soft augmentment path
        """
        # Convert distances to logits (negative distance)
        gumbel_logits = -distance_matrix

        # Apply Gumbel-Softmax sampling
        gumbel_soft_augmenting_path = self.gumbel_softmax_sample(gumbel_logits, temperature=self.temperature)

        return gumbel_soft_augmenting_path

    def gumbel_softmax_sample(self, logits, temperature=1.0):
        """
        Samples from the Gumbel-Softmax distribution with the specified temperature.

        Args:
            logits (Tensor): Input logits.
            temperature (float): Temperature for Gumbel-Softmax. Lower temperatures make it more like hard selection.

        Returns:
            Tensor: Softmax with Gumbel noise added.
        """
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + EPS) + EPS)

        # Add Gumbel noise and apply softmax with temperature scaling
        return F.softmax((logits + gumbel_noise) / temperature, dim=-1)

    def apply_augmenting(self, feature, soft_augmenting_path):
        """
        Apply the augmenting to the feature using the soft augmenting path.

        Args:
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim].
            soft_augmenting_path (Tensor): A tensor of shape [batch, seq_len, seq_len].

        Returns:
            augmented_feature (Tensor): augmented feature of shape [batch, seq_len, feat_dim].
        """
        # Use einsum to apply augmenting across the sequence length dimension
        # Adjusted indices to match the dimensions
        augmented_feature = torch.einsum('bij,bjf->bif', soft_augmenting_path, feature)  # [batch, seq_len // 2, feat_dim]

        return augmented_feature


class NAFA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim


        # Dilated Convolution to learn importance scores
        self.model = DilatedConv(
            in_channels=self.input_f_dim,
            dilation_rate=1,
            input_size=self.input_seq_length,
            kernel_size=5,
            stride=1,
        )
        
        self.frame_augment = FrameAugment(seq_len=self.input_seq_length, feat_dim=self.input_f_dim)

    def forward(self, x):
        ret = {}

        augment_frame, augmenting_path = self.frame_augment(x.exp())
        augment_frame = torch.log(augment_frame + EPS)

        # import ipdb; ipdb.set_trace() 
        # print(gated_score[0:3])

        # Final outputs
        ret["x"] = x
        ret["features"] = augment_frame
        ret["guide_loss"] = torch.tensor([0.0]).to(x.device)

        ret["total_loss"] = ret["guide_loss"]

        return ret