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
    def __init__(self, seq_len, feat_dim, temperature=0.2, window_size=10, frame_reduction_ratio=None, compute_loss=True):
        """
        Initialize the FrameAugment module with a learnable template for augmentation.

        Args:
            seq_len (int): The length of the input sequence.
            feat_dim (int): The dimensionality of each feature.
            temperature (float): The temperature for the Gumbel-Softmax. Lower values make it sharper.
            window_size (int): The size of the local window for augmentation regularization.
            frame_reduction_ratio (float): Ratio to reduce the sequence length (0 < ratio <= 1).
            compute_loss (bool): If True, compute the local window loss; if False, return zero loss.
        """
        super(FrameAugment, self).__init__()

        # Compute reduced sequence length
        self.seq_len = seq_len
        self.frame_reduction_ratio = frame_reduction_ratio
        self.compute_loss = compute_loss

        if frame_reduction_ratio is not None:
            assert 0 < frame_reduction_ratio <= 1, "frame_reduction_ratio must be between 0 and 1"
            self.reduced_len = max(1, int(seq_len * (1 - frame_reduction_ratio)))
        else:
            self.reduced_len = self.seq_len

        # Learnable templates for augmentation (initialized randomly)
        self.learned_template = nn.Parameter(torch.randn(1, self.reduced_len, 1))
        self.score_template = nn.Parameter(torch.randn(1, seq_len, 1))

        self.temperature = temperature
        self.window_size = window_size


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

        # import ipdb; ipdb.set_trace() 
        # print(score.shape)

        # Step 1: Create a pairwise distance matrix between the learned template and the score
        distance_matrix = self.compute_pairwise_distances(self.score_template.expand(batch_size, -1, -1))

        # Step 2: Apply Gumbel-Softmax to get a differentiable augmentment path (soft augmenting path)
        soft_augmenting_path = self.compute_gumbel_soft_augmenting_path(distance_matrix)

        # Step 3: augment the features based on the soft augmenting path
        augmented_feature = self.apply_augmenting(feature, soft_augmenting_path)

        # Step 4: Reconstruct to original sequence length for local window loss computation
        reconstructed_feature = self.reconstruct_feature(augmented_feature, soft_augmenting_path)

        # Step 4: If frame_reduction_ratio is not None, reconstruct the feature to original sequence length
        if self.frame_reduction_ratio is not None:
            reconstructed_feature = self.reconstruct_feature(augmented_feature, soft_augmenting_path)
        else:
            reconstructed_feature = augmented_feature

        # Step 5: Compute local window loss if compute_loss is True
        if self.compute_loss:
            local_window_loss = self.compute_local_window_loss(feature, reconstructed_feature)
        else:
            local_window_loss = torch.tensor(0.0, device=feature.device)

        return augmented_feature, local_window_loss

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

    def reconstruct_feature(self, augmented_feature, soft_augmenting_path):
        """
        Reconstruct the feature back to original sequence length using the soft augmenting path.

        Args:
            augmented_feature (Tensor): Augmented feature of shape [batch, reduced_len, feat_dim].
            soft_augmenting_path (Tensor): Soft augmenting path of shape [batch, reduced_len, seq_len].

        Returns:
            reconstructed_feature (Tensor): Reconstructed feature of shape [batch, seq_len, feat_dim].
        """
        # Transpose the soft_augmenting_path to get path from seq_len to reduced_len
        soft_augmenting_path_T = soft_augmenting_path.transpose(1, 2)  # [batch, seq_len, reduced_len]

        # Use einsum to reconstruct the feature
        reconstructed_feature = torch.einsum('bji,bif->bjf', soft_augmenting_path_T, augmented_feature)  # [batch, seq_len, feat_dim]

        return reconstructed_feature

    def compute_local_window_loss(self, feature, reconstructed_feature):
        """
        Compute the local augmentation loss by measuring the similarity between
        local windows of the input feature and the reconstructed feature.

        Args:
            feature (Tensor): Original input feature of shape [batch, seq_len, feat_dim].
            reconstructed_feature (Tensor): Reconstructed feature of shape [batch, seq_len, feat_dim].

        Returns:
            local_loss (Tensor): Scalar tensor representing the local augmentation loss.
        """
        batch_size, seq_len, feat_dim = feature.size()
        window_size = self.window_size

        # Compute the number of windows
        num_windows = seq_len - window_size + 1

        # Extract local windows and compute cosine similarity
        input_windows = feature.unfold(dimension=1, size=window_size, step=1)  # [batch, num_windows, window_size, feat_dim]
        reconstructed_windows = reconstructed_feature.unfold(dimension=1, size=window_size, step=1)  # [batch, num_windows, window_size, feat_dim]

        # Flatten the windows
        input_windows_flat = input_windows.contiguous().view(batch_size, num_windows, -1)  # [batch, num_windows, window_size * feat_dim]
        reconstructed_windows_flat = reconstructed_windows.contiguous().view(batch_size, num_windows, -1)  # [batch, num_windows, window_size * feat_dim]

        # Normalize the vectors
        input_norm = F.normalize(input_windows_flat, dim=2)
        reconstructed_norm = F.normalize(reconstructed_windows_flat, dim=2)

        # Compute cosine similarity for each window
        cos_sim = (input_norm * reconstructed_norm).sum(dim=2)  # [batch, num_windows]

        # Compute local loss (1 - cosine similarity), averaged over windows and batch
        local_loss = (1 - cos_sim).mean()

        return local_loss


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
        
        self.frame_augment = FrameAugment(
            seq_len=self.input_seq_length, 
            feat_dim=self.input_f_dim,
            temperature=0.2, 
            window_size=10, 
            frame_reduction_ratio=0.6,
            compute_loss=False
            )

    def forward(self, x):
        ret = {}

        augment_frame, local_window_loss = self.frame_augment(x.exp())
        augment_frame = torch.log(augment_frame + EPS)

        # import ipdb; ipdb.set_trace() 
        # print(gated_score[0:3])

        # Final outputs
        ret["x"] = x
        ret["features"] = augment_frame
        ret["local_window_loss"] = local_window_loss

        ret["total_loss"] = ret["local_window_loss"]

        return ret