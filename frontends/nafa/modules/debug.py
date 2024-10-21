import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from frontends.nafa.modules.dilated_convolutions_1d.conv import DilatedConv, DilatedConv_Out_128

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class FrameAlignment(nn.Module):
    def __init__(self, seq_len, feat_dim):
        """
        Initialize the FrameAlignment module with a learnable template for alignment.

        Args:
            seq_len (int): The length of the input sequence.
            feat_dim (int): The dimensionality of each feature.
        """
        super(FrameAlignment, self).__init__()

        # Learnable template for aligning alignment (initialized randomly)
        self.learned_template = nn.Parameter(torch.randn(1, seq_len, 1))

    def forward(self, score, feature):
        """
        Forward function that computes a aligned feature representation
        using a differentiable FrameAlignment mechanism conditioned on the score tensor.

        Args:
            score (Tensor): A tensor of shape [batch, seq_len, 1] representing the conditioning score.
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim] representing the features.
        
        Returns:
            out_feature (Tensor): aligned output feature of shape [batch, seq_len, feat_dim].
        """
        batch_size, seq_len, feat_dim = feature.size()

        # # Replace score with random values (same shape)
        # random_score = torch.randn_like(score)  # Random tensor with same shape as score

        # Step 1: Create a pairwise distance matrix between the score and the learnable template
        distance_matrix = self.compute_pairwise_distances(score)

        # Step 2: Apply softmax to get a differentiable alignment path (soft aligning path)
        soft_aligning_path = self.compute_soft_aligning_path(distance_matrix)

        # Step 3: align the features based on the soft aligning path
        aligned_feature = self.apply_aligning(feature, soft_aligning_path)

        return aligned_feature, soft_aligning_path

    def compute_pairwise_distances(self, score):
        """
        Compute pairwise distances between score matrix and the learned template for alignment.
        
        Args:
            score (Tensor): A tensor of shape [batch, seq_len, 1].
        
        Returns:
            distance_matrix (Tensor): Pairwise distances for alignment, shape [batch, seq_len, seq_len].
        """
        batch_size, seq_len, _ = score.size()

        # Compute pairwise distances between score and the learned template
        distance_matrix = torch.cdist(score, score)  # [batch, seq_len, seq_len]
        
        return distance_matrix

    def compute_soft_aligning_path(self, distance_matrix):
        """
        Compute a soft alignment matrix (soft aligning path) using softmax over the distances.

        Args:
            distance_matrix (Tensor): A tensor of shape [batch, seq_len, seq_len].
        
        Returns:
            soft_aligning_path (Tensor): A soft alignment path, shape [batch, seq_len, seq_len].
        """
        # Apply softmax to get soft aligning path, normalized across sequence length dimension
        # Improve numerical stability by subtracting the max value along the sequence dimension
        soft_aligning_path = F.softmax(-distance_matrix - distance_matrix.max(dim=-1, keepdim=True)[0], dim=-1)  # [batch, seq_len, seq_len]
        return soft_aligning_path

    def apply_aligning(self, feature, soft_aligning_path):
        """
        Apply the aligning to the feature using the soft aligning path.

        Args:
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim].
            soft_aligning_path (Tensor): A tensor of shape [batch, seq_len, seq_len].
        
        Returns:
            aligned_feature (Tensor): aligned feature of shape [batch, seq_len, feat_dim].
        """
        # Use einsum to apply aligning across the sequence length dimension
        aligned_feature = torch.einsum('bij,bjf->bif', soft_aligning_path, feature)  # [batch, seq_len, feat_dim]
        
        return aligned_feature


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

        # Stochastic gating layer to learn when to modulate frames
        self.align = FrameAlignment(seq_len=self.input_seq_length, feat_dim=self.input_f_dim)

    def forward(self, x):
        ret = {}

        # Compute the importance score using the dilated convolutional model
        score = torch.sigmoid(self.model(x.permute(0, 2, 1)).permute(0, 2, 1))
        score, _ = self.monotonic_score_norm(score, total_length=self.input_seq_length)

        # Align frames based on the importance scores
        align_frame, aligning_path = self.align(score, x.exp())
        align_frame = torch.log(align_frame + 1e-8)

        # Compute losses
        guide_loss = self.guide_loss(x, importance_score=score)
        contrastive_loss = self.contrastive_importance_loss(score)

        # Final outputs
        ret["x"] = x
        ret["score"] = score
        ret["features"] = align_frame
        ret["guide_loss"] = guide_loss
        ret["contrastive_loss"] = contrastive_loss

        # Total loss combines guide loss and the new contrastive importance loss
        ret["total_loss"] = ret["guide_loss"] + contrastive_loss

        return ret

    def monotonic_score_norm(self, score, total_length):
        ####################################################################
        # Trying to rescale the total score
        sum_score = torch.sum(score, dim=(1, 2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.
        ####################################################################

        ####################################################################
        # If the weight for one frame is greater than one, rescale the batch
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if torch.sum(dims_need_norm) > 0:
            score[dims_need_norm] = (
                score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
            )
        ####################################################################

        ####################################################################
        # Remove the zero pad at the end, using the rescaling of the weight in between
        # torch.Size([32, 1056, 1])
        if torch.sum(dims_need_norm) > 0:
            sum_score = torch.sum(score, dim=(1, 2), keepdim=True)
            distance_with_target_length = (total_length - sum_score)[:, 0, 0]
            axis = torch.logical_and(
                score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN
            )  # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if distance_with_target_length[i] >= 1:
                    intervel = 1.0 - score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel)
                    if alpha > 1:
                        alpha = 1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def zero_loss_like(self, x):
        return torch.tensor([0.0], device=x.device)

    def guide_loss(self, mel, importance_score):
        if torch.min(mel) < 0:
            x = mel.exp()
        else:
            x = mel
        score_mask = torch.mean(x, dim=-1, keepdim=True)
        score_mask = score_mask < (torch.min(score_mask) + 1e-6)

        guide_loss_final = self.zero_loss_like(mel)

        for id in range(importance_score.size(0)):
            guide_loss = torch.mean(importance_score[id][score_mask[id]])
            if torch.isnan(guide_loss).item():
                continue
            if guide_loss > 0.5:
                guide_loss_final += guide_loss / importance_score.size(0)

        return guide_loss_final

    def contrastive_importance_loss(self, importance_score):
        # Shift the importance scores to get neighbors
        score_prev = torch.zeros_like(importance_score)
        score_next = torch.zeros_like(importance_score)

        score_prev[:, 1:, :] = importance_score[:, :-1, :]
        score_next[:, :-1, :] = importance_score[:, 1:, :]

        # Compute the contrastive loss
        contrast = importance_score - 0.5 * (score_prev + score_next)
        contrast_loss = torch.mean(contrast[:, 1:-1, :] ** 2)  # Exclude first and last frames

        return contrast_loss

if __name__ == "__main__":
    # Define input sizes
    seq_len = 251
    feat_dim = 64
    batch_size = 200

    # Instantiate the model
    model = NAFA(in_t_dim=seq_len, in_f_dim=feat_dim)

    # Create dummy input data
    x = torch.randn(batch_size, seq_len, feat_dim)

    # Forward pass
    outputs = model(x)

    # Print the shapes of the outputs
    print("Input shape:", x.shape)
    print("Score shape:", outputs["score"].shape)
    print("Aligned features shape:", outputs["features"].shape)
    print("Total loss:", outputs["total_loss"].item())