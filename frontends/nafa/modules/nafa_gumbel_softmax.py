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
    def __init__(self, seq_len, feat_dim, temperature=1.0):
        """
        Initialize the FrameAlignment module with a learnable template for alignment.

        Args:
            seq_len (int): The length of the input sequence.
            feat_dim (int): The dimensionality of each feature.
            temperature (float): The temperature for the Gumbel-Softmax. Lower values make it sharper.
        """
        super(FrameAlignment, self).__init__()

        # Learnable template for aligning alignment (initialized randomly)
        self.learned_template = nn.Parameter(torch.randn(1, seq_len, 1))
        self.temperature = temperature

    def forward(self, score, feature):
        """
        Forward function that computes an aligned feature representation
        using a differentiable FrameAlignment mechanism conditioned on the score tensor.

        Args:
            score (Tensor): A tensor of shape [batch, seq_len, 1] representing the conditioning score.
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim] representing the features.

        Returns:
            out_feature (Tensor): aligned output feature of shape [batch, seq_len // 2, feat_dim].
        """
        batch_size, seq_len, feat_dim = feature.size()

        # Step 1: Create a pairwise distance matrix between the learned template and the score
        distance_matrix = self.compute_pairwise_distances(score)

        # Step 2: Apply Gumbel-Softmax to get a differentiable alignment path (soft aligning path)
        soft_aligning_path = self.compute_gumbel_soft_aligning_path(distance_matrix)

        # Step 3: Align the features based on the soft aligning path
        aligned_feature = self.apply_aligning(feature, soft_aligning_path)

        return aligned_feature, soft_aligning_path

    def compute_pairwise_distances(self, score):
        """
        Compute pairwise distances between the learned template and the score matrix for alignment.

        Args:
            score (Tensor): A tensor of shape [batch, seq_len, 1].

        Returns:
            distance_matrix (Tensor): Pairwise distances for alignment, shape [batch, seq_len // 2, seq_len].
        """
        batch_size, seq_len, _ = score.size()

        # Expand the learned_template to match batch size
        learned_template_expanded = self.learned_template.expand(batch_size, -1, -1)  # [batch, seq_len // 2, 1]

        # Compute pairwise distances between learned_template and score
        # The result will be [batch, seq_len // 2, seq_len]
        distance_matrix = torch.cdist(learned_template_expanded, score)  # [batch, seq_len // 2, seq_len]

        return distance_matrix

    def compute_gumbel_soft_aligning_path(self, distance_matrix):
        """
        Compute a soft alignment matrix (soft aligning path) using Gumbel-Softmax over the distances.

        Args:
            distance_matrix (Tensor): A tensor of shape [batch, seq_len // 2, seq_len].

        Returns:
            soft_aligning_path (Tensor): A soft alignment path, shape [batch, seq_len // 2, seq_len].
        """
        # Convert distances to logits (negative distance)
        gumbel_logits = -distance_matrix

        # Apply Gumbel-Softmax sampling
        gumbel_soft_aligning_path = self.gumbel_softmax_sample(gumbel_logits, temperature=self.temperature)

        return gumbel_soft_aligning_path

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

    def apply_aligning(self, feature, soft_aligning_path):
        """
        Apply the aligning to the feature using the soft aligning path.

        Args:
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim].
            soft_aligning_path (Tensor): A tensor of shape [batch, seq_len // 2, seq_len].

        Returns:
            aligned_feature (Tensor): aligned feature of shape [batch, seq_len // 2, feat_dim].
        """
        # Use einsum to apply aligning across the sequence length dimension
        # Adjusted indices to match the dimensions
        aligned_feature = torch.einsum('bij,bjf->bif', soft_aligning_path, feature)  # [batch, seq_len // 2, feat_dim]

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

        # align_frame = self.align_frame(gated_score, x_modulated.exp(), total_length=self.output_seq_length)
        align_frame, aligning_path = self.align(score, x.exp())
        align_frame = torch.log(align_frame + EPS)

        # import ipdb; ipdb.set_trace() 
        # print(gated_score[0:3])

        guide_loss = self.guide_loss(x, importance_score=score)

        # Final outputs
        ret["x"] = x
        ret["score"] = score
        ret["features"] = align_frame
        ret["guide_loss"] = guide_loss

        ret["total_loss"] = ret["guide_loss"]

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
        return torch.tensor([0.0]).to(x.device)

    def guide_loss(self, mel, importance_score):
        # If the mel spectrogram is in log scale
        # mel: [bs, t-steps, f-bins]
        # importance_score: [bs, t-steps, 1]
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
                guide_loss_final = (
                    guide_loss_final + guide_loss / importance_score.size(0)
                )

        return guide_loss_final