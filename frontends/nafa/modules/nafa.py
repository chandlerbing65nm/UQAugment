import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from frontends.nafa.modules.core import Base
from frontends.nafa.modules.dilated_convolutions_1d.conv import DilatedConv, DilatedConv_Out_128

from frontends.nafa.modules.pooling import Pooling_layer

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class FrameAlignment(nn.Module):
    def __init__(self, seq_len, feat_dim, window_size=5):
        super(FrameAlignment, self).__init__()
        self.window_size = window_size

    def forward(self, score, feature):
        batch_size, seq_len, feat_dim = feature.size()
        alignment_matrix = self.compute_alignment_matrix(score)
        soft_aligning_path = self.compute_soft_aligning_path(alignment_matrix)
        aligned_feature = self.apply_aligning(feature, soft_aligning_path)
        local_alignment_loss = self.compute_local_alignment_loss(feature, aligned_feature)
        return aligned_feature, local_alignment_loss

    def compute_alignment_matrix(self, score):
        batch_size, seq_len, _ = score.size()
        projection_template = torch.normal(mean=0, std=1, size=(batch_size, seq_len, seq_len)).to(device=score.device) # [batch, seq_len, seq_len]
        score = torch.diag_embed(score.squeeze(-1)) # [batch, seq_len, seq_len]
        alignment_matrix = torch.matmul(projection_template, score)  # [batch, seq_len, seq_len]
        return alignment_matrix

    def compute_soft_aligning_path(self, alignment_matrix):
        soft_aligning_path = F.softmax(alignment_matrix, dim=-1) # [batch, seq_len, seq_len]
        return soft_aligning_path

    def apply_aligning(self, feature, soft_aligning_path):
        aligned_feature = torch.einsum('bij,bjf->bif', soft_aligning_path, feature)  # [batch, seq_len, feat_dim]
        return aligned_feature

    def compute_local_alignment_loss(self, feature, aligned_feature):
        batch_size, seq_len, feat_dim = feature.size()
        window_size = self.window_size

        num_windows = seq_len - window_size + 1
        input_windows = feature.unfold(dimension=1, size=window_size, step=1)  # [batch, num_windows, window_size, feat_dim]
        aligned_windows = aligned_feature.unfold(dimension=1, size=window_size, step=1)  # [batch, num_windows, window_size, feat_dim]
        input_windows_flat = input_windows.contiguous().view(batch_size, num_windows, -1)  # [batch, num_windows, window_size * feat_dim]
        aligned_windows_flat = aligned_windows.contiguous().view(batch_size, num_windows, -1)  # [batch, num_windows, window_size * feat_dim]
        input_norm = F.normalize(input_windows_flat, dim=2)
        aligned_norm = F.normalize(aligned_windows_flat, dim=2)
        cos_sim = (input_norm * aligned_norm).sum(dim=2)  # [batch, num_windows]
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
            out_channels=1,
            dilation_rate=1,
            input_size=self.input_seq_length,
            kernel_size=5,
            stride=1,
        )
        
        self.align = FrameAlignment(seq_len=self.input_seq_length, feat_dim=self.input_f_dim, window_size=10)

    def forward(self, x):
        ret = {}

        # Compute the importance score using the dilated convolutional model
        score = torch.sigmoid(self.model(x.permute(0, 2, 1)).permute(0, 2, 1))
        score = self.score_norm(score, total_length=self.input_seq_length)

        align_frame, align_loss = self.align(score, x.exp())
        align_frame = torch.log(align_frame + EPS)

        # import ipdb; ipdb.set_trace() 
        # print(align_loss.item())

        # Final outputs
        ret["x"] = x
        ret["score"] = score
        ret["features"] = align_frame
        ret["align_loss"] = align_loss

        ret["total_loss"] = ret["align_loss"]

        return ret

    def score_norm(self, score, total_length):
        sum_score = torch.sum(score, dim=(1, 2), keepdim=True)
        score = (score / sum_score) * total_length
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if torch.sum(dims_need_norm) > 0:
            score[dims_need_norm] = (
                score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
            )
        if torch.sum(dims_need_norm) > 0:
            sum_score = torch.sum(score, dim=(1, 2), keepdim=True)
            distance_with_target_length = (total_length - sum_score)[:, 0, 0]
            axis = torch.logical_and(
                score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN
            )
            for i in range(score.size(0)):
                if distance_with_target_length[i] >= 1:
                    intervel = 1.0 - score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel)
                    if alpha > 1:
                        alpha = 1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score
