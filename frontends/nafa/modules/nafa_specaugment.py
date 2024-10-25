import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from frontends.nafa.modules.dilated_convolutions_1d.conv import DilatedConv, DilatedConv_Out_128

EPS = 1e-12
RESCALE_INTERVEL_MIN = 1e-4
RESCALE_INTERVEL_MAX = 1 - 1e-4

class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes with Gumbel noise.

        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        batch_size = input.shape[0]
        total_width = input.shape[self.dim]

        for n in range(batch_size):
            self.transform_slice(input[n], total_width)

            return input

    def sample_gumbel(self, shape, device, eps=1e-10):
        """Sample Gumbel noise."""
        U = torch.rand(shape, device=device)  # Ensure it's on the correct device
        return -torch.log(-torch.log(U + eps) + eps)

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""
        
        device = e.device  # Get the device from the tensor

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,), device=device)[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,), device=device)[0]

            # Create the mask region for either time or frequency dimension
            if self.dim == 2:
                stripe_shape = e[:, bgn : bgn + distance, :].shape
                gumbel_noise = self.sample_gumbel(stripe_shape, device)
                gumbel_softmax = F.softmax(gumbel_noise, dim=self.dim-1)  # Softmax on time or freq
                e[:, bgn : bgn + distance, :] *= gumbel_softmax
            elif self.dim == 3:
                stripe_shape = e[:, :, bgn : bgn + distance].shape
                gumbel_noise = self.sample_gumbel(stripe_shape, device)
                gumbel_softmax = F.softmax(gumbel_noise, dim=self.dim-1)  # Softmax on time or freq
                e[:, :, bgn : bgn + distance] *= gumbel_softmax

class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num):
        """Spec augmentation with Gumbel noise.

        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """
        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, 
                                        stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, 
                                        stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


class NAFA(nn.Module):
    def __init__(self, in_t_dim, in_f_dim):
        super().__init__()
        self.input_seq_length = in_t_dim
        self.input_f_dim = in_f_dim

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=100, 
            time_stripes_num=3, 
            freq_drop_width=0, 
            freq_stripes_num=0)

    def forward(self, x):
        ret = {}

        augment = self.spec_augmenter(x.unsqueeze(1))

        # Final outputs
        ret["x"] = x
        ret["features"] = augment.squeeze(1)
        ret["dummy"] = torch.tensor(0.0, device=x.device)
        ret["total_loss"] = ret["dummy"]

        return ret
