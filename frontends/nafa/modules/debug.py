import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropStripes(nn.Module):
    def __init__(self, dim, drop_width):
        """Drop stripes with Gumbel noise.

        Args:
          dim: int, dimension along which to drop (2 for time, 3 for frequency)
          drop_width: int, maximum width of stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [1, 2]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 3

        batch_size = input.shape[0]
        total_width = input.shape[self.dim]

        # Compute random distance and start index (bgn)
        device = input.device  # Input device
        distance = self.drop_width
        bgn = torch.randint(low=0, high=total_width - distance, size=(1,), device=device)[0]

        # Create noise matrix: [batch_size, distance, distance]
        stripe_shape = (batch_size, distance, distance)
        gumbel_noise = self.sample_gumbel(stripe_shape, device)
        gumbel_softmax = F.softmax(gumbel_noise, dim=-1)  # Apply softmax along the last dim

        if self.dim == 1:
            # Batch matrix multiplication for time dimension
            input[:, bgn : bgn + distance, :] = torch.einsum(
                'bij,bjk->bik', gumbel_softmax, input[:, bgn : bgn + distance, :]
            )
        elif self.dim == 2:
            # Batch matrix multiplication for frequency dimension
            input[:, :, bgn : bgn + distance] = torch.einsum(
                'bij,bkj->bik', gumbel_softmax, input[:, :, bgn : bgn + distance]
            ).permute(0,2,1)

        return input

    def sample_gumbel(self, shape, device, eps=1e-10):
        """Sample Gumbel noise."""
        U = torch.rand(shape, device=device)  # Ensure it's on the correct device
        return -torch.log(-torch.log(U + eps) + eps)

class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, freq_drop_width):
        """Spec augmentation with Gumbel noise.

        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """
        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=1, drop_width=time_drop_width)

        self.freq_dropper = DropStripes(dim=2, drop_width=freq_drop_width)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    random_state = np.random.RandomState(0)
    np_data = random_state.normal(size=(200, 251, 64))
    pt_data = torch.Tensor(np_data)

    spec_augmenter = SpecAugmentation(time_drop_width=230, 
                                      freq_drop_width=0)

    result = spec_augmenter(pt_data)

    print(result.shape)
