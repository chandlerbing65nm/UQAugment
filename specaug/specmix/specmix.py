import random
import torch
import torch.nn as nn

class SpecMix(nn.Module):
    def __init__(self, prob, min_band_size, max_band_size, max_frequency_bands=2, max_time_bands=2):
        super(SpecMix, self).__init__()
        self.prob = prob
        self.min_band_size = min_band_size
        self.max_band_size = max_band_size
        self.max_frequency_bands = max_frequency_bands
        self.max_time_bands = max_time_bands

    def get_band(self, x, min_band_size, max_band_size, band_type, mask):
        axis = 3 if band_type.lower() == 'freq' else 2
        band_size = random.randint(min_band_size, max_band_size)
        mask_start = random.randint(0, x.size(axis) - band_size)
        mask_end = mask_start + band_size

        if band_type.lower() == 'freq':
            mask[:, mask_start:mask_end] = 1
        else:
            mask[mask_start:mask_end, :] = 1
        return mask

    def forward(self, x):
        k = random.random()
        if k > 1 - self.prob:
            batch_size, _, frames, freq = x.size()
            batch_idx = torch.randperm(batch_size, device=x.device)
            mask = torch.zeros(frames, freq, device=x.device)

            # Generate mask for frequency bands
            num_frequency_bands = random.randint(1, self.max_frequency_bands)
            for _ in range(num_frequency_bands):
                mask = self.get_band(x, self.min_band_size, self.max_band_size, 'freq', mask)
            
            # Generate mask for time bands
            num_time_bands = random.randint(1, self.max_time_bands)
            for _ in range(num_time_bands):
                mask = self.get_band(x, self.min_band_size, self.max_band_size, 'time', mask)
            
            # Mixing ratio
            lam = torch.sum(mask) / (frames * freq)
            lam = lam.expand(batch_size)  # Expand to match batch size for broadcasting

            # Generate mixed input (x')
            x_mixed = x * (1 - mask) + x[batch_idx] * mask

            # Output a dictionary with x', rn_indices, and mixup_lambda
            output_dict = {
                'x_mixed': x_mixed,
                'rn_indices': batch_idx,
                'mixup_lambda': lam
            }
            return x_mixed, batch_idx, lam
        else:
            # If not applied, return the original input with no mixing
            output_dict = {
                'x_mixed': x,
                'rn_indices': None,
                'mixup_lambda': None
            }
            return x, None, None

# Demo code for running SpecMix with GPU support
if __name__ == "__main__":
    # Parameters
    batch_size = 4
    frames = 251
    freq = 64
    min_band_size = 10
    max_band_size = 30
    prob = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize SpecMix instance
    specmix = SpecMix(prob=prob, min_band_size=min_band_size, max_band_size=max_band_size).to(device)

    # Dummy data: Spectrogram input of shape [batch, 1, frames, freq]
    x = torch.randn(batch_size, 1, frames, freq, device=device)

    # Apply SpecMix
    output_dict = specmix(x)

    # Retrieve variables for loss calculation
    x_mixed = output_dict['x_mixed']
    rn_indices = output_dict['rn_indices']
    mixup_lambda = output_dict['mixup_lambda']

    # Print results to verify
    print("Original spectrogram shape:", x.shape)
    print("Mixed spectrogram shape:", x_mixed.shape)
    print("Random indices:", rn_indices)
    print("Mixup lambda:", mixup_lambda)
