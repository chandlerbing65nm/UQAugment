import torch
import torch.nn as nn
import torch.nn.functional as F
from frontends.dmel.dmel import differentiable_spectrogram


class DMel(nn.Module):
    def __init__(self, init_lambd, n_fft, win_length, hop_length, norm=False):
        super(DMel, self).__init__()
        self.lambd = nn.Parameter(torch.tensor(init_lambd, dtype=torch.float))
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.norm = norm

    def forward(self, x):
        # x is expected to be of shape (batch_size, signal_length)
        x = x - x.mean(dim=1, keepdim=True)
        batch_size = x.shape[0]
        spectrograms = []
        for idx in range(batch_size):
            xi = x[idx]
            s = differentiable_spectrogram(
                xi, self.lambd,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                norm=self.norm
            )
            spectrograms.append(s)
        spectrograms = torch.stack(spectrograms, dim=0)
        return spectrograms

def main():
    # Initialize parameters
    init_lambd = 5.0  # Initial value for lambd
    n_fft = 2048
    win_length = n_fft
    hop_length = 1024  # Calculated to get 1025 time steps

    # Create the spectrogram module
    spectrogram_module = DMel(init_lambd, n_fft, win_length, hop_length)

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spectrogram_module.to(device)

    # Generate a batch of signals
    batch_size = 200
    signal_length = 256000
    x = torch.randn(batch_size, signal_length).to(device)

    # Compute the spectrograms
    spectrograms = spectrogram_module(x)

    # Print the shape of the spectrograms
    print(f"Spectrograms shape: {spectrograms.shape}")

if __name__ == "__main__":
    main()