import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Define low and high cutoff frequencies (randomly initialized)
        low_freq_mel = 80  # Initial lower bound for cutoff frequencies in Hz
        high_freq_mel = self.sample_rate // 2  # Nyquist frequency
        
        # Initialize filterbanks (only two parameters: low and high cutoff freq)
        self.low_hz_ = nn.Parameter(torch.linspace(low_freq_mel, high_freq_mel - 1000, out_channels))
        self.band_hz_ = nn.Parameter(torch.linspace(1000, 5000, out_channels))

        # Hamming window to apply to sinc function (size = kernel_size)
        n_lin = torch.linspace(0, (kernel_size - 1), steps=kernel_size)
        self.window = 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / kernel_size)

        # Time axis for the filter (centered, size = kernel_size)
        n = (self.kernel_size) // 2  # Now n is 1024 for kernel_size = 2048
        self.n_ = torch.arange(-n, n).float() / self.sample_rate  # Full kernel_size range
        self.n_ = 2 * np.pi * self.n_

    def forward(self, waveforms):
        # Ensure all tensors are on the same device as the input waveforms
        device = waveforms.device
        self.n_ = self.n_.to(device)
        self.window = self.window.to(device)
        low = torch.abs(self.low_hz_).to(device)
        band = torch.abs(self.band_hz_).to(device)

        high = torch.clamp(low + band, self.sample_rate * 0.1, self.sample_rate / 2)
        filters = []
        for i in range(self.out_channels):
            # Generate band-pass filter using sinc function
            low_pass1 = 2 * low[i] * torch.sinc(low[i] * self.n_)
            low_pass2 = 2 * high[i] * torch.sinc(high[i] * self.n_)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass * self.window  # Apply the window to the band-pass filter
            filters.append(band_pass)
        
        # Stack filters and prepare for 1D convolution
        filters = torch.stack(filters).view(self.out_channels, 1, -1).to(device)
        
        return F.conv1d(waveforms, filters, stride=1024, padding=self.kernel_size // 2)


class SincNet(nn.Module):
    def __init__(self, out_channels, sample_rate=128000, kernel_size=2048, window_size=2048, hop_size=1024):
        super(SincNet, self).__init__()
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.hop_size = hop_size
        self.window_size = window_size
        
        # Sinc-conv layer with 64 filters (equivalent to 64 mel bands)
        self.sinc_conv = SincConv(out_channels=out_channels, kernel_size=self.kernel_size, sample_rate=self.sample_rate)

    def forward(self, x):
        # x: [batch, 1, audio_length], raw waveform input
        x = self.sinc_conv(x)  # Apply sinc-conv
        
        # Output shape after sinc-conv: [batch, 64, frames]
        # Apply any additional processing (e.g., normalization, activation) here if needed.
        return x