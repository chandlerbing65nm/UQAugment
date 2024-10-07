import torch
import torch.nn as nn
import torch.optim as optim
from frontends.sincnet.frontend import SincNet

# Example usage
batch_size =200
input_audio = torch.randn(batch_size, 1, 256000)  # Simulated input batch

# Initialize SincNet frontend
sincnet_frontend = SincNet(out_channels=64, sample_rate=128000, kernel_size=2048, window_size=2048, hop_size=1024)

# Forward pass
output = sincnet_frontend(input_audio)
print(output.shape)  # Expected shape: [batch, 64, 251]

