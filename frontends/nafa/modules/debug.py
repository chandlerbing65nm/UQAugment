import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameAugment(nn.Module):
    def __init__(self, seq_len, feat_dim, device='cuda'):
        super(FrameAugment, self).__init__()
        self.seq_len = seq_len
        self.device = device

        # Initialize learnable parameters for mean and standard deviation
        self.mu = nn.Parameter(torch.zeros(1))
        self.sigma_param = nn.Parameter(torch.ones(1))  # Use softplus to ensure positivity

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()
        augmenting_path = self.compute_augmenting_path(batch_size, seq_len)
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        return augmented_feature

    def compute_augmenting_path(self, batch_size, seq_len):
        # Ensure sigma is positive using softplus
        sigma = F.softplus(self.sigma_param)

        # Sample Gaussian noise with learnable mean and standard deviation
        gaussian_noise = torch.normal(
            self.mu.expand(batch_size, seq_len, seq_len),
            sigma.expand(batch_size, seq_len, seq_len)
        ).to(self.device)

        # Apply softmax normalization along the frame dimension
        return F.softmax(gaussian_noise, dim=-1)

    def apply_augmenting(self, feature, augmenting_path):
        # Perform matrix multiplication to mix frames
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)
        return augmented_feature

if __name__ == "__main__":
    # Define input sizes
    frames = 251 
    feat_dim = 64 
    batch_size = 200

    input_spectrogram = torch.randn(batch_size, frames, feat_dim).cuda()

    # Initialize FrameAugment
    frame_augment = FrameAugment(seq_len=frames, feat_dim=feat_dim, device='cuda').cuda()

    # Apply augmentation
    augmented_spectrogram = frame_augment(input_spectrogram)

    print("Original Spectrogram Shape:", input_spectrogram.shape)
    print("Augmented Spectrogram Shape:", augmented_spectrogram.shape)
