import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameAugmentSSNI(nn.Module):
    def __init__(self, seq_len, feat_dim, device='cuda'):
        super(FrameAugmentSSNI, self).__init__()
        self.seq_len = seq_len
        self.device = device

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()

        # Compute stability-driven noise
        stability_scores = self.compute_stability(feature)
        augmenting_path = self.compute_augmenting_path(batch_size, seq_len, stability_scores)

        # Apply stability-aware noise augmentation
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        
        return augmented_feature

    def compute_stability(self, feature):
        # Compute variance along the frame dimension as stability measure
        variance_scores = torch.var(feature, dim=2, keepdim=True)  # Shape: [batch, seq_len, 1]
        
        # Normalize variance to create stability scores (inverse of variance)
        stability_scores = 1 / (variance_scores + 1e-6)  # Add epsilon to prevent division by zero
        stability_scores = stability_scores / stability_scores.max()  # Normalize to [0, 1]
        
        return stability_scores.squeeze(-1)  # Shape: [batch, seq_len]

    def compute_augmenting_path(self, batch_size, seq_len, stability_scores):
        # Generate Gaussian noise and apply softmax normalization along frame dimension
        gaussian_noise = torch.normal(0, 1, size=(batch_size, seq_len, seq_len), device=self.device)
        
        # Scale noise based on stability scores (higher stability gets more noise)
        noise_scaled = gaussian_noise * stability_scores.unsqueeze(2)
        
        augmenting_path = F.softmax(noise_scaled, dim=-1)
        
        return augmenting_path

    def apply_augmenting(self, feature, augmenting_path):
        # Matrix multiplication of feature with stability-aware noise
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)
        
        return augmented_feature

if __name__ == "__main__":
    # Define input sizes
    frames = 251
    feat_dim = 64
    batch_size = 200
    
    input_spectrogram = torch.randn(batch_size, frames, feat_dim).cuda()
    model = FrameAugmentSSNI(seq_len=frames, feat_dim=feat_dim).cuda()
    
    output = model(input_spectrogram)
    print("Output shape:", output.shape)
