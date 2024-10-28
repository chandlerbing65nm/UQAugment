import torch
import torch.nn as nn
import torch.nn.functional as F
# dummy
class SVEMixup(nn.Module):
    def __init__(self, seq_len, feat_dim, temperature=0.2, frame_reduction_ratio=None, frame_augmentation_ratio=1.0, device='cuda'):
        super(SVEMixup, self).__init__()
        self.seq_len = seq_len
        self.frame_reduction_ratio = frame_reduction_ratio
        if frame_reduction_ratio is not None:
            assert 0 < frame_reduction_ratio <= 1, "frame_reduction_ratio must be between 0 and 1"
            self.reduced_len = max(1, int(seq_len * (1 - frame_reduction_ratio)))
        else:
            self.reduced_len = self.seq_len
        assert 0 <= frame_augmentation_ratio <= 1, "frame_augmentation_ratio must be between 0 and 1"
        self.num_augmented_frames = max(1, int(self.reduced_len * frame_augmentation_ratio))
        self.noise_template = torch.randn(1, self.reduced_len, seq_len).to(device=device)
        self.temperature = temperature

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()
        noise_template = self.noise_template.expand(batch_size, -1, -1)
        augmenting_path = self.compute_augmenting_path(noise_template)
        
        # Apply variance equalization
        augmenting_path = self.apply_variance_equalization(augmenting_path)
        
        if self.num_augmented_frames < self.reduced_len:
            augmenting_path = self.randomly_apply_augmentation(augmenting_path)
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        return augmented_feature

    def compute_augmenting_path(self, noise_template):
        mu, sigma = 0, 1
        gaussian_noise = torch.normal(mu, sigma, size=noise_template.size(), device=noise_template.device)
        return F.softmax(gaussian_noise / self.temperature, dim=-1)

    def apply_variance_equalization(self, augmenting_path):
        batch_variance = augmenting_path.var(dim=-1, unbiased=False)  # Variance per batch sample
        variance_target = batch_variance.mean()  # Mean variance for the batch
        
        scaling_factors = torch.sqrt(variance_target / (batch_variance + 1e-8)).unsqueeze(-1)  # Scaling for variance
        return augmenting_path * scaling_factors  # Scale augmenting path

    def randomly_apply_augmentation(self, augmenting_path):
        batch_size, reduced_len, seq_len = augmenting_path.size()
        mask = torch.zeros(batch_size, reduced_len, 1, device=augmenting_path.device)
        start_index = torch.randint(0, reduced_len - self.num_augmented_frames + 1, (1,)).item()
        mask[:, start_index:start_index + self.num_augmented_frames, :] = 1
        augmenting_path = augmenting_path * mask
        return augmenting_path

    def apply_augmenting(self, feature, augmenting_path):
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)
        return augmented_feature

if __name__ == "__main__":
    # Define input sizes
    frames = 251 
    feat_dim = 64 
    batch_size = 200
    
    # Generate random input
    input_spectrogram = torch.randn(batch_size, frames, feat_dim).cuda()
    
    # Initialize model and forward pass
    model = SVEMixup(seq_len=frames, feat_dim=feat_dim, temperature=0.2, frame_reduction_ratio=None, frame_augmentation_ratio=1.0, device='cuda').cuda()
    output = model(input_spectrogram)
    
    print("Output shape:", output.shape)
