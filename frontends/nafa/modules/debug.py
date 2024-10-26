from torch import nn
import torch
import torch.nn.functional as F

class FrameAugment(nn.Module):
    def __init__(self, seq_len, feat_dim, device='cuda'):
        super(FrameAugment, self).__init__()
        self.seq_len = seq_len
        self.device = device

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()
        augmenting_path = self.compute_augmenting_path(batch_size, seq_len)
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        return augmented_feature

    def compute_augmenting_path(self, batch_size, seq_len):
        gaussian_noise = torch.normal(0, 1, size=(batch_size, seq_len, seq_len), device=self.device)
        augmenting_path = F.softmax(gaussian_noise, dim=-1)
        return augmenting_path

    def apply_augmenting(self, feature, augmenting_path):
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)
        return augmented_feature

    def temporal_wasserstein_loss(self, original, augmented):
        batch_size, seq_len, feat_dim = original.size()
        frame_groups = seq_len // 10
        wasserstein_loss = 0
        for i in range(frame_groups):
            original_group = original[:, i*10:(i+1)*10, :].reshape(batch_size, -1)
            augmented_group = augmented[:, i*10:(i+1)*10, :].reshape(batch_size, -1)
            wasserstein_loss += F.mse_loss(torch.sort(original_group, dim=-1)[0], torch.sort(augmented_group, dim=-1)[0])
        return wasserstein_loss / frame_groups

if __name__ == "__main__":
    frames = 251 
    feat_dim = 64 
    batch_size = 200
    
    input_spectrogram = torch.randn(batch_size, frames, feat_dim).cuda()
    model = FrameAugment(seq_len=frames, feat_dim=feat_dim).cuda()
    augmented_feature = model(input_spectrogram)
    loss = model.temporal_wasserstein_loss(input_spectrogram, augmented_feature)
    print("Wasserstein Consistency Loss:", loss.item())
