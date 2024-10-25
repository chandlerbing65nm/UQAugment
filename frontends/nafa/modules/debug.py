import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9

class FrameAugment(nn.Module):
    def __init__(
        self, 
        seq_len, 
        feat_dim, 
        temperature=0.2, 
        alpha=0.5,    # Alignment weight
        device='cuda'
    ):
        super(FrameAugment, self).__init__()
        
        # Initialize attributes
        self.seq_len = seq_len
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        # Template noise matrix
        self.noise_template = torch.randn(1, seq_len, seq_len, device=device)

    def forward(self, feature):
        batch_size, seq_len, feat_dim = feature.size()
        
        # Expand noise template to batch size
        mixing_matrix = self.noise_template.expand(batch_size, -1, -1)
        
        # Compute alignment-aware augmenting path
        augmenting_path = self.compute_augmenting_path(mixing_matrix, feature)
        
        # Apply augmentation
        augmented_feature = self.apply_augmenting(feature, augmenting_path)
        return augmented_feature

    def compute_augmenting_path(self, mixing_matrix, feature):
        # Generate Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(mixing_matrix) + EPS) + EPS)
        
        # Normalize Gumbel noise with softmax
        normalized_gumbel = F.softmax(gumbel_noise / self.temperature, dim=-1)
        
        # Compute alignment matrix using cosine similarity
        feature_flat = feature.reshape(feature.size(0), feature.size(1), -1)
        alignment_matrix = torch.einsum('bij,bkj->bik', feature_flat, feature_flat)  # Cosine similarity
        alignment_matrix = F.softmax(alignment_matrix, dim=-1)
        
        # Combine normalized Gumbel noise with alignment matrix
        augmenting_path = self.alpha * normalized_gumbel + (1 - self.alpha) * alignment_matrix
        return augmenting_path

    def apply_augmenting(self, feature, augmenting_path):
        # Apply the augmenting path with matrix multiplication
        augmented_feature = torch.einsum('bij,bjf->bif', augmenting_path, feature)
        return augmented_feature

# Testing the implementation
if __name__ == "__main__":
    # Define input sizes
    seq_len = 251
    feat_dim = 64
    batch_size = 200

    # Initialize model and test input
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FrameAugment(seq_len=seq_len, feat_dim=feat_dim, temperature=0.2, alpha=0.5, device=device).to(device)
    test_input = torch.randn(batch_size, seq_len, feat_dim, device=device)

    # Run forward pass
    output = model(test_input)
    print("Output shape:", output.shape)
