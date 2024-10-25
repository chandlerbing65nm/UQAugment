import torch
import torch.nn as nn

class ATIBN(nn.Module):
    def __init__(self, feat_dim, seq_len):
        super(ATIBN, self).__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len

    def forward(self, x):
        # Calculate Mutual Information for temporal dimension (using pairwise cosine similarity as proxy)
        cosine_similarity = torch.cosine_similarity(x[:, :-1, :], x[:, 1:, :], dim=-1)
        mutual_information = 1 - cosine_similarity  # Proxy for MI; high when frames are similar

        # Identify high mutual information frames for noise application
        high_mi_mask = mutual_information > mutual_information.mean(dim=1, keepdim=True)
        high_mi_mask = torch.cat([high_mi_mask, torch.zeros_like(high_mi_mask[:, :1])], dim=1)  # Pad to match x

        # Generate Gaussian noise with shape [batch, seq_len, seq_len]
        noise = torch.randn(x.size(0), self.seq_len, self.seq_len).to(x.device)
        noise = noise * high_mi_mask.unsqueeze(-1)  # Apply noise only on high-MI frames

        # Apply the noise through matrix multiplication
        x_noisy = torch.bmm(noise, x)  # Shape: [batch, seq_len, feat_dim]
        return x_noisy

# Testing the ATIBN Layer
if __name__ == "__main__":
    # Define input sizes
    seq_len = 251
    feat_dim = 64
    batch_size = 200
    
    # Initialize synthetic data and ATIBN layer
    data = torch.randn(batch_size, seq_len, feat_dim)
    atibn_layer = ATIBN(feat_dim, seq_len)
    
    # Apply ATIBN to input data
    augmented_data = atibn_layer(data)
    print("Shape of augmented data:", augmented_data.shape)  # Expected output: [batch_size, seq_len, feat_dim]
