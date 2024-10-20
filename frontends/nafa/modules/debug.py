import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableDTW(nn.Module):
    def __init__(self, seq_len, feat_dim, window_size=5):
        """
        Initialize the DifferentiableDTW module with a learnable template for alignment
        and a local alignment regularization.

        Args:
            seq_len (int): The length of the input sequence.
            feat_dim (int): The dimensionality of each feature.
            window_size (int): The size of the local window for alignment regularization.
        """
        super(DifferentiableDTW, self).__init__()

        # Learnable template for warping alignment (initialized randomly)
        self.learned_template = nn.Parameter(torch.randn(1, seq_len, 1))
        self.window_size = window_size

    def forward(self, score, feature):
        """
        Forward function that computes a warped feature representation
        using a differentiable DTW mechanism conditioned on the score tensor.

        Args:
            score (Tensor): A tensor of shape [batch, seq_len, 1] representing the conditioning score.
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim] representing the features.
        
        Returns:
            warped_feature (Tensor): Warped output feature of shape [batch, seq_len, feat_dim].
            soft_warping_path (Tensor): The soft alignment path.
            local_alignment_loss (Tensor): The local alignment loss value.
        """
        batch_size, seq_len, feat_dim = feature.size()

        # Step 1: Create a pairwise distance matrix between the score and the learnable template
        distance_matrix = self.compute_pairwise_distances(score)

        # Step 2: Apply softmax to get a differentiable alignment path (soft warping path)
        soft_warping_path = self.compute_soft_warping_path(distance_matrix)

        # Step 3: Warp the features based on the soft warping path
        warped_feature = self.apply_warping(feature, soft_warping_path)

        # Step 4: Compute local alignment loss
        local_alignment_loss = self.compute_local_alignment_loss(feature, warped_feature)

        return warped_feature, soft_warping_path, local_alignment_loss

    def compute_pairwise_distances(self, score):
        """
        Compute pairwise distances between score matrix and the learned template for alignment.
        
        Args:
            score (Tensor): A tensor of shape [batch, seq_len, 1].
        
        Returns:
            distance_matrix (Tensor): Pairwise distances for alignment, shape [batch, seq_len, seq_len].
        """
        # Compute pairwise distances between score and the learned template
        distance_matrix = torch.cdist(score, self.learned_template)  # [batch, seq_len, seq_len]
        return distance_matrix

    def compute_soft_warping_path(self, distance_matrix):
        """
        Compute a soft alignment matrix (soft warping path) using softmax over the distances.

        Args:
            distance_matrix (Tensor): A tensor of shape [batch, seq_len, seq_len].
        
        Returns:
            soft_warping_path (Tensor): A soft alignment path, shape [batch, seq_len, seq_len].
        """
        # Apply softmax to get soft warping path, normalized across sequence length dimension
        # Improve numerical stability by subtracting the max value along the sequence dimension
        soft_warping_path = F.softmax(-distance_matrix - distance_matrix.max(dim=-1, keepdim=True)[0], dim=-1)
        return soft_warping_path

    def apply_warping(self, feature, soft_warping_path):
        """
        Apply the warping to the feature using the soft warping path.

        Args:
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim].
            soft_warping_path (Tensor): A tensor of shape [batch, seq_len, seq_len].
        
        Returns:
            warped_feature (Tensor): Warped feature of shape [batch, seq_len, feat_dim].
        """
        # Use einsum to apply warping across the sequence length dimension
        warped_feature = torch.einsum('bij,bjf->bif', soft_warping_path, feature)  # [batch, seq_len, feat_dim]
        return warped_feature

    def compute_local_alignment_loss(self, feature, warped_feature):
        """
        Compute the local alignment loss by measuring the similarity between
        local windows of the input feature and the warped feature.

        Args:
            feature (Tensor): Original input feature of shape [batch, seq_len, feat_dim].
            warped_feature (Tensor): Warped feature of shape [batch, seq_len, feat_dim].

        Returns:
            local_loss (Tensor): Scalar tensor representing the local alignment loss.
        """
        batch_size, seq_len, feat_dim = feature.size()
        window_size = self.window_size

        # Compute the number of windows
        num_windows = seq_len - window_size + 1

        # Extract local windows and compute cosine similarity
        input_windows = feature.unfold(dimension=1, size=window_size, step=1)  # [batch, num_windows, window_size, feat_dim]
        warped_windows = warped_feature.unfold(dimension=1, size=window_size, step=1)  # [batch, num_windows, window_size, feat_dim]

        # Flatten the windows
        input_windows_flat = input_windows.contiguous().view(batch_size, num_windows, -1)  # [batch, num_windows, window_size * feat_dim]
        warped_windows_flat = warped_windows.contiguous().view(batch_size, num_windows, -1)  # [batch, num_windows, window_size * feat_dim]

        # Normalize the vectors
        input_norm = F.normalize(input_windows_flat, dim=2)
        warped_norm = F.normalize(warped_windows_flat, dim=2)

        # Compute cosine similarity for each window
        cos_sim = (input_norm * warped_norm).sum(dim=2)  # [batch, num_windows]

        # Compute local loss (1 - cosine similarity), averaged over windows and batch
        local_loss = (1 - cos_sim).mean()

        return local_loss


if __name__ == "__main__":
    # Define input sizes
    seq_len = 251
    feat_dim = 64
    batch_size = 200
    window_size = 5

    # Initialize the DifferentiableDTW module
    dtw_module = DifferentiableDTW(seq_len, feat_dim, window_size=window_size)

    # Create random input data
    score = torch.randn(batch_size, seq_len, 1)  # learned score
    feature = torch.randn(batch_size, seq_len, feat_dim)  # static log-mel spectrogram

    # Forward pass
    warped_feature, soft_warping_path, local_alignment_loss = dtw_module(score, feature)

    print("Warped Feature Shape:", warped_feature.shape)
    print("Soft Warping Path Shape:", soft_warping_path.shape)
    print("Local Alignment Loss:", local_alignment_loss.item())

