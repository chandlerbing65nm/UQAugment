import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableDTW(nn.Module):
    def __init__(self, seq_len, feat_dim, dropout_rate=0.5):
        """
        Initialize the DifferentiableDTW module with a learnable template for alignment.

        Args:
            seq_len (int): The length of the input sequence.
            feat_dim (int): The dimensionality of each feature.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(DifferentiableDTW, self).__init__()

        # Learnable template for warping alignment (initialized randomly)
        self.learned_template = nn.Parameter(torch.randn(1, seq_len, 1))

        # Batch normalization layer for the warped features
        self.batch_norm = nn.BatchNorm1d(feat_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Store sequence length for alignment regularization
        self.seq_len = seq_len

    def forward(self, score, feature):
        """
        Forward function that computes a warped feature representation
        using a differentiable DTW mechanism conditioned on the score tensor.

        Args:
            score (Tensor): A tensor of shape [batch, seq_len, 1] representing the conditioning score.
            feature (Tensor): A tensor of shape [batch, seq_len, feat_dim] representing the features.
        
        Returns:
            warped_feature (Tensor): Warped output feature of shape [batch, seq_len, feat_dim].
            soft_warping_path (Tensor): Soft alignment path of shape [batch, seq_len, seq_len].
            alignment_reg (Tensor): Alignment regularization term (scalar).
        """
        batch_size, seq_len, feat_dim = feature.size()

        # Step 1: Create a pairwise distance matrix between the score and the learnable template
        distance_matrix = self.compute_pairwise_distances(score)

        # Step 2: Apply softmax to get a differentiable alignment path (soft warping path)
        soft_warping_path = self.compute_soft_warping_path(distance_matrix)

        # Step 3: Warp the features based on the soft warping path
        warped_feature = self.apply_warping(feature, soft_warping_path)

        # Step 4: Apply batch normalization and dropout to the warped features
        warped_feature = self.normalize_and_dropout(warped_feature)

        # Step 5: Compute the alignment regularization term
        alignment_reg = self.compute_alignment_regularization(soft_warping_path)

        return warped_feature, soft_warping_path, alignment_reg

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
        soft_warping_path = F.softmax(
            -distance_matrix - distance_matrix.max(dim=-1, keepdim=True)[0],
            dim=-1
        )  # [batch, seq_len, seq_len]
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

    def normalize_and_dropout(self, warped_feature):
        """
        Apply batch normalization and dropout to the warped features.

        Args:
            warped_feature (Tensor): Warped feature of shape [batch, seq_len, feat_dim].
        
        Returns:
            normalized_feature (Tensor): Normalized and dropout-applied feature of shape [batch, seq_len, feat_dim].
        """
        batch_size, seq_len, feat_dim = warped_feature.size()

        # Reshape for batch normalization: [batch * seq_len, feat_dim]
        warped_feature_flat = warped_feature.view(-1, feat_dim)

        # Apply batch normalization
        normalized_feature = self.batch_norm(warped_feature_flat)

        # Apply dropout
        normalized_feature = self.dropout(normalized_feature)

        # Reshape back to [batch, seq_len, feat_dim]
        normalized_feature = normalized_feature.view(batch_size, seq_len, feat_dim)

        return normalized_feature

    def compute_alignment_regularization(self, soft_warping_path):
        """
        Compute the alignment regularization term to penalize deviations from the identity alignment.

        Args:
            soft_warping_path (Tensor): Soft alignment path, shape [batch, seq_len, seq_len].
        
        Returns:
            alignment_reg (Tensor): Scalar tensor representing the alignment regularization term.
        """
        batch_size, seq_len, _ = soft_warping_path.size()

        # Create identity matrix of shape [seq_len, seq_len]
        identity = torch.eye(seq_len, seq_len, device=soft_warping_path.device)

        # Expand identity matrix to [batch, seq_len, seq_len]
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute mean squared error between soft warping path and identity
        alignment_reg = F.mse_loss(soft_warping_path, identity)

        return alignment_reg


if __name__ == "__main__":
    # Define input sizes
    seq_len = 251
    feat_dim = 64
    batch_size = 200

    # Initialize the module
    dtw_module = DifferentiableDTW(seq_len=seq_len, feat_dim=feat_dim, dropout_rate=0.5)

    # Generate random inputs
    score = torch.randn(batch_size, seq_len, 1)      # learned score
    feature = torch.randn(batch_size, seq_len, feat_dim)  # static log-mel spectrogram

    # Forward pass
    warped_feature, soft_warping_path, alignment_reg = dtw_module(score, feature)

    # Print shapes to verify
    print("Warped Feature Shape:", warped_feature.shape)           # Expected: [batch_size, seq_len, feat_dim]
    print("Soft Warping Path Shape:", soft_warping_path.shape)     # Expected: [batch_size, seq_len, seq_len]
    print("Alignment Regularization:", alignment_reg.item())       # Should be a scalar
