import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.nll_loss(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class SoftBootstrappingLoss(nn.Module):
    """
    Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
        as_pseudo_label (bool): Stop gradient propagation for the term ``(1 - beta) * p``.
            Can be interpreted as pseudo-label.
    """
    def __init__(self, beta=0.95, reduce=True, as_pseudo_label=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label

    def forward(self, y_pred, y):
        # y_pred is expected to be the output from log_softmax
        # Cross entropy = -t * log(p), here p is exp(y_pred)
        beta_xentropy = self.beta * F.nll_loss(y_pred, y, reduction='none')

        # Use y_pred directly as it's log(p)
        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        # Second term = - (1 - beta) * p * log(p) = - (1 - beta) * exp(y_pred_a) * y_pred
        bootstrap = - (1.0 - self.beta) * torch.sum(torch.exp(y_pred_a) * y_pred, dim=1)

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap

class HardBootstrappingLoss(nn.Module):
    """
    Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)
    where z = argmax(p)

    Args:
        beta (float): bootstrap parameter. Default, 0.8
        reduce (bool): computes mean of the loss. Default, True.
    """
    def __init__(self, beta=0.8, reduce=True):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        # y_pred is expected to be the output from log_softmax
        # Cross-entropy loss = -t * log(p)
        beta_xentropy = self.beta * F.nll_loss(y_pred, y, reduction='none')

        # z = argmax(p), where p is the softmax of log probabilities
        z = torch.exp(y_pred).argmax(dim=1)  # Getting the class indices
        z = z.view(-1, 1)  # Reshape for gather

        # Bootstrap term = - (1 - beta) * log(p) where p = exp(y_pred)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)

        # Combine the terms
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap

class InterClassContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_positives=2, num_negatives=2):
        super(InterClassContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_positives = num_positives
        self.num_negatives = num_negatives

    def forward(self, features, labels):
        """
        :param features: Tensor of shape [batch_size, 512] representing the embeddings.
        :param labels: Tensor of shape [batch_size, num_classes] representing the one-hot encoded labels.
        :return: Scalar contrastive loss.
        """
        batch_size = features.shape[0]
        
        # Normalize the feature embeddings
        features = F.normalize(features, p=2, dim=1)
        
        # Compute the cosine similarity matrix between all pairs of features
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Ensure the diagonal elements (self-similarity) are excluded
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Get the actual class labels from one-hot encoded labels
        class_labels = torch.argmax(labels, dim=1)

        # Initialize the total loss
        total_loss = 0.0
        num_pairs = 0
        
        # Iterate over the batch to calculate the loss for each class pair (multiple positives and negatives)
        for i in range(batch_size):
            # Get the current sample's class
            current_class = class_labels[i]

            # Find positive samples in the same class
            positive_indices = (class_labels == current_class).nonzero(as_tuple=True)[0]
            positive_indices = positive_indices[positive_indices != i]  # Exclude self

            # If there are not enough positive samples, skip this iteration
            if positive_indices.numel() < self.num_positives:
                continue

            # Randomly sample multiple positives from the current class
            selected_positives = positive_indices[torch.randperm(len(positive_indices))[:self.num_positives]]

            # Find a negative class (different from the current class)
            negative_classes = (class_labels != current_class).unique()

            if len(negative_classes) > 0:
                # Select one random negative class
                negative_class = negative_classes[torch.randint(len(negative_classes), (1,))]

                # Find all samples from the selected negative class
                negative_indices = (class_labels == negative_class).nonzero(as_tuple=True)[0]

                # If there are not enough negative samples, skip this iteration
                if negative_indices.numel() < self.num_negatives:
                    continue

                # Randomly sample multiple negatives from the selected negative class
                selected_negatives = negative_indices[torch.randperm(len(negative_indices))[:self.num_negatives]]

                # Compute contrastive loss for each combination of positives and negatives
                for pos_idx in selected_positives:
                    for neg_idx in selected_negatives:
                        # Contrast positive vs negative similarity
                        positive_similarity = similarity_matrix[i, pos_idx]
                        negative_similarity = similarity_matrix[i, neg_idx]
                        
                        # Compute the contrastive loss for each positive-negative pair
                        loss = -torch.log(
                            torch.exp(positive_similarity) / 
                            (torch.exp(positive_similarity) + torch.exp(negative_similarity))
                        )
                        total_loss += loss
                        num_pairs += 1

        # Average the loss over the number of valid pairs
        if num_pairs > 0:
            total_loss /= num_pairs
        
        return total_loss