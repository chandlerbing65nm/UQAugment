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