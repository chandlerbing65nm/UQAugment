# File: losses/loss_functions.py

from torch import nn
from losses.loss import FocalLoss, SoftBootstrappingLoss, HardBootstrappingLoss

def get_loss_function(args):
    if args.loss == 'focal':
        criterion = FocalLoss()
    elif args.loss == 'softboot':
        criterion = SoftBootstrappingLoss(beta=0.8)
    elif args.loss == 'hardboot':
        criterion = HardBootstrappingLoss(beta=0.8)
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss}")
    return criterion
