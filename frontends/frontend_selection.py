# File: frontend/frontend.py

import torch
import torch.nn.functional as F

def process_outputs(model, args, inputs, targets, criterion):
    
    # Determine output structure based on model type
    if any(keyword in args.model_name for keyword in ('panns', 'ast')):
        output_dict = model(inputs)
        outputs = output_dict.get('clipwise_output', None) or output_dict
    else:
        outputs = model(inputs)
    
    # Initialize total loss with criterion loss as base
    loss = criterion(outputs, targets.argmax(dim=-1))
    
    # Apply spec augmentation if specified
    if hasattr(args, 'spec_aug') and args.spec_aug == 'mixup':
        mixup_lambda = output_dict.get('mixup_lambda')
        rn_indices = output_dict.get('rn_indices')
        
        if mixup_lambda is not None and rn_indices is not None:
            bs = inputs.size(0)
            labels = targets.argmax(dim=-1)
            samples_loss = (F.cross_entropy(outputs, labels, reduction="none") * mixup_lambda.reshape(bs) +
                            F.cross_entropy(outputs, labels[rn_indices], reduction="none") * (1. - mixup_lambda.reshape(bs)))
            loss = samples_loss.mean() + loss  # Add mixup loss to base loss

    # Apply additional frontend-specific loss if defined
    if hasattr(args, 'frontend'):
        if args.frontend == 'diffres':
            diffres_loss = output_dict.get('diffres_loss')
            if diffres_loss is not None:
                loss += diffres_loss

    return loss, outputs

