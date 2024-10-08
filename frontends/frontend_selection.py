# File: frontend/frontend.py

import torch
import torch.nn.functional as F

def process_outputs(model, args, inputs, targets, criterion):
    
    if any(keyword in args.model_name for keyword in ('panns', 'ast')):
        output_dict = model(inputs)
        outputs = output_dict['clipwise_output']
    else:
        outputs = model(inputs)

    if args.frontend == 'mixup':
        mixup_lambda = output_dict['mixup_lambda']
        rn_indices = output_dict['rn_indices']
        bs = inputs.size(0)
        labels = targets.argmax(dim=-1)
        samples_loss = (F.cross_entropy(outputs, labels, reduction="none") * mixup_lambda.reshape(bs) +
                        F.cross_entropy(outputs, labels[rn_indices], reduction="none") * (1. - mixup_lambda.reshape(bs)))
        return samples_loss.mean(), outputs
    elif args.frontend == 'diffres':
        diffres_loss = output_dict['diffres_loss']
        return diffres_loss + criterion(outputs, targets.argmax(dim=-1)), outputs
    else:
        return criterion(outputs, targets.argmax(dim=-1)), outputs
