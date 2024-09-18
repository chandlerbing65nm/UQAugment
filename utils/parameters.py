import torch
import torch.nn as nn

from methods.panns.template import *
from methods.hugging_face.models import *

import argparse

def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Audio Model')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for testing')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for audio feature extraction')
    parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
    parser.add_argument('--mel_bins', type=int, default=64, help='Number of mel bins for audio feature extraction')
    parser.add_argument('--fmin', type=int, default=50, help='Minimum frequency for mel bins')
    parser.add_argument('--fmax', type=int, default=None, help='Maximum frequency for mel bins')
    parser.add_argument('--model_name', type=str, default='cnn10', help='Model name to test')
    parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/affia3k/')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the saved model checkpoint')

    return parser.parse_args()

def count_parameters(model):
    # Separate trainable and non-trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    def format_params(num):
        if num >= 1e9:
            return f"{num / 1e9:.2f} Billion"
        elif num >= 1e6:
            return f"{num / 1e6:.2f} Million"
        else:
            return f"{num:,}"

    print(f"Trainable parameters: {format_params(trainable_params)}")
    print(f"Non-trainable parameters: {format_params(non_trainable_params)}")
    return trainable_params, non_trainable_params

# Example Usage
if __name__ == "__main__":
    args = parse_test_args()

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    if args.model_name == 'panns_cnn6':
        model = PANNS_CNN6(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
        )
        model.load_finetuned_weights(args.checkpoint_path)
    elif args.model_name == 'panns_resnet22':
        model = PANNS_RESNET22(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
        )
        model.load_finetuned_weights(args.checkpoint_path)
    elif args.model_name == 'panns_mobilenetv1':
        model = PANNS_MOBILENETV1(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
        )
        model.load_finetuned_weights(args.checkpoint_path)
    elif args.model_name == 'panns_wavegram_cnn14':
        # Instantiate the model
        model = PANNS_WAVEGRAM_CNN14(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
            )
        model.load_finetuned_weights(args.checkpoint_path)
    elif args.model_name == 'cnn8rnn':
        # Instantiate the model
        model = CNN8RNN(
            num_classes=args.num_classes
            )
        model.load_finetuned_weights(args.checkpoint_path)
    else: 
        raise ValueError(f"Unknown model name: {args.model_name}")

    # Initialize the model
    model.to(device)

    # Count parameters
    count_parameters(model)
