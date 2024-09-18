import torch
import numpy as np
import argparse

from datasets.affia3k import get_dataloader as affia3k_loader
from sklearn.metrics import accuracy_score, average_precision_score

from methods.panns.template import *
from methods.hugging_face.models import *

import os
import torch.nn.functional as F
from pprint import pprint

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
    parser.add_argument('--frontend', type=str, default='logmel')
    
    return parser.parse_args()

def main():
    args = parse_test_args()

    # Pretty print arguments
    print("Test Arguments:")
    pprint(vars(args))

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

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
            num_classes=args.num_classes,
            frontend=args.frontend,
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

    # Load the saved model checkpoint
    # model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)

    import audiomentations
    transform = audiomentations.Compose([
        audiomentations.LowPassFilter(
            min_cutoff_freq=0.0,
            max_cutoff_freq=12800.0, 
            min_rolloff=12,
            max_rolloff=24,
            zero_phase=False, 
            p=0.10
            ),
    ])

    # Initialize the test data loader
    test_dataset, test_loader = affia3k_loader(
        split='test', 
        batch_size=args.batch_size, 
        sample_rate=args.sample_rate, 
        shuffle=False, 
        seed=args.seed, 
        class_num=args.num_classes, 
        drop_last=False, 
        data_path=args.data_path, 
        transform=None
    )

    # Put the model in evaluation mode
    model.eval()

    test_loss = 0.0
    all_test_targets = []
    all_test_outputs = []

    # Define loss function (this could change based on your training configuration)
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            if 'panns' in args.model_name:
                outputs = model(inputs)['clipwise_output']
            else:
                outputs = model(inputs)

            loss = criterion(outputs, targets.argmax(dim=-1))
            test_loss += loss.item() * inputs.size(0)

            # Store detached predictions and true targets for accuracy and mAP calculation
            all_test_targets.append(targets.argmax(dim=-1).cpu().numpy())
            all_test_outputs.append(outputs.detach().cpu().numpy())

    # Convert stored lists to numpy arrays
    all_test_targets = np.concatenate(all_test_targets, axis=0)
    all_test_outputs = np.concatenate(all_test_outputs, axis=0)

    # Calculate test accuracy and mAP
    test_acc = accuracy_score(all_test_targets, all_test_outputs.argmax(axis=-1))
    test_map = average_precision_score(all_test_targets, all_test_outputs, average='macro')

    # Calculate average loss
    test_loss /= len(test_loader.dataset)

    # Print test results
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test mAP: {test_map:.4f}')

if __name__ == '__main__':
    main()
