import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import argparse
from tqdm import tqdm

from datasets.affia3k import get_dataloader as affia3k_loader
from models.AudioModel import Audio_Frontend

def parse_args():
    parser = argparse.ArgumentParser(description='Train Audio Model with Learning Rate Scheduler')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--classes_num', type=int, default=4, help='Number of classes')
    parser.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for audio feature extraction')
    parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
    parser.add_argument('--mel_bins', type=int, default=128, help='Number of mel bins for audio feature extraction')
    parser.add_argument('--fmin', type=int, default=1, help='Minimum frequency for mel bins')
    parser.add_argument('--fmax', type=int, default=128000, help='Maximum frequency for mel bins')
    parser.add_argument('--patience', type=int, default=5, help='Patience for learning rate scheduler')
    parser.add_argument('--factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced')
    parser.add_argument('--pooling', action='store_true', help='If using a pooling operation')
    parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/affia3k/')
    parser.add_argument('--weights_path', type=str, default='/mnt/users/chadolor/work/Repositories/META-FFIA/pretrained_models/pre_cnn10.pth')
    parser.add_argument('--augment', action='store_true', help='Apply augmentation in the psd')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of CrossEntropyLoss')
    parser.add_argument('--progressive_augmentation', action='store_true', help='Apply progressive augmentation during training')
    parser.add_argument('--model_name', type=str, default='cnn10')
    parser.add_argument('--audiomentations', action='store_true', help='Apply audiomentations')
    parser.add_argument('--min_freq', type=int, default=500)
    parser.add_argument('--filter_chance', type=float, default=0.13)
    parser.add_argument('--wand_mode', type=str, default='offline')
    parser.add_argument('--wand_project', type=str, default='affia-3k')
    
    return parser.parse_args()

args = parse_args()
# Assuming you have a dataloader and a model defined as per your provided code
_, dataloader = affia3k_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, drop_last=True, data_path=args.data_path, transform=None)

def apply_pca_on_dataset(dataloader, model, device, save_path='helpers/pca_visualization.png'):
    model.eval()  # Set model to evaluation mode
    all_outputs = []
    all_targets = []

    # Iterate over the entire dataset
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch['waveform'].to(device)  # [batch, wav_length]
            targets = batch['target'].to(device)   # 4 class one-hot encoded
            outputs = model(inputs)                # [batch, channel, time_steps, mel_bins]
            
            outputs = outputs.reshape(outputs.size(0), -1)
            
            all_outputs.append(outputs.cpu().numpy())  # Append model output (detach to CPU)
            all_targets.append(targets.cpu().numpy())  # Append targets (detach to CPU)

    # Concatenate all collected outputs and targets
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2D for visualization
    pca_result = pca.fit_transform(all_outputs)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=np.argmax(all_targets, axis=1), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    # plt.title('PCA of Model Outputs')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Save figure
    plt.savefig(save_path)
    print(f'PCA plot saved at {save_path}')
    plt.close()  # Close the plot to avoid memory issues

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Example usage
model = Audio_Frontend(args.sample_rate, args.window_size, args.hop_size, args.mel_bins, args.fmin, args.fmax, pooling=args.pooling).to(device)

apply_pca_on_dataset(dataloader, model, device)

