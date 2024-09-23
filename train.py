
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from methods.panns.template import *
from methods.hugging_face.models import *
from losses.loss import *

from datasets.uffia import get_dataloader as uffia_loader
from datasets.affia3k import get_dataloader as affia3k_loader


from tqdm import tqdm
from pprint import pprint
import wandb  # Import wandb
import os
import random
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['WANDB_CONFIG_DIR'] = './wandb'
os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CACHE_DIR'] = './wandb'

def parse_args():
    parser = argparse.ArgumentParser(description='Train Audio Model with Learning Rate Scheduler')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for audio feature extraction')
    parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
    parser.add_argument('--mel_bins', type=int, default=64, help='Number of mel bins for audio feature extraction')
    parser.add_argument('--fmin', type=int, default=50, help='Minimum frequency for mel bins')
    parser.add_argument('--fmax', type=int, default=None, help='Maximum frequency for mel bins')
    parser.add_argument('--patience', type=int, default=500, help='Patience for learning rate scheduler')
    parser.add_argument('--factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced')
    parser.add_argument('--data_path', type=str, default='/mnt/users/chadolor/work/Datasets/affia3k/')
    parser.add_argument('--loss', type=str, default='ce', help='Use other Loss instead of CrossEntropyLoss')
    parser.add_argument('--model_name', type=str, default='cnn10')
    parser.add_argument('--audiomentations', action='store_true', help='Apply audiomentations')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Apply label smoothing with the specified smoothing factor')
    parser.add_argument('--filter_chance', type=float, default=0.13)
    parser.add_argument('--wandb_mode', type=str, default='offline')
    parser.add_argument('--wandb_project', type=str, default='affia-3k')
    parser.add_argument('--lr_warmup', action='store_true', help='Apply learning rate warm-up')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of epochs for learning rate warm-up')
    parser.add_argument('--freq_band', type=str, default='none')
    parser.add_argument('--frontend', type=str, default='logmel')

    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project, 
        config=vars(args), 
        # name=f'{args.model_name}_{args.filter_chance}_{args.min_freq}', 
        name=f'{args.model_name}',
        mode=args.wandb_mode,
        )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pretty print arguments
    print("Arguments:")
    pprint(vars(args))

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Initialize model
    if args.model_name == 'panns_cnn6':
        # Instantiate the model
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
        model.load_from_pretrain("./weights/Cnn6_mAP=0.343.pth")
    # Initialize model
    elif args.model_name == 'panns_resnet22':
        # Instantiate the model
        model = PANNS_RESNET22(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
            )
        model.load_from_pretrain("./weights/ResNet22_mAP=0.430.pth") 
    # Initialize model
    elif args.model_name == 'panns_mobilenetv1':
        # Instantiate the model
        model = PANNS_MOBILENETV1(
            sample_rate=args.sample_rate, 
            window_size=args.window_size, 
            hop_size=args.hop_size, 
            mel_bins=args.mel_bins, 
            fmin=args.fmin, 
            fmax=args.fmax, 
            num_classes=args.num_classes
            )
        model.load_from_pretrain("./weights/MobileNetV1_mAP=0.389.pth") 
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
        model.load_from_pretrain("./weights/Wavegram_Cnn14_mAP=0.389.pth") 
    elif args.model_name == 'cnn8rnn':
        # Instantiate the model
        model = CNN8RNN(
            num_classes=args.num_classes
            )
    else: 
        raise ValueError(f"Unknown model name: {args.model_name}")

    model.to(device)

    transform = None
    if args.audiomentations:
        import audiomentations

        if 'panns' in args.model_name:
            if args.freq_band == 'low':
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
            elif args.freq_band == 'mid':
                transform = audiomentations.Compose([
                    audiomentations.BandPassFilter(
                        min_center_freq=12800.0,  # Start at 12.8 kHz
                        max_center_freq=44800.0,  # End at 44.8 kHz
                        min_bandwidth_fraction=1.0,  # Allow flexibility around 1.11
                        max_bandwidth_fraction=1.2,  # Keep bandwidth fraction close to 1.11
                        min_rolloff=12,  # Gentle rolloff
                        max_rolloff=24,  # Sharper rolloff if needed
                        zero_phase=False,  # No need for phase preservation
                        p=0.10  # Probability of applying the filter
                        ),
                ])
            elif args.freq_band == 'high':
                transform = audiomentations.Compose([
                    audiomentations.HighPassFilter(
                        min_cutoff_freq=44800.0,  # Start filtering at 44.8 kHz
                        max_cutoff_freq=64000.0,  # Maximum cutoff at 64 kHz
                        min_rolloff=12,  # Minimum rolloff
                        max_rolloff=24,  # Maximum rolloff for sharper attenuation
                        zero_phase=False,  # Set to True if phase preservation is required
                        p=0.10  # Probability of applying the filter
                        ),
                ])
            else:
                transform = None
        else:
            transform = None
            # transform = audiomentations.Compose([
            #     audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            #     # audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            #     # audiomentations.Shift(p=0.5),
            # ])

    # Initialize data loaders
    train_dataset, train_loader = affia3k_loader(split='train', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=True, seed=args.seed, class_num=args.num_classes, drop_last=True, data_path=args.data_path, transform=transform)
    val_dataset, val_loader = affia3k_loader(split='test', batch_size=args.batch_size, sample_rate=args.sample_rate, shuffle=False, seed=args.seed, class_num=args.num_classes, drop_last=True, data_path=args.data_path, transform=None)

    # Loss and optimizer
    if args.loss == 'focal':
        criterion = FocalLoss()
    elif args.loss == 'softboot':
        criterion = SoftBootstrappingLoss(beta=0.8)
    elif args.loss == 'hardboot':
        criterion = HardBootstrappingLoss(beta=0.8)
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=args.factor)

    if args.lr_warmup:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: min(1.0, epoch / args.warmup_epochs)
        )

    best_val_loss = np.inf
    best_val_acc = -np.inf

    # Training loop
    for epoch in range(args.max_epoch):
        model.train()
        running_loss = 0.0
        all_train_targets = []
        all_train_outputs = []

        for batch in tqdm(train_loader, "Training:"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            if 'panns' in args.model_name:
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
                loss = samples_loss.mean()
            else:
                loss = criterion(outputs, targets.argmax(dim=-1))


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)

            # Store detached predictions and true targets for accuracy and mAP calculation
            all_train_targets.append(targets.argmax(dim=-1).cpu().numpy())
            all_train_outputs.append(outputs.detach().cpu().numpy())  # detach the outputs here

        # Convert stored lists to numpy arrays
        all_train_targets = np.concatenate(all_train_targets, axis=0)
        all_train_outputs = np.concatenate(all_train_outputs, axis=0)

        # Convert targets to one-hot encoding
        num_classes = all_train_outputs.shape[1]
        all_train_targets_onehot = label_binarize(all_train_targets, classes=np.arange(num_classes))

        # Calculate training accuracy and mAP
        train_acc = accuracy_score(all_train_targets, all_train_outputs.argmax(axis=-1))
        train_map = average_precision_score(all_train_targets_onehot, all_train_outputs, average='macro')

        epoch_loss = running_loss / len(train_loader.dataset)

        # Print training statistics
        print(f'Epoch [{epoch+1}/{args.max_epoch}], '
            f'Total Loss: {epoch_loss:.4f}, '
            f'Accuracy: {train_acc:.4f}, '
            f'mAP: {train_map:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_targets = []
        all_val_outputs = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['waveform'].to(device)
                targets = batch['target'].to(device)

                if 'panns' in args.model_name:
                    outputs = model(inputs)['clipwise_output']
                else:
                    outputs = model(inputs)

                loss = criterion(outputs, targets.argmax(dim=-1))
                val_loss += loss.item() * inputs.size(0)

                # Store detached predictions and true targets for accuracy and mAP calculation
                all_val_targets.append(targets.argmax(dim=-1).cpu().numpy())
                all_val_outputs.append(outputs.detach().cpu().numpy())  # detach the outputs here

        val_loss /= len(val_loader.dataset)

        # Convert stored lists to numpy arrays
        all_val_targets = np.concatenate(all_val_targets, axis=0)
        all_val_outputs = np.concatenate(all_val_outputs, axis=0)

        # Convert targets to one-hot encoding
        num_classes = all_train_outputs.shape[1]
        all_val_targets_onehot = label_binarize(all_val_targets, classes=np.arange(num_classes))

        # Calculate validation accuracy and mAP
        val_acc = accuracy_score(all_val_targets, all_val_outputs.argmax(axis=-1))
        val_map = average_precision_score(all_val_targets_onehot, all_val_outputs, average='macro')

        # Print validation statistics
        print(f'Epoch [{epoch+1}/{args.max_epoch}], '
            f'Val Loss: {val_loss:.4f}, '
            f'Val Accuracy: {val_acc:.4f}, '
            f'Val mAP: {val_map:.4f}')

        # Update learning rate and other metrics
        current_lr = optimizer.param_groups[0]['lr']
        if args.lr_warmup and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"Learning rate changed from {current_lr} to {new_lr}")

        # Log metrics to wandb
        wandb.log({
            "Train Loss": epoch_loss,
            "Train Accuracy": train_acc,
            "Train mAP": train_map,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Validation mAP": val_map
        })

        ckpt_dir = f'checkpoints/{args.frontend}/{args.loss}'
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Best validation loss {best_val_loss:.4f}")
            torch.save(model.state_dict(), f'{ckpt_dir}/{args.model_name}_{args.freq_band.lower()}_band_model_best_loss.pth')

        # Save the best model based on validation loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Best validation accuracy {best_val_acc:.4f}")
            torch.save(model.state_dict(), f'{ckpt_dir}/{args.model_name}_{args.freq_band.lower()}_band_model_best_acc.pth')

    # Save the final model
    torch.save(model.state_dict(), f'{ckpt_dir}/{args.model_name}_{args.freq_band.lower()}_band_model_final.pth')



if __name__ == '__main__':
    main()
