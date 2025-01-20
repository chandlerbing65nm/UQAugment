# File: train.py

import os
import ssl
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.config import parse_args
from methods.model_selection import get_model
from transforms.audio_transforms import get_transforms
from losses.loss_selection import get_loss_function
from specaug.specaug_selection import process_outputs
from loggers.wandb_init import initialize_wandb
from loggers.metrics_logging import log_metrics
from loggers.ckpt_saving import save_checkpoint
from datasets.dataset_selection import get_dataloaders

from datasets.affia3k import get_dataloader as affia3k_loader
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

from datasets.noise import get_dataloader as noise_loader

from tqdm import tqdm
from pprint import pprint

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Set WandB directories
os.environ['WANDB_CONFIG_DIR'] = '/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/wandb'
os.environ['WANDB_DIR'] = '/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/wandb'
os.environ['WANDB_CACHE_DIR'] = '/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/wandb'

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parse_args()

    # Initialize WandB
    initialize_wandb(args)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print arguments
    print("Arguments:")
    pprint(vars(args))

    # Set random seed
    set_seed(args.seed)

    # Initialize model
    model = get_model(args)

    # Get transforms
    transform = get_transforms(args)

    # Initialize data loaders using the new dataset_selection module
    train_dataset, train_loader, val_dataset, val_loader = get_dataloaders(args, transform)

    # Noise data loaders
    _, noisetrain_loader = noise_loader(split='train', batch_size=args.batch_size//10, sample_rate=args.sample_rate, shuffle=False, seed=args.seed, drop_last=True, transform=None, args=args)
    _, noiseval_loader = noise_loader(split='val', batch_size=args.batch_size//10, sample_rate=args.sample_rate, shuffle=False, seed=args.seed, drop_last=True, transform=None, args=args)

    # Loss and optimizer
    criterion = get_loss_function(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.patience, factor=args.factor)

    if args.lr_warmup:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: min(1.0, epoch / args.warmup_epochs)
        )

    # Initialize learning rate tracking
    lr_reduce_count = 0
    prev_lr = args.learning_rate

    best_val_map = -np.inf
    best_val_acc = -np.inf

    # Training loop
    for epoch in range(args.max_epoch):
        model.train()
        running_loss = 0.0
        all_train_targets = []
        all_train_outputs = []

        noise_iter = iter(noisetrain_loader)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epoch} - Training"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

            # Add noise if args.ablation and args.noise are True
            if args.ablation and args.noise:
                try:
                    noise_batch = next(noise_iter)['waveform'].to(device)
                except StopIteration:
                    # Restart noise loader iterator when exhausted
                    noise_iter = iter(noisetrain_loader)
                    noise_batch = next(noise_iter)['waveform'].to(device)
                
                # Repeat noise to match inputs' batch size
                repeat_factor = (inputs.size(0) + noise_batch.size(0) - 1) // noise_batch.size(0)
                noise_batch = noise_batch.repeat(repeat_factor, 1)[:inputs.size(0)]
                
                # Add noise with alpha=0.1
                alpha = 0.1
                inputs = inputs + alpha * noise_batch

            # import ipdb; ipdb.set_trace() 
            # print(inputs.shape)

            loss, outputs = process_outputs(model, args, inputs, targets, criterion)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)

            # Store predictions and targets
            all_train_targets.append(targets.argmax(dim=-1).cpu().numpy())
            all_train_outputs.append(outputs.detach().cpu().numpy())

        # Compute training metrics
        all_train_targets = np.concatenate(all_train_targets, axis=0)
        all_train_outputs = np.concatenate(all_train_outputs, axis=0)

        # import ipdb; ipdb.set_trace() 
        # unique, counts = np.unique(all_train_targets, return_counts=True)
        # print("Training class distribution:", dict(zip(unique, counts)))

        train_acc = accuracy_score(all_train_targets, all_train_outputs.argmax(axis=-1))
        epoch_loss = running_loss / len(train_loader.dataset)

        print(f'Epoch [{epoch+1}/{args.max_epoch}], '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Accuracy: {train_acc:.4f}, ')

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_targets = []
        all_val_outputs = []

        noise_iter = iter(noiseval_loader)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.max_epoch} - Validation"):
                inputs = batch['waveform'].to(device)
                targets = batch['target'].to(device)

                # Add noise if args.ablation and args.noise are True
                if args.ablation and args.noise:
                    try:
                        noise_batch = next(noise_iter)['waveform'].to(device)
                    except StopIteration:
                        # Restart noise loader iterator when exhausted
                        noise_iter = iter(noiseval_loader)
                        noise_batch = next(noise_iter)['waveform'].to(device)
                    
                    # Repeat noise to match inputs' batch size
                    repeat_factor = (inputs.size(0) + noise_batch.size(0) - 1) // noise_batch.size(0)
                    noise_batch = noise_batch.repeat(repeat_factor, 1)[:inputs.size(0)]
                    
                    # Add noise with alpha=0.1
                    alpha = 0.1
                    inputs = inputs + alpha * noise_batch

                if any(keyword in args.model_name for keyword in ('panns', 'ast')):
                    outputs = model(inputs)['clipwise_output']
                else:
                    outputs = model(inputs)

                outputs = torch.softmax(outputs, dim=-1)  # for classification metrics

                # Store predictions and targets
                all_val_targets.append(targets.argmax(dim=-1).cpu().numpy())
                all_val_outputs.append(outputs.detach().cpu().numpy())

        # Compute validation metrics
        all_val_targets = np.concatenate(all_val_targets, axis=0)
        # One-hot encode the targets
        all_val_targets_one_hot = label_binarize(all_val_targets, classes=np.arange(args.num_classes))

        all_val_outputs = np.concatenate(all_val_outputs, axis=0)

        val_acc = accuracy_score(all_val_targets, all_val_outputs.argmax(axis=-1))
        val_map = average_precision_score(all_val_targets_one_hot, all_val_outputs, average='weighted')
        val_f1 = f1_score(all_val_targets, all_val_outputs.argmax(axis=-1), average='weighted')

        print(f'Epoch [{epoch+1}/{args.max_epoch}], '
              f'Val Accuracy: {val_acc:.4f}, '
              f'Val mAP: {val_map:.4f}, '
              f'Val F1: {val_f1:.4f}')

        # Update learning rate
        if args.lr_warmup and epoch < args.warmup_epochs:
            # Apply the warm-up scheduler if within warm-up period
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Warmup Scheduler applied at epoch {epoch+1}, learning rate is {current_lr:.6f}")
        else:
            # Use only the main scheduler after warm-up period
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Main Scheduler applied at epoch {epoch+1}, learning rate is {current_lr:.6f}")

            # Check if learning rate has been reduced
            if current_lr < prev_lr:
                lr_reduce_count += 1
                print(f"Learning rate reduced to {current_lr:.6f} at epoch {epoch+1}")
                if lr_reduce_count >= 3:
                    print("Learning rate has been reduced 3x, stopping training.")
                    break  # Exit the training loop

            prev_lr = current_lr  # Update previous learning rate

        # Log metrics to WandB
        log_metrics({
            "Train Loss": epoch_loss,
            "Train Accuracy": train_acc,
            "Validation Accuracy": val_acc,
            "Validation mAP": val_map,
            "Validation F1": val_f1,
            "Learning Rate": current_lr  # Optionally log the learning rate
        })

        # Save checkpoints
        best_val_map, best_val_acc = save_checkpoint(
            model, args, best_val_map, best_val_acc, val_map, val_acc
        )

    # Optionally, save the final model
    # final_ckpt_path = f'checkpoints/{args.frontend}/{args.loss}/{args.model_name}_{args.freq_band.lower()}_band_model_final.pth'
    # torch.save(model.state_dict(), final_ckpt_path)
    # print(f"Final model saved to {final_ckpt_path}")

if __name__ == '__main__':
    main()