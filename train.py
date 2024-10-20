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
from frontends.frontend_selection import process_outputs
from loggers.wandb_init import initialize_wandb
from loggers.metrics_logging import log_metrics
from loggers.ckpt_saving import save_checkpoint
from datasets.dataset_selection import get_dataloaders

from datasets.affia3k import get_dataloader as affia3k_loader
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize

from tqdm import tqdm
from pprint import pprint

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set WandB directories
os.environ['WANDB_CONFIG_DIR'] = '/scratch/project_465001389/chandler_scratch/Projects/UWAC/wandb'
os.environ['WANDB_DIR'] = '/scratch/project_465001389/chandler_scratch/Projects/UWAC/wandb'
os.environ['WANDB_CACHE_DIR'] = '/scratch/project_465001389/chandler_scratch/Projects/UWAC/wandb'

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

    # Loss and optimizer
    criterion = get_loss_function(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.patience, factor=args.factor)

    if args.lr_warmup:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: min(1.0, epoch / args.warmup_epochs)
        )

    best_val_map = -np.inf
    best_val_acc = -np.inf

    # Training loop
    for epoch in range(args.max_epoch):
        model.train()
        running_loss = 0.0
        all_train_targets = []
        all_train_outputs = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epoch} - Training"):
            inputs = batch['waveform'].to(device)
            targets = batch['target'].to(device)

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

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.max_epoch} - Validation"):
                inputs = batch['waveform'].to(device)
                targets = batch['target'].to(device)

                if any(keyword in args.model_name for keyword in ('panns', 'ast')):
                    outputs = model(inputs)['clipwise_output']
                else:
                    outputs = model(inputs)

                outputs = torch.softmax(outputs, dim=-1) # for classification metrics

                # Store predictions and targets
                all_val_targets.append(targets.argmax(dim=-1).cpu().numpy())
                all_val_outputs.append(outputs.detach().cpu().numpy())

        # Compute validation metrics
        all_val_targets = np.concatenate(all_val_targets, axis=0)
        # One-hot encode the targets
        all_val_targets_one_hot = label_binarize(all_val_targets, classes=np.arange(args.num_classes))

        all_val_outputs = np.concatenate(all_val_outputs, axis=0)

        val_acc = accuracy_score(all_val_targets, all_val_outputs.argmax(axis=-1))
        val_map = average_precision_score(all_val_targets_one_hot, all_val_outputs, average='macro')

        print(f'Epoch [{epoch+1}/{args.max_epoch}], '
              f'Val Accuracy: {val_acc:.4f}, '
              f'Val mAP: {val_map:.4f}')

        # Update learning rate
        if args.lr_warmup and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_acc)

        # Log metrics to WandB
        log_metrics({
            "Train Loss": epoch_loss,
            "Train Accuracy": train_acc,
            "Validation Accuracy": val_acc,
            "Validation mAP": val_map
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
