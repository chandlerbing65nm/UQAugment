# File: config/config.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Audio Model with Learning Rate Scheduler')

    # General Parameters
    general = parser.add_argument_group('General Parameters')
    general.add_argument('--seed', type=int, default=20, help='Random seed')
    general.add_argument('--data_path', type=str, default='/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/datasets/affia3k/', help='Path to the dataset')
    general.add_argument('--csv_path', type=str, default=None, help='Path to the csv metadata')
    general.add_argument('--dataset', type=str, default='affia3k', help='Dataset to use for training and validation')
    general.add_argument('--num_classes', type=int, default=4, help='Number of classes')

    # Training Parameters
    training = parser.add_argument_group('Training Parameters')
    training.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    training.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs')
    training.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    training.add_argument('--loss', type=str, default='ce', help='Loss function to use (ce, focal, softboot, hardboot)')
    training.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')

    # Model Parameters
    model = parser.add_argument_group('Model Parameters')
    model.add_argument('--model_name', type=str, default='cnn10', help='Name of the model to use')
    model.add_argument('--frontend', type=str, default='logmel', help='Frontend type (logmel, mixup, diffres)')

    # Data Processing Parameters
    data_processing = parser.add_argument_group('Data Processing Parameters')
    data_processing.add_argument('--target_duration', type=float, default=2, help='duration of audio in seconds')
    data_processing.add_argument('--sample_rate', type=int, default=128000, help='Sample rate for audio')
    data_processing.add_argument('--window_size', type=int, default=2048, help='Window size for audio feature extraction')
    data_processing.add_argument('--hop_size', type=int, default=1024, help='Hop size for audio feature extraction')
    data_processing.add_argument('--mel_bins', type=int, default=64, help='Number of mel bins for audio feature extraction')
    data_processing.add_argument('--fmin', type=int, default=50, help='Minimum frequency for mel bins')
    data_processing.add_argument('--fmax', type=int, default=None, help='Maximum frequency for mel bins')

    # Scheduler Parameters
    scheduler = parser.add_argument_group('Scheduler Parameters')
    scheduler.add_argument('--patience', type=int, default=500, help='Patience for learning rate scheduler')
    scheduler.add_argument('--factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced')
    scheduler.add_argument('--lr_warmup', action='store_true', help='Apply learning rate warm-up')
    scheduler.add_argument('--warmup_epochs', type=int, default=5, help='Number of warm-up epochs')

    # Augmentation Parameters
    augmentation = parser.add_argument_group('Augmentation Parameters')
    augmentation.add_argument('--audiomentations', action='store_true', help='Apply audiomentations')
    augmentation.add_argument('--spec_aug',  type=str, default='specaugment', help='Name of the spectrogram augmentation')

    # Logging Parameters
    logging_group = parser.add_argument_group('Logging Parameters')
    logging_group.add_argument('--wandb_mode', type=str, default='offline', help='WandB mode (online/offline)')
    logging_group.add_argument('--wandb_project', type=str, default='affia3k', help='WandB project name')

    # Testing Parameters
    testing_group = parser.add_argument_group('Testing Parameters')
    testing_group.add_argument('--checkpoint', type=str, help='checkpoint path for loaded trained model weights')

    return parser.parse_args()
