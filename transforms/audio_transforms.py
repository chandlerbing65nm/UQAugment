# File: transforms/audio_transforms.py

import audiomentations

def get_transforms(args):
    transform = None
    if args.audiomentations:
        transform = audiomentations.Compose([
            audiomentations.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.5),  # Adds Gaussian noise for a 'hiss' effect
        ])
    else:
        transform = None
    return transform
