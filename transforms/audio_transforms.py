# File: transforms/audio_transforms.py

import audiomentations

def get_transforms(args):
    transform = None

    if args.audiomentations:
        augmentations = []

        if 'gaussian_noise' in args.audiomentations:
            augmentations.append(
                audiomentations.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.5)
            )

        if 'pitch_shift' in args.audiomentations:
            augmentations.append(
                audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
            )

        if 'time_stretch' in args.audiomentations:
            augmentations.append(
                audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)
            )

        if augmentations:
            transform = audiomentations.Compose(augmentations)

    return transform

