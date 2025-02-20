import audiomentations

def get_transforms(args):
    transform = None

    if args.audiomentations:
        augmentations = []

        # If using Gaussian noise augmentation:
        if 'gaussian_noise' in args.audiomentations:
            # Parse "min_amplitude, max_amplitude, p" from the string
            min_amplitude, max_amplitude, p = [float(x) for x in args.gaussian_noise_params.split(',')]
            augmentations.append(
                audiomentations.AddGaussianNoise(min_amplitude=min_amplitude,
                                                 max_amplitude=max_amplitude,
                                                 p=p)
            )

        # If using Pitch shift augmentation:
        elif 'pitch_shift' in args.audiomentations:
            # Parse "min_semitones, max_semitones, p" from the string
            min_semitones, max_semitones, p = [float(x) for x in args.pitch_shift_params.split(',')]
            augmentations.append(
                audiomentations.PitchShift(min_semitones=min_semitones,
                                           max_semitones=max_semitones,
                                           p=p)
            )

        # If using Time stretch augmentation:
        elif 'time_stretch' in args.audiomentations:
            # Parse "min_rate, max_rate, p" from the string
            min_rate, max_rate, p = [float(x) for x in args.time_stretch_params.split(',')]
            augmentations.append(
                audiomentations.TimeStretch(min_rate=min_rate,
                                            max_rate=max_rate,
                                            p=p)
            )

        if augmentations:
            transform = audiomentations.Compose(augmentations)

    return transform
