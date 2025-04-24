import audiomentations

def get_transforms(args):
    transform = None

    if args.audiomentations:
        augmentations = []

        # # If using Gaussian noise augmentation:
        if 'gaussian_noise' in args.audiomentations:
            # Parse "min_amplitude, max_amplitude, p" from the string
            min_amplitude, max_amplitude, p = [float(x) for x in args.gaussian_noise_params.split(',')]
            augmentations.append(
                audiomentations.AddGaussianNoise(min_amplitude=min_amplitude,
                                                 max_amplitude=max_amplitude,
                                                 p=p)
            )

        # Pitch Shift
        if 'pitch_shift' in args.audiomentations:
            min_semitones, max_semitones, p = [float(x) for x in args.pitch_shift_params.split(',')]
            augmentations.append(
                audiomentations.PitchShift(min_semitones=min_semitones,
                                           max_semitones=max_semitones,
                                           p=p)
            )

        # Time Stretch
        if 'time_stretch' in args.audiomentations:
            min_rate, max_rate, p = [float(x) for x in args.time_stretch_params.split(',')]
            augmentations.append(
                audiomentations.TimeStretch(min_rate=min_rate,
                                            max_rate=max_rate,
                                            p=p)
            )

        # Time Mask
        if 'time_mask' in args.audiomentations:
            min_band_part, max_band_part, p = [float(x) for x in args.time_mask_params.split(',')]
            augmentations.append(
                audiomentations.TimeMask(min_band_part=min_band_part,
                                         max_band_part=max_band_part,
                                         p=p)
            )

        # Band Stop Filter
        if 'band_stop_filter' in args.audiomentations:
            min_center_freq, max_center_freq, p = [float(x) for x in args.band_stop_filter_params.split(',')]
            augmentations.append(
                audiomentations.BandStopFilter(min_center_freq=min_center_freq,
                                            max_center_freq=max_center_freq,
                                            p=p)
            )

        if augmentations:
            transform = audiomentations.Compose(augmentations)

    return transform
