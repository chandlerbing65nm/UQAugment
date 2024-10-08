# File: transforms/audio_transforms.py

import audiomentations

def get_transforms(args):
    transform = None
    if args.audiomentations:
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
                        min_center_freq=12800.0,  
                        max_center_freq=44800.0,  
                        min_bandwidth_fraction=1.0,  
                        max_bandwidth_fraction=1.2,  
                        min_rolloff=12,  
                        max_rolloff=24,  
                        zero_phase=False,  
                        p=0.10  
                    ),
                ])
            elif args.freq_band == 'high':
                transform = audiomentations.Compose([
                    audiomentations.HighPassFilter(
                        min_cutoff_freq=44800.0,  
                        max_cutoff_freq=64000.0,  
                        min_rolloff=12,  
                        max_rolloff=24,  
                        zero_phase=False,  
                        p=0.10  
                    ),
                ])
            else:
                transform = None
        else:
            transform = None
            # You can add more augmentations here if needed
            # Example:
            # transform = audiomentations.Compose([
            #     audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            #     audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            #     audiomentations.Shift(p=0.5),
            # ])
    return transform
