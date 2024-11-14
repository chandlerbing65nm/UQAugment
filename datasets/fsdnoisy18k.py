import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import math

def load_audio(path, sr=None, offset=0.0, duration=None):
    y, orig_sr = librosa.load(path, sr=sr, offset=offset, duration=duration)
    return y

class FSDNoisy18kDataset(Dataset):
    def __init__(self, data_path, split='train', sample_rate=None, target_duration=None, transform=None):
        """
        Initializes the dataset by reading the metadata and preparing the file paths and labels.

        Args:
            data_path (str): Path to the dataset directory.
            split (str): 'train' or 'test' to specify the dataset split.
            sample_rate (int, optional): Desired sample rate for audio files.
            target_duration (float, optional): Target duration in seconds for each audio segment.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.split = split
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.transform = transform

        # Read the CSV file
        csv_file = os.path.join(data_path, 'meta', f'{split}.csv')
        data_df = pd.read_csv(csv_file)

        if split == 'train':
            # Filter rows where noisy_small == 1
            data_df = data_df[data_df['noisy_small'] == 1]

        # Get list of unique labels
        labels = data_df['label'].unique()
        labels.sort()  # Sort labels for consistent ordering
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        # Prepare index mapping without loading audio files
        self.index_mapping = []
        self.data_info = []  # List to store file paths and labels
        for idx, row in data_df.iterrows():
            fname = row['fname']
            label = row['label']
            label_idx = self.label_to_index[label]
            file_path = os.path.join(data_path, split, fname)

            # Get duration without loading the audio file
            duration = librosa.get_duration(path=file_path)
            if self.sample_rate is not None:
                sr = self.sample_rate
            else:
                # If sample rate is not specified, get the original sample rate
                with open(file_path, 'rb') as f:
                    sr = librosa.get_samplerate(file_path)
            # Calculate number of segments
            if self.target_duration:
                num_segments = math.ceil(duration / self.target_duration)
            else:
                num_segments = 1  # Use the whole audio if no target duration
            # Store data info
            self.data_info.append({'file_path': file_path, 'label_idx': label_idx, 'num_segments': num_segments, 'duration': duration})

            # Update index mapping
            for segment_idx in range(num_segments):
                self.index_mapping.append({'file_idx': len(self.data_info) - 1, 'segment_idx': segment_idx})

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, index):
        mapping = self.index_mapping[index]
        file_info = self.data_info[mapping['file_idx']]
        segment_idx = mapping['segment_idx']

        file_path = file_info['file_path']
        label_idx = file_info['label_idx']
        duration = file_info['duration']

        if self.target_duration:
            offset = segment_idx * self.target_duration
            # Adjust duration for the last segment
            if offset + self.target_duration > duration:
                duration_segment = duration - offset
            else:
                duration_segment = self.target_duration
            # Load audio segment
            waveform = load_audio(file_path, sr=self.sample_rate, offset=offset, duration=duration_segment)
            # Pad if necessary
            expected_length = int(self.sample_rate * self.target_duration)
            waveform = np.pad(waveform, (0, max(0, expected_length - len(waveform))), 'constant')
        else:
            # Load full audio
            waveform = load_audio(file_path, sr=self.sample_rate)

        if self.transform:
            waveform = self.transform(waveform)

        # Convert label to one-hot encoding
        target = np.eye(len(self.label_to_index))[label_idx]

        return {'audio_name': file_path, 'waveform': waveform, 'target': target}

def collate_fn(batch):
    """
    Custom collate function to handle variable-length waveforms by padding them to the same length.
    """
    audio_names = [data['audio_name'] for data in batch]
    waveforms = [data['waveform'] for data in batch]
    targets = [data['target'] for data in batch]

    # Stack waveforms into a 2D tensor
    waveforms = torch.FloatTensor(np.stack(waveforms))
    targets = torch.FloatTensor(np.array(targets))

    return {'audio_name': audio_names, 'waveform': waveforms, 'target': targets}

def get_dataloader(data_path, split, batch_size, sample_rate, target_duration, shuffle=False, num_workers=4, transform=None):
    """
    Returns a DataLoader for the dataset.

    Args:
        data_path (str): Path to the dataset directory.
        split (str): 'train' or 'test' to specify the dataset split.
        batch_size (int): Number of samples per batch.
        sample_rate (int): Desired sample rate for audio files.
        target_duration (float): Target duration in seconds for each audio segment.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    dataset = FSDNoisy18kDataset(
        data_path=data_path,
        split=split,
        sample_rate=sample_rate,
        target_duration=target_duration,
        transform=transform
    )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, collate_fn=collate_fn)
    return dataset, dataloader

if __name__ == '__main__':
    data_path = '/scratch/project_465001389/chandler_scratch/Datasets/fsdnoisy18k'
    sample_rate = 16000  # Desired sample rate
    target_duration = 2.0  # Target duration in seconds
    batch_size = 2
    split = 'train'  # Can be 'train' or 'test'

    dataset, dataloader = get_dataloader(
        data_path=data_path,
        split=split,
        batch_size=batch_size,
        sample_rate=sample_rate,
        target_duration=target_duration,
        shuffle=True,
        transform=None
    )

    for i, batch in enumerate(dataloader):
        audio_names = batch['audio_name']
        waveforms = batch['waveform']
        targets = batch['target']
        print(f"Batch {i}")
        print(f"Audio names: {audio_names}")
        print(f"Waveforms shape: {waveforms.shape}")
        print(f"Targets shape: {targets.shape}")
        # Break after a few batches for demonstration
        if i >= 4:
            break