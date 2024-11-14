import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa

def load_audio(path, sr=None):
    y, orig_sr = librosa.load(path, sr=None)
    if sr is not None and sr != orig_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y

def split_or_pad_audio(waveform, target_length):
    """
    Splits or pads the audio waveform to match the target length.

    Args:
        waveform (np.array): Audio waveform array.
        target_length (int): Target length in samples.

    Returns:
        List of waveforms each of length target_length.
    """
    audio_segments = []
    num_samples = len(waveform)

    if num_samples < target_length:
        # Pad if the waveform is shorter than target length
        padded_waveform = np.pad(waveform, (0, target_length - num_samples), 'constant')
        audio_segments.append(padded_waveform)
    else:
        # Split the waveform if it is longer than target length
        for start in range(0, num_samples, target_length):
            end = start + target_length
            segment = waveform[start:end]
            if len(segment) < target_length:
                # Pad the last segment if it is shorter than target length
                segment = np.pad(segment, (0, target_length - len(segment)), 'constant')
            audio_segments.append(segment)

    return audio_segments

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

        # Calculate target length in samples
        self.target_length = int(sample_rate * target_duration) if sample_rate and target_duration else None

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

        # Create list of file paths and corresponding labels
        self.data_list = []
        for idx, row in data_df.iterrows():
            fname = row['fname']
            label = row['label']
            label_idx = self.label_to_index[label]
            file_path = os.path.join(data_path, split, fname)
            self.data_list.append((file_path, label_idx))

    def __len__(self):
        # Since audio can be split into multiple segments, dynamically calculate length
        return sum(len(self._get_audio_segments(file_path)) for file_path, _ in self.data_list)

    def _get_audio_segments(self, file_path):
        """
        Loads the audio file and splits/pads it based on the target length.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            List of audio segments, each of length self.target_length.
        """
        waveform = load_audio(file_path, sr=self.sample_rate)
        if self.target_length:
            return split_or_pad_audio(waveform, self.target_length)
        else:
            return [waveform]

    def __getitem__(self, index):
        # Locate the file and segment index for the current index
        cumulative_count = 0
        for file_path, label_idx in self.data_list:
            segments = self._get_audio_segments(file_path)
            if cumulative_count + len(segments) > index:
                segment_idx = index - cumulative_count
                segment = segments[segment_idx]
                break
            cumulative_count += len(segments)

        if self.transform:
            segment = self.transform(segment)

        # Convert label to one-hot encoding
        target = np.eye(len(self.label_to_index))[label_idx]

        return {'audio_name': file_path, 'waveform': segment, 'target': target}

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

    dataset, dataloader = get_dataloader(data_path=data_path, split=split,
                                         batch_size=batch_size, sample_rate=sample_rate,
                                         target_duration=target_duration, shuffle=True, transform=None)
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
