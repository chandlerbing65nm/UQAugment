from dataclasses import dataclass
import librosa
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch
from scipy.signal import resample

def load_audio(path, sr=None):
    y, original_sr = librosa.load(path, sr=None)  # Load with original sample rate
    if sr is not None and sr != original_sr:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)  # Resample directly
    return y

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, sample_rate=None, split='train', transform=None, num_segments=30):
        """
        Args:
            csv_path (str): Path to the CSV file.
            audio_dir (str): Directory where audio files are stored.
            sample_rate (int, optional): Desired sample rate. Defaults to None.
            split (str): Split to load ('train' or 'test'). Defaults to 'train'.
            transform (callable, optional): Transform to be applied to the waveform.
            num_segments (int): Number of segments to split each audio file into.
        """
        self.csv_data = pd.read_csv(csv_path)
        self.csv_data = self.csv_data[self.csv_data['split'] == split].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.num_classes = len(self.csv_data['label'].unique())
        self.num_segments = num_segments

    def __len__(self):
        return len(self.csv_data) * self.num_segments

    def __getitem__(self, index):
        # Calculate which audio file and segment to load
        audio_file_index = index // self.num_segments
        segment_index = index % self.num_segments

        row = self.csv_data.iloc[audio_file_index]
        audio_path = os.path.join(self.audio_dir, f"{row['asset_id']}.wav")
        label = row['label']

        # Load the entire audio file
        waveform = load_audio(audio_path, sr=self.sample_rate)

        # Split the waveform into segments
        total_length = len(waveform)
        segment_length = total_length // self.num_segments

        # Calculate start and end indices for the segment
        start_idx = segment_index * segment_length
        if segment_index == self.num_segments - 1:
            # Last segment may include remaining samples
            end_idx = total_length
        else:
            end_idx = start_idx + segment_length

        segment_waveform = waveform[start_idx:end_idx]

        if self.transform:
            segment_waveform = self.transform(segment_waveform)

        target = np.eye(self.num_classes)[label]

        return {
            'audio_name': f"{audio_path}_segment_{segment_index}",
            'waveform': torch.FloatTensor(segment_waveform),
            'target': torch.FloatTensor(target)
        }

def collate_fn(batch):
    audio_names = [data['audio_name'] for data in batch]
    waveforms = [data['waveform'] for data in batch]
    targets = [data['target'] for data in batch]

    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    targets = torch.stack(targets)

    return {'audio_name': audio_names, 'waveform': waveforms, 'target': targets}

def get_dataloader(csv_path, audio_dir, split, batch_size, sample_rate=None, shuffle=False, drop_last=False, num_workers=4, transform=None, num_segments=5):
    dataset = AudioDataset(
        csv_path=csv_path,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        split=split,
        transform=transform,
        num_segments=num_segments
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataset, dataloader

# Example usage
if __name__ == '__main__':
    csv_path = '/scratch/project_465001389/chandler_scratch/Datasets/ssw60/audio_ml.csv'
    audio_dir = '/scratch/project_465001389/chandler_scratch/Datasets/ssw60/audio_ml/'
    
    dataset, dataloader = get_dataloader(
        csv_path=csv_path,
        audio_dir=audio_dir,
        split='train',
        batch_size=2,
        sample_rate=16000,
        shuffle=True,
        drop_last=True,
        num_segments=40  # Ensure this matches the num_segments in the dataset
    )
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}:")
        for j, audio_name in enumerate(batch['audio_name']):
            print(f"  - {audio_name} - Target: {batch['target'][j].argmax().item()}")
        print(f"Waveform shape: {batch['waveform'].shape}\n")
        
        # Stop after a few batches for demonstration purposes
        if i >= 5:
            break
