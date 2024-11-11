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
    def __init__(self, csv_path, audio_dir, sample_rate=None, split='train', transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file.
            audio_dir (str): Directory where audio files are stored.
            sample_rate (int, optional): Desired sample rate. Defaults to None.
            split (str): Split to load ('train' or 'test'). Defaults to 'train'.
            transform (callable, optional): Transform to be applied to the waveform.
        """
        self.csv_data = pd.read_csv(csv_path)
        self.csv_data = self.csv_data[self.csv_data['split'] == split]
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.num_classes = len(self.csv_data['label'].unique())
        
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, index):
        row = self.csv_data.iloc[index]
        audio_path = os.path.join(self.audio_dir, f"{row['asset_id']}.wav")
        label = row['label']
        
        waveform = load_audio(audio_path, sr=self.sample_rate)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        target = np.eye(self.num_classes)[label]
        
        return {
            'audio_name': audio_path,
            'waveform': torch.FloatTensor(waveform),
            'target': torch.FloatTensor(target)
        }

def collate_fn(batch):
    audio_names = [data['audio_name'] for data in batch]
    waveforms = [data['waveform'] for data in batch]
    targets = [data['target'] for data in batch]

    waveforms = torch.stack(waveforms)
    targets = torch.stack(targets)

    return {'audio_name': audio_names, 'waveform': waveforms, 'target': targets}

def get_dataloader(csv_path, audio_dir, split, batch_size, sample_rate=None, shuffle=False, drop_last=False, num_workers=4, transform=None):
    dataset = AudioDataset(csv_path=csv_path, audio_dir=audio_dir, sample_rate=sample_rate, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn)
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
        drop_last=True
    )
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}:")
        for j, audio_name in enumerate(batch['audio_name']):
            print(f"  - {audio_name} - Target: {batch['target'][j].argmax().item()}")
        print(f"Waveform shape: {batch['waveform'].shape}\n")
        
        # Stop after a few batches for demonstration purposes
        if i >= 5:
            break
