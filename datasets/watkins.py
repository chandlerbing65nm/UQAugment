from dataclasses import dataclass
import librosa
import os
import glob
import numpy as np
import torch
from scipy.signal import resample
from itertools import chain
from torch.utils.data import Dataset, DataLoader

# Parameters
target_duration = 2  # Target duration in seconds for padding/resampling

# Define whale classes and labels
whale_classes = {
    'KillerWhale': 0, 
    'SpermWhale': 1, 
    'Long_FinnedPilotWhale': 2, 
    'PantropicalSpottedDolphin': 3, 
    'CommonDolphin': 4,
    'StripedDolphin': 5,
    'Short_Finned(Pacific)PilotWhale': 6,
    'HumpbackWhale': 7,
    'Fin_FinbackWhale': 8,
    'White_sidedDolphin': 9,
}

def load_audio_with_padding(path, target_duration, sr):
    y, orig_sr = librosa.load(path, sr=sr)
    current_duration = librosa.get_duration(y=y, sr=orig_sr)
    
    # Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    
    # Pad or trim audio to match target duration
    target_length = int(target_duration * sr)
    if len(y) > target_length:
        y = y[:target_length]  # Trim
    else:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), 'constant')  # Pad with zeros
    return y

def get_audio_files(data_path='./watkins'):
    audio_files = []
    for folder, label in whale_classes.items():
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} not found.")
            continue
        files = glob.glob(os.path.join(folder_path, '*.wav'))
        for file in files:
            audio_files.append((file, label))
    return audio_files

def data_generator(seed, test_sample_per_class, data_path='./watkins'):
    random_state = np.random.RandomState(seed)
    audio_files = get_audio_files(data_path=data_path)
    
    # Organize and shuffle audio files by class
    audio_by_class = {label: [] for label in whale_classes.values()}
    for file, label in audio_files:
        audio_by_class[label].append(file)
    
    for label, files in audio_by_class.items():
        random_state.shuffle(files)

    train_dict = []
    test_dict = []
    for label, files in audio_by_class.items():
        train_files = files[test_sample_per_class:]
        test_files = files[:test_sample_per_class]
        
        train_dict.extend([(f, label) for f in train_files])
        test_dict.extend([(f, label) for f in test_files])

    random_state.shuffle(train_dict)
    
    return train_dict, test_dict

class WhaleAudioDataset(Dataset):
    def __init__(self, sample_rate, seed, split='train', data_path='./watkins', transform=None):
        self.seed = seed
        self.split = split
        self.data_path = data_path
        self.transform = transform
        train_dict, test_dict = data_generator(self.seed, test_sample_per_class=100, data_path=self.data_path)
        self.data_dict = train_dict if split == 'train' else test_dict
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        wav_name, target = self.data_dict[index]
        wav = load_audio_with_padding(wav_name, target_duration, sr=self.sample_rate)
        
        if self.transform is not None:
            wav = self.transform(samples=wav, sample_rate=self.sample_rate)
        
        # One-hot encode the target
        target_one_hot = np.eye(len(whale_classes))[target]
        data_dict = {'audio_name': wav_name, 'waveform': wav, 'target': target_one_hot}
        return data_dict

def collate_fn(batch):
    wav_name = [data['audio_name'] for data in batch]
    wav = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    wav = torch.FloatTensor(np.array(wav))
    target = torch.FloatTensor(np.array(target))

    return {'audio_name': wav_name, 'waveform': wav, 'target': target}

def get_dataloader(split, batch_size, sample_rate, seed, shuffle=False, drop_last=False, num_workers=4, data_path='./watkins', transform=None):
    dataset = WhaleAudioDataset(split=split, sample_rate=sample_rate, seed=seed, data_path=data_path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn)
    return dataset, dataloader

# Testing section
if __name__ == '__main__':
    # Initialize dataset and dataloader
    dataset, dataloader = get_dataloader(
        split='train', 
        batch_size=2,  # Set a small batch size for testing
        sample_rate=8000, 
        shuffle=True, 
        seed=42, 
        drop_last=True, 
        data_path='./watkins'
    )

    print("Testing which files are combined into each sample:\n")
    for i, batch in enumerate(dataloader):
        audio_names = batch['audio_name']
        waveforms = batch['waveform']
        targets = batch['target']
       
        print(f"Sample {i+1}:")
        for j, file in enumerate(audio_names):
            print(f"  - File: {file}")
            print(f"  - Waveform shape: {waveforms[j].shape}")
            print(f"  - Target: {targets[j]}")
        print("")

        # Break after a few samples to limit output
        if i >= 4:
            break
