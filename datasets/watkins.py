from dataclasses import dataclass
import librosa
import os
import glob
import numpy as np
import torch
from scipy.signal import resample
from itertools import chain
from torch.utils.data import Dataset, DataLoader

# Define whale classes and labels
whale_classes = {
    'Fin_FinbackWhale': 0,
    'HumpbackWhale': 1,
    'BlueWhale': 2,
    'MinkeWhale': 3,
    'BowheadWhale': 4,
}

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

def get_num_segments(path, target_duration):
    duration = librosa.get_duration(path=path)
    n_segments = int(np.ceil(duration / target_duration))
    return n_segments

def data_generator(seed, test_sample_per_class, data_path='./watkins', target_duration=2):
    random_state = np.random.RandomState(seed)
    audio_files = get_audio_files(data_path=data_path)
    
    # Organize and shuffle audio segments by class
    samples_by_class = {label: [] for label in whale_classes.values()}
    for file, label in audio_files:
        n_segments = get_num_segments(file, target_duration)
        for segment_index in range(n_segments):
            samples_by_class[label].append((file, label, segment_index))
    
    # Shuffle samples per class
    for label in samples_by_class:
        random_state.shuffle(samples_by_class[label])
    
    train_samples = []
    test_samples = []
    for label in samples_by_class:
        samples = samples_by_class[label]
        test_samples_per_class = samples[:test_sample_per_class]
        train_samples_per_class = samples[test_sample_per_class:]
        test_samples.extend(test_samples_per_class)
        train_samples.extend(train_samples_per_class)
    
    # Shuffle the train samples
    random_state.shuffle(train_samples)
    
    return train_samples, test_samples

def load_audio_segment(path, target_duration, sr, segment_index):
    y, orig_sr = librosa.load(path, sr=sr, offset=segment_index*target_duration, duration=target_duration)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    # Pad if needed
    target_length = int(target_duration * sr)
    if len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), 'constant')
    else:
        y = y[:target_length]
    return y

class WhaleAudioDataset(Dataset):
    def __init__(self, sample_rate, seed, split='train', data_path='./watkins', transform=None, target_duration=2, test_sample_per_class=100):
        self.seed = seed
        self.split = split
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        train_samples, test_samples = data_generator(self.seed, test_sample_per_class=test_sample_per_class, data_path=self.data_path, target_duration=self.target_duration)
        self.data_samples = train_samples if split == 'train' else test_samples

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, index):
        wav_name, target, segment_index = self.data_samples[index]
        wav = load_audio_segment(wav_name, target_duration=self.target_duration, sr=self.sample_rate, segment_index=segment_index)
        
        if self.transform is not None:
            wav = self.transform(samples=wav, sample_rate=self.sample_rate)
        
        # One-hot encode the target
        target_one_hot = np.eye(len(whale_classes))[target]
        data_dict = {'audio_name': wav_name, 'segment_index': segment_index, 'waveform': wav, 'target': target_one_hot}
        return data_dict

def collate_fn(batch):
    wav_name = [data['audio_name'] for data in batch]
    segment_indices = [data['segment_index'] for data in batch]
    wav = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    wav = torch.FloatTensor(np.array(wav))
    target = torch.FloatTensor(np.array(target))

    return {'audio_name': wav_name, 'segment_index': segment_indices, 'waveform': wav, 'target': target}

def get_dataloader(split, batch_size, sample_rate, seed, shuffle=False, drop_last=False, num_workers=4, data_path='./watkins', transform=None, target_duration=0.5, test_sample_per_class=200):
    dataset = WhaleAudioDataset(split=split, sample_rate=sample_rate, seed=seed, data_path=data_path, transform=transform, target_duration=target_duration, test_sample_per_class=test_sample_per_class)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn)
    return dataset, dataloader

# Testing section
if __name__ == '__main__':
    # Initialize dataset and dataloader for training data
    dataset, dataloader = get_dataloader(
        split='train',
        batch_size=1,  # Set batch_size=5 for testing
        sample_rate=128000,
        shuffle=True,
        seed=42,
        drop_last=True,
        data_path='/scratch/project_465001389/chandler_scratch/Datasets/watkins',
        target_duration=0.5,
    )

    print("Testing if samples are split and padded:\n")
    for i, batch in enumerate(dataloader):
        audio_names = batch['audio_name']
        segment_indices = batch['segment_index']
        waveforms = batch['waveform']
        targets = batch['target']

        print(f"Batch {i+1}:")
        for j, (file, segment_idx) in enumerate(zip(audio_names, segment_indices)):
            print(f"  - File: {file}, Segment Index: {segment_idx}")
            print(f"  - Waveform shape: {waveforms[j].shape}")
            print(f"  - Target: {targets[j]}")
        print("")

        # Break after a few batches to limit output
        if i >= 4:
            break

    # Initialize dataset for test data
    test_dataset, test_dataloader = get_dataloader(
        split='test',
        batch_size=1,
        sample_rate=8000,
        shuffle=False,
        seed=42,
        drop_last=False,
        data_path='/scratch/project_465001389/chandler_scratch/Datasets/watkins',
        target_duration=0.5,
    )

    # Check the number of samples per class in test split
    from collections import Counter
    class_counts = Counter()
    for data in test_dataset:
        target_one_hot = data['target']
        target_label = np.argmax(target_one_hot)
        class_counts[target_label] += 1

    print("Number of samples per class in test split:")
    for label in sorted(class_counts.keys()):
        class_name = [k for k, v in whale_classes.items() if v == label][0]
        print(f"Class {label} ({class_name}): {class_counts[label]}")
