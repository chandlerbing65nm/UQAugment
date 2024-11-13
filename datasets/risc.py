import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict

def load_audio(path, sr=None):
    """
    Load an audio file and resample if needed.
    """
    y, original_sr = librosa.load(path, sr=None)  # Load with original sample rate
    if sr is not None and sr != original_sr:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
    return y

class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, target_duration=3.0, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory containing 'normal' and 'shout' folders.
            sample_rate (int): Desired sample rate.
            target_duration (float): Target duration in seconds.
            split (str): 'train' or 'test'.
            transform (callable, optional): Transform to be applied to the waveform.
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.split = split
        self.transform = transform  # Include transform

        self.file_list = []
        self.labels = []

        # Mapping from class name to integer label
        self.class_to_idx = {'normal': 0, 'low': 1, 'high': 2, 'high/low': 3}
        self.num_classes = len(self.class_to_idx)

        # Prepare dataset by scanning directories and splitting data
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Prepare the dataset by scanning directories, assigning labels, and splitting into train/test sets.
        """
        class_files = defaultdict(list)

        # Process 'normal' folder
        normal_dir = os.path.join(self.root_dir, 'normal')
        for root, _, files in os.walk(normal_dir):
            for file in files:
                if file.endswith('.wav'):
                    filepath = os.path.join(root, file)
                    class_files['normal'].append(filepath)

        # Process 'shout' folder
        shout_dir = os.path.join(self.root_dir, 'shout')
        for root, _, files in os.walk(shout_dir):
            for file in files:
                if file.endswith('.wav'):
                    filepath = os.path.join(root, file)
                    # Extract sentence_index from filename
                    try:
                        _, _, sentence_index = file.split('_')
                        sentence_index = int(sentence_index.split('.')[0])  # Remove .wav and convert to int

                        # Classify based on sentence_index
                        if 1 <= sentence_index <= 10:
                            class_files['high/low'].append(filepath)
                        elif 11 <= sentence_index <= 30:
                            class_files['low'].append(filepath)
                        elif 31 <= sentence_index <= 50:
                            class_files['high'].append(filepath)
                    except ValueError:
                        print(f"Filename format error in {file}")

        # Split each class into train and test sets (80% train, 20% test)
        self.file_list = []
        self.labels = []
        for class_name, files in class_files.items():
            labels = [self.class_to_idx[class_name]] * len(files)
            # Split while preserving class distribution
            train_files, test_files, train_labels, test_labels = train_test_split(
                files, labels, test_size=0.2, random_state=42, stratify=labels)

            if self.split == 'train':
                self.file_list.extend(train_files)
                self.labels.extend(train_labels)
            elif self.split == 'test':
                self.file_list.extend(test_files)
                self.labels.extend(test_labels)
            else:
                raise ValueError("split must be 'train' or 'test'")

        # Convert labels to numpy array for indexing
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]

        waveform = self.load_and_process_audio(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        # Create one-hot encoded target
        target = np.eye(self.num_classes)[label]

        return {
            'audio_name': audio_path,
            'waveform': torch.FloatTensor(waveform),
            'target': torch.FloatTensor(target)
        }

    def load_and_process_audio(self, path):
        """
        Load audio file, resample, and pad or trim to target duration.
        """
        y = load_audio(path, sr=self.sample_rate)

        # Calculate number of samples for target duration
        num_samples = int(self.target_duration * self.sample_rate)

        if len(y) < num_samples:
            # Pad with zeros
            padding = num_samples - len(y)
            y = np.pad(y, (0, padding), 'constant')
        else:
            # Trim to target duration
            y = y[:num_samples]

        return y

def collate_fn(batch):
    """
    Collate function to combine samples into a batch.
    """
    audio_names = [data['audio_name'] for data in batch]
    waveforms = [data['waveform'] for data in batch]
    targets = [data['target'] for data in batch]

    waveforms = torch.stack(waveforms)
    targets = torch.stack(targets)

    return {'audio_name': audio_names, 'waveform': waveforms, 'target': targets}

def get_dataloader(root_dir, split, batch_size, sample_rate=16000, target_duration=3.5,
                   shuffle=False, drop_last=False, num_workers=4, transform=None):
    """
    Create a DataLoader for the dataset.
    """
    dataset = AudioDataset(root_dir=root_dir, sample_rate=sample_rate,
                           target_duration=target_duration, split=split,
                           transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataset, dataloader
# Example usage
if __name__ == '__main__':
    root_dir = '/scratch/project_465001389/chandler_scratch/Datasets/risc/speech/'  # Replace with your dataset directory
    sample_rate = 16000
    target_duration = 3.5  # Target duration in seconds
    batch_size = 1
    num_workers = 4

    # Get training data
    train_dataset, train_loader = get_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=batch_size,
        sample_rate=sample_rate,
        target_duration=target_duration,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    # Get testing data
    test_dataset, test_loader = get_dataloader(
        root_dir=root_dir,
        split='test',
        batch_size=batch_size,
        sample_rate=sample_rate,
        target_duration=target_duration,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # Iterate over the training data
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        for j, audio_name in enumerate(batch['audio_name']):
            print(f"  - {audio_name} - Target: {batch['target'][j].argmax().item()}")
        print(f"Waveform shape: {batch['waveform'].shape}\n")
        
        # Stop after a few batches for demonstration purposes
        if i >= 5:
            break
