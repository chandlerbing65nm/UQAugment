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
    def __init__(self, root_dir, sample_rate=16000, target_duration=3.0, split='train',
                 transform=None, n_samples_per_class=None, seed=42):
        """
        Args:
            root_dir (str): Root directory containing 'normal' and 'shout' folders.
            sample_rate (int): Desired sample rate.
            target_duration (float): Target duration in seconds.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Transform to be applied to the waveform.
            n_samples_per_class (int, optional): Number of samples per class for few-shot training.
            seed (int): Random seed for reproducibility.
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.split = split
        self.transform = transform
        self.n_samples_per_class = n_samples_per_class
        self.seed = seed

        # Mapping from class name to integer label
        self.class_to_idx = {'normal': 0, 'low': 1, 'high': 2, 'high/low': 3}
        self.num_classes = len(self.class_to_idx)

        # Prepare dataset by scanning directories and splitting data
        self.class_segments = defaultdict(list)
        self.class_labels = defaultdict(list)
        self._prepare_dataset()
        self._generate_data_splits()

    def _prepare_dataset(self):
        """
        Prepare the dataset by scanning directories and assigning labels.
        """
        # Process 'normal' folder
        normal_dir = os.path.join(self.root_dir, 'normal')
        for root, _, files in os.walk(normal_dir):
            for file in files:
                if file.endswith('.wav'):
                    filepath = os.path.join(root, file)
                    class_name = 'normal'
                    class_label = self.class_to_idx[class_name]

                    # Get duration
                    try:
                        duration = librosa.get_duration(path=filepath)
                    except:
                        print(f"Error getting duration for {filepath}")
                        continue

                    # Calculate number of segments
                    num_samples = int(self.target_duration * self.sample_rate)
                    total_samples = int(duration * self.sample_rate)
                    num_segments = total_samples // num_samples
                    remainder = total_samples % num_samples
                    if remainder > 0:
                        num_segments += 1  # Include the last partial segment

                    # For each segment, add an entry
                    for segment_idx in range(num_segments):
                        self.class_segments[class_name].append((filepath, segment_idx))
                        self.class_labels[class_name].append(class_label)

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
                            class_name = 'high/low'
                        elif 11 <= sentence_index <= 30:
                            class_name = 'low'
                        elif 31 <= sentence_index <= 50:
                            class_name = 'high'
                        else:
                            continue  # Ignore files not in these ranges
                        class_label = self.class_to_idx[class_name]
                    except ValueError:
                        print(f"Filename format error in {file}")
                        continue

                    # Get duration
                    try:
                        duration = librosa.get_duration(path=filepath)
                    except:
                        print(f"Error getting duration for {filepath}")
                        continue

                    # Calculate number of segments
                    num_samples = int(self.target_duration * self.sample_rate)
                    total_samples = int(duration * self.sample_rate)
                    num_segments = total_samples // num_samples
                    remainder = total_samples % num_samples
                    if remainder > 0:
                        num_segments += 1  # Include the last partial segment

                    # For each segment, add an entry
                    for segment_idx in range(num_segments):
                        self.class_segments[class_name].append((filepath, segment_idx))
                        self.class_labels[class_name].append(class_label)

    def _generate_data_splits(self):
        """
        Generate training, validation, and testing data splits.
        """
        classes = list(self.class_to_idx.keys())
        class_to_label = self.class_to_idx

        train_dict = []
        val_dict = []
        test_dict = []

        test_percentage = 0.2  # Keep the test set size consistent

        for class_name in classes:
            segments = self.class_segments[class_name]
            labels = self.class_labels[class_name]
            class_label = class_to_label[class_name]

            # Shuffle segments and labels together
            class_data = list(zip(segments, labels))
            # Create a random state unique to each class for reproducibility
            class_seed = self.seed + class_label
            random_state_class = np.random.RandomState(class_seed)
            random_state_class.shuffle(class_data)

            total_samples = len(class_data)
            test_samples = int(total_samples * test_percentage)

            test_data = class_data[:test_samples]
            remaining_data = class_data[test_samples:]

            if self.n_samples_per_class is not None:
                train_samples = min(self.n_samples_per_class, len(remaining_data))
                train_data = remaining_data[:train_samples]
                val_data = remaining_data[train_samples:]
            else:
                # Use default percentages for training and validation
                train_samples = int(len(remaining_data) * 0.8)
                train_data = remaining_data[:train_samples]
                val_data = remaining_data[train_samples:]

            train_dict.extend(train_data)
            val_dict.extend(val_data)
            test_dict.extend(test_data)

        # Now, depending on self.split, set self.file_list and self.labels
        if self.split == 'train':
            data_dict = train_dict
        elif self.split == 'val':
            data_dict = val_dict
        elif self.split == 'test':
            data_dict = test_dict
        else:
            raise ValueError(f"Invalid split: {self.split}")

        if len(data_dict) == 0:
            raise ValueError(f"No data available for split '{self.split}'. Check your dataset and parameters.")

        self.file_list, self.labels = zip(*data_dict)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_info = self.file_list[idx]
        label = self.labels[idx]

        audio_path, segment_idx = audio_info

        waveform = self.load_and_process_audio(audio_path, segment_idx)

        if self.transform:
            waveform = self.transform(waveform)

        # Create one-hot encoded target
        target = np.eye(self.num_classes)[label]

        return {
            'audio_name': audio_path,
            'waveform': torch.FloatTensor(waveform),
            'target': torch.FloatTensor(target)
        }

    def load_and_process_audio(self, path, segment_idx):
        """
        Load audio file, resample, and extract the specified segment.
        """
        y = load_audio(path, sr=self.sample_rate)

        # Calculate number of samples for target duration
        num_samples = int(self.target_duration * self.sample_rate)
        start_sample = segment_idx * num_samples
        end_sample = start_sample + num_samples

        if start_sample >= len(y):
            # This shouldn't happen if we calculated num_segments correctly
            # Return zeros
            y_segment = np.zeros(num_samples)
        else:
            y_segment = y[start_sample:end_sample]
            if len(y_segment) < num_samples:
                # Pad with zeros
                padding = num_samples - len(y_segment)
                y_segment = np.pad(y_segment, (0, padding), 'constant')

        return y_segment

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
                   shuffle=False, drop_last=False, num_workers=4, transform=None,
                   n_samples_per_class=None, seed=42):
    """
    Create a DataLoader for the dataset.
    """
    dataset = AudioDataset(root_dir=root_dir, sample_rate=sample_rate,
                           target_duration=target_duration, split=split,
                           transform=transform, n_samples_per_class=n_samples_per_class,
                           seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataset, dataloader

# Example usage
if __name__ == '__main__':
    root_dir = '/scratch/project_465001389/chandler_scratch/Datasets/risc/speech/'  # Replace with your dataset directory
    sample_rate = 16000
    target_duration = 1  # Target duration in seconds
    batch_size = 1
    num_workers = 4

    n_samples_per_class_list = [2, 1, 4]  # Different runs with increasing samples per class

    for n_samples_per_class in n_samples_per_class_list:
        print(f"Run with n_samples_per_class = {n_samples_per_class}")
        train_dataset, train_loader = get_dataloader(
            root_dir=root_dir,
            split='train',
            batch_size=batch_size,
            sample_rate=sample_rate,
            target_duration=target_duration,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            n_samples_per_class=n_samples_per_class,
            seed=42
        )

        # Collect samples per class
        class_samples = {}
        for data in train_dataset:
            label = np.argmax(data['target'])
            audio_name = os.path.basename(data['audio_name'])
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(audio_name)

        for label in sorted(class_samples.keys()):
            print(f"Class {label}: {class_samples[label]}")
        print("")
