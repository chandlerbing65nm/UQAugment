import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample

def load_audio(path, sr=None):
    y, orig_sr = librosa.load(path, sr=None)
    if sr is not None and orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y

def get_file_list(data_path, split='train'):
    data = []
    class_labels = sorted(os.listdir(data_path))
    label_dict = {label: idx for idx, label in enumerate(class_labels)}
    for label in class_labels:
        class_dir = os.path.join(data_path, label, split)
        if os.path.exists(class_dir):
            wav_files = glob.glob(os.path.join(class_dir, '*.wav'))
            for wav_file in wav_files:
                data.append((wav_file, label_dict[label]))
    return data, label_dict

class CustomAudioDataset(Dataset):
    def __init__(self, data_path, split='train', sample_rate=None, target_duration=None, transform=None):
        self.data_path = data_path
        self.split = split
        self.sample_rate = sample_rate
        self.target_duration = target_duration  # Duration in seconds
        self.transform = transform

        self.data, self.label_dict = get_file_list(self.data_path, self.split)
        self.class_num = len(self.label_dict)

        # Calculate target length in samples if target_duration is specified
        self.target_length = int(target_duration * sample_rate) if target_duration and sample_rate else None

    def __len__(self):
        return len(self.data)

    def pad_or_trim(self, waveform):
        if self.target_length is None:
            return waveform
        if len(waveform) < self.target_length:
            # Pad with zeros if shorter than target length
            padding = np.zeros(self.target_length - len(waveform))
            waveform = np.concatenate((waveform, padding), axis=0)
        else:
            # Trim to target length if longer
            waveform = waveform[:self.target_length]
        return waveform

    def __getitem__(self, index):
        wav_path, label = self.data[index]
        waveform = load_audio(wav_path, sr=self.sample_rate)
        
        # Apply padding or trimming to match the target duration
        waveform = self.pad_or_trim(waveform)

        if self.transform is not None:
            waveform = self.transform(waveform)

        # One-hot encode the target
        target = np.eye(self.class_num)[label]

        data_dict = {'audio_name': wav_path, 'waveform': waveform, 'target': target}

        return data_dict

def collate_fn(batch):
    wav_name = [data['audio_name'] for data in batch]
    wav = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    # Stack to create batch tensors
    wav = torch.FloatTensor(np.array(wav))
    target = torch.FloatTensor(np.array(target))

    return {'audio_name': wav_name, 'waveform': wav, 'target': target}

def get_dataloader(data_path, split, batch_size, sample_rate=None, target_duration=None, shuffle=False, drop_last=False, num_workers=4, transform=None):
    dataset = CustomAudioDataset(data_path=data_path, split=split, sample_rate=sample_rate, target_duration=target_duration, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers, collate_fn=collate_fn)
    return dataset, dataloader

if __name__ == '__main__':
    data_path = '/scratch/project_465001389/chandler_scratch/Datasets/dcase20/development'
    dataset, dataloader = get_dataloader(
        data_path=data_path,
        split='test',
        batch_size=1,
        sample_rate=16000,
        target_duration=10,  # Target duration in seconds
        shuffle=True,
        drop_last=True,
        num_workers=4,
        transform=None
    )

    for i, batch in enumerate(dataloader):
        audio_names = batch['audio_name']
        waveforms = batch['waveform']
        targets = batch['target']

        print(f"Batch {i+1}")
        print(f"Audio names: {audio_names}")
        print(f"Waveform shape: {waveforms.shape}")
        print(f"Target shape: {targets.shape}")

        # Break after a few batches for testing
        if i >= 5:
            break
