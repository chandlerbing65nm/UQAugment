import librosa
import glob
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import random

def load_audio(path, sr=None, target_duration=None, normalize=True):
    """
    Load an audio file, normalize its amplitude, and ensure it meets the target duration.
    """
    y, orig_sr = librosa.load(path, sr=None)
    if sr is not None and sr != orig_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    
    if target_duration:
        target_length = int(target_duration * sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        elif len(y) > target_length:
            y = y[:target_length]

    if normalize:
        max_amplitude = np.max(np.abs(y))
        if max_amplitude > 0:
            y = y / max_amplitude  # Normalize to range [-1, 1]
    
    return y

def get_wav_files(data_path, class_label):
    """
    Retrieve all WAV files for a given class label from subfolders.
    """
    wav_dir = os.path.join(data_path, class_label)
    audio = glob.glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True)
    return audio

def data_generator(seed, data_path='./'):
    """
    Generate training and validation data dictionaries using percentages.
    """
    random_state = np.random.RandomState(seed)
    
    classes = ['background', 'foreground']
    class_to_label = {'background': 0, 'foreground': 1}
    
    train_dict = []
    val_dict = []
    
    for class_label in classes:
        wav_list = get_wav_files(data_path, class_label)
        random_state.shuffle(wav_list)
        total_samples = len(wav_list)
        train_samples = int(total_samples * 0.8)
        
        train_list = wav_list[:train_samples]
        val_list = wav_list[train_samples:]
        
        label = class_to_label[class_label]
        
        for wav in train_list:
            train_dict.append([wav, label])
        for wav in val_list:
            val_dict.append([wav, label])
    
    random_state.shuffle(train_dict)
    random_state.shuffle(val_dict)
    
    return train_dict, val_dict

class NoiseDataset(Dataset):
    def __init__(self, sample_rate, seed, split='train', data_path='./', transform=None, target_duration=None):
        """
        Custom Dataset for noise classification.
        """
        self.seed = seed
        self.split = split
        self.data_path = data_path
        self.transform = transform
        self.target_duration = target_duration

        train_dict, val_dict = data_generator(seed=self.seed, data_path=self.data_path)
        
        if self.split == 'train':
            self.data_dict = train_dict
        elif self.split == 'val':
            self.data_dict = val_dict
        else:
            raise ValueError(f"Invalid split: {self.split}")
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        wav_name, target = self.data_dict[index]
        wav = load_audio(wav_name, sr=self.sample_rate, target_duration=self.target_duration)

        wav = np.array(wav)

        if self.transform is not None:
            wav = self.transform(samples=wav, sample_rate=self.sample_rate)

        data_dict = {'audio_name': wav_name, 'waveform': wav, 'target': target}

        return data_dict

def collate_fn(batch):
    wav_name = [data['audio_name'] for data in batch]
    wav = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    # Pad sequences to the same length
    max_length = max([len(w) for w in wav])
    wav_padded = [np.pad(w, (0, max_length - len(w)), 'constant') for w in wav]
    wav = torch.FloatTensor(np.array(wav_padded))
    target = torch.LongTensor(np.array(target))

    return {'audio_name': wav_name, 'waveform': wav, 'target': target}

def get_dataloader(split,
                   batch_size,
                   sample_rate,
                   seed,
                   shuffle=False,
                   drop_last=False,
                   num_workers=4,
                   data_path='/scratch/project_465001389/chandler_scratch/Datasets/qiandaoear22',
                   sampler=None,
                   transform=None,
                   args=None):
    dataset = NoiseDataset(split=split, sample_rate=sample_rate, seed=seed,
                           data_path=data_path, transform=transform, 
                           target_duration=args.target_duration if args else None)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers, sampler=sampler, collate_fn=collate_fn)

    return dataset, dataloader

if __name__ == '__main__':
    class Args:
        target_duration = 3.0  # Example target duration in seconds
    
    args = Args()
    dataset, dataloader = get_dataloader(
        split='train', 
        batch_size=2, 
        sample_rate=22050, 
        shuffle=True, 
        seed=20, 
        drop_last=True, 
        data_path='/scratch/project_465001389/chandler_scratch/Datasets/qiandaoear22',
        args=args
    )
    
    for i, batch in enumerate(dataloader):
        audio_names = batch['audio_name'][0]
        waveforms = batch['waveform'][0]
       
        print(f"Audio Name: {audio_names}")
        print(f"Waveform Shape: {waveforms.shape}")
        print(f"Target: {batch['target'][0]}")
        print("")

        if i >= 4:
            break
