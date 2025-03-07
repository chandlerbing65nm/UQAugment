import librosa
import glob
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import random
import torchaudio

def load_audio(path, sr=None):
    y, orig_sr = librosa.load(path, sr=None)
    if sr is not None and sr != orig_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y

def get_wav_name(class_label, data_path):
    """
    Retrieve all WAV files for a given class label.
    """
    wav_dir = os.path.join(data_path, class_label, '*.wav')
    audio = glob.glob(wav_dir)
    return audio

def data_generator(seed, data_path='./'):
    """
    Generate training, validation, and testing data dictionaries using percentages.
    """
    random_state = np.random.RandomState(seed)
    
    classes = ['none', 'strong', 'medium', 'weak']
    class_to_label = {'none': 0, 'strong':1, 'medium':2, 'weak':3}
    
    train_dict = []
    test_dict = []
    val_dict = []
    
    for class_label in classes:
        wav_list = get_wav_name(class_label=class_label, data_path=data_path)
        random_state.shuffle(wav_list)
        total_samples = len(wav_list)
        train_samples = int(total_samples * 0.8)
        val_samples = int(total_samples * 0.1)
        test_samples = total_samples - train_samples - val_samples  # Ensure all samples are used
        
        train_list = wav_list[:train_samples]
        val_list = wav_list[train_samples:train_samples + val_samples]
        test_list = wav_list[train_samples + val_samples:]
        
        label = class_to_label[class_label]
        
        for wav in train_list:
            train_dict.append([wav, label])
        for wav in val_list:
            val_dict.append([wav, label])
        for wav in test_list:
            test_dict.append([wav, label])
    
    random_state.shuffle(train_dict)
    random_state.shuffle(val_dict)
    random_state.shuffle(test_dict)
    
    return train_dict, val_dict, test_dict

class Fish_Voice_Dataset(Dataset):
    def __init__(self, sample_rate, seed, class_num, split='train', data_path='./', transform=None):
        """
        Custom Dataset for Fish Voice data.
        """
        self.seed = seed
        self.split = split
        self.data_path = data_path
        self.transform = transform
        self.class_num = class_num

        train_dict, val_dict, test_dict = data_generator(
            seed=self.seed,
            data_path=self.data_path)
        
        if self.split == 'train':
            self.data_dict = train_dict
        elif self.split == 'val':
            self.data_dict = val_dict
        elif self.split == 'test':
            self.data_dict = test_dict
        else:
            raise ValueError(f"Invalid split: {self.split}")
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        wav_name, target = self.data_dict[index]
        wav = load_audio(wav_name, sr=self.sample_rate)

        wav = np.array(wav)

        if self.transform is not None:
            wav = self.transform(samples=wav, sample_rate=self.sample_rate)

        target = np.eye(self.class_num)[target]

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
    target = torch.FloatTensor(np.array(target))

    return {'audio_name': wav_name, 'waveform': wav, 'target': target}

def get_dataloader(split,
                   batch_size,
                   sample_rate,
                   seed,
                   shuffle=False,
                   drop_last=False,
                   num_workers=4,
                   class_num=4,
                   data_path='./',
                   sampler=None,
                   transform=None):
    dataset = Fish_Voice_Dataset(split=split, sample_rate=sample_rate, seed=seed, class_num=class_num,
                                 data_path=data_path, transform=transform)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, sampler=sampler, collate_fn=collate_fn)

    return dataset, dataloader

if __name__ == '__main__':
    dataset, dataloader = get_dataloader(
        split='train', 
        batch_size=2, 
        sample_rate=22050, 
        shuffle=True, 
        seed=20, 
        drop_last=True, 
        data_path='/scratch/project_465001389/chandler_scratch/Datasets/mrsffia',
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
