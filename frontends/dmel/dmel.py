import torch
import torch.nn as nn
import torch.nn.functional as F

def differentiable_gaussian_window(lambd, window_length, norm=True):
    m = torch.arange(0, window_length).float().to(lambd.device)
    window = torch.exp(-0.5 * ((m - window_length / 2) / (lambd + 1e-15)) ** 2)
    window_norm = window / torch.sqrt(torch.sum(window ** 2))
    if norm:
        return window_norm
    else:
        return window

def differentiable_spectrogram(x, lambd, n_fft, win_length, hop_length, norm=False):
    window = differentiable_gaussian_window(lambd, window_length=win_length, norm=norm)
    s = torch.stft(
        x, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window, 
        return_complex=True, 
        pad_mode='constant'
    )
    s = torch.pow(torch.abs(s), 2)
    return s