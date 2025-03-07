import os
import torch
import torch.nn as nn
# from pydiffres import DiffRes as pydiffres
from specaug.specmix.specmix import SpecMix as specmix


class SpecMix(nn.Module):
    def __init__(self, prob, min_band_size, max_band_size, max_frequency_bands, max_time_bands):
        super(SpecMix, self).__init__()
        self.model = specmix(
            prob = prob,
            min_band_size = min_band_size,
            max_band_size = max_band_size,
            max_frequency_bands = max_frequency_bands,
            max_time_bands = max_time_bands,
        )

    def forward(self, data):
        return self.model(data)