import os

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

import jaxcrepe


def audio(filename):
    """Load audio from disk"""
    return torchaudio.load(filename)


def model(device, capacity='full'):
    """Preloads model from disk"""
    # Bind model and capacity
    jaxcrepe.infer.capacity = capacity
    jaxcrepe.infer.model = jaxcrepe.Crepe(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pth')
    jaxcrepe.infer.model.load_state_dict(
        torch.load(file, map_location=device, weights_only=True))

    # Place on device
    jaxcrepe.infer.model = jaxcrepe.infer.model.to(torch.device(device))

    # Eval mode
    jaxcrepe.infer.model.eval()
