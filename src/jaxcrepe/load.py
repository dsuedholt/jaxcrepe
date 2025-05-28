import os
import librosa
import equinox as eqx

import jaxcrepe


def audio(filename, sr=None):
    """Load audio from disk"""
    return librosa.load(filename, sr=sr)


def model(capacity='full'):
    """Preloads model from disk"""
    # Bind model and capacity
    jaxcrepe.infer.capacity = capacity
    jaxcrepe.infer.model = jaxcrepe.Crepe(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.eqx')
    jaxcrepe.infer.model = eqx.tree_deserialise_leaves(file, jaxcrepe.infer.model)