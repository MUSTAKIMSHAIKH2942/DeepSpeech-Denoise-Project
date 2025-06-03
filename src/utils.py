import torchaudio
import torch

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform[0]

def save_audio(file_path, audio_tensor, sample_rate=16000):
    torchaudio.save(file_path, audio_tensor.unsqueeze(0), sample_rate)
