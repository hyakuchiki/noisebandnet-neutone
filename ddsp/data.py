import os, pickle
from typing import Any, Optional, List, Dict
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

from ddsp.util import find_audio_files, load_audio_file
from ddsp.features import FeatureProcessor, Feature


class SmallDataset(Dataset):
    """
    Slices audio file(s) (wav, mp3, aif, flac) into segments
    and calculates features on the fly
    """

    def __init__(
        self,
        raw_path: str,
        feat_proc: FeatureProcessor,
        sample_rate: int = 48000,
        length: float = 1.0,
        dataset_size: int = 6400,
        frame_rate: int = 50,
    ):
        self.raw_path = raw_path
        self.sample_rate = sample_rate
        self.length = length
        self.feat_proc = feat_proc
        self.dataset_size = dataset_size
        self.n_samples = int(length * sample_rate)
        self.frame_rate = frame_rate
        if os.path.isdir(raw_path):
            # concat all audio into one
            files = find_audio_files(raw_path)
            audios = []
            for f in files:
                audio, sr = load_audio_file(f)
                if sr != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
                audios.append(audio)
            raw_audio = torch.cat(audios, dim=-1)
        elif os.path.isfile(raw_path):
            audio, sr = load_audio_file(raw_path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            raw_audio = audio
        # tile audio and pad to include head and tail more frequently
        raw_audio = torch.tile(raw_audio, dims=(1, 4))[0]  # (n_samples,)
        # include some silence
        self.raw_audio = F.pad(raw_audio, (0, self.n_samples))
        # generate indices
        last_idx = self.raw_audio.shape[-1] - self.n_samples
        self.start_indices = torch.randint(
            0, last_idx, (dataset_size,), dtype=torch.long
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        audio = self.raw_audio[start_idx : start_idx + self.n_samples]
        feats = self.feat_proc(audio, self.sample_rate)
        feats["audio"] = audio
        return feats
