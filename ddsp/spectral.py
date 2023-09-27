import torch
import torch.nn as nn
import numpy as np
from nnAudio.features import STFT

amp = lambda x: x[..., 0] ** 2 + x[..., 1] ** 2


class Spec(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=1024,
        window="hann",
        center=False,
        power=2,
    ) -> None:
        super().__init__()
        self.stft = STFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            freq_scale="no",
            pad_mode="reflect",
            output_format="Complex",
            verbose=False,
        )
        self.power = power

    def forward(self, audio):
        # audio: (batch, (n_channels=1), time)
        power_spec = amp(self.stft(audio))
        if self.power == 2:
            spec = power_spec
        elif self.power == 1:
            spec = torch.clamp(power_spec, min=1e-8).sqrt()
        return spec


def A_weighting(frequencies, min_db=-80.0):
    # ported from librosa
    f_sq = np.asanyarray(frequencies) ** 2.0
    const = np.array([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    weights = 2.0 + 20.0 * (
        np.log10(const[0])
        + 2 * np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
        - 0.5 * np.log10(f_sq + const[2])
        - 0.5 * np.log10(f_sq + const[3])
    )
    return weights if min_db is None else np.maximum(min_db, weights)


def fft_frequencies(*, sr=22050, n_fft=2048):
    # ported from librosa
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
