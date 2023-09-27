# A. Barahona-Ríos and T. Collins, “NoiseBandNet: Controllable Time-Varying Neural Synthesis of Sound Effects Using Filterbanks.” arXiv, Jul. 16, 2023.
# Uses noise passed through sharp filterbanks
# The mix of each filterbank is estimated
# Wavetables of filtered noise are precomputed and saved

import math
from typing import List
import numpy as np
import scipy
import torch
import torch.nn as nn
from ddsp.dsp import resample_frames


def get_filter(low: float, high: float, sample_rate: int, attenuation_db: float = 30):
    # calculate kaiser window params from transition width (20% of bandwidth) and attenuation
    numtaps, beta = scipy.signal.kaiserord(
        ripple=attenuation_db, width=(high - low) * 0.2 / (sample_rate / 2)
    )
    if numtaps % 2 == 0:
        numtaps += 1  # even taps can't be used for highpass
    if low == 0:
        return scipy.signal.firwin(
            numtaps, high, window=("kaiser", beta), pass_zero="lowpass", fs=sample_rate
        )
    if high == sample_rate / 2:
        return scipy.signal.firwin(
            numtaps, low, window=("kaiser", beta), pass_zero="highpass", fs=sample_rate
        )
    return scipy.signal.firwin(
        numtaps,
        [low, high],
        window=("kaiser", beta),
        pass_zero="bandpass",
        fs=sample_rate,
    )


def create_fbank(sample_rate: int, n_banks: int, attenuation_db: float = 50):
    n_lows = n_banks // 2
    n_highs = n_banks - n_lows
    low_freqs = np.linspace(20, sample_rate / 8, n_lows)
    low_freqs = np.concatenate([[0], low_freqs])
    low_filts = [
        get_filter(low_freqs[i], low_freqs[i + 1], sample_rate) for i in range(n_lows)
    ]
    high_freqs = np.logspace(
        np.log10(sample_rate / 8), np.log10(sample_rate / 2), n_highs + 1
    )
    high_freqs[-1] = sample_rate / 2  # correct floating point error
    high_filts = [
        get_filter(high_freqs[i], high_freqs[i + 1], sample_rate, attenuation_db)
        for i in range(n_highs)
    ]
    filters = low_filts + high_filts
    # pad to same length
    max_numtaps = max([f.shape[-1] for f in filters])
    filters = [np.pad(f, (0, max_numtaps - f.shape[-1])) for f in filters]
    return filters


def get_filtered_noise(filters):
    if isinstance(filters, list):
        filters = torch.tensor(np.array(filters), dtype=torch.float)
    # filters (n_banks, max_numtaps)
    R_filt = torch.fft.rfft(filters).abs()
    # use same noise for all bands
    random_phase = (torch.rand(R_filt.shape[-1]) * 2 - 1) * torch.pi
    random_phase[0] = 0
    random_phase[-1] = 0
    random_phase = random_phase[None, :].repeat(filters.shape[0], 1)
    filtered = torch.fft.irfft(R_filt * torch.exp(1j * random_phase))
    return filtered  # n_banks, max_numtaps


class NoiseBand(nn.Module):
    def __init__(
        self, sample_rate, n_banks, frame_rate, attenuation_db=50, n_splits=32
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_banks = n_banks
        self.frame_rate = frame_rate
        self.hop_size = sample_rate // frame_rate
        # split because amplitudes is too large after upsampling to n_samples
        self.n_splits = n_splits
        filters = create_fbank(sample_rate, n_banks, attenuation_db)
        filtered_noise = get_filtered_noise(filters).permute(1, 0)
        # max_numtaps, n_banks
        self.register_buffer("noises", filtered_noise)
        self.max_numtaps = filtered_noise.shape[0]

    def forward(self, amplitudes: torch.Tensor, n_samples: int) -> torch.Tensor:
        # amplitudes: batch, n_frames, n_banks
        if n_samples > self.max_numtaps:
            rep = math.ceil(n_samples / self.max_numtaps)
            noises = self.noises.tile((rep, 1)).unsqueeze(0)
        else:
            noises = self.noises.unsqueeze(0)
        if self.training:
            noises = noises.roll(
                int(torch.randint(0, self.max_numtaps, (1,)).item()), dims=0
            )
        if self.n_splits == 1:
            amplitudes = resample_frames(amplitudes, float(self.hop_size))
            amplitudes = amplitudes[:, :n_samples, :]
            return torch.sum(
                amplitudes * noises[:, :n_samples, :], dim=-1
            )  # batch, n_samples
        else:
            split_amps = amplitudes.split(self.n_banks // self.n_splits, dim=-1)
            split_noises = noises.split(self.n_banks // self.n_splits, dim=-1)
            bands: List[torch.Tensor] = []
            for amp, noise in zip(split_amps, split_noises):
                amp = resample_frames(amp, float(self.hop_size))
                # crop length in case input isn't divisible by hop_size
                length = min(n_samples, amp.shape[1])
                band = torch.sum(
                    amp[:, :length, :] * noise[:, :length, :],
                    dim=-1,
                )
                bands.append(band)
            return torch.stack(bands, dim=0).sum(dim=0)