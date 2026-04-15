from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchcrepe

from noisebandnet.ddsp.spectral import Spec
from noisebandnet.ddsp.util import center_pad, slice_windows


class Feature(ABC, nn.Module):
    def __init__(
        self, sample_rate: int, window_size: int, frame_rate: int, center: bool = True
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.frame_rate = frame_rate
        self.hop_size = sample_rate // frame_rate
        self.center = center
        # for streaming
        self.streaming = False
        self.register_buffer("cache", torch.zeros(1, window_size // 2 if center else 0))

    @abstractmethod
    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, time)
        # output: (batch_size, )
        pass

    def get_n_frames(self, input_length: int) -> float:
        return float()

    def stream(self, mode: bool = True):
        self.streaming = mode

    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        stream: bool = False,
    ):
        """
        audio: ((batch_size), time)

        Outputs: ((batch_size), n_frames, feat_dim)
        """
        input_ndim = audio.ndim
        if input_ndim == 1:
            audio = audio.unsqueeze(0)
        if self.streaming:
            x = torch.cat([self.cache, audio], dim=-1)
            n_frames = (x.shape[-1] - self.window_size) // self.hop_size + 1
            # starting position of frame that wasn't calculated
            next_pos = self.hop_size * n_frames
            # save as new cache
            self.cache = x[..., next_pos:].clone()
        else:
            if sample_rate is not None and sample_rate != sample_rate:
                # resample
                audio = torchaudio.functional.resample(
                    audio, sample_rate, self.sample_rate
                )
            if self.center:
                x = center_pad(audio, self.window_size, self.hop_size)
            else:
                x = audio
        feat = self.compute_feature(x)
        if input_ndim == 1:
            feat = feat.squeeze(0)
        return feat


class FeatureProcessor(nn.Module):
    def __init__(self, features: Dict[str, Feature]) -> None:
        super().__init__()
        self.features = nn.ModuleDict(features)
        # make sure all frame_rates are the same so feats line up
        fpss = [feat.frame_rate for feat in self.features.values()]
        assert all(x == fpss[0] for x in fpss)
        self.resamples: Dict[Tuple[int, int], nn.Module] = {}
        print("calculating features:", list(self.features.keys()))

    @torch.jit.unused
    def resample(
        self,
        sample_rate: int,
        target_sr: int,
        audio: torch.Tensor,
        inputs: Dict[int, torch.Tensor],
    ):
        # needs resampling
        if (sample_rate, target_sr) not in self.resamples:
            # make resampling kernel only once
            self.resamples[(sample_rate, target_sr)] = torchaudio.transforms.Resample(
                sample_rate, target_sr
            )
        x_resamp = self.resamples[(sample_rate, target_sr)](audio)
        # save resampled audio for other features to maybe use
        inputs[target_sr] = x_resamp

    def forward(self, audio: torch.Tensor, sample_rate: int) -> Dict[str, torch.Tensor]:
        inputs = {sample_rate: audio}
        feature_data: Dict[str, torch.Tensor] = {}
        for feat_name, feat_mod in self.features.items():
            target_sr = feat_mod.sample_rate
            if target_sr not in inputs:
                self.resample(sample_rate, feat_mod.sample_rate, audio, inputs)
            feature_data[feat_name] = feat_mod(inputs[target_sr])
        return feature_data


class SpectralFeature(Feature):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        frame_rate: int,
        center: bool = True,
        n_fft: Optional[int] = None,
    ):
        super().__init__(sample_rate, window_size, frame_rate, center)
        window = torch.hann_window(window_size)
        self.register_buffer("window", window)
        n_fft = n_fft if n_fft else window_size
        self.n_fft = n_fft
        self.spec = Spec(
            n_fft=n_fft,
            win_length=window_size,
            hop_length=self.hop_size,
            center=False,
            power=2,
        )


class SpectralCentroid(SpectralFeature):

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.spec(x)
        freqs = torch.fft.rfftfreq(self.n_fft, 1 / self.sample_rate)[None, :, None]
        cent = (freqs * spec).sum(dim=-2) / (spec.sum(dim=-2) + 1e-5)
        return cent.unsqueeze(-1)  # batch, n_frames, 1


class SpectralBandwidth(SpectralFeature):
    """
    Spectral bandwidth per frame (second central moment of the power spectrum).
    Returns shape (batch, n_frames, 1) with values in Hz.
    """

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.spec(x)
        freqs = torch.fft.rfftfreq(self.n_fft, 1 / self.sample_rate)[None, :, None]
        cent = (freqs * spec).sum(dim=-2) / (spec.sum(dim=-2) + 1e-5)
        freq_diff = freqs - cent[..., None]
        var = (freq_diff**2 * spec).sum(dim=-2) / (spec.sum(dim=-2) + 1e-5)
        bw = torch.sqrt(var)
        return bw.unsqueeze(-1)


class SpectralFlatness(SpectralFeature):
    """
    Spectral flatness (Wiener entropy) per frame.

    Computed as geometric_mean(power_spectrum) / arithmetic_mean(power_spectrum).
    Returns values in [0, 1], shape (batch, n_frames, 1).
    """

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.spec(x)
        # geometric mean across frequency bins
        log_spec = torch.log(spec + 1e-12)
        geometric_mean = torch.exp(log_spec.mean(dim=-2))
        # arithmetic mean across frequency bins
        arithmatic_mean = spec.mean(dim=-2)
        flatness = geometric_mean / (arithmatic_mean + 1e-12)
        return flatness.unsqueeze(-1)


class Volume(Feature):
    # not loudness, just energy
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        frame_rate: int,
        center: bool = True,
    ):
        super().__init__(sample_rate, window_size, frame_rate, center)
        window = torch.hann_window(window_size)
        self.register_buffer("window", window)

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        x_sqr = x**2
        a2_win = slice_windows(
            x_sqr, self.window_size, self.hop_size, "none", pad=False
        )
        rms = a2_win.mean(dim=-1).sqrt()
        return rms.unsqueeze(-1)  # batch, n_frames, 1


class F0(Feature):
    """
    Fundamental frequency (F0) feature using torchcrepe.

    Outputs shape: (batch, n_frames, 1) with F0 in Hz. Requires `torchcrepe` to be
    installed; raises ImportError with instructions if not available.
    """

    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        frame_rate: int,
        center: bool = True,
        fmin: float = 50.0,
        fmax: Optional[float] = None,
    ):
        super().__init__(sample_rate, window_size, frame_rate, center)
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else float(sample_rate // 2)

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        # torchcrepe.predict returns (batch, n_frames) of f0 in Hz
        # Use hop_length = hop_size and run on same device as input
        f0 = torchcrepe.predict(
            x,
            self.sample_rate,
            self.hop_size,
            fmin=self.fmin,
            fmax=self.fmax,
            device=x.device,
            pad=False,
        )

        # ensure shape (batch, n_frames, 1)
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        return f0


class MFCC(Feature):
    """
    Compute first `n_mfcc` MFCCs per frame using `torchaudio.transforms.MFCC`.

    Outputs shape: (batch, n_frames, n_mfcc)
    """

    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        frame_rate: int,
        center: bool = True,
        n_mfcc: int = 10,
        n_mels: int = 80,
        fmin: float = 30.0,
        fmax: Optional[float] = None,
    ):
        super().__init__(sample_rate, window_size, frame_rate, center)
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else float(sample_rate // 2)
        # torchaudio MFCC accepts melkwargs for the MelSpectrogram step
        melkwargs = {
            "n_fft": self.window_size,
            "win_length": self.window_size,
            "hop_length": self.hop_size,
            "n_mels": self.n_mels,
            "f_min": self.fmin,
            "f_max": self.fmax,
        }
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=self.n_mfcc, melkwargs=melkwargs
        )

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        # torchaudio MFCC expects (..., time) and returns (batch, n_mfcc, n_frames)
        mfcc = self.mfcc(x)
        # transpose to (batch, n_frames, n_mfcc)
        mfcc = mfcc.permute(0, 2, 1)
        return mfcc
