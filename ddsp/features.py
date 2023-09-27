from typing import Any, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchaudio

from ddsp.spectral import Spec
from ddsp.util import center_pad, slice_windows


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


class SpectralCentroid(Feature):
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
        # self.spec = Spec(n_fft=n_fft, hop_length=self.hop_size, center=False, power=2)
        spec = Spec(n_fft=n_fft, hop_length=self.hop_size, center=False, power=2)
        self.spec = spec

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.spec(x)
        freqs = torch.fft.rfftfreq(self.window_size, 1 / self.sample_rate)[
            None, :, None
        ]
        cent = (freqs * spec).sum(dim=-2) / (spec.sum(dim=-2) + 1e-5)
        return cent.unsqueeze(-1)  # batch, n_frames, 1


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
