import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features import MelSpectrogram
from ddsp.spectral import fft_frequencies, A_weighting, Spec


class MultiSpecLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[64, 128, 256, 512, 1024, 2048],
        win_lengths=None,
        hop_lengths=None,
        mag_w=1.0,
        log_mag_w=1.0,
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        win_lengths = fft_sizes if win_lengths is None else win_lengths
        if hop_lengths is None:
            overlap = 0.75
            hop_lengths = [int((1 - overlap) * s) for s in fft_sizes]
        self.specs = nn.ModuleList(
            [
                Spec(n_fft, wl, hop_length=hl, center=True, power=2)
                for n_fft, wl, hl in zip(fft_sizes, win_lengths, hop_lengths)
            ]
        )
        self.mag_w = mag_w
        self.log_mag_w = log_mag_w
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths

    def forward(self, input, target):
        loss = 0
        input_audio = input["audio"]
        target_audio = target["audio"]
        if input_audio.ndim == 3:
            input_audio = input_audio.flatten(0, 1)
        if target_audio.ndim == 3:
            target_audio = target_audio.flatten(0, 1)
        for spec in self.specs:
            # print(list(spec.buffers())[0].device)
            x_pow = spec(input_audio)
            y_pow = spec(target_audio)
            loss += self.mag_w * torch.mean(torch.abs(x_pow - y_pow))
            loss += self.log_mag_w * torch.mean(
                torch.abs(torch.log(x_pow + 1e-6) - torch.log(y_pow + 1e-6))
            )
        return loss


class KLLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        logvar = input["logvar"]
        mu = input["mu"]
        loss = (
            -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=(0, 1)).sum()
        )
        if not logvar.requires_grad:
            loss = loss.detach()
        return loss


# Losses for monitoring
class LoudnessLoss(nn.Module):
    def __init__(
        self, sample_rate, n_fft=2048, hop_length=1024, win_length=None, db_range=80.0
    ) -> None:
        super().__init__()
        self.spec = Spec(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window="hann",
            center=True,
            power=2,
        )
        frequencies = fft_frequencies(sr=sample_rate, n_fft=n_fft)
        a_weighting = A_weighting(frequencies + 1e-8)
        self.register_buffer(
            "a_weighting", torch.from_numpy(a_weighting.astype(np.float32))
        )
        self.db_range = db_range

    def get_loudness(self, audio):
        power_spec = self.spec(audio).permute(0, 2, 1)
        weighting = 10 ** (self.a_weighting / 10)  # db to linear
        weighted_power = power_spec * weighting
        avg_power = torch.mean(weighted_power, dim=-1)
        # to db
        min_power = 10 ** -(self.db_range / 10.0)
        power = torch.clamp(avg_power, min=min_power)
        db = 10.0 * torch.log10(power)
        db = torch.clamp(db, min=-self.db_range)
        return db

    def forward(self, input, target) -> torch.Tensor:
        input_l = self.get_loudness(input["audio"])
        target_l = self.get_loudness(target["audio"])
        return F.l1_loss(torch.pow(10, input_l / 10), torch.pow(10, target_l / 10))


class MelLoss(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft=2048,
        win_length=None,
        hop_length=1024,
        n_mels=80,
        fmin=30,
        fmax=8000,
    ) -> None:
        super().__init__()
        self.melspec = MelSpectrogram(
            sample_rate,
            n_fft,
            win_length,
            n_mels,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            verbose=False,
            trainable_mel=False,
        )

    def forward(self, input, target) -> torch.Tensor:
        input_mel = self.melspec(input["audio"])
        target_mel = self.melspec(target["audio"])
        return F.l1_loss(input_mel, target_mel)


class LogSpecDistortion(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=1024,
    ) -> None:
        super().__init__()
        self.spec = Spec(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window="hann",
            center=True,
            power=2,
        )

    def forward(self, input, target) -> torch.Tensor:
        input_spec = self.spec(input["audio"])
        target_spec = self.spec(target["audio"])
        lsd = 10 * (torch.log10(input_spec + 1e-5) - torch.log10(target_spec + 1e-5))
        lsd = torch.sqrt((lsd**2).sum(dim=(-2, -1))) / target_spec.shape[-1]
        return lsd.mean()


class SpectralConvergence(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=1024,
    ) -> None:
        super().__init__()
        self.spec = Spec(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window="hann",
            center=True,
            power=1,
        )

    def forward(self, input, target) -> torch.Tensor:
        input_spec = self.spec(input["audio"])
        target_spec = self.spec(target["audio"])
        sc_loss = torch.linalg.norm(
            input_spec - target_spec, "fro", dim=(1, 2)
        ) / torch.linalg.norm(target_spec, "fro", dim=(1, 2))
        return sc_loss.mean()


class VQLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        return (input["commitment_loss"] + input["codebook_loss"]).mean()
