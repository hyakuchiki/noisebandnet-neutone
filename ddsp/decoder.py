from typing import Dict
import torch
import torch.nn as nn

from ddsp.modules.layers import MLP
from ddsp.modules.reverb import IRReverb
from ddsp.modules.noise import FilteredNoise
from ddsp.modules.noiseband import NoiseBand
from ddsp.util import exp_sigmoid


class DDSPFiltNoiseDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_size=512, filt_taps=1025, sample_rate=48000):
        super().__init__()
        # control encoder
        self.mlp_0 = MLP(latent_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mlp_1 = MLP(hidden_size, hidden_size)
        n_freqs = filt_taps // 2 + 1
        self.out = nn.Linear(hidden_size, n_freqs)
        self.filtnoise = FilteredNoise(filt_taps)
        self.reverb = IRReverb(sr=sample_rate)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, _hidden = self.gru(self.mlp_0(data["z"]))
        freq_response = self.out(self.mlp_1(x))
        freq_response = exp_sigmoid(freq_response, 10.0, 2.0)
        out_audio = self.filtnoise(freq_response, n_samples=data["audio"].shape[-1])
        out_audio = torch.tanh(self.reverb(out_audio))
        output = data.copy()
        output["audio"] = out_audio.unsqueeze(1)  # N, 1, T
        return output


class DDSPNoiseBandDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        frame_rate,
        hidden_size=512,
        n_banks=2048,
        sample_rate=48000,
        attenuation_db=50,
        n_splits=32,
    ):
        super().__init__()
        self.n_banks = n_banks
        # control encoder
        self.mlp_0 = MLP(latent_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mlp_1 = MLP(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_banks)
        self.noiseband = NoiseBand(
            sample_rate, n_banks, frame_rate, attenuation_db, n_splits
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, _hidden = self.gru(self.mlp_0(data["z"]))
        amps = self.out(self.mlp_1(x))
        amps = exp_sigmoid(amps, 10.0, float(self.n_banks))
        out_audio = self.noiseband(amps, n_samples=data["audio"].shape[-1])
        output = data.copy()
        output["audio"] = out_audio.unsqueeze(1)  # N, 1, T
        return output
