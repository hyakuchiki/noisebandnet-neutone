import torch
import torch.nn as nn
import ddsp.dsp as dsp


class IRReverb(nn.Module):
    """
    Learns an IR as model parameter (always applied)
    """

    def __init__(self, ir_s=1.0, sr=48000):
        super().__init__()
        ir_length = int(ir_s * sr)
        noise = torch.rand(1, ir_length) * 2 - 1  # [-1, 1)
        # initial value should be zero to mask dry signal
        self.register_buffer("zero", torch.zeros(1))
        time = torch.linspace(0.0, 1.0, ir_length - 1)
        # initial ir = decaying white noise
        self.ir = nn.Parameter(
            (torch.rand(ir_length - 1) - 0.5) * 0.1 * torch.exp(-5.0 * time),
            requires_grad=True,
        )
        self.ir_length = ir_length

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: input audio (batch, n_samples)
        """
        # initial value should be zero to mask dry signal
        ir = torch.cat([self.zero, self.ir], dim=0)[None, :].expand(audio.shape[0], -1)
        wet = dsp.fft_convolve(audio, ir, padding="same", delay_compensation=0)
        return audio + wet
