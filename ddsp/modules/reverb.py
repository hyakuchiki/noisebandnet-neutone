import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import ddsp.dsp as dsp

MAX_STREAMING_BATCHSIZE = 2


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
        self.register_buffer("cache", torch.zeros(MAX_STREAMING_BATCHSIZE, 16000))
        self.register_buffer("strm_ir", torch.zeros(ir_length))

    def stream(self, mode: bool = True):
        if self.training and mode:
            warnings.warn("Module in streaming mode and training mode")
        self.streaming = mode
        if mode:  # have 0 already concatted
            self.strm_ir.copy_(torch.cat([self.zero, self.ir]))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: input audio (batch, n_samples)
        """
        if self.streaming:
            # convolve with tail
            wet = dsp.fft_convolve(
                audio, self.strm_ir, padding="valid", delay_compensation=0
            )
            length = max(wet.shape[-1], self.cache.shape[-1])
            # add cached reverb tail and new output together
            buffer_wet = F.pad(
                self.cache, (0, max(0, length - self.cache.shape[-1]))
            ) + F.pad(wet, (0, max(0, length - wet.shape[-1])))
            wet = buffer_wet[..., : audio.shape[-1]]
            # save tail for later
            self.cache = buffer_wet[..., audio.shape[-1] :]
        else:
            # initial value should be zero to mask dry signal
            ir = torch.cat([self.zero, self.ir], dim=0)[None, :].expand(
                audio.shape[0], -1
            )
            wet = dsp.fft_convolve(audio, ir, padding="same", delay_compensation=0)
            return audio + wet
