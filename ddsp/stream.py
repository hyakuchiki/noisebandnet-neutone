from typing import Tuple, Dict, Optional
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_STREAM_BATCHSIZE = 2


class StatefulGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first=True)
        self.register_buffer(
            "hidden",
            torch.zeros(MAX_STREAM_BATCHSIZE, num_layers, hidden_size),
            persistent=False,
        )
        self.streaming = False

    def stream(self, mode: bool = True):
        if self.training and mode:
            warnings.warn("Module in streaming mode and training mode")
        self.streaming = mode

    def forward(self, x):
        if self.streaming:
            x, hidden = self.gru(x, self.hidden[: x.shape[0]])
            self.hidden = hidden
        else:
            x, hidden = self.gru(x)
        return x, hidden


class StreamingUpsample(nn.Module):
    def __init__(self, scale_factor, n_channels):
        """
        Streaming (cached) upsampling. Upsamples by scale_factor.
        Keeps the value of the last frame of previous buffer so that it can be smoothly connected with the current buffer.
        This results in a delay of {frame_size}.
        """
        super().__init__()
        self.register_buffer(
            "cache", torch.zeros(MAX_STREAM_BATCHSIZE, n_channels, 1), persistent=False
        )  # initialized with zero padding
        self.streaming = False
        self.scale_factor = float(scale_factor)

    def stream(self, mode: bool = True):
        if self.training and mode:
            warnings.warn("Module in streaming mode and training mode")
        self.streaming = mode

    def forward(self, x_frame, size: Optional[int] = None):
        # input: (batch, n_frames, n_channels)
        x_frame = x_frame.permute(0, 2, 1)
        if self.streaming:
            if size is not None:
                upsample_size = size
            else:
                upsample_size = int(self.scale_factor * (x_frame.shape[-1]))
            if x_frame.shape[-1] == upsample_size:
                # don't perform interpolation
                return x_frame.permute(0, 2, 1)
            x_frame = torch.cat([self.cache[: x_frame.shape[0]], x_frame], dim=-1)
            x_sample = F.interpolate(
                x_frame,
                size=upsample_size + 1,
                mode="linear",
                align_corners=True,
            )
            x_sample = x_sample[:, :, :-1]
            self.cache.copy_(x_frame[:, :, -1:])
        else:
            x_sample = F.interpolate(
                x_frame,
                scale_factor=self.scale_factor,
                mode="linear",
                align_corners=True,
            )
        return x_sample.permute(0, 2, 1)


def switch_streaming_mode(module: nn.Module, is_streaming: bool = True):
    for name, m in module.named_children():
        if len(list(m.children())) > 0:
            switch_streaming_mode(m)
        if hasattr(m, "streaming"):
            m.stream(is_streaming)
