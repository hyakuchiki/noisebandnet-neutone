import torch
import torch.nn as nn
from ddsp.dsp import fir_filter


class FilteredNoise(nn.Module):
    """
    uses frequency sampling
    """

    def __init__(self, filter_size=1025, amplitude=1.0):
        super().__init__()
        self.filter_size = filter_size
        self.amplitude = amplitude
        self.param_sizes = {"freq_response": self.filter_size // 2 + 1}
        self.param_range = {"freq_response": (0, 2.0)}
        self.param_types = {"freq_response": "exp_sigmoid"}

    def forward(self, freq_response: torch.Tensor, n_samples: int):
        """generate Gaussian white noise through FIRfilter
        Args:
            freq_response (torch.Tensor): frequency response (only magnitude) [batch, n_frames, filter_size // 2 + 1]

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        batch_size = freq_response.shape[0]
        audio = (torch.rand(batch_size, n_samples) * 2.0 - 1.0).to(
            freq_response.device
        ) * self.amplitude
        audio = fir_filter(audio, freq_response, self.filter_size)
        return audio
