import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CentVolEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_dim = 2

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        centroid = data["centroid"] / 8000.0
        vol = data["volume"]
        enc_out = torch.cat([centroid, vol], dim=-1)
        output = data.copy()
        output["enc_out"] = enc_out
        return output


class VolEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_dim = 1

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        centroid = data["centroid"] / 8000.0
        vol = data["volume"]
        output = data.copy()
        output["enc_out"] = vol
        return output
