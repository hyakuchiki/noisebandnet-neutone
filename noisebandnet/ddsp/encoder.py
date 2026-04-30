from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        vol = data["volume"]
        output = data.copy()
        output["enc_out"] = vol
        return output


class CentVolBandwidthEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_dim = 3

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        centroid = data["centroid"] / 8000.0
        vol = data["volume"]
        bandwidth = data["bandwidth"] / 8000.0
        enc_out = torch.cat([centroid, vol, bandwidth], dim=-1)
        output = data.copy()
        output["enc_out"] = enc_out
        return output


class MFCCEncoder(nn.Module):
    def __init__(self, n_mfcc=13):
        super().__init__()
        self.encoder_dim = n_mfcc

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mfcc = data["mfcc"]
        output = data.copy()
        output["enc_out"] = mfcc
        return output
