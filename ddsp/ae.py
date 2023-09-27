from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_size: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder.encoder_dim
        self.latent_size = latent_size
        self.lin_z = nn.Linear(self.encoder_dim, latent_size)
        self.decoder = decoder

    def latent(self, enc_out) -> Dict[str, torch.Tensor]:
        z = self.lin_z(enc_out)
        return {"z": z}

    def encode(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc_data = self.encoder(data)
        latent_params = self.latent(enc_data["enc_out"])
        enc_data.update(latent_params)
        return enc_data

    def decode(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dec_data = self.decoder(data)
        return dec_data

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc_data = self.encode(data)
        dec_data = self.decode(enc_data)
        return dec_data


class VAE(AutoEncoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_size: int,
    ):
        super().__init__(encoder, decoder, latent_size)
        del self.lin_z
        self.lin_mu = nn.Linear(self.encoder_dim, latent_size)
        self.lin_var = nn.Linear(self.encoder_dim, latent_size)

    def latent(self, enc_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu = self.lin_mu(enc_out)
        logvar = self.lin_var(enc_out)
        logvar = 10 * torch.tanh(logvar / 10)  # soft clipping
        if self.enc_freeze:
            mu = mu.detach()
            logvar = logvar.detach()
        eps = torch.randn_like(mu)
        z = ((logvar * 0.5).exp() * eps) + mu
        return {"z": z, "mu": mu, "logvar": logvar}
