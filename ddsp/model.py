from typing import Dict, Any
import hydra
from omegaconf import open_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ddsp.losses import MelLoss, LoudnessLoss, LogSpecDistortion, SpectralConvergence
from ddsp.schedules import ParamSchedule
from ddsp.util import pad_or_crop_to_length


class AutoEncoderModel(pl.LightningModule):
    def __init__(self, ae_cfg, losses_cfg):
        super().__init__()
        self.ae = hydra.utils.instantiate(ae_cfg.ae)
        self.losses = hydra.utils.instantiate(losses_cfg.losses)
        self.loss_w_sched = ParamSchedule(losses_cfg.sched)  # loss weighting
        self.sr = ae_cfg.sample_rate
        # optimizer settings
        self.lr = ae_cfg.lr
        self.betas = ae_cfg.betas
        self.lr_decay_steps = ae_cfg.lr_decay_steps
        self.lr_decay_factor = ae_cfg.lr_decay_factor
        # loss functions for validation
        self.monitor_losses = nn.ModuleDict(
            {
                "mel": MelLoss(self.sr),
                "loud": LoudnessLoss(self.sr),
                "lsd": LogSpecDistortion(),
                "sc": SpectralConvergence(),
            }
        )
        self.save_hyperparameters()

    def forward(self, data):
        """
        Args:
            data (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        input_length = data["audio"].shape[-1]
        output = self.ae(data)
        output["audio"] = pad_or_crop_to_length(output["audio"], input_length)
        return output

    def generator_losses(self, output, target, loss_w=None):
        # non_adversarial losses for the autoencoder
        loss_dict = {}
        for k, loss in self.losses.items():
            weight = 1.0 if loss_w is None else loss_w[k]
            if weight > 0.0:
                loss_dict[k] = weight * loss(output, target)
            else:
                loss_dict[k] = 0.0
        return loss_dict

    def training_step(self, batch_dict, batch_idx):
        # get loss weights
        loss_weights = self.loss_w_sched.get_parameters(self.global_step)
        self.log_dict(
            {"loss_weight/" + k: v for k, v in loss_weights.items()},
            on_epoch=True,
            on_step=False,
        )
        # render audio
        output_dict = self(batch_dict)
        g_losses = self.generator_losses(output_dict, batch_dict, loss_weights)
        g_loss = sum(g_losses.values())
        self.log_dict(
            {"train/gen_" + k: v for k, v in g_losses.items()},
            on_epoch=True,
            on_step=False,
        )
        self.log("train/gen_total", g_loss, prog_bar=True, on_epoch=True, on_step=True)
        return g_loss

    def monitor_metrics(self, output, target):
        metrics = {}
        # losses not used for training
        for key, loss_fn in self.monitor_losses.items():
            ## audio losses
            metrics[key] = loss_fn(output, target)
        return metrics

    def validation_step(self, batch_dict, batch_idx, dataloader_idx=0):
        # render audio
        output_dict = self(batch_dict)
        losses = self.generator_losses(output_dict, batch_dict)
        eval_losses = self.monitor_metrics(output_dict, batch_dict)
        losses.update(eval_losses)
        loss_weights = self.loss_w_sched.get_parameters(self.global_step)
        losses = {"val_{0}/{1}".format(dataloader_idx, k): v for k, v in losses.items()}
        self.log_dict(
            losses,
            prog_bar=True,
            on_epoch=True,
            add_dataloader_idx=False,
        )
        return losses

    def test_step(self, batch_dict, batch_idx, dataloader_idx=0):
        # render audio
        outputs = self(batch_dict)
        losses = self.generator_losses(outputs, batch_dict)
        eval_losses = self.monitor_metrics(outputs, batch_dict)
        losses.update(eval_losses)
        losses = {"val_{0}/{1}".format(dataloader_idx, k): v for k, v in losses.items()}
        self.log_dict(
            losses,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            add_dataloader_idx=False,
        )
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.lr, betas=self.betas)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
