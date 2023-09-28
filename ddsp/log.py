import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from torchaudio.functional import resample
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from ddsp.util import find_audio_files, load_audio_file


def plot_recons(
    x, x_tilde, plot_dir, name=None, epochs=None, sr=16000, num=6, save=True
):
    """Plot spectrograms/waveforms of original/reconstructed audio

    Args:
        x (numpy array): [n_samples]
        x_tilde (numpy array): [n_samples]
        sr (int, optional): sample rate. Defaults to 16000.
        dir (str): plot directory.
        name (str, optional): file name.
        epochs (int, optional): no. of epochs.
        num (int, optional): number of spectrograms to plot. Defaults to 6.
    """
    num = min(x.shape[0], num)
    fig, axes = plt.subplots(num, 4, figsize=(15, 5 * num), squeeze=False)
    for i in range(num):
        axes[i, 0].specgram(x[i], Fs=sr, scale="dB")
        axes[i, 1].specgram(x_tilde[i], Fs=sr, scale="dB")
        axes[i, 2].plot(x[i], zorder=-10)  # rasterize waveform
        axes[i, 2].set_rasterization_zorder(-5)
        axes[i, 2].set_ylim(-1, 1)
        axes[i, 3].plot(x_tilde[i], zorder=-10)
        axes[i, 3].set_rasterization_zorder(-5)
        axes[i, 3].set_ylim(-1, 1)
    if save:
        if epochs:
            fig.savefig(os.path.join(plot_dir, "epoch{:0>3}_recons.png".format(epochs)))
            plt.close(fig)
        else:
            fig.savefig(os.path.join(plot_dir, name + ".png"))
            plt.close(fig)
    else:
        return fig


def save_to_board(
    i, name, writer, orig_audio, resyn_audio, plot_num=4, sr=16000, is_train=True
):
    plot_num = min(orig_audio.shape[0], plot_num)
    orig_audio = orig_audio.detach().cpu()
    resyn_audio = resyn_audio.detach().cpu()
    if orig_audio.ndim == 3:
        orig_audio = orig_audio[:, 0, :]
    if resyn_audio.ndim == 3:
        resyn_audio = resyn_audio[:, 0, :]
    for j in range(plot_num):
        if is_train or i == 0:
            # validation example don't change and don't need to be saved every time
            writer.add_audio(
                "{0}_orig/{1}".format(name, j),
                orig_audio[j],
                i,
                sample_rate=sr,
            )
        if i > 0:  # stop saving network output at step 0 (mostly noise)
            writer.add_audio(
                "{0}_resyn/{1}".format(name, j),
                resyn_audio[j],
                i,
                sample_rate=sr,
            )
    fig = plot_recons(
        orig_audio.detach().cpu().numpy(),
        resyn_audio.detach().cpu().numpy(),
        "",
        sr=sr,
        num=plot_num,
        save=False,
    )
    writer.add_figure("plot_recon_{0}".format(name), fig, i)


class AudioLogger(Callback):
    def __init__(self, batch_frequency=5000, sr=16000):
        super().__init__()
        self.batch_freq = batch_frequency
        self.sr = sr

    @rank_zero_only
    def log_local(self, writer, name, current_epoch, orig_audio, resyn_audio):
        save_to_board(
            current_epoch,
            name,
            writer,
            orig_audio,
            resyn_audio,
            plot_num=4,
            sr=self.sr,
            is_train=(name == "train"),
        )

    def log_audio(self, pl_module, batch, name="train"):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        # get audio
        with torch.no_grad():
            outputs = pl_module(batch)
        resyn_audio = torch.clamp(outputs["audio"].detach().cpu(), -1, 1)
        orig_audio = torch.clamp(batch["audio"].detach().cpu(), -1, 1)
        self.log_local(
            pl_module.logger.experiment,
            name,
            pl_module.current_epoch,
            orig_audio,
            resyn_audio,
        )
        if is_train:
            pl_module.train()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            self.log_audio(pl_module, batch, name="val_" + str(dataloader_idx))


class XferLogger(Callback):
    def __init__(self, sounds_dir, sr=24000, feat_proc=None, bkgd_n_batch=8, fad=False):
        """
        Load ood sounds and see the reconstructed (transferred) result
        Also calculate FAD of xferred results
        Args:
            sounds_dir (str): folder containing sounds to test xfer from
            sr (int, optional): sampling rate. Defaults to 24000.
        """
        super().__init__()
        self.sounds_dir = sounds_dir
        self.sound_paths = find_audio_files(sounds_dir)
        self.sr = sr
        self.sounds = []
        max_len = 0
        for s in self.sound_paths:
            audio, orig_sr = load_audio_file(s, mono=True)
            resamp_audio = resample(audio, orig_sr, self.sr)[0]
            self.sounds.append(resamp_audio)
            if resamp_audio.shape[-1] > max_len:
                max_len = resamp_audio.shape[-1]
        sound_tensor = torch.stack(
            [F.pad(ra, (0, max_len - ra.shape[-1])) for ra in self.sounds]
        )
        self.feat_proc = feat_proc
        feats = [feat_proc(st, self.sr) for st in sound_tensor]
        self.example_data = default_collate(feats)
        self.example_data.update({"audio": sound_tensor[:, None, :]})

    @rank_zero_only
    def log_local(self, writer, name, current_epoch, orig_audio, xfer_audio):
        save_to_board(
            current_epoch,
            name,
            writer,
            orig_audio,
            xfer_audio,
            plot_num=len(self.sounds),
            sr=self.sr,
            is_train=False,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        dev = pl_module.device
        pl_module.to("cpu")
        data = {k: v.to(pl_module.device) for k, v in self.example_data.items()}
        with torch.no_grad():
            output = pl_module(data)
        pl_module.to(dev)
        x = torch.clamp(data["audio"].detach().cpu(), min=-1, max=1)
        xfer_res = torch.clamp(output["audio"].detach().cpu(), min=-1, max=1)
        self.log_local(
            pl_module.logger.experiment,
            "val-xfer",
            pl_module.current_epoch,
            x,
            xfer_res,
        )


class CkptEveryNSteps(Callback):
    def __init__(self, every_n, dl_size):
        super().__init__()
        self.every_n_epoch = max(1, every_n // dl_size)

    def on_validation_epoch_end(self, trainer, pl_module):
        cur_epoch = pl_module.current_epoch
        if (cur_epoch + 1) % self.every_n_epoch == 0:
            trainer.save_checkpoint(f"ckpts/step_{pl_module.global_step}.ckpt")
