import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from neutone_sdk import (
    ContinuousNeutoneParameter,
    NeutoneParameter,
    WaveformToWaveformBase,
)
from neutone_sdk.utils import save_neutone_model
from omegaconf import OmegaConf
from torch import Tensor, nn

from noisebandnet.ddsp.model import AutoEncoderModel
from noisebandnet.ddsp.stream import switch_streaming_mode


class NBNStreaming(nn.Module):
    def __init__(self, feat_proc, ae, sr):
        super().__init__()
        self.feat_proc = feat_proc
        self.ae = ae
        self.sr = sr
        self.rand_amp_k = 16
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("amp_noise", torch.zeros(1, 1, 1, dtype=torch.long))

    def forward(self, audio, centroid_shift, rand_amount, tilt_amount):
        feats = self.feat_proc(audio, self.sr)
        feats.update({"audio": audio})
        # determine number of frames from any time-feature (flexible for MFCC-only models)
        # initialize as int to keep TorchScript types consistent
        n_frames = 1
        for v in feats.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                # common feature shape: (batch, n_frames, ...)
                n_frames = int(v.shape[1])
                break
        # shift centroid if present (some exported models use MFCCs and won't have centroid)
        if "centroid" in feats:
            MAX_SHIFT = 48  # semitones
            pshift = (centroid_shift - 0.5) * 2 * MAX_SHIFT  # -24~24
            semishift = torch.round(pshift)
            centroid_mult = 2 ** (semishift / 12)
            feats["centroid"] *= centroid_mult
        # encode
        enc_data = self.ae.encode(feats)
        amps = self.ae.decoder.infer(enc_data["z"])  # amps: batch, n_frames, n_banks
        # randomize amps
        self.count -= n_frames
        N_FRAMES = 30
        if self.count <= 0:  # only change noise every N_FRAMES
            self.count += torch.LongTensor([N_FRAMES], device=amps.device)
            self.amp_noise = torch.rand_like(amps[:, :1, :]) * 20.0 + 0.1
        amps *= torch.clamp(self.amp_noise**rand_amount, 0.1, 10.0)
        # add tilt
        ta = float((tilt_amount - 0.5) * 2)  # -1~1
        amps *= torch.logspace(-ta, ta, amps.shape[-1], device=amps.device)[
            None, None, :
        ]
        audio = self.ae.decoder.synthesize(amps, audio.shape[-1])
        return audio


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NoiseBandNetWrapper(WaveformToWaveformBase):
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "NoiseBandNet.example",
        model_authors: List[str] = ["Author Name"],
        model_desc_short: str = "NoiseBandNet model trained on ...",
        model_desc_long: str = "NoiseBandNet timbre transfer model trained on xxx sounds. Useful for xxx sounds.",
        model_version: str = "1.0",
    ):
        self.model_name = model_name
        self.model_authors = model_authors
        self.model_desc_short = model_desc_short
        self.model_desc_long = model_desc_long
        self.model_version = model_version
        super().__init__(model)

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_authors(self) -> List[str]:
        return self.model_authors

    def get_model_short_description(self) -> str:
        return self.model_desc_short

    def get_model_long_description(self) -> str:
        return self.model_desc_long

    def get_technical_description(self) -> str:
        return "NoiseBandNet proposed by Adrián Barahona-Ríos, Tom Collins"

    def get_technical_links(self) -> Dict[str, str]:
        return {
            # "Paper": "https://arxiv.org/abs/2307.08007",
        }

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "NoiseBandNet"]

    def get_model_version(self) -> str:
        return self.model_version

    def is_experimental(self) -> bool:
        """
        set to True for models in experimental stage
        (status shown on the website)
        """
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            ContinuousNeutoneParameter(
                name="Brightness",
                description="Shift the brightness of input",
                default_value=0.5,
            ),
            ContinuousNeutoneParameter(
                name="Frequency Chaos",
                description="Randomize the amplitude of each frequency band",
                default_value=0.0,
            ),
            ContinuousNeutoneParameter(
                name="Frequency Tilt",
                description="Tilt the amplitude distribution of each frequency band",
                default_value=0.5,
            ),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [48000]  # <-Set to model sr during training

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [960]

    @torch.jit.export
    def get_citation(self) -> str:
        return """Barahona-Ríos, A., & Collins, T. (2023).  NoiseBandNet: Controllable Time-Varying Neural Synthesis of Sound Effects Using Filterbanks. arXiv preprint arXiv:2307.08007."""

    @torch.no_grad()
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # Apply pre-filter
        # x = self.pre_filter(x)
        ## parameters edit the latent variable
        out = self.model(
            x, params["Brightness"], params["Frequency Chaos"], params["Frequency Tilt"]
        )
        out = out.squeeze(1)
        return out


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("-n", "--name", type=str, help="Model name")
    parser.add_argument("-a", "--authors", nargs="+", help="Model authors")
    parser.add_argument(
        "-s", "--short_description", type=str, help="Model short description"
    )
    parser.add_argument(
        "-l", "--long_description", type=str, help="Model long description"
    )
    parser.add_argument("-v", "--version", type=str, help="Model version")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"{os.path.abspath(__file__)}/../exports/test-nm",
    )
    args = parser.parse_args(args)
    full = AutoEncoderModel.load_from_checkpoint(
        args.ckpt,
        map_location="cpu",
        strict=False,
        weights_only=False,
    )
    ae = full.ae.eval()
    switch_streaming_mode(ae)
    ckpt_path = Path(args.ckpt).parents[1]
    conf = OmegaConf.load(ckpt_path / ".hydra/config.yaml")
    proc = hydra.utils.instantiate(conf.data.feat_proc)
    proc = proc.eval()
    if "centroid" in proc.features:
        proc.features["centroid"].spec = torch.jit.trace(
            proc.features["centroid"].spec, torch.randn(1, 48000)
        )
    switch_streaming_mode(proc)
    # join preprocessing and model
    model = NBNStreaming(proc, ae, conf.sample_rate)
    tr_model = torch.jit.script(model)
    wrapper = NoiseBandNetWrapper(
        tr_model,
        model_name=args.name,
        model_authors=args.authors,
        model_desc_short=args.short_description,
        model_desc_long=args.long_description,
        model_version=args.version,
    )
    save_neutone_model(
        wrapper,
        Path(args.output),
        freeze=False,
        dump_samples=True,
        submission=True,
        audio_sample_pairs=None,
    )


if __name__ == "__main__":
    main()
