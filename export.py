import os, argparse, logging
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import OmegaConf
import torch
from torch import Tensor, nn
from torch.nn.utils.weight_norm import WeightNorm
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    render_audio_sample,
)
from neutone_sdk.filters import FIRFilter, FilterType
from neutone_sdk.utils import save_neutone_model

from ddsp.model import AutoEncoderModel
from ddsp.util import load_audio_file, exp_scale
from ddsp.stream import switch_streaming_mode


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
        # shift centroid
        MAX_SHIFT = 48  # semitones
        pshift = (centroid_shift - 0.5) * 2 * MAX_SHIFT  # -24~24
        semishift = torch.round(pshift)
        centroid_mult = 2 ** (semishift / 12)
        feats["centroid"] *= centroid_mult
        # encode
        enc_data = self.ae.encode(feats)
        amps = self.ae.decoder.infer(enc_data["z"])  # amps: batch, n_frames, n_banks
        # randomize amps
        n_frames = feats["centroid"].shape[1]
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
    def get_model_name(self) -> str:
        return "NoiseBandNet.example"

    def get_model_authors(self) -> List[str]:
        return ["Author Name"]

    def get_model_short_description(self) -> str:
        return "NoiseBandNet model trained on ..."

    def get_model_long_description(self) -> str:
        return "NoiseBandNet timbre transfer model trained on xxx sounds. Useful for xxx sounds."  # <-EDIT THIS

    def get_technical_description(self) -> str:
        return "NoiseBandNet proposed by Adrián Barahona-Ríos, Tom Collins"

    def get_technical_links(self) -> Dict[str, str]:
        return {
            # "Paper": "https://arxiv.org/abs/2307.08007",
        }

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "NoiseBandNet"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        """
        set to True for models in experimental stage
        (status shown on the website)
        """
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter(
                name="Brightness",
                description="Shift the brightness of input",
                default_value=0.5,
            ),
            NeutoneParameter(
                name="Frequency Chaos",
                description="Randomize the amplitude of each frequency band",
                default_value=0.0,
            ),
            NeutoneParameter(
                name="Frequency Tilt",
                description="Tilt the amplitude distribution of each frequency band",
                default_value=0.5,
            ),
        ]

    def is_input_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def is_output_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def get_native_sample_rates(self) -> List[int]:
        return [48000]  # <-Set to model sr during training

    def get_native_buffer_sizes(self) -> List[int]:
        return [1200]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("-o", "--output", type=str, default="exports/test-nm")
    args = parser.parse_args()
    full = AutoEncoderModel.load_from_checkpoint(
        args.ckpt,
        map_location="cpu",
        strict=False,
    )
    ae = full.ae.eval()
    switch_streaming_mode(ae)
    ckpt_path = Path(args.ckpt).parents[1]
    conf = OmegaConf.load(ckpt_path / ".hydra/config.yaml")
    proc = hydra.utils.instantiate(conf.data.feat_proc)
    switch_streaming_mode(proc)
    # join preprocessing and model
    model = NBNStreaming(proc, ae, conf.sample_rate)
    tr_model = torch.jit.script(model)
    wrapper = NoiseBandNetWrapper(tr_model)
    save_neutone_model(
        wrapper,
        Path(args.output),
        freeze=False,
        dump_samples=True,
        submission=True,
        audio_sample_pairs=None,
    )
