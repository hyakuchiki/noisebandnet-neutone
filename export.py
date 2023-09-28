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
from ddsp.util import load_audio_file
from ddsp.stream import switch_streaming_mode


class NBNStreaming(nn.Module):
    def __init__(self, feat_proc, ae, sr):
        super().__init__()
        self.feat_proc = feat_proc
        self.ae = ae
        self.sr = sr

    def forward(self, audio):
        feats = self.feat_proc(audio, self.sr)
        feats.update({"audio": audio})
        out = self.ae(feats)
        return out["audio"]


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
        return []

    def is_input_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def is_output_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def get_native_sample_rates(self) -> List[int]:
        return [48000]  # <-Set to model sr during training

    def get_native_buffer_sizes(self) -> List[int]:
        return [1200]

    def calc_model_delay_samples(self) -> int:
        # model latency should also be added if non-causal
        return self.pre_filter.delay

    def set_model_sample_rate_and_buffer_size(
        self, sample_rate: int, n_samples: int
    ) -> bool:
        # Set prefilter samplerate to current sample rate
        self.pre_filter.set_parameters(sample_rate=sample_rate)
        return True

    def get_citation(self) -> str:
        return """Barahona-Ríos, A., & Collins, T. (2023).  NoiseBandNet: Controllable Time-Varying Neural Synthesis of Sound Effects Using Filterbanks. arXiv preprint arXiv:2307.08007."""

    @torch.no_grad()
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # Apply pre-filter
        # x = self.pre_filter(x)
        ## parameters edit the latent variable
        out = self.model(x)
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
