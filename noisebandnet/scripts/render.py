import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torchaudio
from neutone_sdk.audio import AudioSample, render_audio_sample
from neutone_sdk.constants import MAX_N_PARAMS
from neutone_sdk.utils import load_neutone_model
from torch import nn


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def parse_param_override(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid parameter override '{value}'. Expected NAME=VALUE."
        )

    name, raw_value = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(
            f"Invalid parameter override '{value}'. Parameter name cannot be empty."
        )

    try:
        parsed_value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{raw_value}' for parameter '{name}'."
        ) from exc

    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"Value for parameter '{name}' must be between 0.0 and 1.0."
        )

    return name, parsed_value


def build_params_tensor(
    metadata: dict, overrides: list[tuple[str, float]]
) -> torch.Tensor:
    neutone_params = metadata.get("neutone_parameters", {})
    ordered_params = sorted(neutone_params.items())

    values = {f"p{idx + 1}": 0.0 for idx in range(MAX_N_PARAMS)}
    alias_to_key = {}
    for key, spec in ordered_params:
        values[key] = float(spec["default_value"])
        alias_to_key[key.lower()] = key
        alias_to_key[spec["name"].lower()] = key

    for raw_name, value in overrides:
        lookup_name = raw_name.lower()
        if lookup_name not in alias_to_key:
            valid_names = ", ".join(
                [spec["name"] for _, spec in ordered_params]
                + [key for key, _ in ordered_params]
            )
            raise ValueError(
                f"Unknown parameter '{raw_name}'. Available parameters: {valid_names}"
            )
        values[alias_to_key[lookup_name]] = value

    return torch.tensor(
        [values[f"p{idx + 1}"] for idx in range(MAX_N_PARAMS)],
        dtype=torch.float32,
    )


def log_neutone_parameters(metadata: dict) -> None:
    neutone_params = metadata.get("neutone_parameters", {})
    ordered_params = sorted(neutone_params.items())
    log.info("Neutone parameters:")
    for key, spec in ordered_params:
        log.info(
            "  %s (%s): default=%s, type=%s",
            key,
            spec["name"],
            spec["default_value"],
            spec["type"],
        )
    if len(ordered_params) < MAX_N_PARAMS:
        for idx in range(len(ordered_params) + 1, MAX_N_PARAMS + 1):
            log.info("  p%s: unused/reserved slot", idx)


def render_realtime_with_optional_params(
    model,
    input_sample: AudioSample,
    params: torch.Tensor | None,
    output_sr: int,
) -> AudioSample:
    preferred_sr = (
        model.get_native_sample_rates()[0]
        if model.get_native_sample_rates()
        else input_sample.sr
    )
    buffer_size = (
        model.get_native_buffer_sizes()[0] if model.get_native_buffer_sizes() else 512
    )

    model.set_daw_sample_rate_and_buffer_size(
        preferred_sr,
        buffer_size,
        preferred_sr,
        buffer_size,
    )

    audio = input_sample.audio
    if input_sample.sr != preferred_sr:
        audio = torchaudio.transforms.Resample(input_sample.sr, preferred_sr)(audio)

    if model.is_input_mono() and not input_sample.is_mono():
        audio = torch.mean(audio, dim=0, keepdim=True)
    elif not model.is_input_mono() and input_sample.is_mono():
        audio = audio.repeat(2, 1)

    audio_len = audio.size(1)
    padding_amount = math.ceil(audio_len / buffer_size) * buffer_size - audio_len
    padded_audio = nn.functional.pad(audio, [0, padding_amount])
    audio_chunks = padded_audio.split(buffer_size, dim=1)

    if params is None:
        out_chunks = [
            model.forward(audio_chunk, None).clone() for audio_chunk in audio_chunks
        ]
    else:
        params = params.to(dtype=torch.float32)
        if params.dim() == 1:
            params = params.repeat([audio_len, 1]).T
        else:
            expected_shape = (params.shape[0], input_sample.audio.size(1))
            if params.shape != expected_shape:
                raise ValueError(
                    f"Expected automation parameters to have shape {expected_shape}, got {tuple(params.shape)}"
                )
            params = torchaudio.transforms.Resample(input_sample.sr, preferred_sr)(
                params
            )
            params = torch.clamp(params, 0.0, 1.0)

        padded_params = nn.functional.pad(params, [0, padding_amount], mode="replicate")
        param_chunks = padded_params.split(buffer_size, dim=1)
        out_chunks = [
            model.forward(audio_chunk, param_chunk).clone()
            for audio_chunk, param_chunk in zip(audio_chunks, param_chunks)
        ]

    audio_out = torch.hstack(out_chunks)[:, :audio_len]
    model.reset()

    if preferred_sr != output_sr:
        audio_out = torchaudio.transforms.Resample(preferred_sr, output_sr)(audio_out)

    return AudioSample(audio_out, output_sr)


def default_output_path(model_path: Path, input_path: Path) -> Path:
    suffix = input_path.suffix or ".wav"
    return model_path.parent / f"{input_path.stem}_rendered{suffix}"


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=Path, help="Path to an exported Neutone model (.nm)"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the rendered output audio file",
    )
    parser.add_argument(
        "-p",
        "--param",
        action="append",
        default=[],
        type=parse_param_override,
        help="Override a Neutone parameter with NAME=VALUE, where VALUE is in [0, 1]. Repeat this flag for multiple parameters.",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="Print the available Neutone parameters and exit",
    )
    parser.add_argument(
        "--output-sr",
        type=int,
        default=44100,
        help="Sample rate for the rendered output audio",
    )
    parsed = parser.parse_args(args)

    model_path = parsed.model.expanduser().resolve()

    model, metadata = load_neutone_model(str(model_path))
    if parsed.list_params:
        log_neutone_parameters(metadata)
        return

    if parsed.input is None:
        parser.error("the following arguments are required when rendering: input")

    input_path = parsed.input.expanduser().resolve()
    output_path = (
        parsed.output.expanduser().resolve()
        if parsed.output is not None
        else default_output_path(model_path, input_path)
    )

    params = (
        build_params_tensor(metadata, parsed.param)
        if metadata.get("neutone_parameters")
        else None
    )

    log.info("Loading input audio from %s", input_path)
    input_sample = AudioSample.from_file(str(input_path))

    log.info("Rendering with model %s", model_path)
    if model.realtime:
        rendered = render_realtime_with_optional_params(
            model,
            input_sample,
            params,
            parsed.output_sr,
        )
    else:
        rendered = render_audio_sample(
            model,
            input_sample,
            params=params,
            output_sr=parsed.output_sr,
        )[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), rendered.audio.cpu(), rendered.sr)
    log.info("Saved rendered audio to %s", output_path)


if __name__ == "__main__":
    main()
