# NoiseBandNet Neutone

Implementation of **NoiseBandNet** for [Neutone](https://neutone.space/).

NoiseBandNet is a method for controllable time-varying neural synthesis of sound effects using filterbanks. This repository allows you to train NoiseBandNet models on custom audio data and export them for use with the Neutone SDK/Plugin.

## Installation

This project is managed with `uv`.

```bash
# Install package and dependencies
uv sync
```

## Usage

### 1. Training

To train a model on your own audio data, use the `train` command. You can override configuration parameters directly from the command line using Hydra syntax.

**Basic Usage:**

```bash
uv run train name=my_experiment data.raw_path=/path/to/your/audio.wav
```

**Argument List:**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `name` | Name of the experiment (creates logs in `logs/<name>`) | `null` |
| `data.raw_path` | Path to the training audio file (wav/mp3) | `null` |
| `trainer.max_steps` | Total number of training steps | `20000` |
| `batch_size` | Batch size for training | `8` |
| `sample_rate` | Audio sample rate | `48000` |
| `num_workers` | Number of CPU workers for data loading | `8` |
| `trainer.accelerator` | Compute to use during training | `gpu` |

**Example with more options:**

```bash
uv run train \
    name=robot_sfx \
    data.raw_path=assets/robot_sound.wav \
    trainer.max_steps=50000 \
    batch_size=16
```

If training on a local computer, use `trainer.accelerator=cpu`

### 2. Exporting

Once training is complete, you can export your model to a format compatible with the Neutone SDK. The export script wraps the model with Neutone-specific parameters (Brightness, Frequency Chaos, Frequency Tilt).

**Basic Usage:**

```bash
export-neutone {path to checkpoint} -o "{output directory}" -n "{name}" -a "{authors}" -s "{short description}" -l "{long description}" -v "0.0.1"
```

**Arguments:**

| Argument | Flag | Description |
| :--- | :--- | :--- |
| `ckpt` | (Positional) | Path to the PyTorch Lightning checkpoint (`.ckpt` file) |
| `--name` | `-n` | Name of the model (displayed in Neutone) |
| `--authors` | `-a` | Author(s) of the model |
| `--short_description` | `-s` | Brief summary of the model |
| `--long_description` | `-l` | Detailed description of the model |
| `--version` | `-v` | Model version string (e.g., "1.0") |
| `--output` | `-o` | Output directory for the exported model |

**Example:**

```bash
export-neutone \
    logs/robot_sfx/version_0/checkpoints/last.ckpt \
    --name "Robot SFX" \
    --authors "Jane Doe" \
    --short_description "A sci-fi robot texture generator." \
    --output exports/robot_sfx_v1
```

## Project Structure

- **`noisebandnet/`**: Core source code.
  - **`configs/`**: Hydra configuration files (`config.yaml`, `ae/`, `data/`, etc.).
  - **`ddsp/`**: DSP modules, model architecture, and loss functions.
  - **`scripts/`**: Entry points for training and exporting (`train.py`, `export.py`).
- **`logs/`**: Training logs and checkpoints (created at runtime).

## Collab
[Notebook](https://colab.research.google.com/drive/1KJij2CqhLf7ac6aljMckFL71WJrCNg66?usp=sharing)

## Credits

Based on the paper:
*Barahona-Ríos, A., & Collins, T. (2023). NoiseBandNet: Controllable Time-Varying Neural Synthesis of Sound Effects Using Filterbanks. arXiv preprint arXiv:2307.08007.*

Implementation by Naotake Masuda and Benjie Genchel
