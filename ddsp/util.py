import glob, os, itertools, math
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import warnings

EXTENSIONS = ["mp3", "MP3", "wav", "WAV", "aif", "AIF", "flac", "FLAC"]


def find_audio_files(root_dir):
    audio_files = [
        glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True)
        for ext in EXTENSIONS
    ]
    audio_files = sorted(list(itertools.chain(*audio_files)))
    return audio_files


def load_audio_file(filepath, mono=True):
    """
    Args:
        filepath (str): path of sound file to load
        mono (bool, optional): Force sound to be mono. Defaults to True.

    Returns:
        (torch.Tensor, int): Audio tensor (channels, n_samples) and sample rate
    """
    try:
        audio, orig_sr = torchaudio.load(filepath)
    except RuntimeError:
        warnings.warn("Falling back to sndfile because torchaudio loading failed.")
        import soundfile

        audio, orig_sr = soundfile.read(filepath)
        audio = torch.from_numpy(audio)
        if audio.ndim == 1:
            audio = audio[None, :]
        else:
            audio = audio.permute(1, 0)  # channel, time
    if mono:
        audio = audio.mean(dim=0, keepdim=True)  # force mono
    return audio.float(), orig_sr


def pad_or_crop_to_length(x: torch.Tensor, length: int):
    remain = length - x.shape[-1]
    if remain < 0:  # crop
        x = x[..., :length]
    elif remain > 0:  # pad
        x = F.pad(x, (0, remain))
    return x


def center_pad(x: torch.Tensor, win_size: int, hop_size: int, pad_last: bool = True):
    """
    pad so that the k-th window is centered around timestep (k-1)*h
    (or almost because window sizes are even).
    pad_last=True pads end to calculate one extra window
    so that last frame is centered around t > L.
    This seems necessary for align_corners=True
    This doesn't matter if x.shape[-1] % hop_size=0
    """
    pad_left = win_size // 2
    audio_len = x.shape[-1]
    if pad_last and audio_len % hop_size > 0:
        n_wins = math.ceil(audio_len / hop_size)
        pad_right = n_wins * hop_size + win_size // 2 - audio_len
    else:
        pad_right = win_size // 2
    return F.pad(x, (pad_left, pad_right))


def slice_windows(
    signal: torch.Tensor,
    frame_size: int,
    hop_size: int,
    window: str = "none",
    pad: bool = True,
):
    """
    slice signal into overlapping frames
    pads end if pad==True and (l_x - frame_size) % hop_size != 0
    Args:
        signal: [batch, n_samples]
        frame_size (int): size of frames
        hop_size (int): size between frames
    Returns:
        [batch, n_frames, frame_size]
    """
    _batch_dim, l_x = signal.shape
    remainder = (l_x - frame_size) % hop_size
    if pad:
        pad_len = 0 if (remainder == 0) else hop_size - remainder
        signal = F.pad(signal, (0, pad_len), "constant")
    signal = signal[:, None, None, :]  # adding dummy channel/height
    frames = F.unfold(
        signal, (1, frame_size), stride=(1, hop_size)
    )  # batch, frame_size, n_frames
    frames = frames.permute(0, 2, 1)  # batch, n_frames, frame_size
    if window == "hamming":
        win = torch.hamming_window(frame_size)[None, None, :].to(frames.device)
        frames = frames * win
    return frames


def log_eps(x: torch.Tensor, eps: float = 1e-4):
    return torch.log(x + eps)


def exp_scale(
    x: torch.Tensor,
    log_exponent: float = 3.0,
    max_value: float = 2.0,
    threshold: float = 1e-7,
):
    return max_value * x**log_exponent + threshold


def exp_sigmoid(
    x: torch.Tensor,
    exponent: float = 10.0,
    max_value: float = 2.0,
    threshold: float = 1e-5,
):
    """Exponentiated Sigmoid pointwise nonlinearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

    Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
    factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
        pushed to 0.

    Returns:
        A tensor with pointwise nonlinearity applied.
    """
    return max_value * torch.sigmoid(x) ** math.log(exponent) + threshold
