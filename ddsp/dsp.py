import math
import torch
import torch.nn as nn


def resample_frames(
    inputs: torch.Tensor,
    scale_factor: float,
    mode: str = "linear",
):
    """interpolate signals with a value each frame into signal with a value each timestep
    [n_frames] -> [n_timesteps]

    Args:
        inputs (torch.Tensor): [n_frames], [batch_size, n_frames], [batch_size, n_frames, channels]
        n_timesteps (int):
        mode (str): interpolation mode
    Returns:
        torch.Tensor
        [n_timesteps], [batch_size, n_timesteps], or [batch_size, n_timesteps, channels?]
    """
    orig_shape = inputs.shape

    if len(orig_shape) == 1:
        inputs = inputs.unsqueeze(0)  # [dummy_batch, n_frames]
        inputs = inputs.unsqueeze(1)  # [dummy_batch, dummy_channel, n_frames]
    if len(orig_shape) == 2:
        inputs = inputs.unsqueeze(1)  # [batch, dummy_channel, n_frames]
    if len(orig_shape) == 3:
        inputs = inputs.permute(0, 2, 1)  # # [batch, channels, n_frames]

    # interpolate expects [batch_size, channel, (depth, height,) width]

    outputs = nn.functional.interpolate(
        inputs, scale_factor=scale_factor, mode=mode, align_corners=True
    )

    if len(orig_shape) == 1:
        outputs = outputs.squeeze(1)  # get rid of dummy channel
        outputs = outputs.squeeze(0)  # [n_timesteps]
    if len(orig_shape) == 2:
        outputs = outputs.squeeze(
            1
        )  # get rid of dummy channel # [n_frames, n_timesteps]
    if len(orig_shape) == 3:
        outputs = outputs.permute(0, 2, 1)  # [batch, n_frames, channels]

    return outputs


def sin_synthesis(
    frequencies, amplitudes, n_samples=64000, sample_rate=16000, fm_signal=None
):
    """wavetable synthesis similar to the one in DDSP

    Args:
        frequencies (torch.Tensor): Frame-wise frequency in Hz [batch_size, n_frames, 1]
        amplitudes (torch.Tensor): Frame-wise amplitude envelope [batch_size, n_frames, 1]
        fm_signal: audio rate signal for FM (phase modulation)
    Returns:
        audio: Sin at the frequency and amplitude of the inputs. Shape[batch_size, n_samples]
    """
    # upsample frequency/amplitude envelope
    batch_size = frequencies.shape[0]

    if len(frequencies.shape) == 3:
        frequencies = frequencies.squeeze(-1)  # [batch_size, n_frames]
    if len(amplitudes.shape) == 3:
        amplitudes = amplitudes.squeeze(-1)  # [batch_size, n_frames]

    frq_env = resample_frames(
        frequencies, n_samples // frequencies.shape[1]
    )  # [batch_size, n_samples]
    amp_env = resample_frames(
        amplitudes, n_samples // amplitudes.shape[1]
    )  # TODO should really use windows to avoid jaggy envelope

    phase_velocity = frq_env / float(sample_rate)
    phase = torch.cumsum(phase_velocity, 1)[:, :-1] % 1.0  # [batch_size, n_samples]
    phase = torch.cat(
        [torch.zeros(batch_size, 1).to(phase.device), phase], dim=1
    )  # exclusive cumsum starting at 0
    phase_rad = phase * 2 * math.pi
    if fm_signal is not None:
        audio = torch.sin(phase_rad + fm_signal)
    else:
        audio = torch.sin(phase_rad)
    audio *= amp_env
    return audio


def get_harmonic_frequencies(frequencies: torch.Tensor, n_harmonics: int):
    """Create integer multiples of the fundamental frequency.

    Args:
    frequencies: Fundamental frequencies (Hz). Shape [batch_size, time, 1].
    n_harmonics: Number of harmonics.

    Returns:
    harmonic_frequencies: Oscillator frequencies (Hz).
        Shape [batch_size, time, n_harmonics].
    """
    f_ratios = torch.linspace(1.0, float(n_harmonics), int(n_harmonics)).to(
        frequencies.device
    )
    f_ratios = f_ratios[None, None, :]
    harmonic_frequencies = frequencies * f_ratios
    return harmonic_frequencies


def remove_above_nyquist(
    frequency_envelopes: torch.Tensor,
    amplitude_envelopes: torch.Tensor,
    sample_rate: int = 16000,
):
    """Set amplitudes for oscillators above nyquist to 0.

    Args:
        frequency_envelopes: Sample/frame-wise oscillator frequencies (Hz). Shape
        [batch_size, n_samples(or n_frames), n_sinusoids].
        amplitude_envelopes: Sample/frame-wise oscillator amplitude. Shape [batch_size,
        n_samples(or n_frames), n_sinusoids].
        sample_rate: Sample rate in samples per a second.

    Returns:
        amplitude_envelopes: Sample-wise filtered oscillator amplitude.
        Shape [batch_size, n_samples, n_sinusoids].
    """
    if amplitude_envelopes.shape[1] != frequency_envelopes.shape[1]:
        frequency_envelopes = resample_frames(
            frequency_envelopes, amplitude_envelopes.shape[1]
        )

    amplitude_envelopes = torch.where(
        torch.ge(frequency_envelopes, sample_rate / 2.0),
        torch.zeros_like(amplitude_envelopes),
        amplitude_envelopes,
    )
    return amplitude_envelopes


def harmonic_synthesis(
    frequencies: torch.Tensor,
    amplitudes: torch.Tensor,
    harmonic_distribution: torch.Tensor,
    n_samples: int = 64000,
    sample_rate: int = 16000,
):
    """Generate audio from frame-wise monophonic harmonic oscillator bank.

    Args:
        frequencies: Frame-wise fundamental frequency in Hz. Shape [batch_size, n_frame, 1].
        amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size, n_frames, 1].
        harmonic_distribution: Harmonic amplitude variations, ranged zero to one. Total amplitude of a harmonic is equal to (amplitudes * harmonic_distribution). Shape [batch_size, n_frames, n_harmonics].
        n_samples: Total length of output audio. Interpolates and crops to this.
        sample_rate: Sample rate.

    Returns:
        audio: Output audio. Shape [batch_size, n_samples]
    """

    n_harmonics = harmonic_distribution.shape[-1]

    # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
    harmonic_frequencies = get_harmonic_frequencies(frequencies, n_harmonics)

    # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
    if harmonic_distribution is not None:
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    # Create sample-wise envelopes.
    frequency_envelopes = resample_frames(harmonic_frequencies, n_samples)  # cycles/sec
    # window resampling has not been implemented yet
    amplitude_envelopes = resample_frames(harmonic_amplitudes, n_samples)
    # Synthesize from harmonics [batch_size, n_samples].
    audio = oscillator_bank(
        frequency_envelopes, amplitude_envelopes, sample_rate=sample_rate
    )
    return audio


def oscillator_bank_stream(
    frequency_envelopes: torch.Tensor,
    amplitude_envelopes: torch.Tensor,
    init_phase: torch.Tensor,
    sample_rate: int = 16000,
    sum_sinusoids: bool = True,
):
    # Don't exceed Nyquist.
    amplitude_envelopes = remove_above_nyquist(
        frequency_envelopes, amplitude_envelopes, sample_rate
    )

    # Change Hz to radians per sample.
    omegas = frequency_envelopes * (2.0 * math.pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample

    # Accumulate phase and synthesize.
    phases = torch.cumsum(omegas, 1)
    # add initial phase
    phases = phases + init_phase
    last_phase = phases[:, -1, :]
    output = torch.sin(phases)
    output = amplitude_envelopes * output  # [batch_size, n_samples, n_sinusoids]
    if sum_sinusoids:
        output = torch.sum(output, dim=-1)  # [batch_size, n_samples]
    return output, last_phase


def oscillator_bank(
    frequency_envelopes: torch.Tensor,
    amplitude_envelopes: torch.Tensor,
    sample_rate: int = 16000,
    sum_sinusoids: bool = True,
):
    """Generates audio from sample-wise frequencies for a bank of oscillators.

    Args:
        frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape [batch_size, n_samples, n_sinusoids].
        amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size, n_samples, n_sinusoids].
        sample_rate: Sample rate in samples per a second.
        sum_sinusoids: Add up audio from all the sinusoids.
    Returns:
        wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if sum_sinusoids=False, else shape is [batch_size, n_samples].
    """

    # Don't exceed Nyquist.
    amplitude_envelopes = remove_above_nyquist(
        frequency_envelopes, amplitude_envelopes, sample_rate
    )

    # Change Hz to radians per sample.
    omegas = frequency_envelopes * (2.0 * math.pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample

    # Accumulate phase and synthesize.
    phases = torch.cumsum(omegas, 1)
    output = torch.sin(phases)
    output = amplitude_envelopes * output  # [batch_size, n_samples, n_sinusoids]
    if sum_sinusoids:
        output = torch.sum(output, dim=-1)  # [batch_size, n_samples]
    return output


def get_fft_size(frame_size: int, ir_size: int) -> int:
    """Calculate final size for efficient FFT. power of 2
    fft size should be greater than frame_size + ir_size
    Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.

    Returns:
    fft_size: Size for efficient FFT.
    """
    convolved_frame_size = ir_size + frame_size - 1
    # Next power of 2.
    return int(2 ** math.ceil(math.log(convolved_frame_size, 2)))


def frame_signal(signal: torch.Tensor, frame_size: int):
    """
    cut signal into nonoverlapping frames
    Args:
        signal: [batch, n_samples]
        frame_size: int
    Returns:
        [batch, n_frames, frame_size]
    """
    signal_len = signal.shape[-1]
    padding = (frame_size - (signal_len % frame_size)) % frame_size
    signal = torch.nn.functional.pad(signal, (0, padding), "constant", 0.0)
    frames = torch.split(signal.unsqueeze(1), frame_size, dim=-1)
    return torch.cat(frames, dim=1)


def variable_delay(phase: torch.Tensor, audio: torch.Tensor, buf_size: int):
    """delay with variable length

    Args:
        phase (torch.Tensor): 0~1 0: no delay 1: delay=max_length (batch, n_samples)
        audio (torch.Tensor): audio signal (batch, n_samples)
        buf_size (int)    : buffer size in samples = max delay length

    Returns:
        torch.Tensor: delayed audio (batch, n_samples)
    """
    batch_size, n_samples = audio.shape
    audio_4d = audio[:, None, None, :]  # (B, C=1, H=1, W=n_samples)
    delay_ratio = buf_size * 2 / n_samples
    grid_x = torch.linspace(-1, 1, n_samples, device=audio.device)[None, :]
    grid_x = grid_x - delay_ratio + delay_ratio * phase  # B, W=n_samples
    grid_x = grid_x[:, None, :, None]  # B, H=1, W=n_samples, 1
    grid_y = torch.zeros(
        batch_size, 1, n_samples, 1, device=audio.device
    )  # # B, H=1, W=n_samples, 1
    grid = torch.cat([grid_x, grid_y], dim=-1)
    output = torch.nn.functional.grid_sample(audio_4d, grid, align_corners=True)
    # shape: (B, C=1, H=1, W)
    output = output.squeeze(2).squeeze(1)
    return output


def overlap_and_add(signal: torch.Tensor, frame_step: int):
    """overlap-add signals ported from tf.signals

    Args:
        signal (torch.Tensor): (batch_size, frames, frame_length)
        frame_step (int): size of overlap offset
    Returns:
    A `Tensor` with shape `[..., output_size]`
    """
    batch_size = signal.shape[0]
    n_frames = signal.shape[1]
    frame_length = signal.shape[2]

    output_length = frame_length + frame_step * (n_frames - 1)
    if frame_length == frame_step:
        return signal.flatten(-2, -1)
    segments = -(-frame_length // frame_step)
    pad_width = segments * frame_step - frame_length
    # The following code is documented using this example:
    #
    # frame_step = 2
    # signal.shape = (3, 5)
    # a b c d e
    # f g h i j
    # k l m n o

    # Pad the frame_length dimension to a multiple of the frame step.
    # Pad the frames dimension by `segments` so that signal.shape = (6, 6)
    # a b c d e 0
    # f g h i j 0
    # k l m n o 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    signal = nn.functional.pad(signal, (0, pad_width, 0, segments))
    # Reshape so that signal.shape = (6, 3, 2)
    # ab cd e0
    # fg hi j0
    # kl mn o0
    # 00 00 00
    # 00 00 00
    # 00 00 00
    signal = signal.reshape(batch_size, n_frames + segments, segments, frame_step)
    # Transpose dimensions so that signal.shape = (3, 6, 2)
    # ab fg kl 00 00 00
    # cd hi mn 00 00 00
    # e0 j0 o0 00 00 00
    signal = signal.transpose(-2, -3)
    # Reshape so that signal.shape = (18, 2)
    # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0 00 00 00
    signal = torch.flatten(signal, -3, -2)
    signal.shape
    # Truncate so that signal.shape = (15, 2)
    # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0
    signal = signal[..., : (n_frames + segments - 1) * segments, :]
    # Reshape so that signal.shape = (3, 5, 2)
    # ab fg kl 00 00
    # 00 cd hi mn 00
    # 00 00 e0 j0 o0
    signal = signal.reshape(batch_size, segments, (n_frames + segments - 1), frame_step)
    # Now, reduce over the columns, to achieve the desired sum.
    signal = torch.sum(signal, -3)
    # Flatten the array.
    signal = signal.reshape(batch_size, (n_frames + segments - 1) * frame_step)

    # Truncate to final length.
    signal = signal[..., :output_length]
    return signal


def crop_and_compensate_delay(
    audio: torch.Tensor,
    audio_size: int,
    ir_size: int,
    padding: str,
    delay_compensation: int,
):
    """Copied over from ddsp
    Crop audio output from convolution to compensate for group delay.

    Args:
        audio: Audio after convolution. Tensor of shape [batch, time_steps].
        audio_size: Initial size of the audio before convolution.
        ir_size: Size of the convolving impulse response.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation < 0 it
        defaults to automatically calculating a constant group delay of the
        windowed linear phase filter from frequency_impulse_response().

    Returns:
        Tensor of cropped and shifted audio.

    Raises:
        ValueError: If padding is not either 'valid' or 'same'.
    """
    # Crop the output.
    if padding == "valid":
        crop_size = ir_size + audio_size - 1
    elif padding == "same":
        crop_size = audio_size
    else:
        raise ValueError(
            "Padding must be 'valid' or 'same', instead " "of {}.".format(padding)
        )

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = audio.shape[-1]
    crop = total_size - crop_size
    start = (ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation
    end = crop - start
    return audio[:, start:-end]


def fir_filter(
    audio: torch.Tensor,
    freq_response: torch.Tensor,
    filter_size: int,
    padding: str = "same",
):
    # get IR
    ir = torch.fft.irfft(freq_response, n=filter_size, dim=-1)
    # Compute filter windowed impulse response
    # window_size == filter_size
    filter_window = (
        torch.hann_window(filter_size, dtype=torch.float32)
        .roll(filter_size // 2, -1)
        .to(audio.device)
    )
    h = filter_window[None, None, :] * ir
    filtered = fft_convolve(audio, h, padding=padding)
    return filtered


def fft_convolve(
    audio: torch.Tensor,
    impulse_response: torch.Tensor,
    padding: str = "same",
    delay_compensation: int = -1,
):
    """ported from ddsp (Google, MIT License) to pytorch
    original description below
    Filter audio with frames of time-varying impulse responses.

    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.

    Args:
        audio: Input audio. Tensor of shape [batch, n_samples].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (n_samples). For 'valid' the audio is
        extended to include the tail of the impulse response (n_samples +
        ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation is less
        than 0 it defaults to automatically calculating a constant group delay of
        the windowed linear phase filter from frequency_impulse_response().

    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, n_samples + ir_timesteps - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).

    Raises:
        ValueError: If audio and impulse response have different batch size.
        ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
        number of impulse response frames is on the order of the audio size and
        not a multiple of the audio size.)
    """
    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        impulse_response = impulse_response[:, None, :]
        ir_shape = impulse_response.shape

    # Get shapes of audio and impulse response.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.shape

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError(
            "Batch size of audio ({}) and impulse response ({}) must "
            "be the same.".format(batch_size, batch_size_ir)
        )

    # Cut audio into frames.
    frame_size = math.ceil(audio_size / n_ir_frames)

    audio_frames = frame_signal(audio, frame_size)

    # Check that number of frames match.
    n_audio_frames = audio_frames.shape[1]
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            "Number of Audio frames ({}) and impulse response frames ({}) do not "
            "match. For small hop size = ceil(audio_size / n_ir_frames), "
            "number of impulse response frames must be a multiple of the audio "
            "size.".format(n_audio_frames, n_ir_frames)
        )

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size)

    S = torch.fft.rfft(audio_frames, n=fft_size, dim=-1)  # zeropadded
    H = torch.fft.rfft(impulse_response, n=fft_size, dim=-1)

    # Multiply the FFTs (same as convolution in time).
    # Filter the original audio
    audio_ir_fft = S * H

    # Take the IFFT to resynthesize audio.
    # batch_size, n_frames, fft_size
    audio_frames_out = torch.fft.irfft(audio_ir_fft, n=fft_size, dim=-1)
    audio_out = overlap_and_add(audio_frames_out, frame_size)

    # Crop and shift the output audio.
    return crop_and_compensate_delay(
        audio_out, audio_size, ir_size, padding, delay_compensation
    )
