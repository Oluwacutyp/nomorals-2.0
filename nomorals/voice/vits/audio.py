"""Numpy audio front end: WAV loading, STFT and mel spectrograms.

Faithful numpy port of upstream ``mel_processing.py`` + the ``TorchSTFT``
class in ``stft.py``. Two deliberate differences from upstream:

* The mel filterbank is built from scratch (Slaney normalisation, the
  librosa 0.9.2 algorithm) instead of calling ``librosa.filters.mel``,
  so the preparation path needs no librosa. The torch side
  (``mel_processing.py``) falls back to this implementation when librosa
  is absent. Values are pinned against the real librosa output in
  ``vits-parity.toml``.
* Spectrogram caches are ``.spec.npz`` files (numpy) instead of
  upstream's ``.spec.pt`` (torch); the torch trainer reads the ``.npz``
  and converts, so one cache serves both halves.
* Durations are read from the WAV header (stdlib ``wave``) instead of
  ``librosa.get_duration``.
"""

import os
import wave

import numpy as np
from scipy.io.wavfile import read as _wav_read
from scipy.io.wavfile import write as _wav_write

MAX_WAV_VALUE = 32768.0


# --------------------------------------------------------------------- WAV I/O
def wav_duration(path):
    """Duration in seconds, from the header only."""
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError):
        sr, data = _wav_read(path)
        return len(data) / float(sr)


def load_wav(path):
    """Return (sampling_rate, float32 samples in [-1, 1])."""
    sr, data = _wav_read(path)
    return sr, data.astype(np.float32)


def load_wav_normalized(path, target_sr, max_wav_value=MAX_WAV_VALUE):
    """Load a WAV, refuse a sample-rate mismatch, return [1, T] float32.

    Mirrors upstream ``data_utils.get_audio`` / ``utils.load_wav_to_torch``
    including the exact error wording.
    """
    sr, data = load_wav(path)
    if sr != target_sr:
        raise ValueError(
            f"{sr} {target_sr} SR doesn't match target {target_sr} SR"
        )
    audio = data / max_wav_value
    return audio[np.newaxis, :]


def write_wav(path, samples, sampling_rate):
    """Write float32 samples in [-1, 1] as 16-bit PCM WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    _wav_write(path, sampling_rate, (clipped * 32767).astype(np.int16))


# ------------------------------------------------------------------ windows
def hann_window(win_length):
    """Periodic Hann window, as ``torch.hann_window`` / scipy fftbins=True."""
    from scipy.signal import get_window

    return get_window("hann", win_length, fftbins=True).astype(np.float32)


# --------------------------------------------------------------------- STFT
def _frame(y, frame_length, hop_length):
    """Frame a 1-D signal into [n_frames, frame_length] (no padding)."""
    n_frames = 1 + (len(y) - frame_length) // hop_length
    if n_frames <= 0:
        raise ValueError(f"signal too short for STFT: {len(y)} samples, frame {frame_length}")
    shape = (n_frames, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)


def stft(y, n_fft, hop_length, win_length, center=True, pad_mode="reflect"):
    """Complex STFT [1 + n_fft//2, n_frames], matching ``torch.stft``.

    ``center=True`` reflect-pads ``n_fft // 2`` samples on both sides first
    (torch's default ``pad_mode='reflect'``); ``center=False`` frames the
    signal as given.
    """
    y = np.asarray(y, dtype=np.float64)
    if center:
        pad = n_fft // 2
        y = np.pad(y, pad, mode=pad_mode)
    window = hann_window(win_length)
    if win_length < n_fft:
        window = np.pad(window, (n_fft - win_length) // 2)
    frames = _frame(np.ascontiguousarray(y), n_fft, hop_length)
    return np.fft.rfft(frames * window, n=n_fft, axis=1).T


def istft(magnitude, phase, n_fft, hop_length, win_length):
    """Inverse STFT matching ``torch.istft`` with ``center=True``.

    Returns ``(n_frames - 1) * hop_length`` samples: the windowed frames
    are overlap-added, divided by the squared-window envelope, and the
    centre padding (``n_fft // 2`` each side) is trimmed.
    """
    spec = magnitude * np.exp(1j * phase)
    window = hann_window(win_length)
    if win_length < n_fft:
        window = np.pad(window, (n_fft - win_length) // 2)
    n_frames = spec.shape[1]
    expected = (n_frames - 1) * hop_length + n_fft
    signal = np.zeros(expected, dtype=np.float64)
    envelope = np.zeros(expected, dtype=np.float64)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1) * window
    for i in range(n_frames):
        start = i * hop_length
        signal[start:start + n_fft] += frames[i]
        envelope[start:start + n_fft] += window ** 2
    nonzero = envelope > 1e-10
    signal[nonzero] /= envelope[nonzero]
    pad = n_fft // 2
    return signal[pad:pad + (n_frames - 1) * hop_length].astype(np.float32)


def spectrogram(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """Magnitude spectrogram matching upstream ``spectrogram_torch``.

    Upstream reflect-pads ``(n_fft - hop_size) / 2`` on each side, runs
    ``torch.stft`` with ``center=False``, and returns
    ``sqrt(|S|^2 + 1e-6)``.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    pad = int((n_fft - hop_size) / 2)
    y = np.pad(y, ((0, 0), (pad, pad)), mode="reflect")
    out = []
    for row in y:
        spec = stft(row, n_fft, hop_size, win_size, center=False)
        out.append(np.sqrt((spec.real ** 2 + spec.imag ** 2) + 1e-6))
    return np.stack(out).astype(np.float32)


# ------------------------------------------------------------------ mel scale
def hz_to_mel(frequencies, htk=False):
    """librosa 0.9.2 ``hz_to_mel`` (Slaney scale by default)."""
    frequencies = np.asanyarray(frequencies)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels = mels.copy()
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels, htk=False):
    """librosa 0.9.2 ``mel_to_hz`` (Slaney scale by default)."""
    mels = np.asanyarray(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs = freqs.copy()
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def slaney_mel_filterbank(sampling_rate, n_fft, num_mels, fmin, fmax, dtype=np.float32):
    """librosa 0.9.2 ``filters.mel`` with the default Slaney normalisation."""
    if fmax is None:
        fmax = float(sampling_rate) / 2
    weights = np.zeros((num_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = np.linspace(0, float(sampling_rate) / 2, int(1 + n_fft // 2))
    mel_f = mel_to_hz(
        np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), num_mels + 2)
    )
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(num_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:num_mels + 2] - mel_f[:num_mels])
    weights *= enorm[:, np.newaxis]
    return weights


# ------------------------------------------------------- dynamic range (log)
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, clip_val, None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def spectral_normalize(magnitudes):
    return dynamic_range_compression(magnitudes)


def spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """Mel spectrogram from a magnitude spectrogram (upstream ``spec_to_mel_torch``)."""
    mel_basis = slaney_mel_filterbank(sampling_rate, n_fft, num_mels, fmin, fmax)
    return spectral_normalize(np.matmul(mel_basis, spec)).astype(np.float32)


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size,
                    fmin, fmax, center=False):
    """Log-mel spectrogram matching upstream ``mel_spectrogram_torch``."""
    spec = spectrogram(y, n_fft, sampling_rate, hop_size, win_size, center=center)
    return spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax)


# ------------------------------------------------------------- spec caching
def spec_cache_path(wav_path):
    return os.path.splitext(wav_path)[0] + ".spec.npz"


def load_or_compute_spec(wav_path, hps_data):
    """Load a cached ``.spec.npz`` or compute + cache the magnitude spec."""
    cache = spec_cache_path(wav_path)
    if os.path.exists(cache):
        return np.load(cache)["spec"]
    audio = load_wav_normalized(wav_path, hps_data.sampling_rate,
                                hps_data.max_wav_value)
    spec = spectrogram(audio, hps_data.filter_length, hps_data.sampling_rate,
                       hps_data.hop_length, hps_data.win_length, center=False)
    spec = np.squeeze(spec, 0)
    np.savez_compressed(cache, spec=spec)
    return spec
