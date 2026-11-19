"""Numpy mirrors of the pure-math helpers in upstream ``commons.py`` and
``losses.py``.

These exist so the preparation path and the parity tests can exercise the
exact formulas without torch: interspersing blanks, padding masks,
alignment paths, KL divergence and the scalar loss formulas. Every
function is checked against its torch twin in the test battery.
"""

import math

import numpy as np


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    reversed_shape = pad_shape[::-1]
    pad_shape = [item for sublist in reversed_shape for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = int(np.max(length))
    x = np.arange(max_length, dtype=np.int64)
    return x[np.newaxis, :] < np.asarray(length)[:, np.newaxis]


def generate_path(duration, mask):
    """Monotonic alignment path, mirroring ``commons.generate_path``.

    duration: [b, 1, t_x] int; mask: [b, 1, t_y, t_x] float.
    Returns [b, 1, t_y, t_x] float.
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, axis=-1)
    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
    path = path.reshape(b, t_x, t_y)
    padded = np.zeros_like(path)
    padded[:, 1:, :] = path[:, :-1, :]
    path = path - padded
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2) * mask
    return path


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q), mirroring ``commons.kl_divergence``."""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (np.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * np.exp(-2.0 * logs_q)
    return kl


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """Mirroring ``losses.kl_loss``."""
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * np.exp(-2.0 * logs_p)
    kl = np.sum(kl * z_mask)
    return kl / np.sum(z_mask)


def feature_loss(fmap_r, fmap_g):
    """Mirroring ``losses.feature_loss`` (returns loss * 2, as upstream)."""
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g, strict=True):
        for rl, gl in zip(dr, dg, strict=True):
            rl = np.asarray(rl, dtype=np.float64)
            gl = np.asarray(gl, dtype=np.float64)
            if rl.shape[2] > gl.shape[2]:
                rl = rl[:, :, :gl.shape[2]]
            loss += float(np.mean(np.abs(rl - gl)))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=True):
        dr = np.asarray(dr, dtype=np.float64)
        dg = np.asarray(dg, dtype=np.float64)
        r_loss = float(np.mean((1 - dr) ** 2))
        g_loss = float(np.mean(dg ** 2))
        loss += r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        dg = np.asarray(dg, dtype=np.float64)
        disc_loss = float(np.mean((1 - dg) ** 2))
        gen_losses.append(disc_loss)
        loss += disc_loss
    return loss, gen_losses


def slice_segments(x, ids_str, segment_size=4):
    """Mirroring ``commons.slice_segments`` on [b, d, t] arrays."""
    b, d, t = x.shape
    ret = np.zeros((b, d, segment_size), dtype=x.dtype)
    for i in range(b):
        idx_str = int(ids_str[i])
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4, rng=None):
    """Mirroring ``commons.rand_slice_segments`` (rng seeds the choice)."""
    b, d, t = x.shape
    if x_lengths is None:
        x_lengths = np.full(b, t)
    ids_str_max = np.asarray(x_lengths) - segment_size + 1
    rng = np.random.default_rng() if rng is None else rng
    ids_str = (rng.random(b) * ids_str_max).astype(np.int64)
    return slice_segments(x, ids_str, segment_size), ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Mirroring ``commons.get_timing_signal_1d``."""
    position = np.arange(length, dtype=np.float32)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)
    )
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales, dtype=np.float32) * -log_timescale_increment
    )
    scaled_time = position[np.newaxis, :] * inv_timescales[:, np.newaxis]
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], 0)
    signal = np.pad(signal, [(0, 0), (0, channels % 2)])
    return signal.reshape(1, channels, length)
