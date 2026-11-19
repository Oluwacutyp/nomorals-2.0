"""Configuration for the VITS fine-tune port.

``HParams`` is the upstream ``utils.HParams`` (a dict-like attribute bag
that recurses into nested dicts), kept torch-free so configs can be read,
validated and edited anywhere. ``load_config`` reads one of the vendored
configs in ``nomorals/voice/vits/data/configs/`` (or any JSON path) and
validates the fields the training and preparation paths actually read;
upstream never validated them, so a typo silently produced a wrong run.
"""

import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "data", "configs")

# Fields the trainer / preparation path read, with the type each must have.
_REQUIRED = {
    ("train", "segment_size"): int,
    ("train", "batch_size"): int,
    ("train", "epochs"): int,
    ("train", "learning_rate"): float,
    ("train", "betas"): list,
    ("train", "eps"): float,
    ("train", "lr_decay"): float,
    ("train", "c_mel"): (int, float),
    ("train", "c_kl"): (int, float),
    ("train", "log_interval"): int,
    ("train", "eval_interval"): int,
    ("data", "sampling_rate"): int,
    ("data", "filter_length"): int,
    ("data", "hop_length"): int,
    ("data", "win_length"): int,
    ("data", "n_mel_channels"): int,
    # mel_fmin/mel_fmax may be null upstream (ljs_nosdp.json): None
    # means "Nyquist" and slaney_mel_filterbank handles it.
    ("data", "mel_fmin"): (int, float, type(None)),
    ("data", "mel_fmax"): (int, float, type(None)),
    ("data", "max_wav_value"): (int, float),
    ("data", "text_cleaners"): list,
    ("data", "add_blank"): bool,
    ("model", "inter_channels"): int,
    ("model", "hidden_channels"): int,
    ("model", "filter_channels"): int,
    ("model", "n_heads"): int,
    ("model", "n_layers"): int,
    ("model", "kernel_size"): int,
    ("model", "p_dropout"): (int, float),
    ("model", "resblock"): str,
    ("model", "resblock_kernel_sizes"): list,
    ("model", "resblock_dilation_sizes"): list,
    ("model", "upsample_rates"): list,
    ("model", "upsample_initial_channel"): int,
    ("model", "upsample_kernel_sizes"): list,
    # NOT required here: grad_acc_steps (trainer defaults to 1),
    # save_interval (trainer defaults to 10000), gen_istft_n_fft /
    # gen_istft_hop_size (model defaults 16/4), and the bert/moji sizes
    # (only the tmbert configs carry them; the trainer refuses a plain
    # config with a named error instead of a cryptic TypeError).
}


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) is dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def _check_type(path, value, want):
    if not isinstance(value, want):
        raise TypeError(
            "config {} must be {}, got {!r}".format(".".join(path), getattr(want, "__name__", want), value)
        )


def validate(hps, path="<config>"):
    """Validate the fields the trainer and preparation path read."""
    for keys, want in _REQUIRED.items():
        node = hps
        for key in keys:
            if key not in node:
                raise ValueError(
                    "config {} is missing {} (needed by the VITS {} path)".format(path, ".".join(keys), keys[0])
                )
            node = node[key]
        _check_type(keys, node, want)
    if hps.data.filter_length < hps.data.win_length:
        raise ValueError(
            f"config {path}: filter_length ({hps.data.filter_length}) must be >= win_length ({hps.data.win_length})"
        )
    if hps.train.segment_size % hps.data.hop_length != 0:
        raise ValueError(
            f"config {path}: segment_size ({hps.train.segment_size}) must be a multiple of hop_length ({hps.data.hop_length})"
        )
    if not hps.data.text_cleaners:
        raise ValueError(f"config {path}: text_cleaners must name at least one cleaner")
    return hps


def load_config(config_path=None, config_name=None, validate_config=True):
    """Load a config by path or by vendored name (``ljs_li44.json`` etc.)."""
    if config_path is None and config_name is None:
        raise ValueError("load_config needs a config_path or a config_name")
    if config_path is None:
        config_path = os.path.join(CONFIG_DIR, config_name)
    with open(config_path) as f:
        data = f.read()
    try:
        config = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"config {config_path} is not valid JSON: {exc}") from exc
    hps = HParams(**config)
    if validate_config:
        validate(hps, config_path)
    return hps


def list_configs():
    """Names of the vendored configs."""
    return sorted(
        name for name in os.listdir(CONFIG_DIR) if name.endswith(".json")
    )


def save_config(hps, path):
    """Write an HParams tree back out as JSON (for the run directory)."""
    def _plain(node):
        if isinstance(node, HParams):
            return {k: _plain(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_plain(v) for v in node]
        return node

    with open(path, "w") as f:
        json.dump(_plain(hps), f, indent=2)
