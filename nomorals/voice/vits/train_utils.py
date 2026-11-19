"""Checkpointing, logging and plotting helpers for the VITS trainer.

Port of upstream ``utils.py`` minus ``HParams`` (which lives in
``configs.py``, torch-free) and minus the argparse ``get_hparams`` (the
port's CLIs parse their own arguments). Two small fixes: ``np.fromstring``
(replaced by ``np.frombuffer`` — removed in numpy 2.0) and
``logger.warn`` (``logger.warning``).
"""

import glob
import json
import logging
import os
import subprocess
import sys

import numpy as np
from scipy.io.wavfile import read

from ._torch import require_torch

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

MATPLOTLIB_FLAG = False


def load_checkpoint(checkpoint_path, model, optimizer=None):
    torch = require_torch("loading a VITS checkpoint")
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu",
                                 weights_only=False)
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except KeyError:
            logger.info(f"{k} is not in the checkpoint")
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    torch = require_torch("saving a VITS checkpoint")
    logger.info(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"model": state_dict,
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "learning_rate": learning_rate}, checkpoint_path)


def summarize(writer, global_step, scalars=None, histograms=None,
              images=None, audios=None, audio_sampling_rate=22050):
    scalars = scalars or {}
    histograms = histograms or {}
    images = images or {}
    audios = audios or {}
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def _matplotlib():
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    return plt


def _canvas_to_rgb(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram):
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    data = _canvas_to_rgb(fig)
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower",
                   interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    data = _canvas_to_rgb(fig)
    plt.close()
    return data


def load_wav_to_torch(full_path):
    torch = require_torch("loading audio as a tensor")
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = []
        for line in f:
            line_splits = line.strip().split(split)
            if len(line_splits) < 2:
                print(f"Warning: line {line.strip()} invalid split")
                continue
            filepaths_and_text.append(line_splits)
    return filepaths_and_text


def get_hparams_from_dir(model_dir):
    from .configs import HParams
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path) as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    from .configs import HParams
    with open(config_path) as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(f"{source_dir} is not a git repository, therefore hash value comparison will be ignored.")
        return
    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        with open(path) as f:
            saved_hash = f.read()
        if saved_hash != cur_hash:
            logger.warning(f"git hash values are different. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)")
    else:
        with open(path, "w") as f:
            f.write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger
