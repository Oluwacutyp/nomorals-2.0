"""Trainable highlight scorer — NOT "for later."

A tiny MLP (<10K params) that learns highlight weights from data.
Inference is pure numpy (no torch needed at runtime — phone-viable).
Training needs torch; the QVHighlights path is documented in train.py.

Features in (same as score.py's signals, per-second):
  [audio_energy, motion, faces, dialogue, shot_change_rate,
   face_size_mean, audio_spectral_flux, dialogue_wpm]

The shipped heuristic weights in score.py are the fallback. When this
model has trained weights, score.py uses them automatically.
"""

from __future__ import annotations

import json
import os

import numpy as np

N_FEATURES = 8
HIDDEN = 16  # 8*16 + 16 + 16*1 + 1 = 161 params. Tiny on purpose.


class HighlightMLP:
    """161-parameter highlight scorer. Numpy inference, torch training."""

    def __init__(self, weights_path: str | None = None):
        self.W1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.W2: np.ndarray | None = None
        self.b2: np.ndarray | None = None
        if weights_path and os.path.exists(weights_path):
            self.load(weights_path)

    @property
    def loaded(self) -> bool:
        return self.W1 is not None

    def _init_random(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (N_FEATURES, HIDDEN)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = rng.normal(0, 0.5, (HIDDEN, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (n, 8) → (n,) scores in 0-1."""
        if not self.loaded:
            raise RuntimeError("no weights loaded")
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return 1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2).ravel()))  # sigmoid

    def score(self, features: list[list[float]]) -> list[float]:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return [float(v) for v in self.forward(x)]

    def save(self, path: str) -> None:
        assert self.loaded
        np.savez(path,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 meta=json.dumps({"n_features": N_FEATURES,
                                  "hidden": HIDDEN}))

    def load(self, path: str) -> None:
        d = np.load(path, allow_pickle=True)
        self.W1 = d["W1"]
        self.b1 = d["b1"]
        self.W2 = d["W2"]
        self.b2 = d["b2"]

    def export_onnx(self, path: str) -> None:
        """Export for runtimes without numpy if ever needed."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch needed for ONNX export") from exc
        assert self.loaded
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, mlp: "HighlightMLP"):
                super().__init__()
                self.fc1 = nn.Linear(N_FEATURES, HIDDEN)
                self.fc2 = nn.Linear(HIDDEN, 1)
                with torch.no_grad():
                    self.fc1.weight.copy_(torch.from_numpy(mlp.W1.T))
                    self.fc1.bias.copy_(torch.from_numpy(mlp.b1))
                    self.fc2.weight.copy_(torch.from_numpy(mlp.W2.T))
                    self.fc2.bias.copy_(torch.from_numpy(mlp.b2))

            def forward(self, x):  # noqa: D102
                return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

        net = Net(self).eval()
        dummy = torch.zeros(1, N_FEATURES)
        torch.onnx.export(net, dummy, path, input_names=["features"],
                          output_names=["score"],
                          dynamic_axes={"features": {0: "n"}})


def default_weights_path() -> str:
    return os.path.join(os.path.dirname(__file__), "highlight_mlp.npz")
