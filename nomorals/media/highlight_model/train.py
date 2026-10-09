"""Training script for the highlight MLP.

Data paths (in priority order):
1. Owner labels: the owner's "cool" picks are ground truth. Every time the
   owner selects/keeps a scene, that's a positive label. Log via
   log_owner_label().
2. QVHighlights (public academic dataset): saliency annotations for
   moment retrieval + highlight detection. Download from the QVHIGHLIGHTS
   repo; extract features with features.py; train here.

Usage:
    python -m nomorals.media.highlight_model.train --data labels.jsonl
    python -m nomorals.media.highlight_model.train --qvhighlights /path/to/qv

Needs torch for training. Inference stays numpy-only.
"""

from __future__ import annotations

import argparse
import json
import os

from .model import HighlightMLP, N_FEATURES, HIDDEN, default_weights_path


def log_owner_label(features: list[float], label: float,
                    path: str = "owner_labels.jsonl") -> None:
    """Log one owner judgment: features (8-dim) + label (0/1 cool)."""
    assert len(features) == N_FEATURES
    assert 0.0 <= label <= 1.0
    with open(path, "a") as f:
        f.write(json.dumps({"x": features, "y": label}) + "\n")


def load_jsonl(path: str) -> tuple[list[list[float]], list[float]]:
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            xs.append(d["x"])
            ys.append(d["y"])
    return xs, ys


def train(xs: list[list[float]], ys: list[float], *, epochs: int = 200,
          lr: float = 0.05, seed: int = 0,
          out_path: str | None = None) -> HighlightMLP:
    """Train the MLP with torch. Returns the trained model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "torch needed for training — pip install torch") from exc

    import numpy as np
    X = torch.tensor(np.asarray(xs, dtype=np.float32))
    Y = torch.tensor(np.asarray(ys, dtype=np.float32)).unsqueeze(1)

    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(N_FEATURES, HIDDEN)
            self.fc2 = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        opt.step()

    mlp = HighlightMLP()
    with torch.no_grad():
        mlp.W1 = net.fc1.weight.detach().numpy().T.astype(np.float32)
        mlp.b1 = net.fc1.bias.detach().numpy().astype(np.float32)
        mlp.W2 = net.fc2.weight.detach().numpy().T.astype(np.float32)
        mlp.b2 = net.fc2.bias.detach().numpy().astype(np.float32)
    out_path = out_path or default_weights_path()
    mlp.save(out_path)
    print(f"saved {out_path} "
          f"({sum(p.numel() for p in net.parameters())} params)")
    return mlp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="owner_labels.jsonl",
                    help="jsonl of {x: [8 floats], y: 0/1}")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not os.path.exists(args.data):
        raise SystemExit(
            f"no training data at {args.data} — log owner labels first "
            f"via log_owner_label(), or point --data at QVHighlights features")
    xs, ys = load_jsonl(args.data)
    print(f"training on {len(xs)} samples")
    train(xs, ys, epochs=args.epochs, lr=args.lr, out_path=args.out)


if __name__ == "__main__":
    main()
