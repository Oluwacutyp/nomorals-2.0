"""DDPM + DDIM noise schedulers, implemented by hand.

Math reference (Ho et al. 2020, "Denoising Diffusion Probabilistic Models",
arXiv:2006.11239; Song et al. 2020, "Denoising Diffusion Implicit Models",
arXiv:2010.02502):

Forward (fixed, no learning):
    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)

With alpha_t = 1 - beta_t and alpha_bar_t = prod_{s<=t} alpha_s, the
marginal is available in closed form:
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

so a noisy sample is drawn directly (``q_sample``):
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,
    eps ~ N(0, I)

Reverse (learned noise predictor eps_theta):
    mu_theta(x_t, t) = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta)
                       / sqrt(alpha_t)
    p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta, sigma_t^2 * I)

Training loss (the "simple" objective):
    L = E_{t, x_0, eps}[ ||eps - eps_theta(x_t, t)||^2 ]

DDIM replaces the Markovian reverse with a non-Markovian one controlled
by ``eta``: eta = 0 is fully deterministic, eta = 1 recovers DDPM-like
stochasticity. The update (predict x_0, then step to t-1):
    x_0_pred = (x_t - sqrt(1 - a_bar_t) * eps) / sqrt(a_bar_t)
    sigma_t  = eta * sqrt((1 - a_bar_{t-1}) / (1 - a_bar_t))
                         * (1 - a_bar_t / a_bar_{t-1}))
    x_{t-1}  = sqrt(a_bar_{t-1}) * x_0_pred
               + sqrt(1 - a_bar_{t-1} - sigma_t^2) * eps
               + sigma_t * z,   z ~ N(0, I)

which lets sampling skip timesteps (fewer steps, same quality).

Both schedulers work on numpy arrays AND torch tensors: the schedule
itself is pure numpy; tensor ops are dispatched through a tiny backend
shim so the math is testable without torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import ImgGenError

__all__ = [
    "BetaSchedule",
    "DDPMScheduler",
    "DDIMScheduler",
    "linear_beta_schedule",
    "cosine_beta_schedule",
]


def linear_beta_schedule(timesteps: int,
                         beta_start: float = 1e-4,
                         beta_end: float = 2e-2) -> np.ndarray:
    """Ho et al.'s linear schedule: beta_1=1e-4 ... beta_T=2e-2, T=1000."""
    if timesteps < 1:
        raise ImgGenError("timesteps must be >= 1")
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """Nichol & Dhariwal's cosine schedule (arXiv:2102.09672).

    alpha_bar_t = cos^2( ((t/T + s) / (1 + s)) * pi/2 ), clipped so the
    terminal signal-to-noise ratio is sane. Gentler early noising than
    linear; better for small images and fewer steps.
    """
    if timesteps < 1:
        raise ImgGenError("timesteps must be >= 1")
    steps = np.arange(timesteps + 1, dtype=np.float64)
    alphas_cumprod = np.cos(((steps / timesteps) + s) / (1 + s)
                            * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0, 0.999).astype(np.float64)


@dataclass
class BetaSchedule:
    """Precomputed noise schedule and all derived quantities.

    Attributes are numpy float64 arrays of length ``timesteps``:
    betas, alphas, alphas_cumprod (alpha_bar), plus the square roots
    used constantly in q_sample / p_sample.
    """

    timesteps: int
    betas: np.ndarray = field(repr=False)
    alphas: np.ndarray = field(init=False, repr=False)
    alphas_cumprod: np.ndarray = field(init=False, repr=False)
    sqrt_alphas_cumprod: np.ndarray = field(init=False, repr=False)
    sqrt_one_minus_alphas_cumprod: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        betas = np.asarray(self.betas, dtype=np.float64)
        if betas.shape != (self.timesteps,):
            raise ImgGenError(
                f"betas must have shape ({self.timesteps},), "
                f"got {betas.shape}")
        if np.any(betas <= 0) or np.any(betas >= 1):
            raise ImgGenError("betas must lie in (0, 1)")
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(
            1.0 - self.alphas_cumprod)


class _NumpyOps:
    """Tensor-op shim for numpy arrays (lets schedulers run torch-free)."""

    @staticmethod
    def randn_like(x: np.ndarray) -> np.ndarray:
        return np.random.randn(*x.shape).astype(x.dtype)

    @staticmethod
    def sqrt(x):  # passthrough; values already numpy
        return np.sqrt(x)


def _as_numpy(x):
    """Return (array, is_torch). Torch tensors come back as numpy views of
    data for schedule indexing only — elementwise math stays in the
    caller's framework via broadcasting of numpy scalars."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy(), True
    except ImportError:
        pass
    return np.asarray(x), False


class DDPMScheduler:
    """Ancestral (DDPM) sampler with a hand-built beta schedule."""

    def __init__(self, timesteps: int = 1000,
                 schedule: str = "linear") -> None:
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ImgGenError(
                f"unknown schedule {schedule!r}; use 'linear' or 'cosine'")
        self.schedule = BetaSchedule(timesteps=timesteps, betas=betas)
        self.timesteps = timesteps

    # -- forward (training-time) -------------------------------------
    def q_sample(self, x0, t: int, noise=None):
        """x_t ~ q(x_t | x_0): add exactly the right amount of noise.

        x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps
        Works on numpy arrays and torch tensors.
        """
        s = self.schedule
        sqrt_ab = s.sqrt_alphas_cumprod[t]
        sqrt_om = s.sqrt_one_minus_alphas_cumprod[t]
        arr, is_torch = _as_numpy(x0)
        if noise is None:
            if is_torch:
                import torch

                eps = torch.randn_like(x0)
                return (float(sqrt_ab) * x0
                        + float(sqrt_om) * eps)
            eps = np.random.randn(*arr.shape).astype(arr.dtype)
        else:
            eps = noise
        out = sqrt_ab * arr + sqrt_om * np.asarray(eps)
        if is_torch:
            import torch

            return torch.as_tensor(out, dtype=x0.dtype, device=x0.device)
        return out.astype(arr.dtype, copy=False)

    # -- reverse (sampling) ------------------------------------------
    def p_sample(self, model, xt, t: int, eta: float = 0.0):
        """One reverse step x_t -> x_{t-1} using the model's eps prediction.

        ``model`` is a callable (xt, t) -> eps_pred with xt's shape.
        eta=0 is deterministic; eta=1.0 is full DDPM stochasticity.
        """
        s = self.schedule
        a_bar_t = s.alphas_cumprod[t]
        a_bar_tm1 = s.alphas_cumprod[t - 1] if t > 0 else 1.0
        beta_t = s.betas[t]
        alpha_t = s.alphas[t]

        eps = model(xt, t)
        arr, is_torch = _as_numpy(xt)
        eps_a, _ = _as_numpy(eps)

        # Predicted x_0 from the noise estimate.
        x0_pred = ((arr - np.sqrt(1.0 - a_bar_t) * eps_a)
                   / np.sqrt(a_bar_t))
        # DDIM-style sigma; eta=1 matches DDPM's posterior variance.
        sigma = (eta * math.sqrt((1.0 - a_bar_tm1) / (1.0 - a_bar_t))
                 * math.sqrt(max(0.0, 1.0 - a_bar_t / a_bar_tm1)))
        # Direction pointing back to x_t.
        dir_xt = math.sqrt(max(0.0, 1.0 - a_bar_tm1 - sigma ** 2)) * eps_a
        mean = math.sqrt(a_bar_tm1) * x0_pred + dir_xt
        if sigma > 0:
            if is_torch:
                import torch

                z = torch.randn_like(xt).cpu().numpy()
            else:
                z = np.random.randn(*arr.shape)
            mean = mean + sigma * z
        if is_torch:
            import torch

            return torch.as_tensor(mean, dtype=xt.dtype, device=xt.device)
        return mean.astype(arr.dtype, copy=False)

    def sample(self, model, shape, steps: int | None = None,
               eta: float = 0.0, seed: int | None = None):
        """Full reverse chain from pure noise. Returns array/tensor of
        ``shape`` matching the framework of the model's output."""
        rng = np.random.default_rng(seed)
        xt = rng.standard_normal(shape).astype(np.float64)
        ts = (list(range(self.timesteps - 1, -1, -1)) if steps is None
              else np.linspace(self.timesteps - 1, 0, steps).astype(int).tolist())
        for t in ts:
            xt = self.p_sample(model, xt, int(t), eta=eta)
        return xt


class DDIMScheduler(DDPMScheduler):
    """DDIM sampler: same schedule, strided timesteps, eta-controlled.

    The update math lives in :meth:`DDPMScheduler.p_sample` (the DDIM
    formulation); this subclass exists so call sites can ask for DDIM
    explicitly and get strided ``sample()`` by default.
    """

    def sample(self, model, shape, steps: int = 50,
               eta: float = 0.0, seed: int | None = None):
        return super().sample(model, shape, steps=steps, eta=eta,
                              seed=seed)
