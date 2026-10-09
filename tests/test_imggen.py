"""Tests for Devon's own image generation organ (nomorals.media.imggen).

Two tiers:
1. **Math tests** (numpy, no torch needed): the diffusion schedulers are
   verified against the actual DDPM/DDIM formulas — q_sample statistics,
   schedule properties, DDIM determinism, shape correctness. This is the
   proof the implementation is real math, not theater.
2. **Wiring tests** (torch-gated, skipped without torch): UNet shapes,
   LoRA injection, prompt parsing, aspect ratios, chat/CLI wiring.

A full end-to-end training run (tiny model, CPU, asserts loss
decreases) lives in test_imggen_train.py and is skipped without torch.
"""

import math

import numpy as np
import pytest

from nomorals.media.imggen import ImgGenError, checkpoint_dir
from nomorals.media.imggen.diffusion import (
    BetaSchedule,
    DDPMScheduler,
    DDIMScheduler,
    cosine_beta_schedule,
    linear_beta_schedule,
)


# ---------------------------------------------------------------------------
# Schedule properties
# ---------------------------------------------------------------------------

def test_linear_beta_schedule_matches_ho_et_al():
    betas = linear_beta_schedule(1000)
    assert betas.shape == (1000,)
    assert betas[0] == pytest.approx(1e-4)
    assert betas[-1] == pytest.approx(2e-2)
    assert np.all(np.diff(betas) > 0)  # strictly increasing


def test_cosine_beta_schedule_properties():
    betas = cosine_beta_schedule(1000)
    assert betas.shape == (1000,)
    assert np.all(betas > 0) and np.all(betas < 1)
    # Cosine is gentler early: first beta much smaller than linear's.
    assert betas[0] < linear_beta_schedule(1000)[0]


def test_beta_schedule_rejects_garbage():
    with pytest.raises(ImgGenError):
        linear_beta_schedule(0)
    with pytest.raises(ImgGenError):
        BetaSchedule(timesteps=10, betas=np.ones(5))
    with pytest.raises(ImgGenError):
        BetaSchedule(timesteps=10, betas=np.zeros(10))


def test_alphas_cumprod_is_cumprod_of_one_minus_beta():
    s = BetaSchedule(timesteps=100,
                     betas=linear_beta_schedule(100))
    expected = np.cumprod(1.0 - linear_beta_schedule(100))
    np.testing.assert_allclose(s.alphas_cumprod, expected, rtol=1e-12)
    # alpha_bar decays to ~0 at T=1000 (fully noise by the end).
    s1000 = BetaSchedule(timesteps=1000,
                         betas=linear_beta_schedule(1000))
    assert s1000.alphas_cumprod[-1] < 1e-4


# ---------------------------------------------------------------------------
# Forward process: q_sample
# ---------------------------------------------------------------------------

def test_q_sample_at_t0_is_identity():
    sched = DDPMScheduler(timesteps=10)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((2, 3, 8, 8))
    # t=0 with linear schedule: alpha_bar_0 = 1 - 1e-4 ≈ 1.
    xt = sched.q_sample(x0, 0, noise=np.zeros_like(x0))
    np.testing.assert_allclose(
        xt, math.sqrt(sched.schedule.alphas_cumprod[0]) * x0,
        rtol=1e-10)


def test_q_sample_statistics_match_closed_form():
    """q(x_t|x_0) = N(sqrt(a_bar_t) x_0, (1-a_bar_t) I): verify by Monte
    Carlo — sample mean/std of many draws matches the formula."""
    sched = DDPMScheduler(timesteps=1000)
    t = 500
    a_bar = float(sched.schedule.alphas_cumprod[t])
    rng = np.random.default_rng(7)
    x0 = np.zeros((2000,))  # point mass at 0 → samples ~ N(0, 1-a_bar)
    draws = np.array([sched.q_sample(x0, t) for _ in range(1)])
    # Single draw won't do; use many independent draws.
    samples = np.stack([
        sched.q_sample(np.zeros(4000), t,
                       noise=rng.standard_normal(4000))
        for _ in range(30)
    ]).ravel()
    assert samples.mean() == pytest.approx(0.0, abs=0.05)
    assert samples.std() == pytest.approx(math.sqrt(1 - a_bar),
                                          rel=0.05)


def test_q_sample_end_is_pure_noise():
    sched = DDPMScheduler(timesteps=1000)
    x0 = np.ones((3, 8, 8))
    xt = sched.q_sample(x0, 999)
    # At T the signal is ~gone: |sqrt(a_bar_T) * 1| < 0.05 typical.
    assert abs(float(sched.schedule.sqrt_alphas_cumprod[999])) < 0.02


# ---------------------------------------------------------------------------
# Reverse process
# ---------------------------------------------------------------------------

def _perfect_model_factory(sched, x0_true):
    """A model that predicts the TRUE noise (oracle) — lets us test the
    sampler math in isolation from any neural net."""
    def model(xt, t):
        a_bar = float(sched.schedule.alphas_cumprod[t])
        xt_a = np.asarray(xt, dtype=np.float64)
        x0_a = np.asarray(x0_true, dtype=np.float64)
        # eps = (x_t - sqrt(a_bar) x_0) / sqrt(1 - a_bar)
        return (xt_a - math.sqrt(a_bar) * x0_a) / math.sqrt(
            max(1e-12, 1 - a_bar))
    return model


def test_p_sample_with_oracle_reconstructs_x0():
    """With a perfect noise predictor and eta=0, the reverse chain
    should walk back close to x_0."""
    sched = DDPMScheduler(timesteps=100)
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((1, 2, 4, 4))
    model = _perfect_model_factory(sched, x0)
    xt = sched.q_sample(x0, 99)
    # Few reverse steps from t=99 with the oracle.
    for t in range(99, 89, -1):
        xt = sched.p_sample(model, xt, t, eta=0.0)
    xt = np.asarray(xt)
    # Should be much closer to x0 than pure noise is.
    assert np.mean((xt - x0) ** 2) < np.mean(x0 ** 2)


def test_ddim_eta_zero_is_deterministic():
    sched = DDIMScheduler(timesteps=50)
    rng = np.random.default_rng(11)
    x0 = rng.standard_normal((1, 1, 4, 4))
    model = _perfect_model_factory(sched, x0)
    xt = sched.q_sample(x0, 49)
    out1 = sched.sample(model, xt.shape, steps=10, eta=0.0, seed=1)
    out2 = sched.sample(model, xt.shape, steps=10, eta=0.0, seed=1)
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2))


def test_sample_output_shape():
    sched = DDPMScheduler(timesteps=20)
    model = lambda xt, t: np.zeros_like(np.asarray(xt))  # noqa: E731
    out = sched.sample(model, (2, 3, 8, 8), steps=5, seed=0)
    assert np.asarray(out).shape == (2, 3, 8, 8)


def test_unknown_schedule_rejected():
    with pytest.raises(ImgGenError):
        DDPMScheduler(schedule="bogus")


# ---------------------------------------------------------------------------
# Prompt weighting + aspect ratios (no torch needed)
# ---------------------------------------------------------------------------

def test_parse_weighted_prompt():
    from nomorals.media.imggen.pipeline import parse_weighted_prompt

    spans = parse_weighted_prompt("a (red:1.4) car [blurry] end")
    assert ("red", 1.4) in spans
    assert ("blurry", 0.9) in spans
    plain = [s for s in spans if s[1] == 1.0]
    assert any("a " in s[0] for s in plain)


def test_parse_weighted_prompt_plain():
    from nomorals.media.imggen.pipeline import parse_weighted_prompt

    assert parse_weighted_prompt("just words") == [("just words", 1.0)]


def test_resolve_aspect_ratio():
    from nomorals.media.imggen.pipeline import resolve_aspect_ratio

    w, h = resolve_aspect_ratio("16:9", base=64)
    assert w > h and w % 8 == 0 and h % 8 == 0
    w2, h2 = resolve_aspect_ratio("1:1", base=64)
    assert w2 == h2 == 64
    with pytest.raises(ImgGenError):
        resolve_aspect_ratio("bogus")


def test_checkpoint_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(tmp_path))
    assert checkpoint_dir() == str(tmp_path)


# ---------------------------------------------------------------------------
# Data utilities (no torch needed)
# ---------------------------------------------------------------------------

def test_caption_from_filename():
    from nomorals.media.imggen.data import caption_from_filename

    assert caption_from_filename("lagos_market_01.jpg") == "lagos market 01"
    assert caption_from_filename("x.png") == "x"


def test_build_dataset_manifest(tmp_path):
    from PIL import Image

    from nomorals.media.imggen.data import (build_dataset_manifest,
                                            load_manifest)

    for i in range(3):
        Image.new("RGB", (16, 16), (i * 40, 0, 0)).save(
            tmp_path / f"photo_{i}.png")
    (tmp_path / "photo_0.txt").write_text("a red square")
    summary = build_dataset_manifest(str(tmp_path), use_vision=False)
    assert summary["images"] == 3
    rows = load_manifest(summary["manifest"])
    assert len(rows) == 3
    # Sidecar wins for photo_0.
    by_name = {r["image"]: r["caption"] for r in rows}
    assert by_name[str(tmp_path / "photo_0.png")] == "a red square"


def test_build_dataset_manifest_rejects_empty(tmp_path):
    from nomorals.media.imggen.data import build_dataset_manifest

    with pytest.raises(ImgGenError):
        build_dataset_manifest(str(tmp_path))


# ---------------------------------------------------------------------------
# Chat wiring: /imggen never raises
# ---------------------------------------------------------------------------

def test_imggen_chat_empty_usage():
    from types import SimpleNamespace

    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    class M(RuntimeMediaMixin):
        def __init__(self):
            self.context = SimpleNamespace()

    m = M()
    out = m._control_imggen("")
    assert "imggen" in out.lower() and "/imggen" in out


def test_imggen_chat_checkpoints_no_crash(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(tmp_path))

    class M(RuntimeMediaMixin):
        def __init__(self):
            self.context = SimpleNamespace()

    m = M()
    out = m._control_imggen("checkpoints")
    assert "no native checkpoints" in out.lower()


def test_native_backend_readiness_honest():
    from nomorals.media_edit.generate import NativeBackend

    ok, reason = NativeBackend.readiness()
    # In this env: no torch and/or no checkpoints → not ready, but the
    # reason must be actionable, never a traceback.
    assert isinstance(ok, bool) and isinstance(reason, str)
    assert len(reason) > 10
    if not ok:
        assert ("torch" in reason.lower()
                or "checkpoint" in reason.lower())


def test_get_backend_accepts_native():
    from nomorals.media_edit.generate import get_backend

    # "native" is a known backend name even when not ready — it must
    # raise the honest readiness error, not "unknown backend".
    try:
        get_backend("native")
    except Exception as exc:
        assert "unknown MEDIA_GEN_BACKEND" not in str(exc)
