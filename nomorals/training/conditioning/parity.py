"""Parity runner — the machine-checked parity TOML for conditioning.

Each ``[[case]]`` names a real fixture
(``tests/fixtures/finetrainers-conditioning/``) and the operation to run
on it, and the runner EXECUTES it against the real package and reports
pass/fail per case with measured values. Parity is measured, never
asserted: a case whose expectation is wrong fails loudly, a missing
fixture fails (never a silent skip), and an operation that raises
unexpectedly fails with the exception named.

Schema (version 1):

    [schema]
    version = 1

    [[case]]
    id = "two-file-one-row"
    op = "load_dataset"
    fixture = "prompts.txt"          # extra keys per op, see below
    [case.expect]
    rows = 1

Ops and their keys (all keys optional unless noted):

* ``load_dataset`` — ``format`` (two-file/csv/json/jsonl),
  ``caption_column``, ``video_column``, ``dataset_file`` (fixture
  filename, required unless two-file), ``remove_llm_prefixes``.
  Expect: ``rows``, ``first_caption`` (exact), ``first_video`` (tail
  of the resolved path), ``source_format``.
* ``clean_prefixes`` — ``prompt``. Expect: ``cleaned``.
* ``string_to_filename`` — ``text``. Expect: ``filename``.
* ``dir_name`` — ``model_name``, ``model_id``. Expect: ``dirname``.
* ``nearest_resolution`` — ``buckets`` (list of [f,h,w]),
  ``height``, ``width``. Expect: ``nearest`` ([h,w]).
* ``frame_bucket`` — ``buckets``, ``num_frames``, ``max_num_frames``.
  Expect: ``bucket``.
* ``sample_frames`` — ``num_frames``, ``max_num_frames``. Expect:
  ``indices``.
* ``pad_frames`` — ``shape`` ([b,c,f,h,w]), ``patch_size_t``. Expect:
  ``padded_shape``.
* ``pack_latents`` — ``shape``, ``patch_size``, ``patch_size_t``.
  Expect: ``packed_shape``.
* ``normalize_latents`` — ``shape`` ([b,c,f,h,w]) filled with a fixed
  value, ``mean``/``std`` (per-channel lists), ``scaling_factor``.
  Expect: ``first_value`` (rounded to 6dp).
* ``should_precompute`` — ``layout`` ("missing"/"empty"/"matched"/
  "mismatched"). Expect: ``result`` (bool).
* ``bucket_sampler`` — ``shapes`` (list of [f,h,w]), ``batch_size``,
  ``shuffle``, ``drop_last``. Expect: ``batches`` (list of batch
  sizes), ``total`` (items yielded).
* ``spec`` — ``model_name``. Expect: ``model_id``,
  ``max_sequence_length``, ``condition_keys`` (list),
  ``text_encoder``.
* ``precompute`` — ``model_name``, ``fixture`` (two-file caption/video
  filenames). Expect: ``conditions``, ``latents``, ``codec``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from .buckets import (
    BucketSampler,
    VideoItem,
    find_nearest_resolution,
    frame_bucket_for,
    sample_frame_indices,
)
from .configs import ConditioningConfig
from .encoders import (
    get_condition_spec,
    normalize_latents,
    pack_latents,
    pad_frames,
)
from .errors import ConditioningError
from .formats import clean_llm_prefixes, load_dataset
from .precompute import Precomputer, demo_encoders
from .precomputed import (
    NumpyTensorIO,
    PrecomputedStore,
    precomputation_dir_name,
    should_perform_precomputation,
    string_to_filename,
)

__all__ = ["DEFAULT_PARITY_PATH", "load_parity", "run_parity"]

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent.parent.parent.parent / "training" / "conditioning-parity.toml"
FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "finetrainers-conditioning"


def load_parity(path: str | Path | None = None) -> dict[str, Any]:
    """Load a parity TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_PARITY_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConditioningError(f"cannot read parity TOML {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConditioningError(f"cannot parse parity TOML {path}: {exc}") from exc
    if not isinstance(data.get("case"), list) or not data["case"]:
        raise ConditioningError(f"parity TOML {path} has no [[case]] rows")
    return data


def _check_fixture(name: str) -> Path:
    path = FIXTURES / name
    if not path.exists():
        raise ConditioningError(f"parity fixture missing: {path}")
    return path


def _run_case(case: dict[str, Any], tmp: Path) -> dict[str, Any]:
    """Execute one case; returns {"passed": bool, "measured": {...}, "error": str|None}."""
    op = case.get("op")
    expect = case.get("expect", {})
    try:
        if op == "load_dataset":
            measured = _op_load_dataset(case, tmp)
        elif op == "clean_prefixes":
            measured = {"cleaned": clean_llm_prefixes(case["prompt"])}
        elif op == "string_to_filename":
            measured = {"filename": string_to_filename(case["text"])}
        elif op == "dir_name":
            measured = {
                "dirname": precomputation_dir_name(
                    case["model_name"], case["model_id"]
                )
            }
        elif op == "nearest_resolution":
            buckets = [tuple(b) for b in case["buckets"]]
            measured = {
                "nearest": list(
                    find_nearest_resolution(
                        buckets, case["height"], case["width"]
                    )
                )
            }
        elif op == "frame_bucket":
            buckets = [tuple(b) for b in case["buckets"]]
            measured = {
                "bucket": frame_bucket_for(
                    case["num_frames"], buckets, case["max_num_frames"]
                )
            }
        elif op == "sample_frames":
            measured = {
                "indices": sample_frame_indices(
                    case["num_frames"], case["max_num_frames"]
                )
            }
        elif op == "pad_frames":
            array = np.zeros(case["shape"], dtype=np.float32)
            padded = pad_frames(array, case["patch_size_t"])
            measured = {"padded_shape": list(padded.shape)}
        elif op == "pack_latents":
            array = np.zeros(case["shape"], dtype=np.float32)
            packed = pack_latents(
                array, case["patch_size"], case["patch_size_t"]
            )
            measured = {"packed_shape": list(packed.shape)}
        elif op == "normalize_latents":
            shape = case["shape"]
            array = np.full(shape, case.get("fill", 2.0), dtype=np.float64)
            mean = np.array(case["mean"], dtype=np.float64)
            std = np.array(case["std"], dtype=np.float64)
            out = normalize_latents(
                array, mean, std, case.get("scaling_factor", 1.0)
            )
            measured = {"first_value": round(float(out.flat[0]), 6)}
        elif op == "should_precompute":
            measured = {"result": _op_should_precompute(case, tmp)}
        elif op == "bucket_sampler":
            measured = _op_bucket_sampler(case)
        elif op == "spec":
            spec = get_condition_spec(case["model_name"])
            measured = {
                "model_id": spec.model_id,
                "max_sequence_length": spec.max_sequence_length,
                "condition_keys": list(spec.condition_keys),
                "text_encoder": spec.text_encoder,
            }
        elif op == "precompute":
            measured = _op_precompute(case, tmp)
        else:
            return {
                "passed": False,
                "measured": {},
                "error": f"unknown op {op!r}",
            }
    except Exception as exc:  # noqa: BLE001 — parity reports, never hides
        return {"passed": False, "measured": {}, "error": f"{type(exc).__name__}: {exc}"}
    failures = []
    for key, wanted in expect.items():
        got = measured.get(key, "<missing>")
        if got != wanted:
            failures.append(f"{key}: expected {wanted!r}, got {got!r}")
    return {
        "passed": not failures,
        "measured": measured,
        "error": "; ".join(failures) if failures else None,
    }


def _op_load_dataset(case: dict[str, Any], tmp: Path) -> dict[str, Any]:
    fmt = case["format"]
    if fmt == "two-file":
        _check_fixture(case["caption_column"])
        _check_fixture(case["video_column"])
        dataset = load_dataset(
            FIXTURES,
            case["caption_column"],
            case["video_column"],
            remove_llm_prefixes=case.get("remove_llm_prefixes", False),
        )
    else:
        dataset_file = _check_fixture(case["dataset_file"])
        dataset = load_dataset(
            FIXTURES,
            case["caption_column"],
            case["video_column"],
            dataset_file=dataset_file,
            remove_llm_prefixes=case.get("remove_llm_prefixes", False),
        )
    return {
        "rows": len(dataset),
        "first_caption": dataset.prompts[0],
        "first_video": dataset.video_paths[0].name,
        "source_format": dataset.source_format,
    }


def _op_should_precompute(case: dict[str, Any], tmp: Path) -> bool:
    layout = case["layout"]
    root = tmp / f"sp-{layout}"
    store = root / "ltx_video_Lightricks-LTX-Video_precomputed"
    conditions = store / "conditions"
    latents = store / "latents"
    if layout in ("empty", "matched", "mismatched"):
        conditions.mkdir(parents=True, exist_ok=True)
        latents.mkdir(parents=True, exist_ok=True)
    if layout == "matched":
        for name in ("conditions-0-0.npz", "conditions-0-1.npz"):
            (conditions / name).write_bytes(b"x")
        for name in ("latents-0-0.npz", "latents-0-1.npz"):
            (latents / name).write_bytes(b"x")
    elif layout == "mismatched":
        (conditions / "conditions-0-0.npz").write_bytes(b"x")
        (conditions / "conditions-0-1.npz").write_bytes(b"x")
        (latents / "latents-0-0.npz").write_bytes(b"x")
    return should_perform_precomputation(store)


def _op_bucket_sampler(case: dict[str, Any]) -> dict[str, Any]:
    items = [
        VideoItem(frames=s[0], height=s[1], width=s[2], payload=i)
        for i, s in enumerate(case["shapes"])
    ]
    sampler = BucketSampler(
        items,
        batch_size=case["batch_size"],
        shuffle=case.get("shuffle", False),
        drop_last=case.get("drop_last", False),
    )
    batches = sampler.batches()
    return {
        "batches": [len(batch) for batch in batches],
        "total": sum(len(batch) for batch in batches),
    }


def _op_precompute(case: dict[str, Any], tmp: Path) -> dict[str, Any]:
    _check_fixture(case["caption_column"])
    _check_fixture(case["video_column"])
    dataset = load_dataset(
        FIXTURES, case["caption_column"], case["video_column"]
    )
    config = ConditioningConfig(
        data_root=str(FIXTURES),
        model_name=case["model_name"],
        precompute_conditions=True,
        output_dir=str(tmp / f"pc-{case['model_name']}"),
    )
    encode_text, encode_video = demo_encoders(case["model_name"])
    report = Precomputer(
        dataset, config, encode_text, encode_video,
        tensor_io=NumpyTensorIO(),
    ).run()
    store = PrecomputedStore(
        report.store_dir.parent,
        model_name=config.model_name,
        cleaned_model_id=string_to_filename(config.model_id),
        tensor_io=NumpyTensorIO(),
    )
    pair = store.load_pair(0)
    return {
        "conditions": report.conditions,
        "latents": report.latents,
        "codec": report.codec,
        "pair_keys": sorted(pair["text_conditions"].keys()),
        "pair_latent_keys": sorted(pair["latent_conditions"].keys()),
    }


def run_parity(
    path: str | Path | None = None, *, tmp_dir: str | Path | None = None
) -> dict[str, Any]:
    """Run every [[case]]; returns {passed, failed, cases}."""
    import tempfile

    data = load_parity(path)
    results: list[dict[str, Any]] = []
    if tmp_dir is not None:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(tmp_dir) if tmp_dir else None
    ) as tmp:
        tmp_path = Path(tmp)
        for case in data["case"]:
            outcome = _run_case(case, tmp_path)
            results.append(
                {
                    "id": case.get("id", "<unnamed>"),
                    "passed": outcome["passed"],
                    "measured": outcome["measured"],
                    "error": outcome["error"],
                }
            )
    passed = sum(1 for r in results if r["passed"])
    return {"passed": passed, "failed": len(results) - passed, "cases": results}
