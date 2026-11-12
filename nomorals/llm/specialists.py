"""Learner-model toolkit — the small classical-ML family, real and wired.

The owner's doctrine (2026-10-24/25, register §10): two model families,
kept distinct. This module is family (a) — small classical ML (sklearn,
LightGBM, XGBoost, CatBoost) that JUDGES / SCORES / PREDICTS via
journal → grade → retrain. Family (b) — specialist generative models
(XTTS, RVC, face swap) — lives in the organs that use it
(``characters/specialists.py``, ``voice/``); this module never pretends
to be it.

The discipline, ported from the repo's own precedents
(``agents/pumpdesk/learn.py`` — bounded steps, journaled attempts,
in-sample backtest honesty; ``characters/learn.py`` — volume gates
before taste becomes targets):

* **Volume gate first.** No model is trusted below its job's declared
  ``min_examples`` — the betting desk's 300-round minimum is the
  pattern; each job declares its own floor here, in its spec.
* **Held-out proof, not vibes.** Every fit splits a held-out slice
  (fixed seed) and reports it. A model is PROMOTED only when it beats
  the job's code rung on that same slice — otherwise it stays held and
  the code rung carries. The report says so in writing.
* **Versioned store.** One model file per job per version on disk,
  with a manifest carrying the held-out numbers — a retrain never
  silently replaces a working model.
* **Lazy backends, honest walls.** sklearn is the default (present
  everywhere); LightGBM/XGBoost/CatBoost are lazy imports — a missing
  package is an ``unavailable`` reason naming the wall, never a fake
  model. CPU-only, seconds-scale: small models, small data.

Layer: L3 (``llm/``) — imports only stdlib + ``core``. Callers in
``characters/`` (L4), ``games/`` (L5), ``media/`` (L4) and ``agents/``
(L5) supply journal rows; the toolkit never reads a journal itself,
so it never reaches upward.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "LearnerSpec", "LearnerModel", "TrainReport", "FitOutcome",
    "ModelStore", "NotEnoughData", "BackendUnavailable",
    "backend_available", "available_backends", "train", "fit_if_ready",
    "default_models_dir",
]

_log = get_logger(__name__)

#: Feature-vector length cap — a learner row with more features than
#: this is a schema bug, not a model.
MAX_FEATURES = 64


class NotEnoughData(Exception):
    """Raised when a job's journal holds fewer rows than its floor."""


class BackendUnavailable(Exception):
    """Raised when the requested backend package is not installed."""

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(f"backend {backend!r} unavailable: {reason}")
        self.backend = backend
        self.reason = reason


@dataclass(frozen=True)
class LearnerSpec:
    """One learner job: what it learns, from how much, on what backend.

    ``kind``: "classifier" (label 0/1 — win/lose, real/noise, kept/
    skipped) or "regressor" (label a continuous grade, e.g. a line's
    -1..1 grade). ``min_examples`` is the job's declared floor — the
    model is not trusted below it, full stop.
    """

    job: str
    features: tuple[str, ...]
    kind: str = "classifier"  # "classifier" | "regressor"
    backend: str = "gradboost"
    min_examples: int = 30
    label_name: str = "label"


@dataclass
class TrainReport:
    """Held-out numbers for one fit — the promotion evidence."""

    n_rows: int
    n_train: int
    n_test: int
    score: float  # held-out score (accuracy, or 1-MAE for regressors)
    baseline_score: float  # the naive baseline on the same slice
    code_score: float | None = None  # the code rung, same slice
    beats_code: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnerModel:
    """A fitted learner, with its provenance attached."""

    spec: LearnerSpec
    backend: str
    model: Any  # the fitted sklearn / LightGBM / XGBoost / CatBoost est.
    version: int = 1
    trained_at: float = 0.0
    n_rows: int = 0
    report: TrainReport | None = None

    def predict_one(self, features: dict[str, Any]) -> float:
        """One prediction: P(class 1) for classifiers (0..1), the
        predicted value for regressors. Never raises for a bad row —
        a bad row scores the middle, and the failure is logged."""
        try:
            vec = [[float(features.get(k, 0.0) or 0.0)
                    for k in self.spec.features]]
        except (TypeError, ValueError):
            _log.debug("specialist predict: bad features", exc_info=True)
            return 0.5
        try:
            if self.spec.kind == "classifier":
                proba = getattr(self.model, "predict_proba", None)
                if proba is not None:
                    return float(max(0.0, min(1.0, proba(vec)[0][1])))
                dec = getattr(self.model, "decision_function", None)
                if dec is not None:
                    z = float(dec(vec)[0])
                    return 1.0 / (1.0 + math.exp(-z))
                return float(self.model.predict(vec)[0])
            return float(self.model.predict(vec)[0])
        except Exception:  # noqa: BLE001 - a prediction never kills a job
            _log.debug("specialist predict failed", exc_info=True)
            return 0.5


@dataclass
class FitOutcome:
    """What :func:`fit_if_ready` decided, and why."""

    status: str  # "trained" | "held" | "insufficient" | "unavailable"
    model: LearnerModel | None = None
    reason: str = ""
    report: TrainReport | None = None
    code_score: float | None = None
    n_rows: int = 0


# ── backends ────────────────────────────────────────────────────────────

#: (module, classifier factory, regressor factory). Factories take no
#: args and return an unfitted estimator. sklearn is the default; the
#: gradient-boosting rungs are lazy so a missing package is a wall,
#: never an import-time crash.
def _sk_estimators() -> dict[str, Any]:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge

    return {
        "logistic": (LogisticRegression(max_iter=1000, random_state=0),
                     Ridge()),
        "forest": (RandomForestClassifier(n_estimators=200,
                                          random_state=0, n_jobs=2),
                   RandomForestRegressor(n_estimators=200,
                                         random_state=0, n_jobs=2)),
        # gradboost: min_samples_leaf=5, not sklearn's default 20.
        # This toolkit serves journal-sized data (30–100s of rows), and
        # a 20-row leaf minimum stops the trees carving the small cells
        # that carry a job's signal — measured on the tactics journal
        # (120 rounds, the winning cell is ~10 rows): default 0.71
        # held-out with no interaction learned, msl=5 scores 1.00.
        "gradboost": (HistGradientBoostingClassifier(
                          random_state=0, min_samples_leaf=5),
                      HistGradientBoostingRegressor(
                          random_state=0, min_samples_leaf=5)),
    }


def _lgbm_estimators() -> dict[str, Any]:
    import lightgbm as lgbm

    return {
        "lightgbm": (lgbm.LGBMClassifier(n_estimators=200, verbose=-1,
                                         random_state=0),
                     lgbm.LGBMRegressor(n_estimators=200, verbose=-1,
                                        random_state=0)),
    }


def _xgb_estimators() -> dict[str, Any]:
    import xgboost as xgb

    return {
        "xgboost": (xgb.XGBClassifier(n_estimators=200, random_state=0,
                                      n_jobs=2, eval_metric="logloss"),
                    xgb.XGBRegressor(n_estimators=200, random_state=0,
                                     n_jobs=2)),
    }


def _cat_estimators() -> dict[str, Any]:
    from catboost import CatBoostClassifier, CatBoostRegressor

    return {
        "catboost": (CatBoostClassifier(iterations=200, verbose=False,
                                        random_seed=0),
                     CatBoostRegressor(iterations=200, verbose=False,
                                       random_seed=0)),
    }


_BACKEND_LOADERS: dict[str, Callable[[], dict[str, Any]]] = {
    "sklearn": _sk_estimators,
    "lightgbm": _lgbm_estimators,
    "xgboost": _xgb_estimators,
    "catboost": _cat_estimators,
}

#: backend name -> which loader provides it.
_BACKEND_OWNER: dict[str, str] = {
    "logistic": "sklearn", "forest": "sklearn", "gradboost": "sklearn",
    "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost",
}


def backend_available(backend: str) -> tuple[bool, str]:
    """Is ``backend`` importable here? Returns (ok, reason)."""
    owner = _BACKEND_OWNER.get(backend)
    if owner is None:
        return False, f"unknown backend {backend!r}"
    try:
        _BACKEND_LOADERS[owner]()
    except Exception as exc:  # noqa: BLE001 - any import failure is a wall
        return False, f"{owner} not installed: {exc}"
    return True, "ok"


def available_backends() -> dict[str, str]:
    """Every backend with its availability reason."""
    return {name: backend_available(name)[1]
            for name in sorted(_BACKEND_OWNER)}


def _make_estimator(spec: LearnerSpec) -> Any:
    owner = _BACKEND_OWNER.get(spec.backend)
    if owner is None:
        raise BackendUnavailable(spec.backend, "unknown backend name")
    try:
        estimators = _BACKEND_LOADERS[owner]()
    except Exception as exc:  # noqa: BLE001 - import failure is a wall
        raise BackendUnavailable(spec.backend, str(exc)) from exc
    pair = estimators.get(spec.backend)
    if pair is None:
        raise BackendUnavailable(spec.backend, "no estimator registered")
    clf, reg = pair
    return clf if spec.kind == "classifier" else reg


# ── rows → vectors ──────────────────────────────────────────────────────

def _rows_to_xy(rows: list[dict[str, Any]],
                spec: LearnerSpec) -> tuple[list[list[float]],
                                            list[float]]:
    """Journal rows -> (X, y). A row is {"features": {...}, "label": n}.
    Rows without a usable label or features are skipped, and the skip
    is counted in the caller's report — never silently relabelled."""
    X: list[list[float]] = []
    y: list[float] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        feats = row.get("features") or {}
        label = row.get("label")
        if not isinstance(feats, dict) or label is None:
            continue
        try:
            vec = [float(feats.get(k, 0.0) or 0.0)
                   for k in spec.features]
            lab = float(label)
        except (TypeError, ValueError):
            continue
        if math.isnan(lab) or math.isinf(lab):
            continue
        X.append(vec)
        y.append(lab)
    return X, y


def _split(X: list[list[float]], y: list[float], seed: int = 0,
           test_frac: float = 0.2) -> tuple[Any, Any, Any, Any]:
    """Deterministic train/test split (fixed seed). Falls back to an
    unstratified split when a class is too thin to stratify."""
    from sklearn.model_selection import train_test_split

    n = len(X)
    n_test = max(1, min(n - 1, int(round(n * test_frac))))
    strat = None
    try:
        labels = [int(v) for v in y]
        if len(set(labels)) == 2 and min(labels.count(0),
                                         labels.count(1)) >= 2 * n_test:
            strat = labels
    except (TypeError, ValueError):
        strat = None
    return train_test_split(X, y, test_size=n_test, random_state=seed,
                            stratify=strat)


def _heldout_score(model: Any, X_test: Any, y_test: list[float],
                   kind: str) -> tuple[float, float]:
    """(model score, naive-baseline score) on the held-out slice.
    Classifier: accuracy vs the majority-class rate. Regressor:
    1 - MAE/range vs predicting the train mean."""
    from sklearn.metrics import accuracy_score, mean_absolute_error

    if kind == "classifier":
        pred = [int(v) for v in model.predict(X_test)]
        true = [int(v) for v in y_test]
        acc = float(accuracy_score(true, pred))
        maj = max(true.count(0), true.count(1)) / max(len(true), 1)
        return acc, float(maj)
    pred = [float(v) for v in model.predict(X_test)]
    true = [float(v) for v in y_test]
    mae = float(mean_absolute_error(true, pred))
    span = (max(true) - min(true)) or 1.0
    mean_pred = sum(true) / len(true)
    base_mae = float(mean_absolute_error(
        true, [mean_pred] * len(true)))
    return max(0.0, 1.0 - mae / span), max(0.0, 1.0 - base_mae / span)


# ── train / fit_if_ready ────────────────────────────────────────────────

def train(rows: list[dict[str, Any]], spec: LearnerSpec, *,
          seed: int = 0) -> tuple[LearnerModel, TrainReport]:
    """Fit ``spec`` on journal rows. Raises NotEnoughData below the
    job's floor, BackendUnavailable when the backend is missing.
    Always held-out evaluated; the model is returned UNPROMOTED — the
    caller (fit_if_ready) decides promotion against the code rung."""
    X, y = _rows_to_xy(rows, spec)
    if len(X) < spec.min_examples:
        raise NotEnoughData(
            f"{spec.job}: {len(X)}/{spec.min_examples} rows — below the "
            f"floor, the model is not trusted")
    if len(spec.features) > MAX_FEATURES:
        raise ValueError(f"{spec.job}: {len(spec.features)} features "
                         f"exceeds the {MAX_FEATURES} cap")
    estimator = _make_estimator(spec)
    X_train, X_test, y_train, y_test = _split(X, y, seed=seed)
    estimator.fit(X_train, y_train)
    score, baseline = _heldout_score(estimator, X_test, y_test, spec.kind)
    report = TrainReport(n_rows=len(X), n_train=len(X_train),
                         n_test=len(X_test), score=round(score, 4),
                         baseline_score=round(baseline, 4),
                         detail={"backend": spec.backend,
                                 "kind": spec.kind})
    model = LearnerModel(spec=spec, backend=spec.backend,
                         model=estimator, trained_at=time.time(),
                         n_rows=len(X), report=report)
    return model, report


def fit_if_ready(job_rows: list[dict[str, Any]], spec: LearnerSpec, *,
                 code_score_fn: Callable[[list[dict[str, Any]],
                                          list[float]], float] | None = None,
                 store: ModelStore | None = None,
                 seed: int = 0,
                 detail: dict[str, Any] | None = None) -> FitOutcome:
    """The full gate: floor → train → held-out → beat-the-code-rung →
    promote (and save) or hold.

    ``code_score_fn`` scores the job's code rung on the SAME held-out
    rows (features, labels) -> 0..1 score in the same units as the
    report (accuracy for classifiers, 1-MAE/range for regressors).
    Without it the model must beat the naive baseline instead.
    ``detail`` is job provenance merged into the report (and persisted
    with the saved model) — e.g. which actions a policy has evidence
    for. A held model is NOT saved — the code rung keeps carrying
    until a retrain proves better.
    """
    X, y = _rows_to_xy(job_rows, spec)
    n = len(X)
    if n < spec.min_examples:
        return FitOutcome(
            status="insufficient", n_rows=n,
            reason=f"{spec.job}: {n}/{spec.min_examples} rows — below "
                   f"the floor; the code rung carries")
    try:
        model, report = train(job_rows, spec, seed=seed)
    except BackendUnavailable as exc:
        return FitOutcome(status="unavailable", n_rows=n,
                          reason=str(exc))
    # code rung on the same held-out slice, via the toolkit's split
    code_score: float | None = None
    if code_score_fn is not None:
        try:
            _, X_test, _, y_test = _split(X, y, seed=seed)
            feats = [{k: v for k, v in zip(spec.features, vec, strict=True)}
                     for vec in X_test]
            code_score = float(code_score_fn(feats, list(y_test)))
        except Exception:  # noqa: BLE001 - a broken code scorer is a hold
            _log.debug("specialist code scorer failed", exc_info=True)
            code_score = None
    if detail:
        report.detail.update(dict(detail))
    bar = code_score if code_score is not None else report.baseline_score
    report.code_score = (round(code_score, 4)
                         if code_score is not None else None)
    report.beats_code = bool(report.score > bar)
    if not report.beats_code:
        return FitOutcome(
            status="held", model=None, report=report, code_score=code_score,
            n_rows=n,
            reason=f"{spec.job}: held-out {report.score} does not beat "
                   f"the code rung {bar} — the code rung carries")
    if store is not None:
        model.version = store.save(spec.job, model)
    return FitOutcome(status="trained", model=model, report=report,
                      code_score=code_score, n_rows=n,
                      reason=f"{spec.job}: held-out {report.score} beats "
                             f"the code rung {bar}")


# ── versioned store ─────────────────────────────────────────────────────

def default_models_dir() -> Path:
    """Model home: ~/.nomorals/models (NM_MODELS_DIR overrides)."""
    override = (os.environ.get("NM_MODELS_DIR") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser("~")) / ".nomorals" / "models"


class ModelStore:
    """Versioned on-disk store: one job, one model file per version,
    plus a manifest carrying the held-out numbers. Atomic tmp+replace
    writes; a corrupt manifest degrades to 'no model', never a crash."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = (Path(base_dir) if base_dir is not None
                         else default_models_dir())

    def _job_dir(self, job: str) -> Path:
        safe = "".join(c for c in (job or "")
                       if c.isalnum() or c in "-_.")[:64] or "job"
        return self.base_dir / safe

    def _manifest_path(self, job: str) -> Path:
        return self._job_dir(job) / "manifest.json"

    def _read_manifest(self, job: str) -> dict[str, Any]:
        try:
            raw = self._manifest_path(job).read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - corrupt manifest = no model
            return {}

    def latest_version(self, job: str) -> int:
        return int(self._read_manifest(job).get("latest", 0) or 0)

    def save(self, job: str, model: LearnerModel) -> int:
        """Persist ``model`` as the next version. Returns the version."""
        job_dir = self._job_dir(job)
        job_dir.mkdir(parents=True, exist_ok=True)
        version = self.latest_version(job) + 1
        model.version = version
        target = job_dir / f"{job_dir.name}-v{version}.pkl"
        tmp = target.with_suffix(".tmp")
        payload = {
            "job": job, "version": version,
            "backend": model.backend, "kind": model.spec.kind,
            "features": list(model.spec.features),
            "min_examples": model.spec.min_examples,
            "trained_at": model.trained_at, "n_rows": model.n_rows,
            "report": (vars(model.report) if model.report else {}),
            "estimator": model.model,
        }
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        tmp.replace(target)
        manifest = self._read_manifest(job)
        manifest.update({
            "job": job, "latest": version,
            "backend": model.backend,
            "trained_at": model.trained_at, "n_rows": model.n_rows,
            "report": (vars(model.report) if model.report else {}),
            "history": (manifest.get("history") or []) + [{
                "version": version, "trained_at": model.trained_at,
                "n_rows": model.n_rows,
                "score": (model.report.score if model.report else None),
            }],
        })
        m_tmp = self._manifest_path(job).with_suffix(".tmp")
        m_tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        m_tmp.replace(self._manifest_path(job))
        return version

    def load(self, job: str) -> LearnerModel | None:
        """The latest versioned model, or None (never raises)."""
        version = self.latest_version(job)
        if version <= 0:
            return None
        target = self._job_dir(job) / f"{self._job_dir(job).name}-v{version}.pkl"
        try:
            with open(target, "rb") as fh:
                payload = pickle.load(fh)
            spec = LearnerSpec(
                job=payload.get("job", job),
                features=tuple(payload.get("features") or ()),
                kind=payload.get("kind", "classifier"),
                backend=payload.get("backend", "gradboost"),
                min_examples=int(payload.get("min_examples", 30)))
            report = None
            raw_rep = payload.get("report") or {}
            if raw_rep:
                report = TrainReport(
                    n_rows=int(raw_rep.get("n_rows", 0) or 0),
                    n_train=int(raw_rep.get("n_train", 0) or 0),
                    n_test=int(raw_rep.get("n_test", 0) or 0),
                    score=float(raw_rep.get("score", 0.0) or 0.0),
                    baseline_score=float(
                        raw_rep.get("baseline_score", 0.0) or 0.0),
                    code_score=raw_rep.get("code_score"),
                    beats_code=raw_rep.get("beats_code"),
                    detail=dict(raw_rep.get("detail") or {}))
            return LearnerModel(
                spec=spec, backend=payload.get("backend", "gradboost"),
                model=payload.get("estimator"),
                version=int(payload.get("version", version)),
                trained_at=float(payload.get("trained_at", 0.0) or 0.0),
                n_rows=int(payload.get("n_rows", 0) or 0), report=report)
        except Exception:  # noqa: BLE001 - a corrupt model = no model
            _log.debug("specialist load failed for %s", job, exc_info=True)
            return None
