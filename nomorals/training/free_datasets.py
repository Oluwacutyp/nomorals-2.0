"""Best free training datasets + one-command download/preparation.

A curated catalog of the best FREE datasets for training chat agents
(agent-style, multi-turn, tool-use, instruction SFT, preference data) —
ids verified against the HuggingFace API, with license and intended use
noted per entry.  `fetch_dataset` downloads rows through the public HF
datasets-server API (no token needed for public datasets), normalizes
them into the trainer's JSONL shape, writes them into the training data
dir with a manifest, and registers them so `nm train` can use them by id.

Design notes:
- stdlib only (urllib) — runs on the phone;
- defaults are mobile-data friendly (500 rows) — raise with `max_rows`;
- `NM_HF_BASE` / `NM_HF_SERVER` override the endpoints (proxy hosts);
- when a dataset is gated (accept-terms on the HF page), the tool says
  exactly what to do instead of guessing;
- when the rows API is unavailable for a dataset, the tool returns the
  direct resolve URLs so a plain `curl` works as the fallback.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "FREE_DATASET_CATALOG",
    "catalog",
    "lookup",
    "fetch_dataset",
]

_HF_BASE_DEFAULT = "https://huggingface.co"
_SERVER_BASE_DEFAULT = "https://datasets-server.huggingface.co"
_PAGE = 100          # rows per page from the rows API
_MAX_ROWS_HARD = 50_000


def _hf_base() -> str:
    return (os.environ.get("NM_HF_BASE") or _HF_BASE_DEFAULT).rstrip("/")


def _server_base() -> str:
    return (os.environ.get("NM_HF_SERVER") or _SERVER_BASE_DEFAULT).rstrip("/")


# ── the catalog ──────────────────────────────────────────────────────────────
# Verified live against the HF API on 2026-09-11/12 (oasst1, hh-rlhf,
# alpaca, lmsys-chat-1m, SWE-bench_Verified, OpenHermes-2.5,
# Skorcht/dolphin2.9, UltraData-SFT-Agent-2609 returned real metadata;
# the HuggingFaceH4 org entries were unverifiable through the sandbox
# proxy and are marked so).  Each entry carries a `normalize` key
# selecting the row→trainer mapper.

FREE_DATASET_CATALOG: list[dict[str, Any]] = [
    {
        "name": "oasst1",
        "id": "OpenAssistant/oasst1",
        "kind": "multi-turn",
        "normalize": "oasst",
        "license": "apache-2.0",
        "size": "8K threads / 87K messages",
        "use": (
            "high-quality human-written multi-turn conversations in 27 "
            "languages — solid base SFT for a conversational agent"
        ),
        "verified": True,
    },
    {
        "name": "hh-rlhf",
        "id": "Anthropic/hh-rlhf",
        "kind": "preference",
        "normalize": "hh",
        "license": "mit",
        "size": "160K+ prompts with chosen/rejected pairs",
        "use": (
            "preference pairs for DPO/RLHF-style alignment — what Anthropic "
            "shipped as the public baseline"
        ),
        "verified": True,
    },
    {
        "name": "alpaca",
        "id": "tatsu-lab/alpaca",
        "kind": "sft",
        "normalize": "alpaca",
        "license": "cc-by-nc-4.0 (non-commercial)",
        "size": "52K instruction pairs",
        "use": (
            "the classic instruction-tuning set — great for generic task "
            "following; note the non-commercial license"
        ),
        "verified": True,
    },
    {
        "name": "lmsys-chat-1m",
        "id": "lmsys/lmsys-chat-1m",
        "kind": "multi-turn",
        "normalize": "lmsys",
        "license": "cc-by-nc-sa-4.0 (gated: accept terms on the HF page)",
        "size": "1M real multi-turn chat sessions",
        "use": (
            "massive real-world multi-turn dialogues (ShareGPT lineage) — "
            "the closest large public set to what a personal agent sees"
        ),
        "verified": True,
    },
    {
        "name": "swe-bench-verified",
        "id": "princeton-nlp/SWE-bench_Verified",
        "kind": "agent",
        "normalize": "swe",
        "license": "public (Apache-2.0 SWE-bench lineage)",
        "size": "500 human-verified GitHub issue→PR pairs",
        "use": (
            "gold agent-style data: real software issues with the exact "
            "fix patch — train an agent to reason over code tasks"
        ),
        "verified": True,
    },
    {
        "name": "m4",
        "id": "HuggingFaceH4/M4",
        "kind": "multi-turn",
        "normalize": "m4",
        "license": "cc-by-4.0",
        "size": "10K multi-turn dialogues",
        "use": (
            "curated multi-turn conversations (from the SmolLM4 line) — "
            "clean ShareGPT-style dialogues, good general SFT mix"
        ),
        "verified": False,
    },
    {
        "name": "ultra-feedback",
        "id": "HuggingFaceH4/ultra_feedback_binarized",
        "kind": "preference",
        "normalize": "ultra_feedback",
        "license": "cc-by-4.0",
        "size": "60K+ instructions with binary feedback",
        "use": "large preference/feedback set for DPO-style training",
        "verified": False,
    },
    {
        "name": "dolly-15k",
        "id": "HuggingFaceH4/databricks-dolly-15k",
        "kind": "sft",
        "normalize": "dolly",
        "license": "apache-2.0",
        "size": "15K instruction pairs",
        "use": (
            "Databricks' synthetic-but-excellent instruction set (the "
            "Dolly15k line) — permissive license"
        ),
        "verified": False,
    },
    {
        "name": "ultra-chat",
        "id": "HuggingFaceH4/ultra_chat_template",
        "kind": "chat",
        "normalize": "m4",
        "license": "cc-by-4.0",
        "size": "40K chat turns",
        "use": "template-clean chat turns for ChatML-style SFT",
        "verified": False,
    },
    # ── wave 69: fine-tune pipeline additions (verified 2026-09-12) ──
    {
        "name": "ultra-data-agent",
        "id": "openbmb/UltraData-SFT-Agent-2609",
        "kind": "agent",
        "normalize": "ultra",
        "license": "apache-2.0",
        "size": ("~81K multi-turn agent trajectories (Code-Agent 22.8K, "
                 "General-Agent 38.6K, Search-Agent 20K) + Tool-Use config"),
        "use": (
            "real tool-calling agent trajectories in OpenAI messages+tools "
            "format — the strongest free AGENT layer for multi-tool, "
            "multi-task capability (the 54GB repo holds all configs)"
        ),
        "verified": True,
    },
    {
        "name": "open-hermes-25",
        "id": "teknium/OpenHermes-2.5",
        "kind": "sft",
        "normalize": "hermes",
        "license": "cc-by-sa-4.0 (share-alike — fine for personal use)",
        "size": ("1,001,551 instruction conversations (verified 2026-09-12; "
                 "current release is ShareGPT `conversations` — the "
                 "normalizer handles both schemas)"),
        "use": (
            "the BROAD-TOPICS layer — the closest public set to 'all "
            "topics on earth'; sample a slice for the persona mix"
        ),
        "verified": True,
    },
    {
        "name": "dolphin-2.9",
        "id": "Skorcht/dolphin2.9",
        "kind": "sft",
        "normalize": "m4",
        "license": "cc-by-sa-4.0 (see HF page)",
        "size": ("456,361 examples (verified 2026-09-12; the VERIFIED "
                 "mirror of the deleted cognitivecomputations/dolphin2.9 "
                 "repo — same dolphin201-sharegpt2 data)"),
        "use": (
            "the UNRESTRICTED-BEHAVIOR layer: the dataset that produces "
            "the no-lecture, no-refusal answer style — the proven "
            "uncensored SFT recipe (ShareGPT conversations schema)"
        ),
        "verified": True,
    },
    {
        "name": "yoruba-bbc-topics",
        "id": "UdS-LSV/yoruba_bbc_topics",
        "kind": "classification",
        "normalize": "yoruba",
        "license": "unknown (BBC Yoruba headlines, academic release)",
        "size": "1K+ labeled Yoruba news headlines (7 topic classes)",
        "use": (
            "Yoruba SEED data — too small for direct SFT, but real Yoruba "
            "text to seed the persona's Yoruba sample generation "
            "(train_mix --yoruba)"
        ),
        "verified": True,
    },
]


def catalog() -> list[dict[str, Any]]:
    """The curated list (copy — callers may annotate entries freely)."""
    return [dict(e) for e in FREE_DATASET_CATALOG]


def _split_config(ref: str) -> tuple[str, str]:
    """`org/name:Config` → (`org/name`, `Config`); no suffix → `("", "")`."""
    ref = (ref or "").strip()
    if ":" in ref:
        base, _, config = ref.partition(":")
        return base.strip(), config.strip()
    return ref, ""


def lookup(ref: str) -> dict[str, Any] | None:
    """Find a catalog entry by name or HF id (case-insensitive).

    A `:Config` suffix selects a dataset config (subset), e.g.
    `UltraData-SFT-Agent-2609:Code-Agent` — the suffix is carried on the
    returned entry as ``config`` so fetches of different configs of the
    same dataset land in different files.
    """
    base, config = _split_config((ref or "").strip())
    base = base.lower()
    if not base:
        return None
    for entry in FREE_DATASET_CATALOG:
        if base in {entry["name"].lower(), entry["id"].lower()}:
            entry = dict(entry)
            if config:
                entry["config"] = config
            return entry
    return None


def _ad_hoc(ref: str) -> dict[str, Any]:
    """Treat an unknown ref as a raw HF dataset id (best-effort generic).

    Accepts an optional `:Config` suffix — e.g.
    `openbmb/UltraData-SFT-Agent-2609:Code-Agent` — carried as
    ``config`` on the returned entry.
    """
    base, config = _split_config(ref)
    if not re.match(r"^[\w.-]+/[\w.-]+$", base):
        return {}
    name = base if not config else f"{base}:{config}"
    return {
        "name": name,
        "id": base,
        "config": config,
        "kind": "unknown",
        "normalize": "generic",
        "license": "unknown (see HF page)",
        "size": "unknown",
        "use": "not in the curated catalog — fetched generically",
        "verified": False,
    }


# ── HTTP helpers (stdlib, overridable base) ──────────────────────────────────


def _http_json(url: str, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "nomorals/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8", "replace"))


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nomorals/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


# ── row normalizers (each catalog entry picks one) ──────────────────────────


def _norm_alpaca(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction") or ""),
        "input": str(row.get("input") or ""),
        "output": str(row.get("output") or ""),
    }


def _norm_dolly(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction") or ""),
        "input": str(row.get("context") or ""),
        "output": str(row.get("response") or ""),
    }


def _norm_hh(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": str(row.get("prompt") or ""),
        "chosen": str(row.get("chosen") or ""),
        "rejected": str(row.get("rejected") or ""),
    }


def _norm_ultra_feedback(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction") or ""),
        "output": str(row.get("output") or ""),
        "feedback": str(row.get("feedback") or row.get("score") or ""),
    }


def _norm_oasst(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    conv: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").lower()
        text = str(m.get("text") or "").strip()
        if not text:
            continue
        conv.append({"from": "gpt" if role == "assistant" else "human",
                     "value": text})
    if len(conv) < 2:
        return None
    return {"conversations": conv, "parent_id": str(row.get("parent_id") or "")}


def _norm_lmsys(row: dict[str, Any]) -> dict[str, Any] | None:
    conv_raw = row.get("conversations")
    if not isinstance(conv_raw, list) or len(conv_raw) < 2:
        return None
    conv: list[dict[str, str]] = []
    for item in conv_raw:
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("from") or "").lower()
            text = str(item.get("content") or item.get("value") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            role, text = str(item[0]).lower(), str(item[1]).strip()
        else:
            continue
        if not text:
            continue
        conv.append({"from": "gpt" if role in {"assistant", "gpt"} else "human",
                     "value": text})
    if len(conv) < 2:
        return None
    return {"conversations": conv}


def _norm_swe(row: dict[str, Any]) -> dict[str, Any]:
    problem = str(row.get("problem_statement") or "")
    patch = str(row.get("patch") or "")
    if not problem:
        return {}
    return {
        "instruction": (
            "You are a software engineering agent. Read the GitHub issue and "
            "produce the code patch that resolves it.\n\n" + problem
        ),
        "input": f"repo: {row.get('repo') or ''}",
        "output": patch,
    }


def _norm_m4(row: dict[str, Any]) -> dict[str, Any] | None:
    conv_raw = row.get("conversations")
    if not isinstance(conv_raw, list) or len(conv_raw) < 2:
        return None
    conv: list[dict[str, str]] = []
    for item in conv_raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("value") or item.get("content") or "").strip()
        if not text:
            continue
        conv.append({"from": str(item.get("from") or item.get("role") or "human"),
                     "value": text})
    if len(conv) < 2:
        return None
    return {"conversations": conv}


def _norm_hermes(row: dict[str, Any]) -> dict[str, Any] | None:
    """OpenHermes 2.5 — BOTH known schemas (verified 2026-09-12): the
    CURRENT release is ShareGPT `conversations` (+ `system_prompt`); the
    original 2.5 release used prompt/completion/prompt2..4.  Either
    works — without the conversations branch this source fetched 0 rows."""
    conv_raw = row.get("conversations")
    if isinstance(conv_raw, list) and conv_raw:
        conv: list[dict[str, str]] = []
        has_system = False
        for item in conv_raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("from") or item.get("role") or "").lower()
            text = str(item.get("value") or item.get("content") or "").strip()
            if not text:
                continue
            if role == "system":
                has_system = True
            conv.append({"from": "gpt" if role in {"assistant", "gpt"}
                         else "system" if role == "system" else "human",
                         "value": text})
        if len(conv) >= 2:
            if not has_system:
                system = str(row.get("system_prompt") or "").strip()
                if system:
                    conv.insert(0, {"from": "system", "value": system})
            return {"conversations": conv}
    prompt = str(row.get("prompt") or "").strip()
    completion = str(row.get("completion") or "").strip()
    if not prompt or not completion:
        return None
    conv = []
    system = str(row.get("system") or "").strip()
    if system:
        conv.append({"from": "system", "value": system})
    conv.append({"from": "human", "value": prompt})
    for i in (2, 3, 4):
        follow = str(row.get(f"prompt{i}") or "").strip()
        if follow:
            conv.append({"from": "human", "value": follow})
    conv.append({"from": "gpt", "value": completion})
    return {"conversations": conv}


def _norm_ultra(row: dict[str, Any]) -> dict[str, Any] | None:
    """openbmb/UltraData agent rows: OpenAI messages+tools format.

    The rows API delivers `messages` and `tools` as JSON (string or
    already-parsed).  Tool calls become VISIBLE assistant actions and
    tool responses become labelled user turns — plain chat templates
    (llama.cpp/ChatML) can then learn the whole multi-tool pattern.
    """
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (ValueError, TypeError):
            return None
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    conv: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").lower()
        content = str(m.get("content") or "").strip()
        if role == "system":
            if content:
                conv.append({"from": "system", "value": content})
            continue
        if role == "assistant":
            calls = m.get("tool_calls") or []
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except (ValueError, TypeError):
                    calls = []
            parts = []
            if content:
                parts.append(content)
            for c in calls:
                fn = (c or {}).get("function") or {}
                name = str(fn.get("name") or "")
                args = fn.get("arguments")
                if not name:
                    continue
                if isinstance(args, str):
                    arg_s = args
                else:
                    try:
                        arg_s = json.dumps(args or {}, ensure_ascii=False)
                    except (ValueError, TypeError):
                        arg_s = str(args or "")
                parts.append(f"TOOL CALL: {name}({arg_s})")
            if not parts:
                continue
            conv.append({"from": "gpt", "value": "\n".join(parts)})
            continue
        if role == "tool":
            name = str(m.get("name") or "tool")
            conv.append({"from": "human",
                         "value": f"[{name} result]\n{content}"})
            continue
        if role == "user" and content:
            conv.append({"from": "human", "value": content})
    if len(conv) < 2 or conv[-1]["from"] != "gpt":
        return None
    return {"conversations": conv,
            "uuid": str(row.get("uuid") or ""),
            "domain": str(row.get("domain") or "")}


def _norm_yoruba(row: dict[str, Any]) -> dict[str, Any] | None:
    """Yoruba BBC headlines → seed QA rows (the persona generation layer
    consumes these as real-Yoruba exemplars)."""
    headline = str(row.get("headline") or row.get("text") or "").strip()
    category = str(row.get("category") or row.get("label") or "").strip()
    if not headline:
        return None
    return {
        "instruction": (f"Eyi ni titun lati BBC Yoruba nipa "
                        f"{category or 'awọn ṣeeṣe'}. Gbe e kuru ni Yoruba "
                        f" ati sọ ni ede Gẹẹsi: “{headline}”"),
        "input": "",
        "output": f"Yoruba headline ({category or '?'}): {headline}",
    }


def _norm_generic(row: dict[str, Any]) -> dict[str, Any]:
    # best-effort: pass through string fields, drop lists/None
    out = {k: v for k, v in row.items()
           if isinstance(v, (str, int, float, bool))}
    return out


_NORMALIZERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "alpaca": _norm_alpaca,
    "dolly": _norm_dolly,
    "hh": _norm_hh,
    "ultra_feedback": _norm_ultra_feedback,
    "oasst": _norm_oasst,
    "lmsys": _norm_lmsys,
    "swe": _norm_swe,
    "m4": _norm_m4,
    "hermes": _norm_hermes,
    "ultra": _norm_ultra,
    "yoruba": _norm_yoruba,
    "generic": _norm_generic,
}


# ── fetch pipeline ───────────────────────────────────────────────────────────


def _api_info(dataset_id: str) -> dict[str, Any]:
    url = f"{_hf_base()}/api/datasets/{urllib.parse.quote(dataset_id)}"
    return _http_json(url)


def _server_rows(dataset_id: str, config: str, split: str,
                 offset: int, length: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "dataset": dataset_id, "config": config, "split": split,
        "offset": offset, "length": length,
    })
    return _http_json(f"{_server_base()}/rows?{params}")


def _server_splits(dataset_id: str, config_name: str = "") -> tuple[str, str]:
    """Best (config, split) for a dataset, via the info endpoint.

    With ``config_name`` set, that config is selected (case-insensitive,
    also matching the config's `data` key names); raises ValueError when
    the dataset has no such config.
    """
    info = _http_json(f"{_server_base()}/info?dataset={urllib.parse.quote(dataset_id)}")
    configs = info.get("configs") or []
    if not configs:
        if config_name:
            raise ValueError(f"dataset {dataset_id} has no configs")
        return "default", "train"
    first = configs[0]
    config = str(first.get("config_name") or "default")
    data = (first.get("data") or {})
    splits = list(data.keys()) if isinstance(data, dict) else []
    split = "train" if "train" in splits else (splits[0] if splits else "train")
    if not config_name:
        return config, split
    wanted = config_name.lower()
    for candidate in configs:
        cname = str(candidate.get("config_name") or "").lower()
        if cname == wanted:
            cdata = candidate.get("data") or {}
            csplits = list(cdata.keys()) if isinstance(cdata, dict) else []
            csplit = ("train" if "train" in csplits
                      else (csplits[0] if csplits else "train"))
            return str(candidate.get("config_name") or config), csplit
    raise ValueError(
        f"config {config_name!r} not found in {dataset_id} — available: "
        + ", ".join(str(c.get("config_name") or "?") for c in configs))


def _resolve_fallback_urls(dataset_id: str) -> list[str]:
    """Direct resolve URLs (siblings) for a manual `curl` fallback."""
    try:
        info = _api_info(dataset_id)
        siblings = info.get("siblings") or []
        urls = [
            f"{_hf_base()}/datasets/{dataset_id}/resolve/main/{s['rfilename']}"
            for s in siblings
            if isinstance(s, dict) and str(s.get("rfilename", "")).endswith(
                (".parquet", ".json", ".jsonl", ".csv"))
        ][:8]
        return urls
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
            OSError):
        return []


def fetch_dataset(
    ref: str,
    dest_dir: str | Path,
    *,
    max_rows: int = 500,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch rows for `ref` (catalog name or raw HF id) and normalize them.

    Writes `<dest>/<name>.jsonl` + `<name>.manifest.json`.  Registration in
    the DatasetRegistry is done by the caller via `make_registry_register`
    (keeps this module context-free).
    """
    entry = lookup(ref) or _ad_hoc(ref)
    if not entry:
        raise ToolError(
            f"unknown dataset {ref!r} — pass a catalog name (see "
            "train_datasets → free_catalog) or an HF id like org/name"
            " (add :Config for a dataset subset, e.g. "
            "UltraData-SFT-Agent-2609:Code-Agent)")
    dataset_id = entry["id"]
    wanted_config = str(entry.get("config") or "").strip()
    normalize = _NORMALIZERS.get(str(entry.get("normalize") or "generic"),
                                 _norm_generic)
    max_rows = max(1, min(int(max_rows or 500), _MAX_ROWS_HARD))

    # 1. existence / gate check
    try:
        info = _api_info(dataset_id)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            page = f"{_hf_base()}/datasets/{dataset_id}"
            raise ToolError(
                f"{dataset_id} is gated — accept its terms at {page} in a "
                "browser (with your HF account), then retry") from None
        if exc.code == 404:
            raise ToolError(f"dataset {dataset_id} not found on HuggingFace") from None
        raise ToolError(f"HF API error {exc.code} for {dataset_id}") from None
    if info.get("gated") in {True, "auto", "manual"}:
        page = f"{_hf_base()}/datasets/{dataset_id}"
        raise ToolError(
            f"{dataset_id} requires accepting terms — open {page}, sign in "
            f"and accept, then retry (license: {entry.get('license', '?')})")
    if info.get("private") or info.get("disabled"):
        raise ToolError(f"{dataset_id} is not publicly fetchable")

    # 2. rows (paged)
    try:
        config, split = _server_splits(dataset_id, wanted_config)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        urls = _resolve_fallback_urls(dataset_id)
        fallback = "\n".join(urls) or f"{_hf_base()}/datasets/{dataset_id}"
        raise ToolError(
            f"the rows API is unavailable for {dataset_id} ({exc}); fetch "
            f"manually with one of: {fallback}") from None
    rows_out: list[dict[str, Any]] = []
    offset = 0
    dropped = 0
    total = 0
    while len(rows_out) < max_rows:
        page_len = min(_PAGE, max_rows - len(rows_out))
        try:
            page = _server_rows(dataset_id, config, split, offset, page_len)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
                OSError) as exc:
            if offset == 0:
                urls = _resolve_fallback_urls(dataset_id)
                raise ToolError(
                    f"rows fetch failed for {dataset_id} ({exc}); fallback "
                    "urls: " + ("\n".join(urls) or "—")) from None
            break  # keep what we have on a mid-stream failure
        features = page.get("features") or []
        total = int(page.get("num_rows_total") or 0)
        page_rows = page.get("rows") or []
        for item in page_rows:
            if not isinstance(item, dict):
                continue
            raw = item.get("row")
            if isinstance(raw, list):
                # datasets-server returns rows positionally
                row = {
                    str(f.get("name") or f"col{i}"): v
                    for i, (f, v) in enumerate(zip(features, raw))
                }
            elif isinstance(raw, dict):
                row = raw
            else:
                continue
            norm = normalize(row)
            if not norm:
                dropped += 1
                continue
            rows_out.append(norm)
            if len(rows_out) >= max_rows:
                break
        offset += len(page_rows)
        if not page_rows or (total and offset >= total):
            break
        time.sleep(0.1)  # be gentle with the shared server

    if not rows_out:
        raise ToolError(
            f"fetched 0 usable rows from {dataset_id} (config={config}, "
            f"split={split}) — the schema may not match its normalizer")

    # 3. write + manifest
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-z0-9._-]", "_", entry["name"].lower())
    if config and config.lower() not in name.lower():
        name = f"{name}-{re.sub(r'[^a-z0-9._-]', '-', config.lower())}"
    jsonl_path = dest / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "name": entry["name"],
        "source": dataset_id,
        "kind": entry["kind"],
        "license": entry.get("license", ""),
        "rows": len(rows_out),
        "dropped": dropped,
        "config": config,
        "split": split,
        "fetched_at": time.time(),
        "path": str(jsonl_path),
    }
    (dest / f"{name}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    return {"ok": True, **manifest}


def make_registry_register(context: Any) -> Callable[[dict[str, Any]], None]:
    """Return a register-callback bound to this context's dataset registry.

    Kept as a factory so the module never touches context plumbing at
    import time (tools pass `register_cb(out)` after a fetch).
    """
    def _register(out: dict[str, Any]) -> None:
        try:
            from ..training.dataset import DatasetRegistry

            path = out["path"]
            kind = "chat"
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    first = json.loads(fh.readline())
                if "conversations" in first:
                    kind = "sharegpt"
                elif "instruction" in first:
                    kind = "alpaca"
            except (OSError, ValueError) as e:
                _log.debug("dataset kind sniff failed for %s: %s", path, e)
            DatasetRegistry(context.db).register(
                name=f"{out['name']}-fetched",
                path=path,
                kind=kind,
                metadata={
                    "source": out.get("source", ""),
                    "license": out.get("license", ""),
                    "fetched_at": out.get("fetched_at", 0.0),
                },
            )
        except Exception as exc:  # noqa: BLE001 - never fail the fetch on this
            _log.warning("dataset registration failed: %s", exc)

    return _register
