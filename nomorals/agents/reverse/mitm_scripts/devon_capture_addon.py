"""Devon's mitmproxy capture addon — owner-invoked, local, hers.

Run it where mitmproxy already lives, only on traffic the owner
routes through their own proxy:

    mitmdump -s devon_capture_addon.py --set devon_out=flows.devon.jsonl

Every completed HTTP flow is appended to the out file as one JSON
line carrying the facts Devon's traffic rung folds into Evidence:
who talked to whom, the request line and host, the status, sizes,
content types and timing. Bodies are NEVER copied — sizes and a
sha256 of each body are the record; the content itself stays in
the owner's own capture.

This file is deliberately dependency-light: it runs inside
mitmproxy's process, imports mitmproxy the way every addon does,
and touches nothing else. It opens no listeners of its own, starts
nothing in the background, and intercepts nothing by itself —
mitmproxy the owner started is the only interceptor in the room.
Folding the JSONL into the evidence store is the organ's job
(:mod:`nomorals.agents.reverse.traffic`), never the addon's.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

try:  # mitmproxy is present exactly where this file is meant to run
    from mitmproxy import ctx, http
except ImportError:  # pragma: no cover - outside mitmproxy
    ctx = None  # type: ignore[assignment]
    http = None  # type: ignore[assignment]

ADDON_ID = "devon.mitm-capture-addon"
ADDON_VERSION = "1"


def _out_path() -> str:
    configured = ""
    if ctx is not None:
        try:
            configured = str(ctx.options.devon_out or "")
        except Exception:  # noqa: BLE001 - options not loaded yet
            configured = ""
    return (configured or os.environ.get("DEVON_RE_EVIDENCE")
            or "devon-flows.jsonl")


def _body_facts(content: bytes | None) -> dict:
    raw = content or b""
    return {"size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest() if raw else ""}


def flow_record(flow: http.HTTPFlow) -> dict:
    """One completed flow → the JSONL record. Pure over the flow
    object, so the organ's tests drive this function directly
    with real mitmproxy flows."""
    req = flow.request
    resp = flow.response
    record = {
        "addon": ADDON_ID,
        "addon_version": ADDON_VERSION,
        "ts": float(req.timestamp_start or time.time()),
        "client": list(flow.client_conn.peername or ()),
        "server": list(flow.server_conn.address or ()),
        "method": req.method,
        "scheme": req.scheme,
        "host": req.host,
        "port": req.port,
        "path": req.path,
        "http_version": req.http_version,
        "request_headers": [[k, v] for k, v in req.headers.items()],
        "request_body": _body_facts(req.raw_content),
        "status": getattr(resp, "status_code", None),
        "response_headers": (
            [[k, v] for k, v in resp.headers.items()] if resp
            else []),
        "response_body": _body_facts(
            getattr(resp, "raw_content", None)),
        "duration_s": (
            round(float(resp.timestamp_end - req.timestamp_start), 6)
            if resp is not None and resp.timestamp_end
            and req.timestamp_start else None),
        "error": (str(flow.error.msg) if flow.error else ""),
    }
    return record


def _append(record: dict) -> None:
    path = _out_path()
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # an addon must never take the owner's proxy down with it;
        # the failure is said out loud on the console instead
        if ctx is not None:
            ctx.log.warn(
                f"devon capture addon: could not append to "
                f"{path}: {exc}")


class DevonCapture:
    """The addon object mitmproxy loads (addons list below)."""

    def load(self, loader) -> None:  # noqa: ANN001 - mitmproxy API
        loader.add_option(
            "devon_out", str, "",
            "file the Devon capture addon appends flow JSONL to "
            "(default: $DEVON_RE_EVIDENCE or ./devon-flows.jsonl)")

    def response(self, flow: http.HTTPFlow) -> None:
        _append(flow_record(flow))

    def error(self, flow: http.HTTPFlow) -> None:
        _append(flow_record(flow))


addons = [DevonCapture()]
