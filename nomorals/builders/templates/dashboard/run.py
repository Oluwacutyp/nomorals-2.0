"""$PROJECT_NAME -- stdlib-only data dashboard (no third-party dependencies).

Serves a single-page HTML dashboard with an inline canvas chart (no CDN,
works offline) plus JSON endpoints:

    GET  /            dashboard page (canvas chart + summary cards)
    GET  /api/health  JSON health check
    GET  /api/data    the dataset with computed summary stats

The dataset lives in data.json next to run.py; edit it and refresh.

Run:
    python run.py [--port 8000]
    PORT=9000 python run.py
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT = "$PROJECT_NAME"
VERSION = "0.1.0"

DATA_PATH = Path(__file__).with_name("data.json")


def load_dataset() -> dict:
    """Read data.json; fail fast with a clear error, never a guess."""
    try:
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"data.json not found next to {Path(__file__).name}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"data.json is not valid JSON: {exc}")


def summarize(rows: list[dict]) -> dict:
    """Compute summary stats; empty input yields zeros, not a crash."""
    if not rows:
        return {"points": 0, "total_requests": 0, "total_errors": 0,
                "error_rate": 0.0, "avg_latency_ms": 0.0,
                "max_latency_ms": 0.0}
    total_requests = sum(int(r.get("requests", 0)) for r in rows)
    total_errors = sum(int(r.get("errors", 0)) for r in rows)
    latencies = [float(r.get("latency_ms", 0.0)) for r in rows]
    return {
        "points": len(rows),
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_requests, 4)
        if total_requests else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
    }


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@PROJECT@@ dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0d1117;
         color: #e6edf3; }
  main { max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { font-size: 1.8rem; margin-bottom: .25rem; }
  p.sub { color: #8b949e; margin-bottom: 1.5rem; }
  .cards { display: grid; grid-template-columns: repeat(3, 1fr);
           gap: 1rem; margin-bottom: 1.5rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
          padding: 1rem 1.25rem; }
  .card .label { font-size: .8rem; color: #8b949e; }
  .card .value { font-size: 1.6rem; font-variant-numeric: tabular-nums; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
           padding: 1.25rem 1.5rem; }
  canvas { width: 100%; height: 260px; display: block; }
  .err { color: #f85149; }
</style>
</head>
<body>
<main>
  <h1>@@PROJECT@@</h1>
  <p class="sub">Live dashboard &mdash; data from <code>/api/data</code>.</p>
  <div class="cards">
    <div class="card"><div class="label">Total requests</div>
      <div class="value" id="c-req">&ndash;</div></div>
    <div class="card"><div class="label">Error rate</div>
      <div class="value err" id="c-err">&ndash;</div></div>
    <div class="card"><div class="label">Avg latency</div>
      <div class="value" id="c-lat">&ndash;</div></div>
  </div>
  <div class="panel">
    <canvas id="chart" width="860" height="260"></canvas>
  </div>
</main>
<script>
async function load() {
  const res = await fetch("/api/data");
  const data = await res.json();
  const s = data.summary;
  document.getElementById("c-req").textContent = s.total_requests;
  document.getElementById("c-err").textContent =
    (s.error_rate * 100).toFixed(2) + "%";
  document.getElementById("c-lat").textContent = s.avg_latency_ms + " ms";
  draw(data.rows);
}
function draw(rows) {
  const cv = document.getElementById("chart");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, pad = 36;
  ctx.clearRect(0, 0, W, H);
  if (!rows.length) {
    ctx.fillStyle = "#8b949e";
    ctx.fillText("no data", pad, H / 2);
    return;
  }
  const maxR = Math.max.apply(null, rows.map(function (r) { return r.requests; }));
  const maxL = Math.max.apply(null, rows.map(function (r) { return r.latency_ms; }));
  const x = function (i) {
    return pad + (i * (W - 2 * pad)) / Math.max(rows.length - 1, 1);
  };
  const yR = function (v) { return H - pad - (v / maxR) * (H - 2 * pad); };
  const yL = function (v) { return H - pad - (v / maxL) * (H - 2 * pad); };
  ctx.strokeStyle = "#30363d";
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad);
  ctx.lineTo(W - pad, H - pad); ctx.stroke();
  ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 2; ctx.beginPath();
  rows.forEach(function (r, i) {
    if (i === 0) ctx.moveTo(x(i), yR(r.requests));
    else ctx.lineTo(x(i), yR(r.requests));
  });
  ctx.stroke();
  ctx.strokeStyle = "#f85149"; ctx.lineWidth = 2; ctx.beginPath();
  rows.forEach(function (r, i) {
    if (i === 0) ctx.moveTo(x(i), yL(r.latency_ms));
    else ctx.lineTo(x(i), yL(r.latency_ms));
  });
  ctx.stroke();
  ctx.fillStyle = "#8b949e"; ctx.font = "12px system-ui";
  ctx.fillText("requests", pad + 6, pad - 8);
  ctx.fillStyle = "#f85149";
  ctx.fillText("latency ms", pad + 80, pad - 8);
}
load().catch(function (e) {
  document.querySelector("p.sub").textContent = "failed to load data: " + e;
});
</script>
</body>
</html>
"""


def _page() -> str:
    return _PAGE.replace("@@PROJECT@@", PROJECT)


class Handler(BaseHTTPRequestHandler):
    server_version = f"{PROJECT}/{VERSION}"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, _page().encode("utf-8"),
                       "text/html; charset=utf-8")
        elif path == "/api/health":
            self._json(200, {"status": "ok", "project": PROJECT,
                             "version": VERSION})
        elif path == "/api/data":
            dataset = load_dataset()
            rows = dataset.get("rows", [])
            self._json(200, {"project": PROJECT, "rows": rows,
                             "summary": summarize(rows)})
        else:
            self._json(404, {"error": "not found", "path": path})

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
        pass  # keep test/console output clean


def make_server(host: str = "127.0.0.1",
                port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="$PROJECT_NAME",
        description=f"{PROJECT} -- stdlib-only data dashboard.")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--host",
                        default=os.environ.get("HOST", "127.0.0.1"))
    args = parser.parse_args(argv)
    load_dataset()  # fail fast if the dataset is missing/broken
    server = make_server(args.host, args.port)
    print(f"{PROJECT} dashboard on "
          f"http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # noqa: E103, E106 - deliberate shutdown hook
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
