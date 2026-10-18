# $PROJECT_NAME

A tiny stdlib-only data dashboard: one HTML page with an inline canvas
chart (no CDN, works offline) plus JSON endpoints. No third-party
dependencies — Python's standard library is enough.

## Run

```sh
python run.py                 # serves on 127.0.0.1:8000
python run.py --port 9000     # pick a port
PORT=9000 python run.py       # or via env
```

Then open http://127.0.0.1:8000/ — the page fetches `/api/data`,
renders summary cards and draws requests vs. latency on a canvas.

## Endpoints

| Method | Path         | Description                              |
|--------|--------------|------------------------------------------|
| GET    | `/`          | Dashboard page (chart + summary cards)   |
| GET    | `/api/health`| JSON health check                        |
| GET    | `/api/data`  | Dataset rows plus computed summary stats |

## Your data

Edit `data.json` (rows of `{"t", "requests", "errors", "latency_ms"}`)
and refresh — the chart and the summary cards update from it.

## Test

```sh
python -m unittest discover -s tests -t .
```
