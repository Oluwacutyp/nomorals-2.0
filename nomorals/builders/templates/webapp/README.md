# $PROJECT_NAME

A tiny stdlib-only web app: an HTML landing page plus a JSON API.
No third-party dependencies — Python's standard library is enough.

## Run

```sh
python run.py                 # listens on 127.0.0.1:8000
python run.py --port 9000     # pick a port
PORT=9000 python run.py       # or via env
```

Then open http://127.0.0.1:8000/ or:

```sh
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/echo \
     -H 'Content-Type: application/json' -d '{"hello":"world"}'
```

## Routes

| Method | Path         | Description            |
|--------|--------------|------------------------|
| GET    | `/`          | HTML landing page      |
| GET    | `/api/health`| JSON health check      |
| POST   | `/api/echo`  | Echoes the JSON body   |

## Test

```sh
python -m unittest discover -s tests -t .
```
