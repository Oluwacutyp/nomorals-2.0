# $PROJECT_NAME

A tiny stdlib-only REST API: JSON CRUD over an in-memory item store.
No third-party dependencies — Python's standard library is enough.

## Run

```sh
python run.py                 # listens on 127.0.0.1:8000
python run.py --port 9000     # pick a port
PORT=9000 python run.py       # or via env
```

## Endpoints

| Method | Path              | Description                    |
|--------|-------------------|--------------------------------|
| GET    | `/`               | API index (project + endpoints) |
| GET    | `/api/health`     | JSON health check              |
| GET    | `/api/items`      | List all items                 |
| POST   | `/api/items`      | Create an item `{"name": "..."}` |
| GET    | `/api/items/<id>` | Fetch one item                 |
| PATCH  | `/api/items/<id>` | Partial update                 |
| DELETE | `/api/items/<id>` | Delete an item                 |

Errors are JSON envelopes (`{"error": "..."}`); wrong methods get 405
with an `Allow` header.

```sh
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/items \
     -H 'Content-Type: application/json' -d '{"name":"first"}'
curl http://127.0.0.1:8000/api/items
```

## Test

```sh
python -m unittest discover -s tests -t .
```
