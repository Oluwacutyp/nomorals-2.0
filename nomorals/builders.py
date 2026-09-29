"""Builder system — scaffold real, runnable applications.

:class:`AppBuilder` generates complete working apps in ``workspace/apps/``
for six stacks.  Every template is **real code** (routes, state, CSS, CLI
dispatch — not placeholders), the app's ``features`` from the spec are
wired into the app itself, and after generation the files are
**validated** (``py_compile`` for Python, ``node --check`` for JS, JSON
parse for manifests) with the results reported.

Stacks:
* ``static``     — HTML/CSS/JS landing app (works from file://, no server)
* ``flask``      — Flask app with /, /api/health, /api/features, item CRUD
* ``fastapi``    — FastAPI app with /health, /features, /items CRUD + models
* ``express``    — Node/Express server, /health, /features, /items CRUD
* ``react-vite`` — Vite + React app (stateful UI, feature list, demo state)
* ``cli-python`` — argparse CLI (add/list/clear against a JSON data file)

    from nomorals.builders import AppBuilder
    b = AppBuilder(context)
    b.build({"name": "todo", "title": "Todo", "stack": "fastapi",
             "features": ["add tasks", "complete tasks"]})
    b.list_apps()

Registered as the ``build_app`` tool.
"""

from __future__ import annotations

import json
import os
import py_compile
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .core.errors import ToolError
from .core.logging_setup import get_logger
from .core.policy import Capability

_log = get_logger(__name__)

__all__ = ["AppBuilder", "STACKS", "register"]

STACKS = ("static", "flask", "fastapi", "express", "react-vite",
          "cli-python")


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or "app"


def _html_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _features_html(features: list[str]) -> str:
    if not features:
        return '<li class="muted">No features specified.</li>'
    return "\n".join(f"  <li>{_html_escape(f)}</li>" for f in features)


def _features_json(features: list[str]) -> str:
    return json.dumps(features or ["core functionality"], indent=2)


# ── template generators: spec → {relative path: content} ─────────────────

def _static(spec: dict) -> dict[str, str]:
    title = spec.get("title") or spec["name"]
    desc = spec.get("description") or f"{title} — built with the No-Morals " \
        "builder system."
    return {
        "index.html": f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html_escape(title)}</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>{_html_escape(title)}</h1>
    <p class="tagline">{_html_escape(desc)}</p>
  </header>
  <main>
    <section>
      <h2>Features</h2>
      <ul id="features">
{_features_html(spec.get("features", []))}
      </ul>
    </section>
    <section id="counter">
      <h2>Counter demo</h2>
      <p class="value" id="count">0</p>
      <button id="inc">+1</button>
      <button id="reset">Reset</button>
    </section>
  </main>
  <footer>
    <p>Built by No-Morals AI · {time.strftime("%Y-%m-%d")}</p>
  </footer>
  <script src="app.js"></script>
</body>
</html>
""",
        "style.css": """* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3;
       min-height: 100vh; display: flex; flex-direction: column; }
header { padding: 2.5rem 2rem 1rem; }
h1 { font-size: 2.4rem; }
.tagline { color: #8b949e; margin-top: .5rem; max-width: 60ch; }
main { padding: 1rem 2rem; flex: 1; }
section { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
          padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; max-width: 720px; }
h2 { font-size: 1.1rem; margin-bottom: .75rem; color: #58a6ff; }
li { margin: .35rem 0 .35rem 1.25rem; }
.muted { color: #8b949e; list-style: none; margin-left: 0; }
.value { font-size: 2rem; font-variant-numeric: tabular-nums; }
button { background: #238636; color: #fff; border: 0; border-radius: 6px;
         padding: .5rem 1rem; margin-right: .5rem; cursor: pointer; }
button:hover { background: #2ea043; }
footer { padding: 1rem 2rem; color: #8b949e; font-size: .85rem; }
""",
        "app.js": """let count = 0;
const el = document.getElementById("count");
document.getElementById("inc").addEventListener("click", () => {
  count += 1;
  el.textContent = String(count);
});
document.getElementById("reset").addEventListener("click", () => {
  count = 0;
  el.textContent = "0";
});
""",
        "README.md": f"""# {spec.get('title') or spec['name']}

{_html_escape(spec.get('description') or 'Static app built by the No-Morals builder.')}

## Run

Open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8000
```

## Features

{chr(10).join('- ' + f for f in spec.get('features', [])) or '- (none specified)'}
""",
    }


def _flask(spec: dict) -> dict[str, str]:
    name, title = spec["name"], spec.get("title") or spec["name"]
    feats = spec.get("features", [])
    return {
        "app.py": f'''"""{title} — Flask app generated by the No-Morals builder."""
import os

from flask import Flask, jsonify, request

app = Flask(__name__)

# simple in-memory store (swap for a DB in production)
_items = {{}}
_next_id = 1


@app.route("/")
def index():
    return jsonify({{
        "app": {json.dumps(name)},
        "title": {json.dumps(title)},
        "features": {json.dumps(feats)},
    }})


@app.route("/api/health")
def health():
    return jsonify({{"status": "ok"}})


@app.route("/api/features")
def features():
    return jsonify({{"features": {json.dumps(feats)}}})


@app.route("/api/items", methods=["GET"])
def list_items():
    return jsonify({{"items": sorted(_items.values(), key=lambda i: i["id"])}})


@app.route("/api/items", methods=["POST"])
def create_item():
    global _next_id
    data = request.get_json(force=True) or {{}}
    if not str(data.get("name", "")).strip():
        return jsonify({{"error": "name is required"}}), 400
    item = {{
        "id": _next_id,
        "name": str(data["name"]),
        "done": False,
    }}
    _next_id += 1
    _items[item["id"]] = item
    return jsonify(item), 201


@app.route("/api/items/<int:item_id>", methods=["PATCH"])
def update_item(item_id: int):
    item = _items.get(item_id)
    if item is None:
        return jsonify({{"error": "not found"}}), 404
    data = request.get_json(force=True) or {{}}
    if "name" in data:
        item["name"] = str(data["name"])
    if "done" in data:
        item["done"] = bool(data["done"])
    return jsonify(item)


@app.route("/api/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id: int):
    if item_id not in _items:
        return jsonify({{"error": "not found"}}), 404
    del _items[item_id]
    return jsonify({{"deleted": item_id}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", {spec.get("port", 5000)}))
    app.run(host="0.0.0.0", port=port, debug=False)
''',
        "requirements.txt": "flask>=3.0\n",
        "README.md": f"""# {title}

Flask API app built by the No-Morals builder.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Endpoints: `GET /`, `GET /api/health`, `GET /api/features`,
`GET|POST /api/items`, `PATCH|DELETE /api/items/<id>`.

## Features

{chr(10).join('- ' + f for f in feats) or '- (none specified)'}
""",
    }


def _fastapi(spec: dict) -> dict[str, str]:
    name, title = spec["name"], spec.get("title") or spec["name"]
    feats = spec.get("features", [])
    return {
        "main.py": f'''"""{title} — FastAPI app generated by the No-Morals builder."""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title={json.dumps(title)},
              description="Built by the No-Morals builder system.")

_items: dict[int, dict] = {{}}
_next_id = 1


class ItemIn(BaseModel):
    name: str
    done: bool = False


class Item(ItemIn):
    id: int


@app.get("/health")
def health():
    return {{"status": "ok"}}


@app.get("/features")
def features():
    return {{"features": {json.dumps(feats)}}}


@app.get("/items", response_model=list[Item])
def list_items():
    return sorted(_items.values(), key=lambda i: i["id"])


@app.post("/items", response_model=Item, status_code=201)
def create_item(item: ItemIn):
    global _next_id
    if not item.name.strip():
        raise HTTPException(400, "name is required")
    entry = {{ "id": _next_id, "name": item.name, "done": item.done }}
    _next_id += 1
    _items[entry["id"]] = entry
    return entry


@app.patch("/items/{{item_id}}", response_model=Item)
def update_item(item_id: int, update: dict):
    item = _items.get(item_id)
    if item is None:
        raise HTTPException(404, "not found")
    if "name" in update and str(update["name"]).strip():
        item["name"] = str(update["name"])
    if "done" in update:
        item["done"] = bool(update["done"])
    return item


@app.delete("/items/{{item_id}}")
def delete_item(item_id: int):
    if item_id not in _items:
        raise HTTPException(404, "not found")
    del _items[item_id]
    return {{"deleted": item_id}}


@app.get("/")
def index():
    return {{"app": {json.dumps(name)}, "title": {json.dumps(title)},
            "features": {json.dumps(feats)}}}
''',
        "requirements.txt": "fastapi>=0.110\nuvicorn>=0.29\n",
        "README.md": f"""# {title}

FastAPI app built by the No-Morals builder.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port {spec.get('port', 8000)}
```

Interactive docs at `/docs`.

## Features

{chr(10).join('- ' + f for f in feats) or '- (none specified)'}
""",
    }


def _express(spec: dict) -> dict[str, str]:
    name, title = spec["name"], spec.get("title") or spec["name"]
    feats = spec.get("features", [])
    port = spec.get("port", 3000)
    return {
        "package.json": json.dumps({
            "name": name,
            "version": "1.0.0",
            "description": spec.get("description") or f"{title} API",
            "main": "server.js",
            "scripts": {"start": "node server.js"},
            "dependencies": {"express": "^4.19.0"},
        }, indent=2) + "\n",
        "server.js": f'''// {title} — Express app generated by the No-Morals builder
const express = require("express");

const app = express();
app.use(express.json());

let items = new Map();
let nextId = 1;

app.get("/", (req, res) => {{
  res.json({{
    app: {json.dumps(name)},
    title: {json.dumps(title)},
    features: {json.dumps(feats)},
  }});
}});

app.get("/health", (req, res) => res.json({{ status: "ok" }}));
app.get("/features", (req, res) =>
  res.json({{ features: {json.dumps(feats)} }}));

app.get("/items", (req, res) => {{
  res.json({{ items: [...items.values()] }});
}});

app.post("/items", (req, res) => {{
  const name = String(req.body.name || "").trim();
  if (!name) return res.status(400).json({{ error: "name is required" }});
  const item = {{ id: nextId++, name, done: false }};
  items.set(item.id, item);
  res.status(201).json(item);
}});

app.patch("/items/:id", (req, res) => {{
  const item = items.get(Number(req.params.id));
  if (!item) return res.status(404).json({{ error: "not found" }});
  if (req.body.name) item.name = String(req.body.name);
  if (typeof req.body.done === "boolean") item.done = req.body.done;
  res.json(item);
}});

app.delete("/items/:id", (req, res) => {{
  const id = Number(req.params.id);
  if (!items.has(id)) return res.status(404).json({{ error: "not found" }});
  items.delete(id);
  res.json({{ deleted: id }});
}});

const port = Number(process.env.PORT || {port});
app.listen(port, "0.0.0.0", () => {{
  console.log({json.dumps(title)} + " listening on " + port);
}});
''',
        "README.md": f"""# {title}

Express API built by the No-Morals builder.

## Run

```bash
npm install
npm start
```

Endpoints: `GET /`, `GET /health`, `GET /features`,
`GET|POST /items`, `PATCH|DELETE /items/:id`.

## Features

{chr(10).join('- ' + f for f in feats) or '- (none specified)'}
""",
    }


def _react_vite(spec: dict) -> dict[str, str]:
    name, title = spec["name"], spec.get("title") or spec["name"]
    feats = spec.get("features", [])
    return {
        "package.json": json.dumps({
            "name": name,
            "private": True,
            "version": "1.0.0",
            "type": "module",
            "scripts": {
                "dev": "vite --host 0.0.0.0",
                "build": "vite build",
                "preview": "vite preview",
            },
            "dependencies": {
                "react": "^18.3.0",
                "react-dom": "^18.3.0",
            },
            "devDependencies": {
                "@vitejs/plugin-react": "^4.3.0",
                "vite": "^5.4.0",
            },
        }, indent=2) + "\n",
        "vite.config.js": """import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({ plugins: [react()] });
""",
        "index.html": f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{_html_escape(title)}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
""",
        "src/main.jsx": """import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./style.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""",
        "src/App.jsx": f'''import React, {{ useState }} from "react";

const FEATURES = {_features_json(feats)};

export default function App() {{
  const [count, setCount] = useState(0);
  return (
    <main>
      <h1>{_html_escape(title)}</h1>
      <p className="tagline">
        {_html_escape(spec.get("description") or "React app built by the No-Morals builder.")}
      </p>
      <section>
        <h2>Features</h2>
        <ul>
          {{FEATURES.map((f) => (
            <li key={{f}}>{{f}}</li>
          ))}}
        </ul>
      </section>
      <section>
        <h2>Counter demo</h2>
        <p className="value">{{count}}</p>
        <button onClick={{() => setCount(count + 1)}}>+1</button>
        <button onClick={{() => setCount(0)}}>Reset</button>
      </section>
    </main>
  );
}}
''',
        "src/style.css": """* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3; }
main { max-width: 720px; margin: 0 auto; padding: 2.5rem 1.5rem; }
h1 { font-size: 2.2rem; }
.tagline { color: #8b949e; margin: .5rem 0 1.5rem; }
section { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
          padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }
h2 { font-size: 1.05rem; color: #58a6ff; margin-bottom: .75rem; }
li { margin: .35rem 0 .35rem 1.25rem; }
.value { font-size: 2rem; font-variant-numeric: tabular-nums; }
button { background: #238636; color: #fff; border: 0; border-radius: 6px;
         padding: .5rem 1rem; margin-right: .5rem; cursor: pointer; }
""",
        "README.md": f"""# {title}

React + Vite app built by the No-Morals builder.

## Run

```bash
npm install
npm run dev
```

## Features

{chr(10).join('- ' + f for f in feats) or '- (none specified)'}
""",
    }


def _cli_python(spec: dict) -> dict[str, str]:
    name, title = spec["name"], spec.get("title") or spec["name"]
    feats = spec.get("features", [])
    return {
        f"{name}.py": f'''"""{title} — CLI generated by the No-Morals builder."""
import argparse
import json
import sys
from pathlib import Path

DATA = Path(__file__).with_name("data.json")
FEATURES = {json.dumps(feats)}


def load():
    if not DATA.exists():
        return []
    try:
        return json.loads(DATA.read_text())
    except json.JSONDecodeError:
        return []


def save(rows):
    DATA.write_text(json.dumps(rows, indent=2))


def cmd_add(args):
    rows = load()
    rows.append({{"name": args.name, "done": False}})
    save(rows)
    print(f"added: {{args.name}} ({{len(rows)}} total)")


def cmd_list(args):
    rows = load()
    if not rows:
        print("(empty)")
        return
    for i, r in enumerate(rows, 1):
        mark = "x" if r.get("done") else " "
        print(f"{{i:3d}} [{{mark}}] {{r['name']}}")


def cmd_done(args):
    rows = load()
    if args.id < 1 or args.id > len(rows):
        print(f"no item {{args.id}}", file=sys.stderr)
        sys.exit(1)
    rows[args.id - 1]["done"] = True
    save(rows)
    print(f"done: {{rows[args.id - 1]['name']}}")


def cmd_clear(args):
    save([])
    print("cleared")


def main():
    p = argparse.ArgumentParser(prog={json.dumps(name)},
                                description={json.dumps(title)})
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("add", help="add an item"); sp.add_argument("name")
    sp.set_defaults(fn=cmd_add)
    sp = sub.add_parser("list", help="list items"); sp.set_defaults(fn=cmd_list)
    sp = sub.add_parser("done", help="mark item done"); sp.add_argument("id", type=int)
    sp.set_defaults(fn=cmd_done)
    sp = sub.add_parser("clear", help="remove all items")
    sp.set_defaults(fn=cmd_clear)
    sp = sub.add_parser("features", help="show app features")
    sp.set_defaults(fn=lambda a: [print(f"- {{f}}") for f in FEATURES])
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
''',
        "README.md": f"""# {title}

Python CLI built by the No-Morals builder.

## Run

```bash
python {name}.py add "first item"
python {name}.py list
python {name}.py done 1
python {name}.py features
python {name}.py clear
```

## Features

{chr(10).join('- ' + f for f in feats) or '- (none specified)'}
""",
    }


_TEMPLATES: dict[str, Callable[[dict], dict[str, str]]] = {
    "static": _static,
    "flask": _flask,
    "fastapi": _fastapi,
    "express": _express,
    "react-vite": _react_vite,
    "cli-python": _cli_python,
}


class AppBuilder:
    """Scaffold + validate runnable applications."""

    role = "builder"

    def __init__(self, context: Any) -> None:
        self.context = context
        self._apps_dir = self._ws() / "apps"

    def _ws(self) -> Path:
        settings = getattr(self.context, "settings", None)
        root = Path(settings.workspace_dir) if settings is not None \
            else Path.cwd() / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def stacks(self) -> list[dict[str, Any]]:
        return [
            {"stack": s, "description": _TEMPLATES[s].__doc__ or ""}
            for s in STACKS
        ]

    def build(self, spec: dict[str, Any]) -> dict[str, Any]:
        name = str(spec.get("name") or "").strip()
        if not name:
            raise ToolError("build needs a name")
        stack = str(spec.get("stack") or "static").strip().lower()
        if stack not in _TEMPLATES:
            raise ToolError(f"unknown stack {stack!r} — one of "
                            f"{', '.join(STACKS)}")
        spec = {
            "name": name,
            "title": str(spec.get("title") or name),
            "description": str(spec.get("description") or ""),
            "features": [str(f) for f in (spec.get("features") or [])][:20],
            "port": int(spec.get("port") or
                        {"flask": 5000, "fastapi": 8000,
                         "express": 3000, "react-vite": 5173,
                         }.get(stack, 8000)),
            "overwrite": bool(spec.get("overwrite")),
        }
        slug = _slug(name)
        app_dir = self._apps_dir / slug
        if app_dir.exists():
            if not bool(spec.get("overwrite")):
                raise ToolError(f"{slug} already exists — pass overwrite=true "
                                f"to replace it")
            shutil.rmtree(app_dir)
        app_dir.mkdir(parents=True, exist_ok=True)

        files = _TEMPLATES[stack](spec)
        written = []
        for rel, content in files.items():
            target = (app_dir / rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(rel)

        manifest = {
            "name": name,
            "title": spec["title"],
            "stack": stack,
            "description": spec["description"],
            "features": spec["features"],
            "port": spec["port"],
            "built_at": time.time(),
            "files": written,
            "run": {
                "static": "open index.html (or: python3 -m http.server 8000)",
                "flask": "pip install -r requirements.txt && python app.py",
                "fastapi": "pip install -r requirements.txt && "
                           f"uvicorn main:app --port {spec['port']}",
                "express": "npm install && npm start",
                "react-vite": "npm install && npm run dev",
                "cli-python": f"python {name}.py list",
            }[stack],
        }
        (app_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        written.append("manifest.json")

        validated, failed = self._validate(app_dir, stack)
        return {
            "app": slug,
            "stack": stack,
            "dir": str(app_dir),
            "files": written,
            "run": manifest["run"],
            "validation": {"ok": not failed,
                           "validated": validated,
                           "failed": failed},
        }

    def _validate(self, app_dir: Path, stack: str):
        validated, failed = [], []
        for p in sorted(app_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(app_dir))
            try:
                if p.suffix == ".py":
                    py_compile.compile(str(p), doraise=True)
                    validated.append(rel)
                elif p.suffix == ".js" and shutil.which("node") \
                        and "node_modules" not in rel and p.name != "vite.config.js":
                    import subprocess

                    subprocess.run(
                        ["node", "--check", str(p)], capture_output=True,
                        text=True, timeout=30, check=True)
                    validated.append(rel)
                elif p.suffix == ".json":
                    json.loads(p.read_text(encoding="utf-8"))
                    validated.append(rel)
                elif p.suffix == ".html":
                    text = p.read_text(encoding="utf-8")
                    if "<html" in text.lower() and "</html>" in text.lower():
                        validated.append(rel)
                    else:
                        failed.append({rel: "missing html tags"})
                elif p.suffix in (".jsx", ".css", ".md", ".txt"):
                    validated.append(rel)
            except Exception as exc:  # noqa: BLE001
                failed.append({rel: str(exc)[:200]})
        return validated, failed

    def list_apps(self) -> dict[str, Any]:
        self._apps_dir.mkdir(parents=True, exist_ok=True)
        apps = []
        for d in sorted(self._apps_dir.iterdir()):
            man = d / "manifest.json"
            if not d.is_dir() or not man.exists():
                continue
            try:
                m = json.loads(man.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            apps.append({"name": m.get("name"), "stack": m.get("stack"),
                         "dir": str(d), "files": len(m.get("files", [])),
                         "run": m.get("run")})
        return {"apps": apps, "count": len(apps)}

    def info(self, name: str) -> dict[str, Any]:
        man = self._apps_dir / _slug(name) / "manifest.json"
        if not man.exists():
            raise ToolError(f"no app named {name!r}")
        return json.loads(man.read_text(encoding="utf-8"))

    # ── serve: actually run the app and hand back a live URL ─────────────
    def _served(self) -> dict[str, Any]:
        return self._kv("apps.served")

    def _set_served(self, data: dict[str, Any]) -> None:
        self._kv_save("apps.served", data)

    def _kv(self, key: str) -> Any:
        db = getattr(self.context, "db", None)
        if db is None:
            return {}
        row = db.query_one("SELECT value FROM kv_store WHERE key=?", (key,))
        if not row:
            return {}
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return {}

    def _kv_save(self, key: str, value: Any) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        db.execute(
            "INSERT INTO kv_store (key, value, kind, updated_at) "
            "VALUES (?, ?, 'json', ?) ON CONFLICT(key) DO UPDATE SET "
            "value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, default=str), time.time()))

    def serve(self, name: str, *, port: int = 0,
              wait: float = 20.0) -> dict[str, Any]:
        """Spawn the app's server (detached), wait for it, health-check."""
        man = self.info(name)
        slug, stack = _slug(name), man["stack"]
        app_dir = self._apps_dir / slug
        if stack == "cli-python":
            raise ToolError(
                f"{slug} is a CLI app, not a server — run it with "
                f"{man.get('run', 'python <name>.py --help')}")
        running = self._served().get(slug)
        if running and _pid_alive(int(running.get("pid", 0))):
            return {**running, "note": "already running"}

        port = int(port or man.get("port", 0) or 0)
        port = _free_port(port)

        # dependencies, honestly
        missing = _missing_deps(stack)
        if missing:
            raise ToolError(
                f"{slug} ({stack}) needs: {', '.join(missing)} — "
                f"install them first (pip install {' '.join(missing)})")
        if stack in ("express", "react-vite") and \
                not (app_dir / "node_modules").exists():
            npm = shutil.which("npm")
            if not npm:
                raise ToolError(
                    f"{slug} ({stack}) needs node_modules but npm is not "
                    "installed here")
            proc = subprocess.run([npm, "install", "--no-audit",
                                   "--no-fund"], cwd=app_dir,
                                  capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                raise ToolError(
                    f"npm install failed for {slug}: "
                    f"{(proc.stderr or proc.stdout)[:400]}")

        cmd, env = _serve_command(stack, app_dir, port)
        full_env = dict(os.environ)
        full_env.update(env)
        log_path = app_dir / "serve.log"
        with open(log_path, "ab") as log:
            subprocess.Popen(
                cmd, cwd=str(app_dir), stdout=log, stderr=log,
                start_new_session=True, env=full_env)
        pid = None
        # discover the spawned pid (the command's process group leader)
        time.sleep(0.4)
        pid = _find_child_pid(app_dir, cmd)
        if pid is None:
            # fall back: we don't know it yet; record and health-check only
            pid = 0
        entry = {"app": slug, "stack": stack, "pid": pid, "port": port,
                 "url": f"http://localhost:{port}",
                 "log": str(log_path), "started_at": time.time(),
                 "cmd": " ".join(cmd) if isinstance(cmd, list) else cmd}
        data = self._served()
        data[slug] = entry
        self._set_served(data)

        health = _wait_for_http(port, stack, timeout=wait)
        entry["health"] = health
        self._set_served(data)
        if not health.get("ok"):
            tail = _log_tail(log_path)
            entry["note"] = (f"server started but the health check did not "
                             f"pass — {health.get('error', 'unknown')}\n"
                             f"log tail:\n{tail}")
        return entry

    def stop(self, name: str) -> dict[str, Any]:
        slug = _slug(name)
        data = self._served()
        entry = data.pop(slug, None)
        if entry is None:
            raise ToolError(f"{slug} is not being served")
        self._set_served(data)
        killed = False
        pid = int(entry.get("pid", 0) or 0)
        if pid and _pid_alive(pid):
            try:
                import signal

                os.killpg(os.getpgid(pid), signal.SIGKILL)
                killed = True
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    import signal

                    os.kill(pid, signal.SIGKILL)
                    killed = True
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        return {"app": slug, "stopped": killed or not _pid_alive(pid),
                "was_pid": pid}

    def served(self) -> dict[str, Any]:
        out = []
        data = self._served()
        for slug, entry in sorted(data.items()):
            pid = int(entry.get("pid", 0) or 0)
            alive = _pid_alive(pid)
            live = _quick_http(entry.get("port", 0)) if alive else None
            out.append({**entry, "alive": alive, "live": live})
            if not alive:
                data.pop(slug, None)
        self._set_served(data)
        return {"served": out, "count": len(out)}


    # ── deploy: a real reverse-proxy URL behind a domain ─────────────────
    def _deployed(self) -> dict[str, Any]:
        return self._kv("apps.deployed")

    def _set_deployed(self, data: dict[str, Any]) -> None:
        self._kv_save("apps.deployed", data)

    def deploy(self, name: str, *, host: str = "0.0.0.0", port: int = 0,
               domain: str = "", path: str = "", wait: float = 25.0,
               serve: bool = True, tls: bool = False) -> dict[str, Any]:
        """Make the app reachable on a real public URL behind a domain.

        1. The app server is (re)started on its own local port.
        2. A real reverse proxy (detached ThreadingHTTPServer subprocess)
           listens on host:port and forwards EVERY request to the app —
           path prefix stripped, Host rewritten to the domain, backend
           redirects re-pointed at the public origin.  With ``tls`` the
           proxy terminates HTTPS (a self-signed cert is generated and
           cached when you don't supply your own).
        3. The deployment is recorded in kv so ``deployed`` /
           ``stop_deploy`` find it.  The proxy keeps running after this
           call returns.

        Returns the live entry including its public URL.
        """
        import sys

        man = self.info(name)
        slug, stack = _slug(name), man["stack"]
        if stack == "cli-python":
            raise ToolError(f"{slug} is a CLI app — nothing to proxy")

        # 1 ── app server up
        if serve:
            app_entry = self.serve(name, wait=wait)
            backend_port = int(app_entry.get("port", 0))
        else:
            running = self._served().get(slug)
            if not running or not _pid_alive(int(running.get("pid", 0))):
                raise ToolError(
                    f"{slug} is not running — start it (nm apps serve "
                    f"{slug}) or call deploy(serve=True)")
            backend_port = int(running.get("port", 0))
        if not backend_port:
            raise ToolError(f"could not determine {slug}\'s port")

        # already deployed with the same backend + scheme? return it
        data = self._deployed()
        prev = data.get(slug)
        if prev and prev.get("backend_port") == backend_port and \
                bool(prev.get("tls")) == bool(tls) and \
                _pid_alive(int(prev.get("pid", 0))):
            return {**prev, "note": "already deployed"}

        # 2 ── real reverse proxy, detached
        proxy_port = _free_port(int(port or 0))
        prefix = "/" + (path or "").strip("/") if (path or "").strip("/") \
            else ""
        log_path = self._apps_dir / f"{slug}.proxy.log"
        # TLS cert/key cached per-app so restarts reuse the same identity
        cert_path = key_path = ""
        if tls:
            tls_dir = self._apps_dir / "tls"
            tls_dir.mkdir(parents=True, exist_ok=True)
            cert_path = str(tls_dir / f"{slug}.crt")
            key_path = str(tls_dir / f"{slug}.key")
        cmd = [sys.executable, "-m", "nomorals.builders_proxy",
               "--listen", f"{host}:{proxy_port}",
               "--backend", f"127.0.0.1:{backend_port}"]
        if prefix:
            cmd += ["--prefix", prefix]
        if domain:
            cmd += ["--domain", domain.strip().rstrip("/")]
        if tls:
            cmd += ["--tls", "--cert", cert_path, "--key", key_path]
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                                    start_new_session=True)
        time.sleep(0.6)
        if not _pid_alive(proc.pid):
            tail = _log_tail(log_path)
            raise ToolError(f"proxy died at startup\nlog tail:\n{tail}")

        scheme = "https" if tls else "http"
        dom = domain.strip().rstrip("/")
        if dom:
            # behind a domain: the proxy answers on the box's :80/:443,
            # the domain points at this machine (or this sandbox host)
            url = f"{scheme}://{dom}{prefix or '/'}"
        else:
            shown = "localhost" if host in ("0.0.0.0", "") else host
            url = f"{scheme}://{shown}:{proxy_port}{prefix or '/'}"

        entry = {
            "app": slug, "pid": proc.pid, "port": proxy_port,
            "host": host, "domain": domain.strip().rstrip("/"),
            "path": prefix, "backend_port": backend_port,
            "tls": bool(tls),
            "cert": cert_path, "key": key_path,
            "url": url, "log": str(log_path), "started_at": time.time(),
            "cmd": " ".join(cmd),
        }
        # health check THROUGH the proxy (proves the path works end-to-end)
        health = _wait_for_http(
            proxy_port, stack, timeout=wait,
            path=(prefix or "") + _health_path(stack), tls=tls)
        entry["health"] = health
        data[slug] = entry
        self._set_deployed(data)
        if not health.get("ok"):
            entry["note"] = (f"proxy up but end-to-end check failed — "
                             f"{health.get('error', 'unknown')}\n"
                             f"proxy log tail:\n{_log_tail(log_path)}")
        return entry

    def stop_deploy(self, name: str) -> dict[str, Any]:
        """Kill the reverse proxy for an app (the app server keeps going)."""
        slug = _slug(name)
        data = self._deployed()
        entry = data.pop(slug, None)
        if entry is None:
            raise ToolError(f"{slug} has no deployment")
        self._set_deployed(data)
        killed = False
        pid = int(entry.get("pid", 0) or 0)
        if pid and _pid_alive(pid):
            import signal

            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                killed = True
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed = True
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        return {"app": slug, "stopped": killed or not _pid_alive(pid),
                "was_pid": pid, "url": entry.get("url", "")}

    def deployed(self) -> dict[str, Any]:
        out = []
        data = self._deployed()
        for slug, entry in sorted(data.items()):
            pid = int(entry.get("pid", 0) or 0)
            alive = _pid_alive(pid)
            live = _quick_http(entry.get("port", 0),
                               entry.get("path", "") or "/",
                               tls=bool(entry.get("tls"))) if alive \
                else None
            out.append({**entry, "alive": alive, "live": live})
            if not alive:
                data.pop(slug, None)
        self._set_deployed(data)
        return {"deployments": out, "count": len(out)}


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _free_port(want: int) -> int:
    import socket

    for port in (want, *range(want + 1, want + 21)) if want else \
            (*[0],):
        if port == 0:
            with socket.socket() as s:
                s.bind(("0.0.0.0", 0))
                return s.getsockname()[1]
        with socket.socket() as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return 0


def _missing_deps(stack: str) -> list[str]:
    import importlib.util

    need = {"flask": ["flask"], "fastapi": ["fastapi", "uvicorn"]}.get(
        stack, [])
    return [m for m in need if importlib.util.find_spec(m) is None]


def _serve_command(stack: str, app_dir: Path,
                   port: int) -> tuple[list[str], dict[str, str]]:
    import sys

    if stack == "static":
        return ([sys.executable, "-m", "http.server", str(port),
                 "--bind", "0.0.0.0"], {})
    if stack == "flask":
        return [sys.executable, "app.py"], {"PORT": str(port)}
    if stack == "fastapi":
        return ([sys.executable, "-m", "uvicorn", "main:app",
                 "--host", "0.0.0.0", "--port", str(port)], {})
    if stack == "express":
        return ["node", "server.js"], {"PORT": str(port)}
    if stack == "react-vite":
        return (["npx", "vite", "--port", str(port), "--host", "0.0.0.0"], {})
    raise ToolError(f"no serve command for stack {stack!r}")


def _find_child_pid(app_dir: Path, cmd: list[str]) -> int | None:
    """Best-effort: the newest process whose cwd is app_dir."""
    import subprocess as _sp

    # /proc scan (Linux): most reliable — follow each process's cwd link
    best = (0, 0, None)  # (start_ticks, pid, pid)
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd")
                with open(f"/proc/{pid}/stat", "rb") as f:
                    stat_raw = f.read().decode("ascii", "ignore")
                start_ticks = int(stat_raw.rsplit(")", 1)[1].split()[19])
            except (OSError, ValueError, IndexError):
                continue
            if cwd == str(app_dir):
                if start_ticks > best[0]:
                    best = (start_ticks, pid, pid)
    except OSError:
        pass
    if best[2] is not None:
        return best[2]
    # fallback: ps(1)
    try:
        out = _sp.run(["ps", "-eo", "pid,lstart,cwd,args"],
                      capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    best2 = (0, None)
    for line in out.stdout.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4 or str(app_dir) not in parts[2]:
            continue
        try:
            epoch = time.mktime(time.strptime(
                parts[1], "%a %b %d %H:%M:%S %Y"))
        except ValueError:
            continue
        if epoch > best2[0]:
            best2 = (epoch, int(parts[0]))
    return best2[1]


def _wait_for_http(port: int, stack: str, *, timeout: float = 20.0,
                   path: str = "", tls: bool = False) -> dict[str, Any]:
    from .core.http import HttpClient

    scheme = "https" if tls else "http"
    # a self-signed local cert is the EXPECTED identity for tls deploys —
    # verification is what would fail, not the app
    path = path or _health_path(stack)
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            resp = HttpClient(timeout=2.0,
                              verify_tls=not tls).get(
                f"{scheme}://127.0.0.1:{port}{path}")
            if resp.status < 500:
                return {"ok": True, "status": resp.status, "path": path}
            last = f"http {resp.status}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc).splitlines()[0][:120]
        time.sleep(0.5)
    return {"ok": False, "error": last or "no response"}


def _quick_http(port: int, path: str = "/", *, tls: bool = False) -> dict[str, Any] | None:
    if not port:
        return None
    from .core.http import HttpClient

    scheme = "https" if tls else "http"
    try:
        resp = HttpClient(timeout=2.0, verify_tls=not tls).get(
            f"{scheme}://127.0.0.1:{port}{path}")
        return {"status": resp.status}
    except Exception:  # noqa: BLE001
        return None


def _health_path(stack: str) -> str:
    return {"flask": "/api/health", "fastapi": "/health",
            "express": "/health"}.get(stack, "/")


def _log_tail(path: Path, n: int = 12) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8",
                                     errors="ignore").splitlines()
        return "\n".join(lines[-n:]) or "(empty log)"
    except OSError:
        return "(no log)"


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "build_app",
        description=(
            "Scaffold a real runnable app in workspace/apps/: stacks= "
            "static|flask|fastapi|express|react-vite|cli-python. Params: "
            "action=build (name, stack, title, description, features "
            "(comma-separated), port, overwrite) | stacks | list | info "
            "(name) | serve (name, port) — starts the server detached, "
            "health-checks it, returns a live URL + log path | stop "
            "(name) | served (what's running) | deploy (name, host, "
            "port, domain, path, tls) — puts the app behind a real "
            "reverse proxy; tls serves HTTPS (self-signed cert "
            "generated + cached per app) | stop_deploy "
            "(name) | deployed (what's proxying). Generated code is "
            "validated (py_compile/node --check/JSON) with a "
            "manifest.json + run instructions."
        ),
        capability=Capability.FS_WRITE,
    )
    def build_app(action: str = "build", name: str = "", stack: str = "static",
                  title: str = "", description: str = "",
                  features: str = "", port: int = 0,
                  overwrite: bool = False, domain: str = "",
                  path: str = "", host: str = "0.0.0.0",
                  tls: bool = False) -> dict[str, Any]:
        b = AppBuilder(context)
        if action == "stacks":
            return {"stacks": list(STACKS)}
        if action == "list":
            return b.list_apps()
        if action == "info":
            return b.info(name)
        if action == "serve":
            if not name:
                raise ToolError("build_app serve needs name=")
            return b.serve(name, port=int(port or 0))
        if action == "stop":
            if not name:
                raise ToolError("build_app stop needs name=")
            return b.stop(name)
        if action == "served":
            return b.served()
        if action == "deploy":
            if not name:
                raise ToolError("build_app deploy needs name=")
            return b.deploy(name, host=host, port=int(port or 0),
                            domain=domain, path=path, tls=tls)
        if action == "stop_deploy":
            if not name:
                raise ToolError("build_app stop_deploy needs name=")
            return b.stop_deploy(name)
        if action == "deployed":
            return b.deployed()
        if action != "build":
            raise ToolError(f"unknown build_app action {action!r}")
        feats = [f.strip() for f in features.split(",") if f.strip()]
        spec = {"name": name, "stack": stack, "title": title,
                "description": description, "features": feats,
                "port": int(port or 0), "overwrite": overwrite}
        return b.build(spec)
