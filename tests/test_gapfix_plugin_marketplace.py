"""Gapfix R1-15: the plugin marketplace must gate installs and be reachable.

Before the fix, ``PluginRegistry.install`` never ran a plugin's own
tests (``Marketplace`` had zero production importers) and
``nm plugin`` had no ``listings``/``review`` verbs. These tests pin:

* registry install/upgrade refuse a plugin whose own tests fail —
  atomically (no DB row, no directory) — and report the failure;
* passing-tests and no-tests plugins install, with the test outcome
  recorded honestly on the registry;
* zip sources are test-gated on their extracted contents;
* ``nm plugin listings`` shows installed plugins with ratings;
* ``nm plugin review`` records a review (one per reviewer, overwrite);
* a corrupt ``marketplace_reviews.json`` loads empty, never crashes;
* the test sandbox writes no ``.pytest_cache`` into the source or the
  installed plugin dir.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.cmdline.commands.plugin import _cmd_plugin
from nomorals.plugins import PluginRegistry
from nomorals.plugins.errors import PluginError
from nomorals.plugins.marketplace import Marketplace, PluginTestFailed
from nomorals.storage.db import Database


def _plugin_dir(base: Path, name: str = "demo", version: str = "1.0.0",
                test_body: str | None = None,
                code: str = "") -> Path:
    """A registry-style plugin dir; ``test_body`` adds tests/test_x.py."""
    d = base / f"{name}-{version}-src"
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "version": version,
        "description": "demo plugin",
        "author": "test",
        "entry_points": {"main": "demo:run"},
        "permissions": [],
    }
    (d / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "demo.py").write_text(
        code or "def run(caps):\n    return {'ok': True}\n",
        encoding="utf-8")
    if test_body is not None:
        (d / "tests").mkdir(exist_ok=True)
        (d / "tests" / "test_demo.py").write_text(test_body, encoding="utf-8")
    return d


def _zip_plugin(src: Path, dest: Path) -> Path:
    zp = dest / f"{src.name}.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src))
    return zp


@pytest.fixture()
def ctx(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    workspace = tmp_path / "workspace"
    settings = SimpleNamespace(workspace_dir=str(workspace))
    context = SimpleNamespace(settings=settings, db=db)
    reg = PluginRegistry(db, workspace / "plugins")
    mp = Marketplace(plugins_dir=workspace / "plugins")
    return SimpleNamespace(tmp=tmp_path, db=db, context=context,
                           reg=reg, mp=mp, workspace=workspace)


def _args(task, json_out=False):
    return SimpleNamespace(task=list(task), json=json_out)


# ── the gate: registry install ────────────────────────────────────────────

def test_install_refuses_failing_tests_atomically(ctx):
    bad = _plugin_dir(ctx.tmp, test_body="def test_no():\n    assert False\n")
    with pytest.raises(PluginTestFailed):
        ctx.reg.install(bad)
    assert ctx.reg.list() == []
    assert not (ctx.workspace / "plugins" / "demo-1.0.0").exists()


def test_install_failure_is_a_plugin_error(ctx):
    assert issubclass(PluginTestFailed, PluginError)
    bad = _plugin_dir(ctx.tmp, test_body="def test_no():\n    assert False\n")
    with pytest.raises(PluginError):
        ctx.reg.install(bad)


def test_install_passing_tests_records_outcome(ctx):
    good = _plugin_dir(ctx.tmp, test_body="def test_yes():\n    assert True\n")
    p = ctx.reg.install(good)
    assert p.name == "demo"
    assert ctx.reg.get("demo").version == "1.0.0"
    assert ctx.reg.last_test_result["passed"] is True
    assert ctx.reg.last_test_result["tests_found"] == 1


def test_install_no_tests_vacuous_pass(ctx):
    plain = _plugin_dir(ctx.tmp)
    p = ctx.reg.install(plain)
    assert p.name == "demo"
    assert ctx.reg.last_test_result["passed"] is True
    assert ctx.reg.last_test_result["tests_found"] == 0


def test_install_zip_source_failing_tests_refused(ctx):
    bad = _plugin_dir(ctx.tmp, test_body="def test_no():\n    assert False\n")
    zp = _zip_plugin(bad, ctx.tmp)
    with pytest.raises(PluginTestFailed):
        ctx.reg.install(zp)
    assert ctx.reg.list() == []
    assert not (ctx.workspace / "plugins" / "demo-1.0.0").exists()


def test_upgrade_refuses_failing_tests_old_version_intact(ctx):
    v1 = _plugin_dir(ctx.tmp, version="1.0.0",
                     test_body="def test_yes():\n    assert True\n")
    ctx.reg.install(v1)
    v2 = _plugin_dir(ctx.tmp, version="1.1.0",
                     test_body="def test_no():\n    assert False\n")
    with pytest.raises(PluginTestFailed):
        ctx.reg.upgrade(v2)
    assert ctx.reg.get("demo").version == "1.0.0"
    assert not (ctx.workspace / "plugins" / "demo-1.1.0").exists()


def test_no_pytest_cache_left_behind(ctx):
    good = _plugin_dir(ctx.tmp, test_body="def test_yes():\n    assert True\n")
    ctx.reg.install(good)
    assert not (good / ".pytest_cache").exists()
    assert not (ctx.workspace / "plugins" / "demo-1.0.0" / ".pytest_cache").exists()


# ── the gate: CLI install ─────────────────────────────────────────────────

def test_cli_install_reports_test_failure(ctx, capsys):
    bad = _plugin_dir(ctx.tmp, test_body="def test_no():\n    assert False\n")
    rc = _cmd_plugin(_args(["install", str(bad)]), ctx.context)
    assert rc == 1
    assert "test" in capsys.readouterr().err.lower()
    assert ctx.reg.list() == []


def test_cli_install_reports_tests_passed(ctx, capsys):
    good = _plugin_dir(ctx.tmp, test_body="def test_yes():\n    assert True\n")
    rc = _cmd_plugin(_args(["install", str(good)]), ctx.context)
    assert rc == 0
    out = capsys.readouterr().out
    assert "installed demo 1.0.0" in out
    assert "test" in out.lower()


# ── listings / review verbs ───────────────────────────────────────────────

def test_cli_listings_shows_ratings(ctx, capsys):
    rated = _plugin_dir(ctx.tmp, name="rated")
    plain = _plugin_dir(ctx.tmp, name="plain")
    ctx.reg.install(rated)
    ctx.reg.install(plain)
    ctx.mp.add_review("rated", "owner", 5, "great")
    rc = _cmd_plugin(_args(["listings"]), ctx.context)
    assert rc == 0
    out = capsys.readouterr().out
    assert "rated" in out and "plain" in out
    assert "5" in out  # the rating shows
    assert "unrated" in out  # the unreviewed plugin says so honestly


def test_cli_listings_json(ctx, capsys):
    good = _plugin_dir(ctx.tmp)
    ctx.reg.install(good)
    ctx.mp.add_review("demo", "owner", 4)
    rc = _cmd_plugin(_args(["listings"], json_out=True), ctx.context)
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "demo"
    assert rows[0]["average_rating"] == 4.0
    assert rows[0]["review_count"] == 1


def test_cli_review_records_and_overwrites(ctx, capsys):
    ctx.reg.install(_plugin_dir(ctx.tmp))
    rc = _cmd_plugin(_args(["review", "demo", "5", "great"]), ctx.context)
    assert rc == 0
    rc = _cmd_plugin(_args(["review", "demo", "3", "meh"]), ctx.context)
    assert rc == 0
    assert ctx.mp.average_rating("demo") == 3.0
    assert len(ctx.mp.reviews("demo")) == 1  # one review per reviewer


def test_cli_review_rejects_bad_rating_and_unknown_plugin(ctx, capsys):
    ctx.reg.install(_plugin_dir(ctx.tmp))
    rc = _cmd_plugin(_args(["review", "demo", "9"]), ctx.context)
    assert rc == 1
    rc = _cmd_plugin(_args(["review", "ghost", "5"]), ctx.context)
    assert rc == 1
    assert "not installed" in capsys.readouterr().err.lower()


def test_cli_listings_unknown_verb_still_usage(ctx, capsys):
    rc = _cmd_plugin(_args(["bogus-verb"]), ctx.context)
    assert rc == 2


# ── reviews store ─────────────────────────────────────────────────────────

def test_corrupt_reviews_file_loads_empty(ctx):
    reviews_path = ctx.workspace / "plugins" / "marketplace_reviews.json"
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    reviews_path.write_text("{not json", encoding="utf-8")
    assert ctx.mp.listings() == []  # must not raise on corrupt reviews
    assert ctx.mp.average_rating("demo") is None
    ctx.mp.add_review("demo", "owner", 5)
    assert ctx.mp.average_rating("demo") == 5.0
    assert json.loads(reviews_path.read_text(encoding="utf-8"))


def test_marketplace_exported_from_package():
    from nomorals.plugins import Marketplace as MP
    from nomorals.plugins import PluginTestFailed as PTF
    from nomorals.plugins import run_plugin_tests

    assert MP is Marketplace
    assert PTF is PluginTestFailed
    assert callable(run_plugin_tests)
