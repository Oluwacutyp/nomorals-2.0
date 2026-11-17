"""Gapfix R3-47: failure classifier wired into the failures-ledger read path.

Failing-first: ``triaged_failures`` / triage-aware ``format_failures``
do not exist yet, and there is no ``nm failures`` command. The tests
below fail on the un-wired tree and pass once the classifier is loaded
into the ledger read path and the grading surface exists.
"""
from types import SimpleNamespace

import pytest

from nomorals.agents import failures as af


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NM_FAILURE_GRADES",
                       str(tmp_path / "grades" / "failure_grades.jsonl"))
    monkeypatch.setenv("NM_MODELS_DIR", str(tmp_path / "models"))
    af.clear_failures()
    yield
    af.clear_failures()


class _OddError(Exception):
    """Noise-class exception: deliberately NOT in _KNOWN_EXC."""


def _record_pair():
    af.record_failure("agents/news.py", TimeoutError("x" * 120))
    af.record_failure("agents/news.py", _OddError("bad row"))


def _grade_training_set(n_real=20, n_noise=20):
    """Grade separable rows straight into the journal: real = known
    timeout + long message, noise = unknown exc + short message."""
    from nomorals.agents import failure_models as fm

    for _ in range(n_real):
        fm.grade_failure(site="agents/news.py", exc_type="TimeoutError",
                         message="poll timed out after 300s " + "x" * 120,
                         real=True)
    for _ in range(n_noise):
        fm.grade_failure(site="agents/news.py", exc_type="OddError",
                         message="bad row", real=False)


def _train(tmp_path):
    from nomorals.agents.failure_models import fit_error_classifier
    from nomorals.llm.specialists import ModelStore

    outcome = fit_error_classifier(store=ModelStore(tmp_path / "models"))
    assert outcome.status == "trained", outcome.reason
    return outcome


# ---- read path ---------------------------------------------------------

def test_read_path_classifies_with_loaded_classifier(tmp_path):
    _grade_training_set()
    _train(tmp_path)
    _record_pair()

    rows = af.triaged_failures()
    assert len(rows) == 2
    real_row = next(r for r in rows if r.failure.exc_type == "TimeoutError")
    noise_row = next(r for r in rows if r.failure.exc_type == "_OddError")
    assert real_row.verdict == "real"
    assert real_row.p_real >= 0.5
    assert noise_row.verdict == "noise"
    assert noise_row.p_real < 0.5


def test_format_shows_triage_when_model_trained(tmp_path):
    _grade_training_set()
    _train(tmp_path)
    _record_pair()

    text = af.format_failures()
    assert "[real" in text
    assert "[noise" in text
    assert "real" in text.splitlines()[0]  # header carries the split


def test_ungraded_ledger_reads_exactly_as_before():
    _record_pair()
    text = af.format_failures()
    assert "[real" not in text
    assert "[noise" not in text
    assert "ungraded" not in text
    lines = text.splitlines()
    assert lines[0] == "⚠️ agents failures (2 total, showing 2 newest):"
    assert lines[1].endswith("agents/news.py: _OddError: bad row")
    assert lines[2].endswith("agents/news.py: TimeoutError: " + "x" * 120)


def test_triaged_failures_without_model_is_ungraded():
    _record_pair()
    rows = af.triaged_failures()
    assert [r.verdict for r in rows] == ["ungraded", "ungraded"]
    assert all(r.p_real == 0.5 for r in rows)


# ---- grading surface (CLI) ---------------------------------------------

def _run_cli(words, json_out=False):
    from nomorals.cmdline.commands.failures import _cmd_failures

    args = SimpleNamespace(task=list(words), json=json_out)
    return _cmd_failures(args, None)


def test_cli_grade_records_durably(capsys):
    _record_pair()
    rc = _run_cli(["grade", "1", "noise"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "noise" in out
    assert "1/30" in out  # running count against the volume gate

    from nomorals.agents.failure_models import graded_rows
    rows = graded_rows()
    assert len(rows) == 1
    assert rows[0]["label"] == 0


def test_cli_grade_by_site_substring(capsys):
    af.record_failure("agents/news.py", TimeoutError("boom"))
    af.record_failure("media/dj.py", ValueError("bad"))
    rc = _run_cli(["grade", "media/dj", "real"])
    assert rc == 0
    from nomorals.agents.failure_models import graded_rows
    assert len(graded_rows()) == 1


def test_cli_grade_refuses_bad_index(capsys):
    _record_pair()
    rc = _run_cli(["grade", "9", "real"])
    assert rc == 1
    assert "no failure" in capsys.readouterr().out
    from nomorals.agents.failure_models import graded_rows
    assert graded_rows() == []


def test_cli_grade_refuses_ambiguous_site(capsys):
    af.record_failure("agents/news.py", TimeoutError("boom"))
    af.record_failure("agents/scheduler.py", TimeoutError("boom"))
    rc = _run_cli(["grade", "agents/", "real"])
    assert rc == 1
    assert "2 failures match" in capsys.readouterr().out
    from nomorals.agents.failure_models import graded_rows
    assert graded_rows() == []


def test_grades_accumulate_past_gate_and_refit_loads(tmp_path, capsys):
    _grade_training_set()
    rc = _run_cli(["train"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "trained" in out

    # The read path now loads the NEW classifier, not the old absence.
    _record_pair()
    rows = af.triaged_failures()
    assert {r.verdict for r in rows} == {"real", "noise"}


def test_cli_train_below_gate_refuses(tmp_path, capsys):
    from nomorals.agents import failure_models as fm

    fm.grade_failure(site="agents/news.py", exc_type="TimeoutError",
                     message="boom", real=True)
    rc = _run_cli(["train"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "insufficient" in out
    assert "1/30" in out


def test_cli_list_json_carries_verdicts(tmp_path, capsys):
    _grade_training_set()
    _train(tmp_path)
    _record_pair()
    rc = _run_cli(["list"], json_out=True)
    assert rc == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["classifier"] == "trained"
    verdicts = {row["verdict"] for row in payload["failures"]}
    assert verdicts == {"real", "noise"}


def test_cli_list_json_without_model_is_ungraded(capsys):
    _record_pair()
    rc = _run_cli(["list"], json_out=True)
    assert rc == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["classifier"] == "none"
    assert all(row["verdict"] == "ungraded"
               for row in payload["failures"])


def test_failures_command_registered_in_dispatch_and_parser():
    from nomorals.cmdline.dispatch import _COMMANDS
    from nomorals.cmdline.parser import CLI_ALIASES, _parser

    assert "failures" in _COMMANDS
    assert "failures" in CLI_ALIASES
    parser = _parser()
    assert "failures" in parser._subparsers._group_actions[0].choices
