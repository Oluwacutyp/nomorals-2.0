"""Computational-answer organ: stdlib exact rung, sympy CAS rung,
WolframAlpha API rung, chat verb, brain tool, CLI.

Failing-first: every import below names code that does not exist yet.
"""

from __future__ import annotations

import json

# A recorded WolframAlpha v2/query JSON body (shape, not a live call):
# "integrate x^2" with an Input pod, a Result pod and one assumption.
WOLFRAM_FIXTURE = {
    "queryresult": {
        "success": True,
        "error": False,
        "numpods": 2,
        "timing": 1.234,
        "inputstring": "integrate x^2",
        "pods": [
            {
                "title": "Input interpretation",
                "scanner": "Identity",
                "id": "Input",
                "position": 100,
                "error": False,
                "numsubpods": 1,
                "subpods": [
                    {
                        "title": "",
                        "img": {"src": "https://example.invalid/in.png",
                                "alt": "integrate x^2"},
                        "plaintext": "integrate x^2",
                    }
                ],
            },
            {
                "title": "Result",
                "scanner": "Integral",
                "id": "Result",
                "position": 200,
                "error": False,
                "numsubpods": 1,
                "subpods": [
                    {
                        "title": "",
                        "img": {"src": "https://example.invalid/r.png",
                                "alt": "x^3/3 + constant"},
                        "plaintext": "x^3/3 + constant",
                    }
                ],
            },
        ],
        "assumptions": [
            {
                "type": "Clash",
                "word": "x",
                "template": "Assuming ${desc1}",
                "count": 1,
                "values": [
                    {"name": "Variable", "desc": "x is a variable",
                     "input": "Variable"}
                ],
            }
        ],
    }
}

WOLFRAM_FAILURE_FIXTURE = {
    "queryresult": {
        "success": False,
        "error": False,
        "numpods": 0,
        "timing": 0.4,
        "inputstring": "blorp zorp",
        "pods": [],
        "didyoumeans": [{"score": "0.8", "level": "medium",
                         "val": "integrate x^2"}],
    }
}


def _fake_fetcher(body: dict):
    def fetch(url: str) -> str:
        assert "api.wolframalpha.com" in url
        assert "appid=" in url
        return json.dumps(body)

    return fetch


class TestStdlibRung:
    def test_exact_arithmetic(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("2 + 3 * 4")
        assert comp.ok is True
        assert comp.rung == "stdlib"
        assert comp.result == "14"
        assert comp.steps

    def test_exact_fractions(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("1/3 + 1/6")
        assert comp.ok is True
        assert comp.result == "1/2"

    def test_decimal_is_exact_not_floaty(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("0.1 + 0.2")
        assert comp.ok is True
        assert comp.result == "0.3"

    def test_power_and_percent(self):
        from nomorals.codews.compute import compute_answer

        assert compute_answer("2^10").result == "1024"
        assert compute_answer("15% of 200").result == "30"

    def test_units(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("5 miles to km")
        assert comp.ok is True
        assert "8.04672" in comp.result
        assert compute_answer("32 fahrenheit to celsius").result == "0"
        assert compute_answer("2 hours to minutes").result == "120"

    def test_dates(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("days between 2026-01-01 and 2026-11-17")
        assert comp.ok is True
        assert comp.result == "320"
        comp = compute_answer("what day of the week is 2026-11-17")
        assert comp.ok is True
        assert comp.result == "Tuesday"

    def test_unsafe_input_never_executes(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("__import__('os').system('echo pwned')")
        assert comp.ok is False


class TestSympyRung:
    def test_integrate(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("integrate x^2")
        assert comp.ok is True
        assert comp.rung == "sympy"
        assert "x**3" in comp.result or "x^3" in comp.result

    def test_solve_quadratic(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("solve x^2 - 5*x + 6 = 0")
        assert comp.ok is True
        assert comp.rung == "sympy"
        assert "2" in comp.result and "3" in comp.result

    def test_differentiate(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("differentiate sin(x^2)")
        assert comp.ok is True
        assert "cos" in comp.result

    def test_factor(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("factor x^2 - 1")
        assert comp.ok is True
        assert "x - 1" in comp.result and "x + 1" in comp.result


class TestWolframRung:
    def test_client_parses_pods_and_assumptions(self):
        from nomorals.codews.wolfram import WolframClient

        client = WolframClient(
            appid="TEST-APPID", fetcher=_fake_fetcher(WOLFRAM_FIXTURE))
        res = client.query("integrate x^2")
        assert res["ok"] is True
        assert len(res["pods"]) == 2
        assert res["pods"][1]["title"] == "Result"
        assert res["pods"][1]["subpods"][0]["plaintext"] == "x^3/3 + constant"
        assert res["assumptions"] == ["x is a variable"]

    def test_client_failure_is_honest(self):
        from nomorals.codews.wolfram import WolframClient

        client = WolframClient(
            appid="TEST-APPID",
            fetcher=_fake_fetcher(WOLFRAM_FAILURE_FIXTURE))
        res = client.query("blorp zorp")
        assert res["ok"] is False
        assert "did you mean" in res["error"].lower() or res.get("didyoumeans")

    def test_client_without_appid_refuses_in_words(self):
        from nomorals.codews.wolfram import WolframClient

        client = WolframClient(appid="")
        res = client.query("integrate x^2")
        assert res["ok"] is False
        assert "AppID" in res["error"]

    def test_orchestrator_prefers_wolfram_when_keyed(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer(
            "integrate x^2", appid="TEST-APPID",
            fetcher=_fake_fetcher(WOLFRAM_FIXTURE))
        assert comp.ok is True
        assert comp.rung == "wolfram"
        assert comp.result == "x^3/3 + constant"
        assert comp.assumptions == ["x is a variable"]
        assert len(comp.pods) == 2

    def test_orchestrator_falls_back_local_when_unkeyed(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("integrate x^2")
        assert comp.ok is True
        assert comp.rung == "sympy"

    def test_curated_fact_without_wolfram_is_never_invented(self):
        from nomorals.codews.compute import compute_answer

        comp = compute_answer("population of Lagos in 1990")
        assert comp.ok is False
        assert comp.rung == "unanswered"
        assert "WolframAlpha" in (comp.error or "")


class TestReplyShape:
    def test_reply_names_rung_steps_and_result(self):
        from nomorals.codews.compute import (
            compute_answer,
            format_compute_reply,
        )

        reply = format_compute_reply(compute_answer("2 + 3 * 4"))
        assert "14" in reply
        assert "stdlib" in reply.lower()
        assert "step" in reply.lower()

    def test_unanswered_reply_says_what_would_help(self):
        from nomorals.codews.compute import (
            compute_answer,
            format_compute_reply,
        )

        reply = format_compute_reply(
            compute_answer("population of Lagos in 1990"))
        assert "WolframAlpha" in reply
        assert "AppID" in reply


class TestChatRegistration:
    def test_compute_command_registered(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            PUBLIC_COMMANDS,
            list_catalog,
        )

        assert "compute" in CONTROL_COMMANDS
        assert CONTROL_COMMANDS["compute"] == (1, None)
        assert "compute" in PUBLIC_COMMANDS
        assert len(PUBLIC_COMMANDS) == 270
        assert "compute" in COMMAND_DETAILS
        assert "compute" in LIST_ONELINERS
        assert "compute" in [c for _, kinds in LIST_GROUPS
                             for c in kinds]
        assert "compute" in list_catalog()

    def test_compute_handler_answers_and_never_raises(self):
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        runtime = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        handler = runtime._control_compute
        assert "14" in handler("2 + 3 * 4")
        assert "usage" in handler("").lower()
        assert "WolframAlpha" in handler("population of Lagos in 1990")


class TestCli:
    def test_nm_compute_answers(self, capsys):
        import argparse

        from nomorals.cmdline.commands.compute import _cmd_compute

        args = argparse.Namespace(question="2 + 3 * 4", json=False,
                                  rung="auto")
        assert _cmd_compute(args, None) == 0
        assert "14" in capsys.readouterr().out

    def test_nm_compute_json(self, capsys):
        import argparse

        from nomorals.cmdline.commands.compute import _cmd_compute

        args = argparse.Namespace(question="2 + 3 * 4", json=True,
                                  rung="auto")
        assert _cmd_compute(args, None) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["result"] == "14"
        assert payload["rung"] == "stdlib"

    def test_nm_compute_unanswered_exits_nonzero(self, capsys):
        import argparse

        from nomorals.cmdline.commands.compute import _cmd_compute

        args = argparse.Namespace(
            question="population of Lagos in 1990", json=False, rung="auto")
        assert _cmd_compute(args, None) == 1
        assert "WolframAlpha" in capsys.readouterr().out


class TestConfig:
    def test_wolfram_appid_setting_exists(self):
        from nomorals.core.config import Settings, load_settings

        assert hasattr(Settings(), "wolfram")
        assert Settings().wolfram.appid == ""
        settings = load_settings(
            env={"NM_WOLFRAM_APPID": "TEST-APPID"}, use_env_file=False)
        assert settings.wolfram.appid == "TEST-APPID"


class TestBrainTool:
    def test_compute_tool_registered(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register_builtins()
        names = {s["name"] for s in registry.schemas()}
        assert "compute" in names
