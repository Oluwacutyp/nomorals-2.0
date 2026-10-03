"""Tests for CAPTCHA-solver wiring into the account creator.

The solver (nomorals.tools.captcha, L4) is injected into AccountCreator
(L2) — never imported — so the creator tries the solver FIRST when it
hits a CAPTCHA and only falls back to a human checkpoint when the
solver is disabled or fails. Offline by design: the solver is faked.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from nomorals.accounts import (
    AccountCheckpointPending,
    AccountCreator,
    CheckpointKind,
    CredentialVault,
)
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test-pass")


def _ok_solver(token: str = "tok-123", backend: str = "service"):
    def _solve(challenge: dict) -> dict:
        return {
            "ok": True, "kind": challenge.get("kind", ""),
            "backend": backend, "token": token, "text": "",
            "takeover": False, "elapsed_ms": 10, "detail": "",
        }
    _solve.calls = []  # type: ignore[attr-defined]
    orig = _solve

    def _tracked(challenge: dict) -> dict:
        _tracked.calls.append(challenge)  # type: ignore[attr-defined]
        return orig(challenge)
    _tracked.calls = []  # type: ignore[attr-defined]
    return _tracked


def _takeover_solver():
    def _solve(challenge: dict) -> dict:
        return {
            "ok": False, "kind": challenge.get("kind", ""),
            "backend": "takeover", "token": "", "text": "",
            "takeover": True, "elapsed_ms": 5,
            "detail": "owner takeover needed",
        }
    _solve.calls = []  # type: ignore[attr-defined]

    def _tracked(challenge: dict) -> dict:
        _tracked.calls.append(challenge)  # type: ignore[attr-defined]
        return _solve(challenge)
    _tracked.calls = []  # type: ignore[attr-defined]
    return _tracked


class SolverFirstTests(unittest.TestCase):
    """attempt_captcha_solve tries the solver before the human fallback."""

    def setUp(self):
        self.creator = AccountCreator(
            _vault(), captcha_solver=_ok_solver("tok-abc"))

    def test_solver_success_returns_token(self):
        token = self.creator.attempt_captcha_solve(
            kind="recaptcha_v2", sitekey="site-1",
            page_url="https://example.com/signup", service="github")
        self.assertEqual(token, "tok-abc")

    def test_solver_receives_challenge_details(self):
        solver = _ok_solver()
        creator = AccountCreator(_vault(), captcha_solver=solver)
        creator.attempt_captcha_solve(
            kind="hcaptcha", sitekey="hk-1",
            page_url="https://example.com/", service="x")
        self.assertEqual(len(solver.calls), 1)
        self.assertEqual(solver.calls[0]["kind"], "hcaptcha")
        self.assertEqual(solver.calls[0]["sitekey"], "hk-1")
        self.assertEqual(solver.calls[0]["page_url"], "https://example.com/")

    def test_takeover_result_raises_checkpoint(self):
        creator = AccountCreator(_vault(), captcha_solver=_takeover_solver())
        with self.assertRaises(AccountCheckpointPending) as ctx:
            creator.attempt_captcha_solve(
                kind="recaptcha_v2", service="github")
        cp = ctx.exception.checkpoint
        self.assertEqual(cp.kind, CheckpointKind.CAPTCHA)
        self.assertEqual(cp.service, "github")
        # checkpoint persisted and listed
        pending = creator.get_pending_checkpoints(service="github")
        self.assertEqual(len(pending), 1)

    def test_solver_exception_falls_back_to_checkpoint(self):
        def _boom(challenge: dict) -> dict:
            raise RuntimeError("solver exploded")
        creator = AccountCreator(_vault(), captcha_solver=_boom)
        with self.assertRaises(AccountCheckpointPending) as ctx:
            creator.attempt_captcha_solve(kind="recaptcha_v2")
        self.assertEqual(ctx.exception.checkpoint.kind,
                         CheckpointKind.CAPTCHA)

    def test_no_solver_injected_raises_checkpoint(self):
        creator = AccountCreator(_vault())  # no solver
        with self.assertRaises(AccountCheckpointPending):
            creator.attempt_captcha_solve(kind="recaptcha_v2")


class SolverDisabledTests(unittest.TestCase):
    """solver_enabled=False / NM_CAPTCHA_SOLVER=0 → straight to takeover."""

    def test_explicit_flag_skips_solver(self):
        solver = _ok_solver()
        creator = AccountCreator(
            _vault(), captcha_solver=solver, solver_enabled=False)
        with self.assertRaises(AccountCheckpointPending):
            creator.attempt_captcha_solve(kind="recaptcha_v2")
        self.assertEqual(solver.calls, [])

    def test_env_var_skips_solver(self):
        solver = _ok_solver()
        creator = AccountCreator(_vault(), captcha_solver=solver)
        with mock.patch.dict(os.environ, {"NM_CAPTCHA_SOLVER": "0"}):
            with self.assertRaises(AccountCheckpointPending):
                creator.attempt_captcha_solve(kind="recaptcha_v2")
        self.assertEqual(solver.calls, [])

    def test_env_var_on_by_default(self):
        solver = _ok_solver("tok-env")
        creator = AccountCreator(_vault(), captcha_solver=solver)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_CAPTCHA_SOLVER", None)
            token = creator.attempt_captcha_solve(kind="recaptcha_v2")
        self.assertEqual(token, "tok-env")
        self.assertEqual(len(solver.calls), 1)

    def test_solve_captcha_raises_when_disabled(self):
        from nomorals.core.errors import NoMoralsError
        creator = AccountCreator(
            _vault(), captcha_solver=_ok_solver(), solver_enabled=False)
        with self.assertRaises(NoMoralsError):
            creator.solve_captcha({"kind": "recaptcha_v2"})

    def test_solve_captcha_raises_without_solver(self):
        from nomorals.core.errors import NoMoralsError
        creator = AccountCreator(_vault())
        with self.assertRaises(NoMoralsError):
            creator.solve_captcha({"kind": "recaptcha_v2"})


class AdapterTests(unittest.TestCase):
    """creator_solver_adapter bridges tools.captcha → the creator protocol."""

    def test_adapter_calls_solve_with_auto(self):
        from nomorals.tools import captcha as cap

        seen = {}

        def _fake_solve(challenge, backend="auto", settings=None,
                        solver_enabled=True):
            seen["backend"] = backend
            seen["solver_enabled"] = solver_enabled
            seen["kind"] = challenge.kind
            seen["sitekey"] = challenge.sitekey
            result = mock.Mock()
            result.to_dict.return_value = {"ok": True, "token": "t",
                                           "takeover": False}
            return result

        with mock.patch.object(cap, "solve", _fake_solve):
            adapter = cap.creator_solver_adapter()
            out = adapter({"kind": "recaptcha_v2", "sitekey": "sk",
                           "page_url": "https://example.com/"})
        self.assertEqual(seen["backend"], "auto")
        self.assertTrue(seen["solver_enabled"])
        self.assertEqual(seen["kind"], "recaptcha_v2")
        self.assertEqual(seen["sitekey"], "sk")
        self.assertEqual(out["token"], "t")

    def test_adapter_honors_disabled_flag(self):
        from nomorals.tools import captcha as cap

        seen = {}

        def _fake_solve(challenge, backend="auto", settings=None,
                        solver_enabled=True):
            seen["solver_enabled"] = solver_enabled
            result = mock.Mock()
            result.to_dict.return_value = {"ok": False, "takeover": True}
            return result

        with mock.patch.object(cap, "solve", _fake_solve):
            adapter = cap.creator_solver_adapter(solver_enabled=False)
            adapter({"kind": "recaptcha_v2"})
        self.assertFalse(seen["solver_enabled"])

    def test_adapter_env_disables(self):
        from nomorals.tools import captcha as cap

        seen = {}

        def _fake_solve(challenge, backend="auto", settings=None,
                        solver_enabled=True):
            seen["solver_enabled"] = solver_enabled
            result = mock.Mock()
            result.to_dict.return_value = {"ok": False, "takeover": True}
            return result

        with mock.patch.object(cap, "solve", _fake_solve):
            with mock.patch.dict(os.environ, {"NM_CAPTCHA_SOLVER": "0"}):
                adapter = cap.creator_solver_adapter()
                adapter({"kind": "recaptcha_v2"})
        self.assertFalse(seen["solver_enabled"])


class LayeringTests(unittest.TestCase):
    """accounts (L2) must not import tools (L4) — solver is injected."""

    def test_creator_has_no_tools_import(self):
        import ast
        from pathlib import Path

        path = (Path(__file__).resolve().parent.parent
                / "nomorals" / "accounts" / "creator.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name.startswith("nomorals.tools"),
                        f"direct tools import at line {node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                self.assertFalse(
                    mod.startswith("tools")
                    or mod.startswith("nomorals.tools"),
                    f"direct tools import at line {node.lineno}")


class AccountCliTests(unittest.TestCase):
    """`nm account` parser wiring, incl. --solver/--no-solver."""

    def _parse(self, argv):
        from nomorals.cmdline.parser import _parser
        return _parser().parse_args(argv)

    def test_create_parses(self):
        args = self._parse(["account", "create", "--service", "github"])
        self.assertEqual(args.account_action, "create")
        self.assertEqual(args.service, "github")
        self.assertIsNone(args.solver_enabled)  # default → env

    def test_solver_flags(self):
        args = self._parse(
            ["account", "create", "--service", "github", "--solver"])
        self.assertTrue(args.solver_enabled)
        args = self._parse(
            ["account", "create", "--service", "github", "--no-solver"])
        self.assertFalse(args.solver_enabled)

    def test_resume_pending_parse(self):
        args = self._parse(["account", "resume", "--id", "achk_1"])
        self.assertEqual(args.account_action, "resume")
        args = self._parse(["account", "pending"])
        self.assertEqual(args.account_action, "pending")

    def test_alias_registered(self):
        from nomorals.cmdline.parser import CLI_ALIASES
        self.assertIn("account", CLI_ALIASES)


if __name__ == "__main__":
    unittest.main()
