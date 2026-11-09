"""The surfaces: the ``computer`` tool registration, the command
catalog entries, and the /computer chat handler.

The catalog assertions pin the owner-only shape (in CONTROL_COMMANDS
with the owner tuple, detailed, grouped, one-lined — and NOT in
PUBLIC_COMMANDS). The handler is exercised through a stub runtime
``self`` the way the partner runtime invokes it: verbs in, owner's
words out, against a real (test-owned) display.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest

from nomorals.computer import provision


class TestCatalog:
    def test_command_registered_owner_only(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            PUBLIC_COMMANDS,
        )

        assert CONTROL_COMMANDS["computer"] == (0, None)
        assert "computer" not in PUBLIC_COMMANDS
        detail = COMMAND_DETAILS["computer"]
        assert "/computer open" in detail["usage"]
        assert "takeover" in detail["what"]
        grouped = [name for _title, names in LIST_GROUPS for name in names]
        assert "computer" in grouped
        assert "computer" in LIST_ONELINERS

    def test_dispatch_routes_to_the_handler(self):
        import inspect

        from nomorals.agents.partner import runtime
        source = inspect.getsource(runtime)
        assert 'kind == "computer"' in source
        assert "_control_computer" in source


class TestToolRegistration:
    def test_computer_tool_registers(self):
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry(context=SimpleNamespace(settings=None),
                           enforce=False)
        reg.register_builtins()
        assert "computer" in reg.names()
        failed = [row["module"] for row in reg.failed_modules()]
        assert "computer" not in failed

    def test_tool_status_speaks(self, monkeypatch, tmp_path):
        from nomorals.tools.registry import ToolRegistry

        monkeypatch.setenv("NM_HOME", str(tmp_path / "nmhome"))
        reg = ToolRegistry(context=SimpleNamespace(settings=None),
                           enforce=False)
        reg.register_builtins()
        out = reg._tools["computer"].fn(action="status")
        assert out["ok"] is True
        assert "computer:" in out["text"]
        assert "profile" in out and "sessions" in out

    def test_tool_rejects_bad_args_json(self):
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry(context=SimpleNamespace(settings=None),
                           enforce=False)
        reg.register_builtins()
        out = reg._tools["computer"].fn(action="act", kind="click",
                                        args="{not json")
        assert out["ok"] is False
        assert "not valid JSON" in out["error"]


class _StubRuntime(SimpleNamespace):
    """The two computer handler methods on a plain attribute bag —
    exactly how the real runtime composes them from the mixin."""

    from nomorals.agents.partner.runtime_expression import RuntimeExpressionMixin as _Mixin
    _control_computer = _Mixin._control_computer
    _computer_send_frame = _Mixin._computer_send_frame


@pytest.mark.skipif(shutil.which("Xvfb") is None,
                    reason="Xvfb not installed on this host")
class TestChatHandler:
    @pytest.fixture()
    def runtime_self(self, tmp_path, monkeypatch):
        # The verbs are the subject here, not the profile gate (the
        # gate has its own test below): pin a servable profile so
        # the round trip runs on any CI box size.
        from nomorals.computer import service as service_mod
        monkeypatch.setattr(
            service_mod._profile_mod, "detect_profile",
            lambda: SimpleNamespace(kind="vps"))
        number = provision.find_free_display()
        auth = tmp_path / "xauth"
        provision.write_xauthority(auth, number)
        server = provision.start_xvfb(number, 1280, 1024, auth)
        monkeypatch.setenv("NM_COMPUTER_DISPLAY", f":{number}")
        monkeypatch.setenv("XAUTHORITY", str(auth))
        monkeypatch.setenv("NM_HOME", str(tmp_path / "nmhome"))
        sent = []
        gateway = SimpleNamespace(
            send_file=lambda *a, **kw: sent.append((a, kw)))
        stub = _StubRuntime(context=None, gateway=gateway, sent=sent)
        yield stub
        provision.stop_process(server.pid)

    def _handler(self):
        return _StubRuntime._control_computer

    def test_status_open_view_close_round_trip(self, runtime_self):
        handler = self._handler()
        text = handler(runtime_self, "status")
        assert "computer:" in text
        text = handler(runtime_self, "open")
        assert "is open" in text, text
        text = handler(runtime_self, "status")
        assert "desk-" in text
        # view perceives and pushes the live frame through the gateway.
        message = SimpleNamespace(chat=SimpleNamespace(
            platform="telegram", key="owner-1"))
        text = handler(runtime_self, "view", message)
        assert "window(s)" in text
        assert runtime_self.sent, "the frame was not sent"
        args, kwargs = runtime_self.sent[-1]
        assert kwargs.get("kind") == "photo"
        text = handler(runtime_self, "takeover")
        assert "hands are off" in text
        text = handler(runtime_self, "handback")
        assert "handed back" in text
        text = handler(runtime_self, "stop")
        assert "sticky" in text
        text = handler(runtime_self, "resume")
        assert "resumed" in text
        text = handler(runtime_self, "close")
        assert "closed" in text

    def test_unknown_verb_lists_the_verbs(self, runtime_self):
        text = self._handler()(runtime_self, "frobnicate")
        assert "verbs:" in text


class TestProfileGate:
    """The organ reports itself unavailable WITH the reason on
    profiles that have no display facility — capability present in
    code, gated by the profile runtime, never faked."""

    def test_termux_profile_reports_unavailable_with_reason(
            self, monkeypatch):
        from nomorals.computer import service as service_mod

        monkeypatch.setattr(
            service_mod._profile_mod, "detect_profile",
            lambda: SimpleNamespace(kind="termux"))
        svc = service_mod.ComputerService()
        avail = svc.availability()
        assert avail["available"] is False
        assert "Termux" in avail["reason"]
        opened = svc.open()
        assert opened["ok"] is False
        assert "Termux" in opened["error"]
