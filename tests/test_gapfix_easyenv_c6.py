"""§3C.6 env helpers (easyenv.rs port, MIT).

The typed env surface is proved against real ``os.environ`` state
(via the organ's own ``env_override`` + explicit ``env=`` mappings,
never the ambient process env) and against real ``.env`` files on
disk. No fake success: a required variable that is missing raises
by name, an unparseable value raises by name, and a secret value is
never printed in full by any surface the tests exercise.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.core import easyenv
from nomorals.core.easyenv import (
    EnvNotUnicode,
    EnvParseError,
    EnvRequiredNotPresent,
)

# -- string -----------------------------------------------------------------


class TestString:
    def test_optional(self):
        assert easyenv.get_env_string_optional(
            "C6_MISSING", env={}) is None
        assert easyenv.get_env_string_optional(
            "C6_NAME", env={"C6_NAME": "devon"}) == "devon"

    def test_or_default(self):
        assert easyenv.get_env_string_or_default(
            "C6_MISSING", "fallback", env={}) == "fallback"
        assert easyenv.get_env_string_or_default(
            "C6_NAME", "fallback", env={"C6_NAME": "devon"}) == "devon"

    def test_required(self):
        assert easyenv.get_env_string_required(
            "C6_NAME", env={"C6_NAME": "devon"}) == "devon"
        with pytest.raises(EnvRequiredNotPresent) as exc:
            easyenv.get_env_string_required("C6_MISSING", env={})
        assert "C6_MISSING" in str(exc.value)

    def test_empty_string_is_a_real_value(self):
        # The string family keeps "" (the _coerce convention: string
        # fields keep empty strings as-is, unlike every other type).
        assert easyenv.get_env_string_optional(
            "C6_EMPTY", env={"C6_EMPTY": ""}) == ""
        assert easyenv.get_env_string_required(
            "C6_EMPTY", env={"C6_EMPTY": ""}) == ""


# -- bool -------------------------------------------------------------------


class TestBool:
    @pytest.mark.parametrize("raw", [
        "1", "true", "TRUE", "True", "yes", "YES", "on", "ON", "y", "Y",
        "  true  ",
    ])
    def test_true_spellings(self, raw):
        assert easyenv.get_env_bool_required(
            "C6_BOOL", env={"C6_BOOL": raw}) is True

    @pytest.mark.parametrize("raw", [
        "0", "false", "FALSE", "False", "no", "NO", "off", "OFF", "n", "N",
    ])
    def test_false_spellings(self, raw):
        assert easyenv.get_env_bool_required(
            "C6_BOOL", env={"C6_BOOL": raw}) is False

    def test_garbage_is_an_error_not_a_silent_false(self):
        # The whole point of the port over ad-hoc `== "1"` checks:
        # "maybe" is neither true nor false, and must say so.
        with pytest.raises(EnvParseError) as exc:
            easyenv.get_env_bool_required(
                "C6_BOOL", env={"C6_BOOL": "maybe"})
        assert "C6_BOOL" in str(exc.value)
        assert "maybe" in str(exc.value)
        assert easyenv.get_env_bool_optional(
            "C6_BOOL", env={"C6_BOOL": "maybe"}) is None
        assert easyenv.get_env_bool_or_default(
            "C6_BOOL", True, env={"C6_BOOL": "maybe"}) is True

    def test_empty_means_not_set(self):
        assert easyenv.get_env_bool_optional(
            "C6_BOOL", env={"C6_BOOL": ""}) is None
        assert easyenv.get_env_bool_or_default(
            "C6_BOOL", True, env={"C6_BOOL": "   "}) is True
        with pytest.raises(EnvRequiredNotPresent):
            easyenv.get_env_bool_required("C6_BOOL", env={"C6_BOOL": ""})

    def test_missing(self):
        assert easyenv.get_env_bool_optional("C6_MISSING", env={}) is None
        assert easyenv.get_env_bool_or_default(
            "C6_MISSING", False, env={}) is False
        with pytest.raises(EnvRequiredNotPresent):
            easyenv.get_env_bool_required("C6_MISSING", env={})


# -- num --------------------------------------------------------------------


class TestNum:
    def test_default_type_drives_the_parse(self):
        assert easyenv.get_env_num(
            "C6_NUM", 7, env={"C6_NUM": "42"}) == 42
        assert isinstance(
            easyenv.get_env_num("C6_NUM", 7, env={"C6_NUM": "42"}), int)
        assert easyenv.get_env_num(
            "C6_NUM", 7.5, env={"C6_NUM": "42"}) == 42.0
        assert isinstance(
            easyenv.get_env_num("C6_NUM", 7.5, env={"C6_NUM": "42"}),
            float)

    def test_absent_returns_default(self):
        assert easyenv.get_env_num("C6_MISSING", 7, env={}) == 7

    def test_bad_parse_raises_instead_of_returning_default(self):
        with pytest.raises(EnvParseError):
            easyenv.get_env_num("C6_NUM", 7, env={"C6_NUM": "3.5"})
        with pytest.raises(EnvParseError):
            easyenv.get_env_num("C6_NUM", 7, env={"C6_NUM": "abc"})

    def test_optional_and_required(self):
        assert easyenv.get_env_num_optional(
            "C6_NUM", int, env={"C6_NUM": "9"}) == 9
        assert easyenv.get_env_num_optional(
            "C6_NUM", int, env={"C6_NUM": "abc"}) is None
        assert easyenv.get_env_num_required(
            "C6_NUM", float, env={"C6_NUM": "1.5"}) == 1.5
        with pytest.raises(EnvRequiredNotPresent):
            easyenv.get_env_num_required("C6_MISSING", int, env={})

    def test_bool_default_is_rejected(self):
        # bool is a subclass of int — accepting it would silently
        # parse numbers as True/False.
        with pytest.raises(TypeError):
            easyenv.get_env_num("C6_NUM", True, env={})
        with pytest.raises(TypeError):
            easyenv.get_env_num_optional("C6_NUM", str, env={})


# -- duration ---------------------------------------------------------------


class TestDuration:
    def test_plain_seconds(self):
        assert easyenv.get_env_duration_seconds_required(
            "C6_DUR", env={"C6_DUR": "90"}) == timedelta(seconds=90)

    @pytest.mark.parametrize("raw,seconds", [
        ("90s", 90), ("5m", 300), ("2h", 7200), ("1d", 86400),
        ("1.5h", 5400), (" 30m ", 1800),
    ])
    def test_suffixes(self, raw, seconds):
        assert easyenv.get_env_duration_seconds_required(
            "C6_DUR", env={"C6_DUR": raw}) == timedelta(seconds=seconds)

    def test_bad_and_missing(self):
        with pytest.raises(EnvParseError):
            easyenv.get_env_duration_seconds_required(
                "C6_DUR", env={"C6_DUR": "soon"})
        with pytest.raises(EnvParseError):
            easyenv.get_env_duration_seconds_required(
                "C6_DUR", env={"C6_DUR": "-5"})
        assert easyenv.get_env_duration_seconds_optional(
            "C6_MISSING", env={}) is None
        assert easyenv.get_env_duration_seconds_or_default(
            "C6_MISSING", timedelta(seconds=30), env={}
        ) == timedelta(seconds=30)


# -- path -------------------------------------------------------------------


class TestPath:
    def test_path_family(self):
        assert easyenv.get_env_pathbuf_required(
            "C6_PATH", env={"C6_PATH": "/tmp/x"}) == Path("/tmp/x")
        assert easyenv.get_env_pathbuf_optional(
            "C6_MISSING", env={}) is None
        assert easyenv.get_env_pathbuf_or_default(
            "C6_MISSING", "/tmp/d", env={}) == Path("/tmp/d")
        with pytest.raises(EnvRequiredNotPresent):
            easyenv.get_env_pathbuf_required("C6_MISSING", env={})


# -- errors name themselves (the Rust Display bug, fixed) -------------------


class TestErrors:
    def test_each_variant_names_itself_and_the_var(self):
        required = EnvRequiredNotPresent("C6_REQ")
        assert "C6_REQ" in str(required)
        assert "not set" in str(required)
        assert "ParseError" not in str(required)
        parse = EnvParseError("C6_BAD", "couldn't parse as bool: 'x'")
        assert "C6_BAD" in str(parse)
        assert "couldn't parse as bool" in str(parse)
        assert "RequiredNotPresent" not in str(parse)
        not_unicode = EnvNotUnicode("C6_UNI")
        assert "C6_UNI" in str(not_unicode)
        assert "unicode" in str(not_unicode)


# -- prefix + setters -------------------------------------------------------


class TestPrefixAndSetters:
    def test_prefixed(self):
        env = {"NM_A": "1", "NM_B": "2", "OTHER": "3"}
        assert easyenv.prefixed("NM_", env=env) == {"A": "1", "B": "2"}
        assert easyenv.prefixed("NM_", env=env, strip=False) == {
            "NM_A": "1", "NM_B": "2"}
        assert easyenv.prefixed("ZZ_", env=env) == {}

    def test_set_and_unset_return_previous(self):
        with easyenv.env_override():
            assert easyenv.set_env("C6_SET", "first") is None
            assert os.environ["C6_SET"] == "first"
            assert easyenv.set_env("C6_SET", "second") == "first"
            assert easyenv.unset_env("C6_SET") == "second"
            assert easyenv.unset_env("C6_SET") is None
            assert "C6_SET" not in os.environ

    def test_env_override_restores(self):
        with easyenv.env_override():
            os.environ["C6_KEEP"] = "orig"
            assert "C6_GONE" not in os.environ
            with easyenv.env_override(C6_KEEP="temp", C6_GONE="x"):
                assert os.environ["C6_KEEP"] == "temp"
                assert os.environ["C6_GONE"] == "x"
            assert os.environ["C6_KEEP"] == "orig"
            assert "C6_GONE" not in os.environ

    def test_env_override_restores_on_exception(self):
        with easyenv.env_override():
            os.environ["C6_KEEP"] = "orig"
            with pytest.raises(RuntimeError), easyenv.env_override(
                    C6_KEEP="temp"):
                raise RuntimeError("boom")
            assert os.environ["C6_KEEP"] == "orig"

    def test_env_override_can_delete(self):
        with easyenv.env_override():
            os.environ["C6_DEL"] = "v"
            with easyenv.env_override(C6_DEL=None):
                assert "C6_DEL" not in os.environ
            assert os.environ["C6_DEL"] == "v"


# -- .env files -------------------------------------------------------------


class TestDotenv:
    def test_parse_env_file(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text(
            "# comment\n"
            "PLAIN=hello\n"
            "QUOTED=\"with spaces\" # trailing\n"
            "SINGLE='single'\n"
            "export EXPORTED=1\n"
            "EMPTY=\n"
            "COMMENTED=   # note\n"
            "NO_EQUALS_HERE\n"
            "HASHED=\"a # b\"\n"
            "SPACED = padded \n",
            encoding="utf-8")
        parsed = easyenv.parse_env_file(dotenv)
        assert parsed["PLAIN"] == "hello"
        assert parsed["QUOTED"] == "with spaces"
        assert parsed["HASHED"] == "a # b"
        assert parsed["SINGLE"] == "single"
        assert parsed["EXPORTED"] == "1"
        assert parsed["EMPTY"] == ""
        assert parsed["COMMENTED"] == ""
        assert "NO_EQUALS_HERE" not in parsed
        assert parsed["SPACED"] == "padded"

    def test_parse_missing_file_is_empty(self, tmp_path):
        assert easyenv.parse_env_file(tmp_path / "nope.env") == {}

    def test_init_dotenv_real_env_wins(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text("C6_FROM_FILE=fileval\nC6_SHARED=fileval\n",
                          encoding="utf-8")
        monkeypatch.setenv("C6_SHARED", "realval")
        monkeypatch.delenv("C6_FROM_FILE", raising=False)
        loaded = easyenv.init_dotenv(dotenv)
        assert loaded == {"C6_FROM_FILE": "fileval"}
        assert os.environ["C6_FROM_FILE"] == "fileval"
        assert os.environ["C6_SHARED"] == "realval"
        monkeypatch.delenv("C6_FROM_FILE", raising=False)

    def test_init_dotenv_overwrite(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text("C6_SHARED=fileval\n", encoding="utf-8")
        monkeypatch.setenv("C6_SHARED", "realval")
        loaded = easyenv.init_dotenv(dotenv, overwrite=True)
        assert loaded == {"C6_SHARED": "fileval"}
        assert os.environ["C6_SHARED"] == "fileval"

    def test_init_dotenv_missing_file(self, tmp_path):
        assert easyenv.init_dotenv(tmp_path / "nope.env") == {}


# -- log level --------------------------------------------------------------


class TestLogLevel:
    def test_ensure_log_level(self, monkeypatch):
        monkeypatch.delenv("NM_LOG_LEVEL", raising=False)
        assert easyenv.ensure_log_level() == "INFO"
        assert os.environ["NM_LOG_LEVEL"] == "INFO"
        monkeypatch.setenv("NM_LOG_LEVEL", "DEBUG")
        assert easyenv.ensure_log_level() == "DEBUG"

    def test_init_all(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NM_LOG_LEVEL", raising=False)
        dotenv = tmp_path / ".env"
        dotenv.write_text("C6_INIT_ALL=1\n", encoding="utf-8")
        result = easyenv.init_all(dotenv_path=dotenv)
        assert result["log_level"] == "INFO"
        assert result["dotenv"] == {"C6_INIT_ALL": "1"}
        monkeypatch.delenv("C6_INIT_ALL", raising=False)


# -- secret hygiene ---------------------------------------------------------


class TestSecrets:
    @pytest.mark.parametrize("name", [
        "NM_GROQ_API_KEY", "MY_TOKEN", "X_SECRET", "DB_PASSWORD",
        "PASSWD", "PRIVATE_KEY", "AUTH_HEADER", "BEARER", "CREDENTIALS",
        "apikey",
    ])
    def test_secret_names(self, name):
        assert easyenv.is_secret_name(name) is True

    @pytest.mark.parametrize("name", [
        "NM_LOG_LEVEL", "NM_PROFILE", "HOME", "PATH",
    ])
    def test_non_secret_names(self, name):
        assert easyenv.is_secret_name(name) is False

    def test_substring_matching_is_deliberately_broad(self):
        # "TOKENIZER" contains "token" — it IS treated as secret.
        # Over-masking a display name is harmless; under-masking a
        # credential is not.
        assert easyenv.is_secret_name("TOKENIZER") is True

    def test_mask_secret(self):
        assert easyenv.mask_secret("short") == "***"
        assert easyenv.mask_secret("12345678") == "***"
        assert easyenv.mask_secret("supersecretvalue12345") == "supe…2345"

    def test_display_value_never_leaks(self):
        masked = easyenv.display_value(
            "NM_GROQ_API_KEY", "sk-live-abcdef")
        assert masked == "sk-l…cdef"
        assert "sk-live-abcdef" not in masked
        assert easyenv.display_value("NM_PROFILE", "workstation") == \
            "workstation"

    def test_describe_masks(self):
        info = easyenv.describe(
            "NM_GROQ_API_KEY", env={"NM_GROQ_API_KEY": "sk-live-abcdef"})
        assert info["present"] is True
        assert info["secret"] is True
        assert info["value"] == "sk-l…cdef"
        assert "sk-live-abcdef" not in str(info)
        assert easyenv.describe("C6_MISSING", env={}) == {
            "name": "C6_MISSING", "present": False, "secret": False,
            "value": None}


# -- config.py delegation (the moved parser + unified hints) ----------------


class TestConfigDelegation:
    def test_load_settings_honours_dotenv(self, tmp_path, monkeypatch):
        from nomorals.core.config import load_settings

        dotenv = tmp_path / ".env"
        dotenv.write_text(
            "NM_LOG_LEVEL=DEBUG # note\n"
            "export NM_PARTNER_PLATFORMS=local\n",
            encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NM_LOG_LEVEL", raising=False)
        monkeypatch.delenv("NM_PARTNER_PLATFORMS", raising=False)
        settings = load_settings()
        assert settings.log.level == "DEBUG"
        assert settings.partner.platforms == "local"

    def test_config_secret_check_is_the_unified_set(self):
        # The two hint copies in config.py diverged (6 vs 10 hints):
        # the old 6-hint `_looks_secret` did NOT mask auth/bearer/
        # passwd/private_key names. Both names now point at the one
        # canonical set in core.easyenv.
        from nomorals.core.config import _is_secret_key, _looks_secret

        for name in ("BEARER", "AUTH_HEADER", "PASSWD", "PRIVATE_KEY"):
            assert _looks_secret(name) is True
            assert _is_secret_key(name) is True
        assert _looks_secret("NM_LOG_LEVEL") is False

    def test_settings_dump_masks_secrets(self):
        from nomorals.core.config import Settings, redacted_dict

        settings = Settings()
        settings.llm.groq_api_key = "gsk-secret-value-12345"
        settings.llm.hf_token = "hf-secret-value-12345"
        dumped = redacted_dict(settings)
        assert "gsk-secret" not in str(dumped)
        assert "hf-secret" not in str(dumped)
        assert dumped["llm"]["groq_api_key"] == "***"
        assert dumped["llm"]["hf_token"] == "***"


# -- CLI --------------------------------------------------------------------


class TestCli:
    def test_cli_parser(self):
        from nomorals.cmdline.parser import _parser

        ns = _parser().parse_args(["env", "get", "NM_LOG_LEVEL"])
        assert ns.env_action == "get"
        assert ns.name == "NM_LOG_LEVEL"
        ns = _parser().parse_args(["env", "check", "A", "B",
                                   "--as", "bool"])
        assert ns.env_action == "check"
        assert ns.names == ["A", "B"]
        assert ns.as_kind == "bool"
        ns = _parser().parse_args(["env", "list", "NM_"])
        assert ns.env_action == "list"
        assert ns.prefix == "NM_"

    def test_cli_get_masks_secret(self, capsys):
        from nomorals.cmdline.commands.env import _cmd_env
        from nomorals.cmdline.parser import _parser

        with easyenv.env_override(NM_C6_TOKEN="supersecretvalue12345",
                                  NM_C6_FLAG="1"):
            args = _parser().parse_args(["env", "get", "NM_C6_TOKEN"])
            assert _cmd_env(args, None) == 0
            out = capsys.readouterr().out
            assert "supersecret" not in out
            assert "supe…2345" in out
            args = _parser().parse_args(
                ["env", "get", "NM_C6_TOKEN", "--show"])
            assert _cmd_env(args, None) == 0
            assert "supersecretvalue12345" in capsys.readouterr().out
            args = _parser().parse_args(["env", "get", "NM_C6_FLAG"])
            assert _cmd_env(args, None) == 0
            assert "NM_C6_FLAG=1" in capsys.readouterr().out

    def test_cli_get_missing_is_rc1(self, capsys):
        from nomorals.cmdline.commands.env import _cmd_env
        from nomorals.cmdline.parser import _parser

        args = _parser().parse_args(["env", "get", "C6_DEFINITELY_MISSING"])
        assert _cmd_env(args, None) == 1
        assert "(not set)" in capsys.readouterr().out

    def test_cli_check_rc(self, capsys):
        from nomorals.cmdline.commands.env import _cmd_env
        from nomorals.cmdline.parser import _parser

        with easyenv.env_override(C6_FLAG="1"):
            args = _parser().parse_args(
                ["env", "check", "C6_FLAG", "--as", "bool"])
            assert _cmd_env(args, None) == 0
            assert "ok" in capsys.readouterr().out
        with easyenv.env_override(C6_FLAG="maybe"):
            args = _parser().parse_args(
                ["env", "check", "C6_FLAG", "--as", "bool"])
            assert _cmd_env(args, None) == 1
            assert "FAIL" in capsys.readouterr().out
        args = _parser().parse_args(["env", "check", "C6_MISSING"])
        assert _cmd_env(args, None) == 1

    def test_cli_status_and_dotenv(self, capsys, tmp_path):
        from nomorals.cmdline.commands.env import _cmd_env
        from nomorals.cmdline.parser import _parser

        args = _parser().parse_args(["env", "status"])
        assert _cmd_env(args, None) == 0
        assert "mapped set:" in capsys.readouterr().out
        dotenv = tmp_path / ".env"
        dotenv.write_text("C6_DOTENV_NEW=hello\n", encoding="utf-8")
        with easyenv.env_override():
            args = _parser().parse_args(["env", "dotenv", str(dotenv)])
            assert _cmd_env(args, None) == 0
            out = capsys.readouterr().out
            assert "would load" in out
            assert "C6_DOTENV_NEW" in out


# -- chat -------------------------------------------------------------------


class TestChat:
    def test_chat_catalog(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            list_catalog,
        )

        assert "env" in CONTROL_COMMANDS
        assert CONTROL_COMMANDS["env"][0] == 0  # owner only
        page = COMMAND_DETAILS["env"]
        assert page["what"].strip()
        assert page["usage"].strip()
        assert page["example"].strip()
        assert "run /env." not in page["what"]  # no placeholder page
        assert "env" in LIST_ONELINERS
        groups = [title for title, kinds in LIST_GROUPS if "env" in kinds]
        assert groups == ["tools & automation"]
        assert "/env" in list_catalog()

    def test_chat_handler_masks_secrets(self):
        from nomorals.agents.partner.runtime_env import RuntimeEnvMixin

        mixin = RuntimeEnvMixin()
        with easyenv.env_override(NM_C6_TOKEN="supersecretvalue12345",
                                  NM_C6_FLAG="1"):
            found = mixin._control_env("get NM_C6_TOKEN", "test-key")
            assert "supersecret" not in found
            assert "supe…2345" in found
            found = mixin._control_env("get NM_C6_FLAG", "test-key")
            assert found == "NM_C6_FLAG=1"
            found = mixin._control_env("get C6_MISSING", "test-key")
            assert "(not set)" in found
            found = mixin._control_env(
                "check NM_C6_FLAG --as bool", "test-key")
            assert "ok" in found
            assert "all ok" in found
            found = mixin._control_env("check C6_MISSING", "test-key")
            assert "FAIL" in found
            found = mixin._control_env("status", "test-key")
            assert "mapped set:" in found
            found = mixin._control_env("", "test-key")
            assert "/env get" in found


# -- tool count: this lane adds NO brain tool, so the count stays 784 -------


class TestToolCount:
    def test_tool_count_still_784(self):
        # 782 after C.2 registered the (previously dead) tonal_g2p tool;
        # 783 after C.1 added smarthome; 784 after C.4 added camwall.
        # C.6 adds no tool — the env surface is `nm env` + `/env` over
        # the core/easyenv organ — so the count does not move.
        from nomorals.tools.registry import ToolRegistry

        count = len(ToolRegistry(SimpleNamespace()).register_builtins()
                    ._tools)
        assert count == 784
