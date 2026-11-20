"""Typed environment-variable helpers — the easyenv.rs port.

Ported from echelon/easyenv.rs (Brandon Thomas, 2020-2023), MIT
licence — see ``LICENSE``. The crate's ``Cargo.toml`` declares
``BSD-4-Clause`` but the LICENCE file shipped with the crate is
MIT; the FILE wins, and this port carries the MIT attribution.

The Rust crate is five small getter families (string / bool / num /
duration-seconds / path) in three flavours — optional, or-default,
required — plus a ``.env`` loader and an env-logger initializer.
This port keeps every one of those shapes and names, then extends
where Devon needs more:

* booleans use Devon's truth table (``core.config._coerce``): true is
  ``1/true/yes/on/y``, false is ``0/false/no/off/n`` — the Rust
  crate only accepted ``true/TRUE/false/FALSE``. Anything else is a
  parse error that names the value, never a silent ``False``.
* durations accept a ``s/m/h/d`` suffix (``5m``, ``2h``) on top of
  the crate's plain-seconds form.
* ``EnvError``'s display is fixed: the Rust ``Display`` impl always
  printed ``EnvError::ParseError`` for every variant. Here each
  variant names itself, the variable, and the reason.
* the crate's ``env_logger`` initializer has no Devon equivalent —
  Devon configures logging at boot from ``Settings``. The analog is
  :func:`ensure_log_level`, which guarantees the ``NM_LOG_LEVEL``
  variable the settings tree reads (``_ENV_MAP``) is set, and
  nothing else.
* two conveniences the crate never had, because Devon needs them:
  prefix-scoped reads (:func:`prefixed`) and a safe setter /
  restore-on-exit override (:func:`set_env`, :func:`unset_env`,
  :func:`env_override`) for provisioning and tests.
* secret hygiene: :func:`is_secret_name` / :func:`mask_secret` are
  the single home for the credential-name hints ``core.config``
  used to carry twice (two divergent copies). Values for
  secret-named variables are never shown in full — see
  :func:`display_value`.

Empty / whitespace-only values mean "not set" for every non-string
type (the ``_coerce`` convention) — the string family keeps an empty
string as a real value, exactly as the settings tree does.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

__all__ = [
    "LOG_LEVEL_ENV",
    "DEFAULT_LOG_LEVEL",
    "EnvError",
    "EnvNotUnicode",
    "EnvParseError",
    "EnvRequiredNotPresent",
    "SECRET_HINTS",
    "get_env_string_optional",
    "get_env_string_or_default",
    "get_env_string_required",
    "get_env_bool_optional",
    "get_env_bool_or_default",
    "get_env_bool_required",
    "get_env_num",
    "get_env_num_optional",
    "get_env_num_required",
    "get_env_duration_seconds_optional",
    "get_env_duration_seconds_or_default",
    "get_env_duration_seconds_required",
    "get_env_pathbuf_optional",
    "get_env_pathbuf_or_default",
    "get_env_pathbuf_required",
    "prefixed",
    "set_env",
    "unset_env",
    "env_override",
    "parse_env_file",
    "env_file_candidates",
    "init_dotenv",
    "ensure_log_level",
    "init_all",
    "is_secret_name",
    "mask_secret",
    "display_value",
    "describe",
]

_log = logging.getLogger(__name__)

#: The variable the settings tree reads for the log level
#: (``core.config._ENV_MAP``: ``NM_LOG_LEVEL`` -> ``log.level``).
#: The Rust crate's ``RUST_LOG`` analog for Devon.
LOG_LEVEL_ENV = "NM_LOG_LEVEL"

#: Fallback log level when neither the environment nor the caller
#: supplies one — the crate's ``DEFAULT_LOG_LEVEL`` (``"info"``),
#: upper-cased to match Devon's ``LogSettings.level``.
DEFAULT_LOG_LEVEL = "INFO"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "y"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "n"})
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$",
                          re.IGNORECASE)
_DURATION_MULTIPLIERS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0,
                         "d": 86400.0}

#: Name fragments that mark a value as a credential
#: (case-insensitive substring match). The union of the two copies
#: ``core.config`` used to carry (``_looks_secret``'s six hints and
#: ``_is_secret_key``'s ten) — one canonical set now.
SECRET_HINTS = ("token", "secret", "password", "passwd", "api_key",
                "apikey", "private_key", "auth", "credential", "bearer")


# ── errors ─────────────────────────────────────────────────────────────────


class EnvError(Exception):
    """Base class for every easyenv failure. Carries ``name``."""


class EnvNotUnicode(EnvError):
    """The variable's value is not valid unicode."""

    def __init__(self, name: str) -> None:
        super().__init__(f"env var {name!r} is not valid unicode")
        self.name = name


class EnvParseError(EnvError):
    """The variable is set but cannot be parsed as the desired type."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"env var {name!r}: {reason}")
        self.name = name
        self.reason = reason


class EnvRequiredNotPresent(EnvError):
    """The required variable is not set (or is empty)."""

    def __init__(self, name: str) -> None:
        super().__init__(f"required env var {name!r} is not set")
        self.name = name


# ── internals ──────────────────────────────────────────────────────────────


def _environ(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _raw(name: str, env: Mapping[str, str] | None) -> str | None:
    """The raw value, or ``None`` when absent.

    A value that cannot be encoded as UTF-8 (undecodable bytes
    behind ``os.environ``) raises :class:`EnvNotUnicode` — the
    ``NotUnicode`` variant the Rust crate returns from ``env::var``.
    """
    mapping = _environ(env)
    if name not in mapping:
        return None
    value = mapping[name]
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EnvNotUnicode(name) from exc
    return value


def _present(name: str, env: Mapping[str, str] | None) -> str | None:
    """The raw value, or ``None`` when absent or empty/whitespace-only.

    The non-string families treat an empty value as "not set"
    (``core.config._coerce``'s ``_EMPTY`` convention) — a template
    placeholder like ``NM_FLOAT=`` must not crash a parse.
    """
    value = _raw(name, env)
    if value is None or not value.strip():
        return None
    return value


def _required(name: str, env: Mapping[str, str] | None) -> str:
    value = _present(name, env)
    if value is None:
        _log.warning("Env var %r not supplied.", name)
        raise EnvRequiredNotPresent(name)
    return value


# ── string ─────────────────────────────────────────────────────────────────


def get_env_string_optional(name: str, *,
                            env: Mapping[str, str] | None = None,
                            ) -> str | None:
    """The variable as a string, or ``None`` when not set."""
    value = _raw(name, env)
    if value is None:
        _log.warning("Env var %r not supplied.", name)
        return None
    return value


def get_env_string_or_default(name: str, default: str, *,
                              env: Mapping[str, str] | None = None,
                              ) -> str:
    """The variable as a string, or ``default`` when not set."""
    value = _raw(name, env)
    if value is None:
        _log.warning("Env var %r not supplied. Using default %r.",
                     name, default)
        return default
    return value


def get_env_string_required(name: str, *,
                            env: Mapping[str, str] | None = None) -> str:
    """The variable as a string, or raise :class:`EnvRequiredNotPresent`."""
    value = _raw(name, env)
    if value is None:
        _log.warning("Required env var %r not supplied.", name)
        raise EnvRequiredNotPresent(name)
    return value


# ── bool ───────────────────────────────────────────────────────────────────


def _parse_bool(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise EnvParseError(
        name,
        f"couldn't parse as bool: {value!r} "
        f"(expected one of {sorted(_TRUE_VALUES | _FALSE_VALUES)})")


def get_env_bool_optional(name: str, *,
                          env: Mapping[str, str] | None = None,
                          ) -> bool | None:
    """The variable as a bool, or ``None`` when not set / unparseable."""
    value = _present(name, env)
    if value is None:
        _log.warning("Env var %r not supplied.", name)
        return None
    try:
        return _parse_bool(name, value)
    except EnvParseError as exc:
        _log.warning("Env var %r: error parsing boolean value: %s. "
                     "Returning no value.", name, exc.reason)
        return None


def get_env_bool_or_default(name: str, default: bool, *,
                            env: Mapping[str, str] | None = None) -> bool:
    """The variable as a bool, or ``default`` when not set / unparseable."""
    value = _present(name, env)
    if value is None:
        _log.warning("Env var %r not supplied. Using default %r.",
                     name, default)
        return default
    try:
        return _parse_bool(name, value)
    except EnvParseError as exc:
        _log.warning("Env var %r: error parsing boolean value: %s. "
                     "Using default %r.", name, exc.reason, default)
        return default


def get_env_bool_required(name: str, *,
                          env: Mapping[str, str] | None = None) -> bool:
    """The variable as a bool, or raise on absent / unparseable."""
    return _parse_bool(name, _required(name, env))


# ── num ────────────────────────────────────────────────────────────────────


def _parse_num(name: str, value: str, kind: type[int] | type[float]):
    try:
        return kind(value.strip())
    except (ValueError, OverflowError) as exc:
        raise EnvParseError(
            name,
            f"couldn't parse as {kind.__name__}: {value!r} ({exc})"
        ) from exc


def get_env_num(name: str, default: int | float, *,
                env: Mapping[str, str] | None = None) -> int | float:
    """The variable as a number, or ``default`` when not set.

    The type of ``default`` drives the parse (``int`` default parses
    as ``int``, ``float`` default as ``float``) — the Rust crate's
    single generic ``get_env_num<T>``. Present-but-unparseable raises
    :class:`EnvParseError` instead of silently returning the default.
    """
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        raise TypeError(
            f"get_env_num default must be int or float, got "
            f"{type(default).__name__}")
    value = _present(name, env)
    if value is None:
        _log.warning("Env var %r not supplied. Using default %r.",
                     name, default)
        return default
    return _parse_num(name, value, type(default))


def get_env_num_optional(name: str, kind: type[int] | type[float], *,
                         env: Mapping[str, str] | None = None,
                         ) -> int | float | None:
    """The variable as ``kind`` (``int``/``float``), or ``None``."""
    if kind not in (int, float):
        raise TypeError(f"kind must be int or float, got {kind!r}")
    value = _present(name, env)
    if value is None:
        _log.warning("Env var %r not supplied.", name)
        return None
    try:
        return _parse_num(name, value, kind)
    except EnvParseError as exc:
        _log.warning("Env var %r: error parsing numeric value: %s. "
                     "Returning no value.", name, exc.reason)
        return None


def get_env_num_required(name: str, kind: type[int] | type[float], *,
                         env: Mapping[str, str] | None = None,
                         ) -> int | float:
    """The variable as ``kind`` (``int``/``float``), or raise."""
    if kind not in (int, float):
        raise TypeError(f"kind must be int or float, got {kind!r}")
    return _parse_num(name, _required(name, env), kind)


# ── duration (seconds) ─────────────────────────────────────────────────────


def _parse_duration(name: str, value: str) -> timedelta:
    """Plain seconds (the crate's form) or a ``s/m/h/d`` suffix."""
    match = _DURATION_RE.match(value)
    if match is None:
        raise EnvParseError(
            name,
            f"couldn't parse as seconds: {value!r} "
            "(expected a number, optionally suffixed s/m/h/d)")
    amount, suffix = match.groups()
    try:
        seconds = float(amount) * _DURATION_MULTIPLIERS[suffix.lower()]
    except (ValueError, OverflowError) as exc:
        raise EnvParseError(
            name, f"couldn't parse as seconds: {value!r} ({exc})") from exc
    return timedelta(seconds=seconds)


def get_env_duration_seconds_optional(
        name: str, *,
        env: Mapping[str, str] | None = None) -> timedelta | None:
    """The variable as a duration in seconds, or ``None``."""
    value = _present(name, env)
    if value is None:
        _log.warning("Env var %r not present.", name)
        return None
    try:
        return _parse_duration(name, value)
    except EnvParseError as exc:
        _log.warning("Env var %r: error parsing numeric value: %s. "
                     "Returning no value.", name, exc.reason)
        return None


def get_env_duration_seconds_or_default(
        name: str, default: timedelta, *,
        env: Mapping[str, str] | None = None) -> timedelta:
    """The variable as a duration in seconds, or ``default``."""
    value = _present(name, env)
    if value is None:
        _log.warning("Env var %r not supplied. Using default %r.",
                     name, default)
        return default
    try:
        return _parse_duration(name, value)
    except EnvParseError as exc:
        _log.warning("Env var %r: error parsing numeric value: %s. "
                     "Using default %r.", name, exc.reason, default)
        return default


def get_env_duration_seconds_required(
        name: str, *,
        env: Mapping[str, str] | None = None) -> timedelta:
    """The variable as a duration in seconds, or raise."""
    return _parse_duration(name, _required(name, env))


# ── path ───────────────────────────────────────────────────────────────────


def get_env_pathbuf_optional(name: str, *,
                             env: Mapping[str, str] | None = None,
                             ) -> Path | None:
    """The variable as a path, or ``None`` when not set."""
    value = _present(name, env)
    if value is None:
        _log.warning("Env var %r not present.", name)
        return None
    return Path(value)


def get_env_pathbuf_or_default(name: str, default: str | Path, *,
                               env: Mapping[str, str] | None = None) -> Path:
    """The variable as a path, or ``default`` when not set."""
    value = _present(name, env)
    if value is None:
        default_path = Path(default)
        _log.warning("Env var %r not supplied. Using default %r.",
                     name, default_path)
        return default_path
    return Path(value)


def get_env_pathbuf_required(name: str, *,
                             env: Mapping[str, str] | None = None) -> Path:
    """The variable as a path, or raise :class:`EnvRequiredNotPresent`."""
    return Path(_required(name, env))


# ── prefix-scoped reads + safe setters ─────────────────────────────────────


def prefixed(prefix: str, *, env: Mapping[str, str] | None = None,
             strip: bool = True) -> dict[str, str]:
    """Every set variable starting with ``prefix``.

    With ``strip`` (default) the keys come back without the prefix —
    ``prefixed("NM_")`` returns ``{"LOG_LEVEL": "INFO", ...}`` — so a
    section reads like its own namespace. Values are raw strings;
    never pass the result anywhere that prints secrets unmasked —
    see :func:`display_value`.
    """
    mapping = _environ(env)
    out: dict[str, str] = {}
    for key, value in mapping.items():
        if not key.startswith(prefix):
            continue
        out[key[len(prefix):] if strip else key] = value
    return out


def set_env(name: str, value: str) -> str | None:
    """Set ``name`` in the process environment, returning the previous
    value (``None`` when it was unset). The inverse of
    :func:`unset_env`; prefer :func:`env_override` in tests so the
    old value is always restored."""
    previous = os.environ.get(name)
    os.environ[name] = value
    return previous


def unset_env(name: str) -> str | None:
    """Delete ``name`` from the process environment, returning the
    previous value (``None`` when it was already unset)."""
    return os.environ.pop(name, None)


@contextmanager
def env_override(**overrides: str | None) -> Iterator[None]:
    """Temporarily set environment variables, restoring on exit.

    ``None`` deletes the variable for the duration. Every previous
    value — including "was unset" — is restored even when the body
    raises. This is the setter tests and provisioning should use:
    a plain :func:`set_env` that is never undone leaks into every
    test that runs after it in the same process.
    """
    previous: dict[str, str | None] = {}
    for name in overrides:
        previous[name] = os.environ.get(name)
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ── .env files ─────────────────────────────────────────────────────────────


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a ``.env`` file: ``KEY=VALUE``, ``#`` comments, optional
    quotes, ``export`` prefix, inline ``#`` comments.

    The parser ``core.config`` has always used inside
    ``load_settings`` (moved here so there is one canonical copy),
    with one fix: a quoted value followed by a comment
    (``KEY="a b" # note``) now parses as ``a b`` — the old code
    checked whole-value quoting first and kept the leading quote.
    A ``#`` inside quotes is kept (``KEY="a # b"`` -> ``a # b``).
    ``KEY=   # note`` reads as empty (the value IS the comment), and
    a line without ``=`` is skipped. ``{}`` when the file does not
    exist.
    """
    values: dict[str, str] = {}
    path = Path(path)
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8",
                              errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'":
            # Quoted value: the closing quote ends the value; anything
            # after it is a comment (`KEY="a b" # note` -> `a b`).
            # The config parser used to check whole-value quoting first,
            # so that line parsed as `"a b` — a corrupted value.
            closing = value.find(value[0], 1)
            if closing != -1:
                value = value[1:closing]
                if key:
                    values[key] = value
                continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif value.startswith("#"):
            # `KEY=   # note` — the value IS the comment: empty, not "#" text
            value = ""
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key:
            values[key] = value
    return values


def env_file_candidates(home: str | None = None) -> list[Path]:
    """The ``.env`` files a boot consults, in order. First existing wins.

    The working directory's ``.env``, then ``$NM_HOME/.env`` — the
    same order ``core.config.load_settings`` uses.
    """
    env_home = os.environ.get("NM_HOME") or home or "~/.nomorals"
    return [Path.cwd() / ".env",
            Path(os.path.expanduser(env_home)) / ".env"]


def init_dotenv(path: str | Path | None = None, *,
                overwrite: bool = False) -> dict[str, str]:
    """Load a ``.env`` file into the process environment.

    With no ``path``, the first existing candidate from
    :func:`env_file_candidates` wins. Real environment variables
    always win over the file (``setdefault`` semantics — the same
    precedence ``load_settings`` applies); ``overwrite=True`` flips
    that for callers that explicitly want the file to win. Returns
    the keys actually applied. The crate's ``init_dotenv`` printed
    either way; this one returns the result and logs it.
    """
    candidates = ([Path(path)] if path is not None
                  else env_file_candidates())
    loaded: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for key, value in parse_env_file(candidate).items():
            if overwrite or key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
        break
    if loaded:
        _log.info("dotenv configs initialized from %s (%d keys)",
                  candidates[0] if path is None else path, len(loaded))
    else:
        _log.info("Could not initialize dotenv: no .env file found")
    return loaded


# ── logging level ──────────────────────────────────────────────────────────


def ensure_log_level(default: str = DEFAULT_LOG_LEVEL) -> str:
    """Guarantee ``NM_LOG_LEVEL`` is set, returning the effective level.

    The crate's ``init_env_logger`` analog: the crate defaulted
    ``RUST_LOG`` and then initialized ``env_logger``. Devon has no
    ``env_logger`` — logging is configured at boot from ``Settings``
    (``log.level``, fed by this very variable) — so this stops at
    guaranteeing the variable itself, and never touches a logger.
    """
    level = os.environ.get(LOG_LEVEL_ENV)
    if level is None or not level.strip():
        os.environ[LOG_LEVEL_ENV] = default
        _log.info("Setting default logging level to %r, override with "
                  "env var %s.", default, LOG_LEVEL_ENV)
        return default
    return level


def init_all(default_log_level: str = DEFAULT_LOG_LEVEL, *,
             dotenv_path: str | Path | None = None) -> dict[str, object]:
    """Load ``.env`` and guarantee a log level — the crate's
    ``init_all_with_default_logging``. Returns ``{"dotenv": {...},
    "log_level": ...}``."""
    return {"dotenv": init_dotenv(dotenv_path),
            "log_level": ensure_log_level(default_log_level)}


# ── secret hygiene ─────────────────────────────────────────────────────────


def is_secret_name(name: str) -> bool:
    """True when ``name`` smells like a credential (token, secret,
    password, api_key, auth, bearer, …). Case-insensitive substring
    match over :data:`SECRET_HINTS`."""
    lowered = name.lower()
    return any(hint in lowered for hint in SECRET_HINTS)


def mask_secret(value: str, *, placeholder: str = "***") -> str:
    """Mask a secret for display: short values become ``***``; longer
    ones keep the first and last four characters (``abcd…wxyz``) so
    the owner can tell two keys apart without seeing either."""
    if len(value) <= 8:
        return placeholder
    return f"{value[:4]}…{value[-4:]}"


def display_value(name: str, value: str) -> str:
    """The value as it may be shown: masked when ``name`` is secret,
    verbatim otherwise. Every surface that prints an env value —
    the ``nm env`` CLI, the ``/env`` chat verb — goes through here,
    so a secret is never printed in full by accident."""
    return mask_secret(value) if is_secret_name(name) else value


def describe(name: str, *,
             env: Mapping[str, str] | None = None) -> dict[str, object]:
    """One variable, inspection-ready: ``{"name", "present",
    "secret", "value"}`` — ``value`` is the masked display form when
    the name is secret, ``None`` when the variable is not set."""
    value = _raw(name, env)
    return {"name": name,
            "present": value is not None,
            "secret": is_secret_name(name),
            "value": (None if value is None
                      else display_value(name, value))}


# ── MIT licence (echelon/easyenv.rs, Copyright (c) 2020 Brandon Thomas) ──
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
