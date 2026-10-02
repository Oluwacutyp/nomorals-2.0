"""$PROJECT_NAME -- Telegram bot on the raw Bot HTTP API (stdlib only).

No python-telegram-bot, no framework: urllib long-polling plus a command
router.  Adding a command is one method named ``cmd_<command>`` on
:class:`Bot`.

The token comes from the ``BOT_TOKEN`` environment variable (get one
from @BotFather).  Everything is unit-testable without network:
:class:`TelegramClient` accepts a fake transport and
:meth:`Bot.handle_update` takes plain update dicts.

Run:
    export BOT_TOKEN=<token from @BotFather>
    python run.py
    python run.py --self-test   # offline: canned updates, no network
    python run.py --once        # fetch one update batch, then exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable

PROJECT = "$PROJECT_NAME"
API_BASE = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org")


class TelegramError(Exception):
    """The Bot API answered ok:false, or the transport failed."""


class TelegramClient:
    """Minimal Bot API client.  ``post`` is swappable for tests."""

    def __init__(self, token: str,
                 post: Callable[[str, dict], dict] | None = None) -> None:
        if not token:
            raise TelegramError("BOT_TOKEN is empty")
        self.token = token
        self._post = post or self._http_post

    def _url(self, method: str) -> str:
        return f"{API_BASE}/bot{self.token}/{method}"

    def _http_post(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise TelegramError(f"transport failed for {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise TelegramError(f"bad JSON from {url}: {exc}") from exc

    def call(self, method: str, payload: dict | None = None) -> Any:
        result = self._post(self._url(method), payload or {})
        if not isinstance(result, dict) or not result.get("ok"):
            raise TelegramError(f"Bot API {method} failed: {result!r}"[:300])
        return result["result"]

    def send_message(self, chat_id: int | str, text: str) -> Any:
        return self.call("sendMessage", {"chat_id": chat_id, "text": text})

    def get_updates(self, offset: int = 0, timeout: int = 30) -> list[dict]:
        return self.call("getUpdates", {"offset": offset, "timeout": timeout,
                                        "allowed_updates": ["message"]})


class Bot:
    """Command router.  Each ``cmd_<name>`` method handles ``/<name>``."""

    def __init__(self, client: TelegramClient, name: str = PROJECT) -> None:
        self.client = client
        self.name = name

    # -- update handling ---------------------------------------------------
    def handle_update(self, update: dict) -> str | None:
        """Process one getUpdates dict; send the reply, return its text."""
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        text = str(message.get("text") or "").strip()
        if chat_id is None or not text:
            return None
        reply = self.dispatch(text)
        if reply is not None:
            self.client.send_message(chat_id, reply)
        return reply

    def dispatch(self, text: str) -> str:
        """Map raw message text to a reply string (no I/O)."""
        text = (text or "").strip()
        if text.startswith("/"):
            parts = text[1:].split(None, 1)
            command = parts[0].lower().split("@", 1)[0]  # strip @bot suffix
            arg = parts[1] if len(parts) > 1 else ""
            handler = getattr(self, f"cmd_{command}", None)
            if handler is not None:
                return handler(arg)
            return f"Unknown command /{command}. Try /help."
        return f"You said: {text}"

    # -- commands ----------------------------------------------------------
    def cmd_start(self, _arg: str) -> str:
        return (f"Hello! I am {self.name}. "
                "Send any text and I will echo it, or try /help.")

    def cmd_help(self, _arg: str) -> str:
        return (
            "Commands:\n"
            "/start -- greet\n"
            "/help -- this message\n"
            "/ping -- liveness check\n"
            "/echo <text> -- repeat <text>\n"
            "/reverse <text> -- reverse <text>\n"
            "/upper <text> -- shout <text>\n"
            "Anything else is echoed back."
        )

    def cmd_ping(self, _arg: str) -> str:
        return "pong"

    def cmd_echo(self, arg: str) -> str:
        return arg if arg else "Usage: /echo <text>"

    def cmd_reverse(self, arg: str) -> str:
        return arg[::-1] if arg else "Usage: /reverse <text>"

    def cmd_upper(self, arg: str) -> str:
        return arg.upper() if arg else "Usage: /upper <text>"


def poll(client: TelegramClient, bot: Bot, *, once: bool = False,
         poll_timeout: int = 30) -> int:
    """Long-poll getUpdates and route every message through the bot."""
    offset = 0
    while True:
        updates = client.get_updates(offset=offset, timeout=poll_timeout)
        for update in updates:
            offset = max(offset, int(update.get("update_id", 0)) + 1)
            try:
                bot.handle_update(update)
            except TelegramError as exc:
                # one bad update must not kill the loop; the next
                # batch still advances the offset
                print(f"update failed: {exc}", file=sys.stderr)
        if once:
            return 0


def _recording_post(calls: list[tuple[str, dict]]) -> Callable[[str, dict], dict]:
    def post(url: str, payload: dict) -> dict:
        calls.append((url, payload))
        if url.endswith("/getUpdates"):
            return {"ok": True, "result": []}
        return {"ok": True, "result": {"message_id": len(calls)}}
    return post


def self_test() -> int:
    """Offline sanity check: canned updates through a recording client."""
    calls: list[tuple[str, dict]] = []
    client = TelegramClient("self-test-token", post=_recording_post(calls))
    bot = Bot(client)
    updates = [
        {"update_id": 1, "message": {"chat": {"id": 7}, "text": "/start"}},
        {"update_id": 2, "message": {"chat": {"id": 7}, "text": "/ping"}},
        {"update_id": 3, "message": {"chat": {"id": 7}, "text": "/echo hi"}},
        {"update_id": 4, "message": {"chat": {"id": 7}, "text": "plain"}},
        {"update_id": 5, "message": {"chat": {"id": 7}, "text": "/nope"}},
    ]
    replies = [bot.handle_update(u) for u in updates]
    sent = [url for url, _ in calls if url.endswith("/sendMessage")]
    ok = (all(r is not None for r in replies)
          and len(sent) == len(updates)
          and replies[1] == "pong"
          and replies[2] == "hi")
    print(f"{PROJECT} self-test: "
          f"{len(updates)} updates, {len(sent)} messages sent -> "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="$PROJECT_NAME",
        description=f"{PROJECT} -- Telegram bot on the raw Bot HTTP API.")
    parser.add_argument("--self-test", action="store_true",
                        help="run canned updates through a fake client "
                             "(no network, no token)")
    parser.add_argument("--once", action="store_true",
                        help="fetch one batch of updates, then exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    token = os.environ.get("BOT_TOKEN", "")
    if not token:
        print("BOT_TOKEN is not set -- get a token from @BotFather, then run:\n"
              "  export BOT_TOKEN=<your token>\n"
              "  python run.py",
              file=sys.stderr)
        return 2

    client = TelegramClient(token)
    bot = Bot(client)
    try:
        me = client.call("getMe")
        print(f"{PROJECT} polling as @{me.get('username', '?')} "
              f"({bot.name}) -- Ctrl-C to stop", flush=True)
    except TelegramError as exc:
        print(f"could not reach Telegram: {exc}", file=sys.stderr)
        return 1
    try:
        return poll(client, bot, once=args.once)
    except KeyboardInterrupt:  # noqa: E103, E106 - deliberate shutdown hook
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
