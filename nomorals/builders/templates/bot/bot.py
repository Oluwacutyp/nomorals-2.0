"""$PROJECT_NAME -- a chat bot with a clean adapter interface.

The whole bot is one method: :meth:`Bot.on_message`.  Wire any chat
platform (Telegram, Discord, WhatsApp, ...) by calling ``on_message``
with the incoming text and sending back the returned string.

Run with zero credentials against the console:

    python run.py
"""

from __future__ import annotations


class Bot:
    """Message handler.  Stateless, synchronous, dependency-free."""

    def __init__(self, name: str = "$PROJECT_NAME") -> None:
        self.name = name

    # -- adapter interface -------------------------------------------------
    def on_message(self, text: str) -> str:
        """Handle one incoming message, return the reply text."""
        text = (text or "").strip()
        if not text:
            return "Say something -- try /help."
        if text.startswith("/"):
            parts = text[1:].split(None, 1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            handler = getattr(self, f"cmd_{command}", None)
            if handler is not None:
                return handler(arg)
            return f"Unknown command /{command}. Try /help."
        return f"You said: {text}"

    # -- built-in commands ---------------------------------------------------
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


class ConsoleRunner:
    """Zero-credential runner: type messages, get replies."""

    def __init__(self, bot: Bot | None = None) -> None:
        self.bot = bot or Bot()

    def run(self) -> int:
        print(f"{self.bot.name} console. Type /quit to exit.", flush=True)
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):  # noqa: E106 - deliberate REPL exit
                print()
                break
            if line.strip().lower() in ("/quit", "/exit"):
                break
            print(self.bot.on_message(line), flush=True)
        return 0


def main() -> int:
    return ConsoleRunner().run()
