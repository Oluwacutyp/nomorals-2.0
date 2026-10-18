# $PROJECT_NAME

A Telegram bot built on the **raw Bot HTTP API** with the Python standard
library only — no `python-telegram-bot`, no framework. It long-polls
`getUpdates` and routes every message through a command router.

## Run

```sh
export BOT_TOKEN=<token from @BotFather>
python run.py              # long-poll until Ctrl-C
python run.py --once       # fetch one batch of updates, then exit
python run.py --self-test  # offline check: canned updates, no network/token
```

You can point at a local Bot API server with `TELEGRAM_API_BASE`.

## Adding a command

One method on `Bot`, named `cmd_<command>`:

```python
def cmd_roll(self, arg: str) -> str:
    import random
    return f"You rolled {random.randint(1, 6)}"
```

`/roll` now works. `Bot.dispatch(text) -> str` is pure (no I/O), and
`Bot.handle_update(update)` takes plain `getUpdates` dicts, so commands
are unit-testable without touching the network — pass a fake `post`
callable to `TelegramClient`.

Built-in commands: `/start` `/help` `/ping` `/echo` `/reverse` `/upper`.

## Test

```sh
python -m unittest discover -s tests -t .
```
