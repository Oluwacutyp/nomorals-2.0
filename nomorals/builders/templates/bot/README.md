# $PROJECT_NAME

A chat-bot scaffold with a clean adapter interface and a zero-credential
console runner. No third-party dependencies.

## The adapter interface

The entire bot is `Bot.on_message(text) -> str`:

```python
from bot import Bot

bot = Bot()
reply = bot.on_message("/echo hello")   # -> "hello"
```

Wire any chat platform (Telegram, Discord, WhatsApp, ...) by feeding
incoming text to `on_message` and sending the returned string back.

## Run (console, no credentials)

```sh
python run.py
```

Built-in commands: `/start` `/help` `/ping` `/echo` `/reverse` `/upper`.

## Test

```sh
python -m unittest discover -s tests -t .
```
