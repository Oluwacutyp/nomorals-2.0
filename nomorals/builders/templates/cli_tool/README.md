# $PROJECT_NAME

An argparse-based command-line tool scaffold. No third-party dependencies.

## Run

```sh
python run.py --help
python run.py greet --name Ada
python run.py greet --name Ada --shout
python run.py --version
```

## Commands

| Command | Description        |
|---------|--------------------|
| `greet` | Greet someone (`--name`, `--shout`) |

## Test

```sh
python -m unittest discover -s tests -t .
```
