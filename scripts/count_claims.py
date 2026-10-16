"""Regenerate the measured numbers quoted in README.md / Makefile.

Every number printed here is produced by running the same command the docs
cite next to it, so the docs cannot drift from reality without this script
disagreeing. Suite runtime is NOT measured here (a full run takes minutes);
it is recorded by whoever runs the suite in docs/SUITE_RUNTIME.md, and this
script reports that file's contents.

Usage: python3 scripts/count_claims.py
"""
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(args: list[str]) -> str:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    ).stdout


def main() -> int:
    tools_out = run(["nomorals", "tools"])
    tools = len(re.findall(r"(?m)^\S+ +\[[^\]]+\]", tools_out))
    doctor_out = run(["nomorals", "doctor"])
    m = re.search(r"schema v(\d+), (\d+) tables", doctor_out)
    schema = m.group(1) if m else "?"
    tables = int(m.group(2)) if m else -1
    today = datetime.date.today().isoformat()
    print(f"measured: {today}")
    print(f"tools: {tools}  (python3 -m nomorals tools | grep -cE '^\\S+ +\\[')")
    print(f"tables: {tables}  schema v{schema}  (python3 -m nomorals doctor)")
    runtime_file = ROOT / "docs" / "SUITE_RUNTIME.md"
    if runtime_file.exists():
        print("--- docs/SUITE_RUNTIME.md ---")
        print(runtime_file.read_text(encoding="utf-8").strip())
    else:
        print("suite runtime: no docs/SUITE_RUNTIME.md yet — run the suite, then record it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
