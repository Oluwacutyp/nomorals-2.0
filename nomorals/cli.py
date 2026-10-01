"""Command-line interface: ``nm`` / ``python -m nomorals``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .compat import feature_report, report_as_text
from .core.logging_setup import get_logger, setup_logging

_log = get_logger(__name__)
from .version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nm", description="NoMorals Core — self-hosted multi-agent AI substrate."
    )
    parser.add_argument("--version", action="version", version=f"nomorals {__version__}")
    parser.add_argument("--config", help="path to a TOML config file")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="show environment capabilities and health")
    sub.add_parser("config", help="print the effective configuration")
    sub.add_parser("setup", help="guided model setup wizard")

    models = sub.add_parser("models", help="inspect the model registry and catalog")
    models.add_argument("--catalog", action="store_true", help="list the curated catalog")
    models.add_argument("--search", default="")
    models.add_argument("--kind", default="")
    models.add_argument("--max-params", type=int, default=0)
    models.add_argument("--activate", default="", help="activate a registered model by name")
    models.add_argument("--promote-local", default="",
                        help="point the whole system at a local GGUF (writes the .env contract)")
    models.add_argument("--lora", default="",
                        help="with --promote-local: GGUF LoRA adapter(s) to run "
                             "on top of the base model (comma-separated)")

    data = sub.add_parser("data", help="fine-tune data: catalog, base models, persona mix")
    data.add_argument("action", nargs="?", default="catalog",
                      choices=["catalog", "models", "mix"],
                      help="what to do (default: catalog)")
    data.add_argument("sources", nargs="?", default="",
                      help="mix: comma-separated dataset names (files under the "
                           "training data dir, <name>.jsonl)")
    data.add_argument("--rows", type=int, default=0,
                      help="mix: target row count (default: the full recipe target)")
    data.add_argument("--persona", default="", help="mix: persona text override")
    data.add_argument("--persona-file", default="",
                      help="mix: read the persona from this file (built-in default otherwise)")
    data.add_argument("--base", default="",
                      help="mix: base model id for the Colab script (default: the abliterated Qwen)")
    data.add_argument("--out", default="",
                      help="mix: output base path (default: <data dir>/persona-mix)")
    data.add_argument("--json", action="store_true", help="Output as JSON")

    tools = sub.add_parser("tools", help="list available tools and their capabilities")
    tools.add_argument("--schema", action="store_true", help="emit full JSON schemas")

    memory = sub.add_parser("memory", help="inspect and query memory")
    memory.add_argument("--query", default="")
    memory.add_argument("--limit", type=int, default=8)
    memory.add_argument("--consolidate", action="store_true")
    memory.add_argument("--stats", action="store_true")
    memory.add_argument("--remember", default="")

    agent = sub.add_parser("run", help="run a goal through the orchestrator")
    agent.add_argument("goal", nargs="+")
    agent.add_argument("--max-steps", type=int, default=8)
    agent.add_argument("--no-reflect", action="store_true")

    ask = sub.add_parser("ask", help="single-turn chat with the active model")
    ask.add_argument("prompt", nargs="+")
    ask.add_argument("--system", default="")
    ask.add_argument("--max-tokens", type=int, default=1024)
    ask.add_argument("--temperature", type=float, default=0.7)
    ask.add_argument("--model", default="", help="hot-swap to this provider for this call")

    backup = sub.add_parser("backup", help="create, list, verify, or restore backups")
    backup.add_argument("--create", action="store_true")
    backup.add_argument("--list", action="store_true")
    backup.add_argument("--verify", action="store_true")
    backup.add_argument("--restore", default="")
    backup.add_argument("--push", action="store_true", help="push the latest backup to git")

    missions = sub.add_parser("missions", help="list, run, resume, or inspect missions")
    missions.add_argument("--start", default="", help="start a new mission with this goal")
    missions.add_argument("--resume", default="", help="resume a mission by id")
    missions.add_argument("--resume-all", action="store_true", help="resume every interrupted mission")
    missions.add_argument("--status", default="", help="filter by status")
    missions.add_argument("--show", default="", help="show one mission's detail and checkpoints")
    missions.add_argument("--max-iterations", type=int, default=8)
    missions.add_argument("--budget-wall", type=float, default=0.0)
    missions.add_argument("--budget-tokens", type=int, default=0)
    missions.add_argument("--no-reflect", action="store_true")
    missions.add_argument("--pause", default="", help="pause a mission by id (cross-process)")
    missions.add_argument("--cancel", default="", help="cancel a mission by id (terminal)")
    missions.add_argument("--resume-status", default="",
                          help="lift a pause and report the mission's status")

    sub.add_parser("serve", help="start the HTTP API server")
    sub.add_parser("tui", help="start the interactive terminal UI")
    queue = sub.add_parser("queue", help="inspect the durable work queue")
    queue.add_argument("--topic", default="")
    
    # Additional subcommands expected by tests
    commands = sub.add_parser("commands", help="list available commands")
    commands.add_argument("filter", nargs="?", default="")
    
    zip_cmd = sub.add_parser("zip", help="create and manage zip archives")
    zip_cmd.add_argument("action", nargs="?", default="list")
    zip_cmd.add_argument("path", nargs="?", default="")
    zip_cmd.add_argument("--dest", default="")
    

    # Agent-tool subcommands
    autonomy = sub.add_parser("autonomy", help="Manage autonomous agent operations")
    autonomy.add_argument("action", nargs="?", default="status",
                         choices=["status", "tick", "report", "enable", "disable", "budget", "on", "off"],
                         help="Action to perform")
    autonomy.add_argument("--json", action="store_true", help="Output as JSON")
    autonomy.add_argument("--cap", default="",
                          help="set the daily model-call cap (budget action; persisted)")
    autonomy.add_argument("--unlimited", action="store_true",
                          help="lift the daily model-call cap (budget action)")
    
    goal = sub.add_parser("goal", help="Create and manage goals")
    goal.add_argument("action", nargs="?", default="list",
                     choices=["list", "create", "get", "update", "delete", "replan",
                              "depends", "next", "priority", "advance", "adapt",
                              "tick", "status", "spawn", "complete", "pause", "resume",
                              "reflect"],
                     help="Action to perform")
    goal.add_argument("arg", nargs="?", default="",
                     help="goal title (create) or goal id (most actions)")
    goal.add_argument("arg2", nargs="?", default="",
                     help="second operand: dependency id (depends), value (priority)")
    goal.add_argument("--id", default="", help="Goal ID (alternative to positional)")
    goal.add_argument("--title", default="", help="Goal title")
    goal.add_argument("--description", default="", help="Goal description")
    goal.add_argument("--priority", default="", help="Goal priority (int)")
    goal.add_argument("--project", action="store_true",
                      help="spawn the autonomous project cascade for this goal")
    goal.add_argument("--remove", action="store_true",
                      help="remove the dependency (depends action)")
    goal.add_argument("--reason", default="", help="why to adapt (adapt action)")
    goal.add_argument("--status", dest="filter_status", default="",
                      help="filter list by status")
    goal.add_argument("--limit", default="20", help="max rows to list")
    goal.add_argument("--json", action="store_true", help="Output as JSON")
    
    mission = sub.add_parser("mission", help="Mission control and planning")
    mission.add_argument("action", nargs="?", default="plan",
                        choices=["plan", "next", "status", "health"],
                        help="Action to perform")
    mission.add_argument("--json", action="store_true", help="Output as JSON")
    
    skill = sub.add_parser("skill", help="Manage reusable skills")
    skill.add_argument("action", nargs="?", default="list",
                      choices=["list", "create", "run", "delete"],
                      help="Action to perform")
    
    project = sub.add_parser("project", help="Manage projects")
    project.add_argument("action", nargs="?", default="list",
                        choices=["list", "create", "get", "update", "delete",
                                 "heal", "replan", "status", "plan", "run", "advance"],
                        help="Action to perform")
    project.add_argument("title", nargs="?", default="", help="Project title (or id for other actions)")
    project.add_argument("description", nargs="?", default="", help="Project objective")
    project.add_argument("--id", default="", help="Project ID (alternative to positional)")
    project.add_argument("--json", action="store_true", help="Output as JSON")
    
    kg = sub.add_parser("kg", help="Knowledge graph operations")
    kg.add_argument("action", nargs="?", default="stats",
                    choices=["stats", "consolidate", "communities", "top",
                             "decay"],
                    help="Action to perform")
    kg.add_argument("--limit", default="10", help="rows to show (top/communities)")
    kg.add_argument("--json", action="store_true", help="Output as JSON")
    sub.add_parser("simulate", help="Sandbox code execution")
    code = sub.add_parser(
        "code",
        help="Coding agent: run a task, review diffs, run tests",
        description=("nm code \"<task>\" [--file F] [--accept CMD] [--max-iters N] [--root DIR]\n"
                     "nm code review [ref] [--root DIR]\n"
                     "nm code test [--changed] [--root DIR]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # `task` takes the remaining words so `nm code review` / `nm code test`
    # work as subcommands without argparse subparser conflicts.
    code.add_argument("task", nargs="*", default=[],
                      help='task to run, or "review [ref]" / "test"')
    code.add_argument("--file", default="main.py", help="target file for the task")
    code.add_argument("--accept", default="",
                      help="acceptance command (default: python3 <file>)")
    code.add_argument("--max-iters", type=int, default=5, help="max draft iterations")
    code.add_argument("--root", default=".", help="project root the agent works in")
    code.add_argument("--changed", action="store_true",
                      help="with test: only run tests for changed files")
    code.add_argument("--json", action="store_true", help="Output as JSON")

    media = sub.add_parser(
        "media",
        help="Edit images and video from plain language",
        description=("nm media edit <file> \"<instruction>\" [--dry-run] [--wait]\n"
                     "nm media probe <file>\n"
                     "nm media jobs [--limit N]\n"
                     "nm media convert <file> <fmt>"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    media_sub = media.add_subparsers(dest="media_action", required=True)
    m_edit = media_sub.add_parser("edit", help="edit an image or video")
    m_edit.add_argument("file", help="image or video file (workspace-relative)")
    m_edit.add_argument("instruction", help="plain-language edit instruction")
    m_edit.add_argument("--dry-run", action="store_true",
                        help="show the planned op chain without executing")
    m_edit.add_argument("--wait", action="store_true",
                        help="with video: block until the background job finishes")
    m_edit.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait with --wait (default 900)")
    m_edit.add_argument("--video", action="store_true",
                        help="force video handling regardless of extension")
    m_edit.add_argument("--image", action="store_true",
                        help="force image handling regardless of extension")
    m_edit.add_argument("--json", action="store_true", help="Output as JSON")
    m_probe = media_sub.add_parser("probe", help="probe an image or video file")
    m_probe.add_argument("file", help="file to probe (workspace-relative)")
    m_probe.add_argument("--json", action="store_true", help="Output as JSON")
    m_jobs = media_sub.add_parser("jobs", help="list recent media jobs")
    m_jobs.add_argument("--limit", type=int, default=20)
    m_jobs.add_argument("--json", action="store_true", help="Output as JSON")
    m_conv = media_sub.add_parser("convert", help="convert to another format")
    m_conv.add_argument("file", help="file to convert (workspace-relative)")
    m_conv.add_argument("fmt", help="target format: webp, png, mp4, ...")
    m_conv.add_argument("--wait", action="store_true",
                        help="with video: block until the background job finishes")
    m_conv.add_argument("--json", action="store_true", help="Output as JSON")

    inbox = sub.add_parser(
        "inbox",
        help="Drop-in inbox: drop a file, link, or note — Devon acts",
        description=("nm inbox list [--status pending] [--room slug]\n"
                     "nm inbox show <id>\n"
                     "nm inbox retry <id>\n"
                     "nm inbox release <id>\n"
                     "nm inbox add-link <url> [--room slug] [--note ...]\n"
                     "nm inbox sweep"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inbox_sub = inbox.add_subparsers(dest="inbox_action", required=True)
    i_list = inbox_sub.add_parser("list", help="list inbox items")
    i_list.add_argument("--status", default="",
                        help="filter: pending|processing|done|needs_input|quarantined|failed")
    i_list.add_argument("--room", default="", help="filter by room slug")
    i_list.add_argument("--limit", type=int, default=20)
    i_list.add_argument("--json", action="store_true", help="Output as JSON")
    i_show = inbox_sub.add_parser("show", help="show one inbox item")
    i_show.add_argument("id", help="item id")
    i_show.add_argument("--json", action="store_true", help="Output as JSON")
    i_retry = inbox_sub.add_parser("retry", help="re-queue a failed/needs_input item")
    i_retry.add_argument("id", help="item id")
    i_retry.add_argument("--json", action="store_true", help="Output as JSON")
    i_release = inbox_sub.add_parser("release",
                                     help="release a quarantined item (explicit, logged)")
    i_release.add_argument("id", help="item id")
    i_release.add_argument("--json", action="store_true", help="Output as JSON")
    i_addlink = inbox_sub.add_parser("add-link", help="drop a link into the inbox")
    i_addlink.add_argument("url", help="http(s) URL")
    i_addlink.add_argument("--room", default="", help="room slug")
    i_addlink.add_argument("--note", default="", help="note attached to the link")
    i_addlink.add_argument("--json", action="store_true", help="Output as JSON")
    i_sweep = inbox_sub.add_parser("sweep", help="run one inbox sweep cycle now")
    i_sweep.add_argument("--json", action="store_true", help="Output as JSON")


    improve = sub.add_parser(
        "improve",
        help="Self-improvement engine: skill rewrites, lessons, canaries",
        description=("nm improve status\n"
                     "nm improve lessons [--query Q]\n"
                     "nm improve skill <name> --propose\n"
                     "nm improve rollback <skill> <hash>"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    improve_sub = improve.add_subparsers(dest="improve_action", required=True)
    imp_status = improve_sub.add_parser("status", help="loop state, recent edits, lesson stats")
    imp_status.add_argument("--json", action="store_true", help="Output as JSON")
    imp_lessons = improve_sub.add_parser("lessons", help="list lessons, optionally filtered by relevance")
    imp_lessons.add_argument("--query", default="", help="filter by relevance to Q")
    imp_lessons.add_argument("--limit", type=int, default=20)
    imp_lessons.add_argument("--json", action="store_true", help="Output as JSON")
    imp_skill = improve_sub.add_parser("skill", help="skill operations")
    imp_skill.add_argument("name", help="skill name")
    imp_skill.add_argument("--propose", action="store_true",
                           help="manually trigger a skill-edit proposal (gate only, no apply)")
    imp_skill.add_argument("--json", action="store_true", help="Output as JSON")
    imp_rollback = improve_sub.add_parser("rollback", help="restore a previous skill version")
    imp_rollback.add_argument("skill", help="skill name")
    imp_rollback.add_argument("hash", help="version hash to restore")
    imp_rollback.add_argument("--json", action="store_true", help="Output as JSON")

    trade = sub.add_parser(
        "trade",
        help="FinancialExpert trading brain (Sentinel.py engine)",
        description=("nm trade analyze BTC/USDT [--market crypto] [--timeframe 1h]\n"
                     "nm trade backtest XAUUSD --market forex [--profile aggressive]\n"
                     "nm trade signal ETH/USDT\n"
                     "nm trade compare BTC/USDT,ETH/USDT,SOL/USDT\n"
                     "nm trade strategies --market crypto\n"
                     "nm trade paper start BTC/USDT --capital 10000\n"
                     "nm trade paper status [--session ID]\n"
                     "nm trade paper stop [--session ID]\n"
                     "nm trade live unlock --confirm \"I understand\"\n"
                     "nm trade live order BTC/USDT --side buy --size 0.01\n"
                     "nm trade kill\n"
                     "nm trade doctor"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    trade_sub = trade.add_subparsers(dest="trade_action", required=True)
    t_analyze = trade_sub.add_parser("analyze", help="plain-language regime + strategy brief")
    t_analyze.add_argument("symbol", help="e.g. BTC/USDT, XAUUSD")
    t_analyze.add_argument("--market", default="crypto", help="crypto|forex|stocks")
    t_analyze.add_argument("--timeframe", default="1h", help="bar timeframe")
    t_analyze.add_argument("--bars", type=int, default=2000, help="bars to load")
    t_analyze.add_argument("--json", action="store_true", help="Output as JSON")
    t_backtest = trade_sub.add_parser("backtest", help="backtest + pass/fail verdict vs the bar")
    t_backtest.add_argument("symbol", help="e.g. XAUUSD")
    t_backtest.add_argument("--market", default="crypto", help="crypto|forex|stocks")
    t_backtest.add_argument("--profile", default="default",
                            help="default|aggressive|conservative")
    t_backtest.add_argument("--json", action="store_true", help="Output as JSON")
    t_signal = trade_sub.add_parser("signal", help="directional bias + invalidation")
    t_signal.add_argument("symbol", help="e.g. ETH/USDT")
    t_signal.add_argument("--market", default="crypto", help="crypto|forex|stocks")
    t_signal.add_argument("--json", action="store_true", help="Output as JSON")
    t_compare = trade_sub.add_parser("compare", help="side-by-side regime + strategy fit")
    t_compare.add_argument("symbols", help="comma-separated symbols")
    t_compare.add_argument("--market", default="crypto", help="crypto|forex|stocks")
    t_compare.add_argument("--json", action="store_true", help="Output as JSON")
    t_strat = trade_sub.add_parser("strategies", help="list Sentinel strategies")
    t_strat.add_argument("--market", default="crypto", help="crypto|forex|stocks")
    t_strat.add_argument("--limit", type=int, default=40)
    t_strat.add_argument("--json", action="store_true", help="Output as JSON")
    t_paper = trade_sub.add_parser("paper", help="paper-trading sessions (simulated)")
    t_paper_sub = t_paper.add_subparsers(dest="paper_action", required=True)
    t_pstart = t_paper_sub.add_parser("start", help="open a paper session")
    t_pstart.add_argument("symbol", help="e.g. BTC/USDT")
    t_pstart.add_argument("--market", default="crypto")
    t_pstart.add_argument("--capital", type=float, default=10000.0)
    t_pstart.add_argument("--json", action="store_true", help="Output as JSON")
    t_pstatus = t_paper_sub.add_parser("status", help="mark session to market")
    t_pstatus.add_argument("--session", default="", help="session id (default: latest open)")
    t_pstatus.add_argument("--symbol", default="", help="symbol of open session")
    t_pstatus.add_argument("--json", action="store_true", help="Output as JSON")
    t_pstop = t_paper_sub.add_parser("stop", help="close a paper session")
    t_pstop.add_argument("--session", default="", help="session id (default: latest open)")
    t_pstop.add_argument("--json", action="store_true", help="Output as JSON")
    t_live = trade_sub.add_parser("live", help="live-trading gates (inert by default)")
    t_live_sub = t_live.add_subparsers(dest="live_action", required=True)
    t_lunlock = t_live_sub.add_parser("unlock", help="unlock live for 24h")
    t_lunlock.add_argument("--confirm", default="",
                           help='must be exactly "I understand"')
    t_lunlock.add_argument("--json", action="store_true", help="Output as JSON")
    t_lorder = t_live_sub.add_parser("order", help="place ONE live market order")
    t_lorder.add_argument("symbol", help="e.g. BTC/USDT")
    t_lorder.add_argument("--side", required=True, help="buy|sell")
    t_lorder.add_argument("--size", type=float, required=True,
                          help="size in base units")
    t_lorder.add_argument("--exchange", default="binance")
    t_lorder.add_argument("--json", action="store_true", help="Output as JSON")
    t_kill = trade_sub.add_parser("kill", help="kill switch: cancel live, revoke unlocks")
    t_kill.add_argument("--json", action="store_true", help="Output as JSON")
    t_doctor = trade_sub.add_parser("doctor", help="check the Sentinel integration")
    t_doctor.add_argument("--json", action="store_true", help="Output as JSON")

    swarm = sub.add_parser(
        "swarm",
        help="Agent swarm: specialist roles, debate, fan-out/fan-in",
        description=("nm swarm run \"<goal>\" [--roles researcher,coder,critic] [--rounds N] [--fanout N]\n"
                     "nm swarm roles\n"
                     "nm swarm debate \"<work description>\""),
    )
    swarm_sub = swarm.add_subparsers(dest="swarm_action", required=True)
    s_run = swarm_sub.add_parser("run", help="run a goal through the full swarm")
    s_run.add_argument("goal", help="goal to execute")
    s_run.add_argument("--roles", default="researcher,coder,critic",
                       help="comma-separated roles for the plan")
    s_run.add_argument("--rounds", type=int, default=3,
                       help="max debate rounds")
    s_run.add_argument("--fanout", type=int, default=3,
                       help="research fan-out angles")
    s_run.add_argument("--json", action="store_true", help="Output as JSON")
    s_roles = swarm_sub.add_parser("roles", help="list registered roles")
    s_roles.add_argument("--json", action="store_true", help="Output as JSON")
    s_debate = swarm_sub.add_parser("debate", help="run a standalone debate")
    s_debate.add_argument("work", help="work description / artifact to debate")
    s_debate.add_argument("--rounds", type=int, default=3,
                          help="max debate rounds")
    s_debate.add_argument("--json", action="store_true", help="Output as JSON")


    # Additional subcommands
    book = sub.add_parser("book", help="AI-assisted book writing")
    book.add_argument("action", nargs="?", default="list",
                     choices=["list", "create", "run", "status", "build"],
                     help="Action to perform")
    book.add_argument("topic", nargs="?", default="", help="Book topic")
    book.add_argument("--chapters", type=int, default=5, help="Number of chapters")
    book.add_argument("--words", type=int, default=2000, help="Words per chapter")
    book.add_argument("--no-research", action="store_true", help="Skip research phase")
    book.add_argument("--slug", default="", help="Book slug")
    sub.add_parser("hub", help="Model hub operations")
    cipher = sub.add_parser("cipher",
                            help="nmc1 encryption: encrypt, decrypt, classic ciphers, hmac")
    cipher.add_argument("action", nargs="?", default="",
                        choices=["", "encrypt", "decrypt", "classic", "hmac",
                                 "formats", "vault_put", "vault_get",
                                 "vault_list", "vault_rm",
                                 "vault_export", "vault_import"])
    cipher.add_argument("data", nargs="?", default="",
                        help="text to encrypt / classic-cipher input")
    cipher.add_argument("secret", nargs="?", default="",
                        help="vault: secret value to store under <data> as name")
    cipher.add_argument("--blob", default="", help="nmc1:v1 blob to decrypt")
    cipher.add_argument("--passphrase", default="", help="KDF passphrase")
    cipher.add_argument("--key", default="", help="raw key (hex or text) instead of a passphrase")
    cipher.add_argument("--mode", default="", choices=["", "ctr", "cbc"],
                        help="aes mode (default ctr)")
    cipher.add_argument("--algorithm", default="caesar",
                        choices=["caesar", "vigenere", "atbash", "b64"],
                        help="classic: which cipher")
    cipher.add_argument("--shift", type=int, default=1, help="classic: caesar shift")
    cipher.add_argument("--keyword", default="", help="classic: vigenere keyword")
    cipher.add_argument("--decrypt", action="store_true",
                        help="classic: decrypt instead of encrypt")
    cipher.add_argument("--entry-pass", default="",
                        help="vault_export: passphrase of 'pass'-scheme entries")
    cipher.add_argument("--json", action="store_true", help="Output as JSON")
    osint_p = sub.add_parser("osint",
                             help="OSINT reports + the identity graph")
    osub = osint_p.add_subparsers(dest="subcommand")
    for _name, _help in (("report", "full read-only OSINT report for a target"),
                         ("domain", "domain intel"), ("ip", "IP intel"),
                         ("url", "URL intel"), ("email", "email intel")):
        _op = osub.add_parser(_name, help=_help)
        _op.add_argument("target")
    _op = osub.add_parser("campaign", help="run an automated investigation walk")
    _op.add_argument("seeds", nargs="*")
    _op = osub.add_parser("decoder", help="ingest a decoder report into the graph")
    _op.add_argument("report")
    _op.add_argument("--source", default="cli")
    for _name in ("stats", "clusters", "timeline", "clear"):
        osub.add_parser(_name, help=f"identity graph: {_name}")
    _op = osub.add_parser("node", help="identity graph: one entity's neighborhood")
    _op.add_argument("ref")
    _op = osub.add_parser("graph", help="identity graph: subcommand")
    _op.add_argument("action", nargs="?", default="stats")
    _op.add_argument("args", nargs="*")
    _op.add_argument("--source", default="cli")
    structure_p = sub.add_parser("structure", help="Structure an objective into a brief")
    structure_p.add_argument("objective", nargs="?", default="",
                             help="task text to structure into a brief")
    structure_p.add_argument("--for", dest="for_", default="mission",
                             help="shape the brief for: mission|goal|chat")
    structure_p.add_argument("--polish", action="store_true",
                             help="let the model rewrite the brief")
    structure_p.add_argument("--json", action="store_true", help="Output as JSON")
    arena_parser = sub.add_parser("arena", help="Self-improvement arena")
    arena_sub = arena_parser.add_subparsers(dest="arena_command")
    arena_sub.add_parser("status", help="Show arena status")
    arena_sub.add_parser("approve", help="Approve arena actions")
    trial_parser = sub.add_parser("trial", help="Single-account trial flow")
    trial_sub = trial_parser.add_subparsers(dest="trial_command")
    trial_sub.add_parser("save", help="Save trial data")
    trial_sub.add_parser("list", help="List trial data")
    train = sub.add_parser("train", help="model training: backends, runs")
    train.add_argument("--backends", action="store_true",
                       help="list training backends with honest machine availability")
    train.add_argument("--run", action="store_true",
                       help="kick off one pipeline run (collect→train→evaluate→promote)")
    train.add_argument("--backend", default="",
                       help="override the training backend for this run")
    train.add_argument("--base-model", default="",
                       help="HF base model id (required for external backends)")
    train.add_argument("--json", action="store_true", help="Output as JSON")
    help_p = sub.add_parser("help", help="Show help")
    help_p.add_argument("topic", nargs="?", default="",
                        help="command or topic page (e.g. code, budget)")
    help_p.add_argument("--json", action="store_true", help="Output as JSON")
    cookies_p = sub.add_parser("cookies", help="Cookie lab: parse, classify, ingest")
    cookies_p.add_argument("text", nargs="*", default=[],
                           help="cookie/header text, or 'ingest <text>'")
    cookies_p.add_argument("--source", default="cookies",
                           help="source label for ingest")
    cookies_p.add_argument("--json", action="store_true", help="Output as JSON")
    sub.add_parser("reason", help="Reasoning engine")
    sub.add_parser("workspace", help="Workspace management")
    monitor = sub.add_parser("monitor", help="watch files/URLs for changes")
    monitor.add_argument("action", nargs="?", default="status",
                         choices=["add", "list", "tick", "status", "remove",
                                  "enable", "disable", "alert", "webhook_test",
                                  "webhook-test"])
    monitor.add_argument("ref", nargs="?", default="",
                         help="target or ref (add/remove/enable/disable)")
    monitor.add_argument("--interval", type=float, default=300.0,
                         help="seconds between checks (floor 30)")
    monitor.add_argument("--webhook", default="", help="URL to POST alerts to")
    monitor.add_argument("--secret", default="",
                         help="HMAC secret for webhook signatures")
    monitor.add_argument("--min-gap", type=int, default=None,
                         help="minimum seconds between alerts for this watch")
    monitor.add_argument("--watch", default="content", choices=["content", "size"])
    monitor.add_argument("--json", action="store_true", help="Output as JSON")
    crack = sub.add_parser("crack", help="Offline hash cracking (md5/sha1/sha256/…)")
    crack.add_argument("hashes", nargs="*", help="digest(s) to attack")
    crack.add_argument("--hash", default="", metavar="DIGEST",
                       help="one more digest (script-friendly)")
    crack.add_argument("--algo", default="",
                       choices=["", "md5", "sha1", "sha256", "sha512", "ntlm"],
                       help="force one algorithm (default: detect)")
    crack.add_argument("--mode", default="",
                       choices=["", "hybrid", "dictionary", "brute", "known"],
                       help="attack mode (default hybrid)")
    crack.add_argument("--words", default="", metavar="FILE",
                       help="extra wordlist file")
    crack.add_argument("--json", action="store_true", help="Output as JSON")
    decode = sub.add_parser("decode",
                            help="decode/identify encodings, chains, hashes")
    decode.add_argument("text", nargs="?", default="",
                        help="data (or file:path) to decode")
    decode.add_argument("--hash", default="", metavar="DIGEST",
                        help="known-plaintext hash lookup")
    decode.add_argument("--mode", default="", choices=["", "decoders"],
                        help="'decoders' lists the decoder registry")
    decode.add_argument("--history", action="store_true",
                        help="list the most recent decode reports")
    decode.add_argument("--show", default="", metavar="ID",
                        help="print one stored decode report (JSON)")
    decode.add_argument("--json", action="store_true", help="Output as JSON")
    music = sub.add_parser("music", help="Music generation")
    music.add_argument("action", nargs="?", default="styles",
                       choices=["styles", "compose", "songs"],
                       help="styles | compose <topic> | songs [name]")
    music.add_argument("topic", nargs="?", default="",
                       help="song topic (compose) or lookup (songs)")
    music.add_argument("--style", default="pop")
    music.add_argument("--title", default="")
    music.add_argument("--key", default="")
    music.add_argument("--seed", default="0")
    music.add_argument("--json", action="store_true", help="Output as JSON")

    exec_cmd = sub.add_parser("exec", help="run code in the sandbox")
    exec_cmd.add_argument("code", nargs="?", default="",
                          help="source to run (or: languages)")
    exec_cmd.add_argument("--lang", default="", help="python|bash|node|… (auto by default)")
    exec_cmd.add_argument("--timeout", default="30")
    exec_cmd.add_argument("--file", default="", help="run a workspace file instead")
    exec_cmd.add_argument("--json", action="store_true", help="Output as JSON")

    apps = sub.add_parser("apps", help="build and manage local web apps")
    apps.add_argument("action", nargs="?", default="list",
                      choices=["build", "list", "info", "stacks", "serve",
                               "stop", "served"],
                      help="what to do")
    apps.add_argument("name", nargs="?", default="", help="app name")
    apps.add_argument("--stack", default="static")
    apps.add_argument("--features", default="", help="comma-separated feature list")
    apps.add_argument("--title", default="")
    apps.add_argument("--port", default="")
    apps.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Connector commands
    connectors = sub.add_parser("connectors", help="Manage external service connectors")
    connectors.add_argument("action", nargs="?", default="list",
                           choices=["list", "status", "connect", "disconnect"],
                           help="Action to perform")
    connectors.add_argument("--name", help="Connector name")
    connectors.add_argument("--provider", help="Provider (mono, plaid, etc.)")
    
    # Finance commands
    finance = sub.add_parser("finance", help="Bank account linking and transactions")
    finance.add_argument("action", nargs="?", default="status",
                        choices=["status", "link", "accounts", "transactions", "balance"],
                        help="Action to perform")
    finance.add_argument("--provider", default="auto", help="Provider (mono, plaid, auto)")
    finance.add_argument("--account-id", help="Account ID")
    finance.add_argument("--days", type=int, default=30, help="Transaction history days")
    
    # Native extensions command
    native = sub.add_parser("native", help="native kernels: status and build")
    native.add_argument("--build", action="store_true",
                        help="compile the C++ kernels when a compiler is present")
    native.add_argument("--benchmark", action="store_true",
                        help="time the vector-search kernel native vs pure Python and check agreement")
    
        # Virtual cards commands
    cards = sub.add_parser("cards", help="Virtual card management")
    cards.add_argument("action", nargs="?", default="list",
                      choices=["list", "create", "pause", "close", "status"],
                      help="Action to perform")
    cards.add_argument("--token", help="Card token")
    cards.add_argument("--type", default="UNLOCKED", help="Card type")
    cards.add_argument("--limit", type=int, help="Spend limit in cents")
    cards.add_argument("--merchant", help="Merchant name")
    
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    setup_logging(args.log_level, force=True)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI is the last line of defence
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    # bare Namespaces (tests, embedders) may not carry --json at all
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    else:
        print(text)


def _cmd_data(args: argparse.Namespace, context: Any) -> int:
    """Fine-tune data: what's free, what to base on, and the persona mix."""
    from pathlib import Path as _P

    from .training import finetune as ft
    from .training.free_datasets import catalog as data_catalog

    action = getattr(args, "action", "") or "catalog"
    data_dir = _P(context.settings.resolve(context.settings.training.data_dir))

    if action == "catalog":
        entries = data_catalog()
        print("free fine-tune datasets "
              "([v] = format verified against the live repo, [?] = unverified):")
        for e in entries:
            mark = "[v]" if e.get("verified") else "[?]"
            print(f"  {mark} {e.get('name', '?'):26s} {e.get('kind', ''):12s} "
                  f"{e.get('license', ''):14s} {e.get('size', ''):24s} "
                  f"{e.get('id', '')}")
        return 0

    if action == "models":
        print("Colab-safe base models (free-tier VRAM, permissive licenses):")
        for m in ft.COLAB_BASE_MODELS:
            why = str(m.get('why', '')).replace(chr(10), ' ')
            print(f"  {m['id']}")
            print(f"      {m.get('license', '')} - {why[:160]}")
        return 0

    if action == "mix":
        names = [s.strip() for s in (getattr(args, "sources", "") or "").split(",")
                 if s.strip()]
        if not names:
            print("usage: nm data mix <dataset,dataset,...> [--rows N] "
                  "[--persona-file PATH] [--base MODEL] [--out BASE] [--json]")
            print(f"hint: `nm data catalog` lists fetchable datasets; mix reads "
                  f"fetched files from {data_dir} as <name>.jsonl")
            return 2
        found: list[str] = []
        missing: list[str] = []
        for n in names:
            f = data_dir / f"{n}.jsonl"
            (found if f.is_file() else missing).append(str(f) if f.is_file() else n)
        if not found:
            print(f"no fetched files for: {', '.join(missing)} - the datasets must "
                  f"be fetched into {data_dir} first")
            return 1
        if missing:
            print(f"warning: no fetched files for {', '.join(missing)} - "
                  f"mixing without them")
        persona = getattr(args, "persona", "") or ""
        if not persona:
            persona = ft.load_persona(getattr(args, "persona_file", "") or "")
        out_base = _P(args.out) if getattr(args, "out", "") else data_dir / "persona-mix"
        sources = [ft.MixSource(name=_P(fpath).stem, path=fpath) for fpath in found]
        # A fetched source carries <name>.manifest.json (rows, HF id,
        # config): recipe-complete.  For those, a tiny --rows is treated as
        # "at least the Colab floor" — 100 rows is the smallest sample that
        # moves a QLoRA — clamped down to what the sources actually hold.
        # Hand-made files with no manifest take the number literally.
        requested = int(getattr(args, "rows", 0) or 0)
        metas: list[dict] = []
        for fpath in found:
            mf = _P(fpath).with_suffix(".manifest.json")
            try:
                metas.append(json.loads(mf.read_text(encoding="utf-8"))
                             if mf.is_file() else {})
            except Exception:  # noqa: BLE001 — a corrupt manifest = no manifest
                metas.append({})
        recipe = bool(metas) and all(m and m.get("rows") for m in metas)
        kwargs: dict = {}
        notebook_target = 0
        if recipe:
            available = sum(int(m.get("rows") or 0) for m in metas)
            floor = min(ft.NOTEBOOK_MIN_TARGET_ROWS, available)
            eff = max(requested or floor, floor)
            kwargs["target_rows"] = eff
            notebook_target = max(requested, ft.NOTEBOOK_MIN_TARGET_ROWS)
        elif requested:
            kwargs["target_rows"] = requested
            notebook_target = requested
        manifest = ft.build_persona_mix(sources, persona, out_base, **kwargs)
        colab = ft.write_colab_script(
            out_base,
            base_model=getattr(args, 'base', '') or ft.DEFAULT_COLAB_BASE,
            train_file=str(manifest["outputs"]["messages_jsonl"]))
        manifest["colab_script"] = colab
        nb_sources = [
            {"name": _P(fpath).stem,
             "id": str((m or {}).get("source") or ""),
             "config": str((m or {}).get("config") or ""),
             "rows": int((m or {}).get("rows") or 0)}
            for fpath, m in zip(found, metas)
        ]
        notebook = ft.write_colab_notebook(
            out_base,
            base_model=getattr(args, "base", "") or ft.DEFAULT_COLAB_BASE,
            persona=persona,
            sources=nb_sources,
            target_rows=notebook_target or ft.DEFAULT_TARGET_ROWS)
        manifest["colab_notebook"] = notebook
        per_src = ", ".join(f"{k}={v['rows']}" for k, v in manifest["per_source"].items())
        payload = manifest
        if getattr(args, 'json', False):
            print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
            return 0
        print(f"persona mix ready — {manifest['rows']} rows (target "
              f"{manifest['target_rows']}) from {len(sources)} source(s)")
        print(f"  {out_base.with_suffix('.jsonl')}")
        print(f"  filtered: {manifest['filtered']}  dupes dropped: "
              f"{manifest['deduped']}  per source: {per_src}")
        print(f"  colab script: {colab}")
        print(f"  colab nb:   {notebook}")
        return 0

    print(f"unknown data action: {action}")
    return 2


def _cmd_native(args, context) -> int:
    """Native hot-path status: which kernels run on C++, which fell back."""
    from . import native as native_mod

    if getattr(args, "build", False):
        ok, msg = native_mod.build(force=True)
        print(f"native build: {'OK' if ok else 'FAILED'} \u2014 {msg}")
    if getattr(args, "benchmark", False):
        bench = native_mod.benchmark()
        agreement = "agreement: True" if bench.get("match") else "agreement: False (top-10 index lists differ)"
        _emit(args, {"benchmark": bench},
              "\n".join([
                  f"benchmark: {bench['vectors']} vectors x {bench['dim']} dim, "
                  f"top-10, backend {bench.get('backend')}",
                  f"  python: {bench['python_ms']} ms | native: {bench['native_ms']} ms"
                  + (f" | speedup: {bench['speedup']}x" if bench.get('speedup') else ""),
                  agreement,
              ]))
        return 0
    info = native_mod.info()
    mlp = info.get("mlp") or {}
    search_backend = info.get("backend", "pure-python")
    train_backend = mlp.get("backend", "pure-python")
    _emit(args, info,
          f"native vector search: {search_backend}\n"
          f"native mlp training: {train_backend}\n"
          f"compiler: {info.get('compiler') or 'not found (pure-python fallback is fine)'}")
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    from .core.config import load_settings

    settings = load_settings(args.config)

    if args.command == "doctor":
        return _cmd_doctor(args, settings)
    if args.command == "config":
        _emit(args, settings.to_dict(), _render_config(settings))
        return 0
    if args.command == "setup":
        from .agents.context import build_context
        with build_context(settings) as context:
            return _cmd_setup(args, context)

    from .agents.context import build_context

    with build_context(settings) as context:
        if args.command == "models":
            return _cmd_models(args, context)
        if args.command == "data":
            return _cmd_data(args, context)
        if args.command == "tools":
            return _cmd_tools(args, context)
        if args.command == "memory":
            return _cmd_memory(args, context)
        if args.command == "run":
            return _cmd_run(args, context)
        if args.command == "ask":
            return _cmd_ask(args, context)
        if args.command == "backup":
            return _cmd_backup(args, context)
        if args.command == "missions":
            return _cmd_missions(args, context)
        if args.command == "tui":
            return _cmd_tui(args, context)
        if args.command == "serve":
            return _cmd_serve(args, context)
        if args.command == "queue":
            return _cmd_queue(args, context)
        if args.command == "commands":
            return _cmd_commands(args, context)
        if args.command == "zip":
            return _cmd_zip(args, context)

        if args.command == "book":
            return _cmd_stub(args, context, "book")
        if args.command == "hub":
            return _cmd_stub(args, context, "hub")
        if args.command == "cipher":
            return _cmd_cipher(args, context)
        if args.command == "osint":
            return _cmd_osint(args, context)
        if args.command == "structure":
            return _cmd_structure(args, context)
        if args.command == "arena":
            if getattr(args, "arena_command", None) == "status":
                return _cmd_stub(args, context, "arena status")
            elif getattr(args, "arena_command", None) == "approve":
                # approve requires arguments
                _emit(args, {"error": "approve requires arguments"}, "arena approve: missing required arguments")
                return 2
            return _cmd_stub(args, context, "arena")
        if args.command == "trial":
            if getattr(args, "trial_command", None) == "save":
                # save requires arguments
                _emit(args, {"error": "save requires arguments"}, "trial save: missing required arguments")
                return 2
            return _cmd_stub(args, context, "trial")
        if args.command == "train":
            return _cmd_train(args, context)
        if args.command == "help":
            return _cmd_help(args, context)
        if args.command == "cookies":
            return _cmd_cookies(args, context)
        if args.command == "reason":
            return _cmd_reason(args, context)
        if args.command == "workspace":
            return _cmd_workspace(args, context)
        if args.command == "monitor":
            return _cmd_monitor(args, context)
        if args.command == "crack":
            return _cmd_crack(args, context)
        if args.command == "decode":
            return _cmd_decode(args, context)
        if args.command == "music":
            return _cmd_music(args, context)
        if args.command == "exec":
            return _cmd_exec(args, context)
        if args.command == "apps":
            return _cmd_apps(args, context)
        if args.command == "connectors":
            return _cmd_connectors(args, context)
        if args.command == "finance":
            return _cmd_finance(args, context)
        if args.command == "improve":
            return _cmd_improve(args, context)
        if args.command == "trade":
            return _cmd_trade(args, context)
        if args.command == "swarm":
            return _cmd_swarm(args, context)
        if args.command == "native":
            return _cmd_native(args, context)
        if args.command == "cards":
            return _cmd_cards(args, context)
        if args.command == "autonomy":
            return _cmd_autonomy(args, context)
        if args.command == "goal":
            return _cmd_goal(args, context)
        if args.command == "skill":
            return _cmd_stub(args, context, "skill")
        if args.command == "project":
            return _cmd_project(args, context)
        if args.command == "mission":
            return _cmd_mission(args, context)
        if args.command == "kg":
            return _cmd_kg(args, context)
        if args.command == "simulate":
            return _cmd_stub(args, context, "simulate")
        if args.command == "code":
            return _cmd_code(args, context)
        if args.command == "media":
            return _cmd_media(args, context)
        if args.command == "inbox":
            return _cmd_inbox(args, context)
    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


# ── commands ───────────────────────────────────────────────────────────────────


def _cmd_doctor(args: argparse.Namespace, settings: Any) -> int:
    report = feature_report()
    from .storage.db import Database

    db = Database(settings.db_path)
    summary = db.migrate()
    health = {
        "version": __version__,
        "profile": settings.profile,
        "database": {
            "path": str(settings.db_path),
            "schema_version": summary.version,
            "tables": len(db.tables()),
            "integrity": db.integrity_check(),
        },
        "cpu_count": __import__("os").cpu_count(),
    }
    db.close()
    payload = {**health, **report}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"NoMorals Core {__version__}  profile={settings.profile}")
    print(f"database: {health['database']['path']}")
    print(f"  schema v{summary.version}, {health['database']['tables']} tables, "
          f"integrity {health['database']['integrity']}")
    print()
    print(report_as_text(report))
    return 0


def _render_config(settings: Any) -> str:
    data = settings.to_dict()
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key} = {sub_value!r}")
        else:
            lines.append(f"{key} = {value!r}")
    return "\n".join(lines)


def _env_update_home(context: Any, pairs: dict[str, str]) -> "Path":
    """Set KEY=VALUE lines in the home ``.env`` (replace-or-append, one line
    per key). The durable contract the next process reads on boot."""
    import os
    from pathlib import Path

    home = Path(os.path.expanduser(os.environ.get("NM_HOME")
                                    or getattr(context.settings, "home",
                                               "~/.nomorals")))
    home.mkdir(parents=True, exist_ok=True)
    env_path = home / ".env"
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    remaining = dict(pairs)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"
    for key, value in remaining.items():
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def _cmd_models(args: argparse.Namespace, context: Any) -> int:
    from .llm.registry import ModelRegistry, search_catalog

    registry = ModelRegistry(context.db)
    if getattr(args, "promote_local", ""):
        path = str(args.promote_local).strip()
        lora = str(getattr(args, "lora", "") or "").strip()
        update = {
            "NM_LLM_PROVIDER": "llama_cpp",
            "NM_LLM_LOCAL_MODEL": path,
            "NM_LLM_LOCAL_AUTO_START": "1",
        }
        if lora:
            update["NM_LLM_LOCAL_LORA"] = lora
        _env_update_home(context, update)
        try:
            context.settings.llm.provider = "llama_cpp"
            context.settings.llm.local_model = path
            if lora:
                context.settings.llm.local_lora = lora
        except Exception:  # noqa: BLE001 — the file is the durable truth
            pass
        payload = {"ok": True, "local_model": path, "lora": lora,
                   "provider": "llama_cpp"}
        lora_line = f"  lora on top: {lora}\n" if lora else ""
        _emit(args, payload,
              f"local model promoted: {path}\n" + lora_line +
              "  provider -> llama_cpp, auto-start on — written to the home "
              ".env; the next boot runs entirely on it")
        return 0
    if args.activate:
        record = registry.activate(args.activate)
        _emit(args, {"activated": record.name}, f"activated {record.name}")
        return 0
    if args.catalog or args.search or args.kind:
        entries = search_catalog(
            args.search,
            kind=args.kind,
            max_params=args.max_params or None,
        )
        payload = [
            {
                "repo_id": e.repo_id, "family": e.family, "kind": e.kind,
                "params": e.params, "context": e.context_length,
                "size_gb": e.size_hint_gb, "license": e.license, "notes": e.notes,
            }
            for e in entries
        ]
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        for entry in entries:
            print(f"{entry.repo_id}")
            print(f"    {entry.kind:<9} {entry.params/1e9:>5.1f}B  ctx {entry.context_length:>6}  "
                  f"~{entry.size_hint_gb}GB  {entry.license}")
            print(f"    {entry.notes}")
        return 0
    stats = registry.stats()
    rows = registry.list(limit=50)
    if args.json:
        print(json.dumps({"stats": stats, "models": [r.__dict__ for r in rows]}, indent=2, default=str))
        return 0
    print(f"active: {stats['active'] or '(none)'}   registered: {stats['total']}   "
          f"finetunes: {stats['finetunes']}")
    for record in rows:
        mark = "*" if record.active else " "
        print(f" {mark} {record.name:<52} {record.kind:<10} {record.source}")
    return 0


def _cmd_tools(args: argparse.Namespace, context: Any) -> int:
    tools = context.tools.register_builtins()
    if args.schema:
        print(json.dumps(tools.schemas(), indent=2))
        return 0
    for schema in tools.schemas():
        params = ", ".join(schema["parameters"])
        print(f"{schema['name']:<20} [{schema['capability'] or '-'}]  {schema['description']}")
        if params:
            print(f"{'':<20} ({params})")
    return 0


def _cmd_memory(args: argparse.Namespace, context: Any) -> int:
    memory = context.memory
    if args.remember:
        record_id = memory.remember(args.remember, source="cli")
        _emit(args, {"id": record_id}, f"remembered as {record_id}")
        return 0
    if args.consolidate:
        report = memory.consolidate()
        _emit(args, report, f"consolidated: {report}")
        return 0
    if args.stats or not args.query:
        stats = memory.stats_snapshot()
        _emit(args, stats, json.dumps(stats, indent=2, default=str))
        return 0
    result = memory.recall(args.query, limit=args.limit)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0
    if not result.records:
        print("no matches")
        return 0
    for record in result.records:
        print(f"{record.score:.3f} [{record.kind}] {record.content[:160]}")
    return 0


def _cmd_code(args: argparse.Namespace, context: Any) -> int:
    """Route `nm code` to run / review / test."""
    words = list(args.task or [])
    if words and words[0] == "review":
        ref = words[1] if len(words) > 1 else None
        return _cmd_code_review(args, context, ref)
    if words and words[0] == "test":
        return _cmd_code_test(args, context)
    task = " ".join(words).strip()
    if not task:
        print('usage: nm code "<task>" [--file F] [--accept CMD] [--max-iters N] [--root DIR]',
              file=sys.stderr)
        print('       nm code review [ref] [--root DIR]', file=sys.stderr)
        print('       nm code test [--changed] [--root DIR]', file=sys.stderr)
        return 2
    return _cmd_code_run(args, context, task)


def _cmd_code_run(args: argparse.Namespace, context: Any, task: str) -> int:
    """Run a coding task directly through the CodingAgent (no orchestrator)."""
    from .agents.coding import CodingAgent

    root = str(Path(args.root).expanduser().resolve())
    agent = CodingAgent(context, root=root)
    result = agent.run(
        task,
        filename=args.file,
        accept=args.accept,
        max_iterations=args.max_iters,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    print(f"task: {task}")
    print(f"file: {args.file}  iterations: {result.iterations}  "
          f"seconds: {result.seconds:.1f}")
    if result.ok:
        print("OK — acceptance command passed")
    else:
        print(f"FAILED: {result.error}")
    # Always end the session with the diff on screen; committing is a
    # separate, explicit `nm` step — never automatic.
    from .tools.git import git_diff
    try:
        diff = git_diff(None, [args.file], root)
        if diff["diff"].strip():
            print("\n--- diff ---")
            print(diff["diff"] if not diff["truncated"]
                  else diff["diff"] + "\n... [truncated]")
        else:
            print("\n(no changes)")
    except Exception as exc:  # noqa: BLE001 — diff is best-effort
        print(f"\n(could not render diff: {exc})")
    return 0 if result.ok else 1


def _critic_verdict(context: Any, diff_text: str) -> tuple[str, list[str]]:
    """Run the harsh reviewer over a diff.  Never raises — a failed review
    means 'no verdict', which is the conservative outcome."""
    if not diff_text.strip():
        return "no changes — nothing to review", []
    try:
        from .agents.coding import CodingAgent, _DIFF_REVIEW_FOCUS

        flaws = CodingAgent(context)._review_flaws(
            "review the working tree diff", diff_text,
            focus=_DIFF_REVIEW_FOCUS)
    except Exception:  # noqa: BLE001 — review must never break the CLI
        return "critic unavailable", []
    if flaws:
        return "VERDICT: FAIL — do not commit as-is", flaws
    return "VERDICT: PASS", []


def _cmd_code_review(args: argparse.Namespace, context: Any, ref: str | None) -> int:
    """Render a readable diff of the working tree (or ref) for review.

    Phase C: the diff is ALWAYS shown with the critic's verdict before
    any commit is proposed, and the owner gets an explicit commit prompt.
    """
    from .tools.git import git_diff, git_status

    root = str(Path(args.root).expanduser().resolve())
    try:
        status = git_status(root)
        diff = git_diff(ref, None, root)
    except Exception as exc:
        print(f"review failed: {exc}", file=sys.stderr)
        return 1
    body = diff["diff"]
    if args.json:
        verdict, flaws = _critic_verdict(context, body)
        print(json.dumps({"status": status, "diff": diff, "verdict": verdict,
                          "flaws": flaws}, indent=2, default=str))
        return 0
    print(f"repo: {root}  branch: {status['branch']}")
    dirty = status["staged"] + status["unstaged"] + status["untracked"]
    print(f"dirty files ({len(dirty)}): "
          + (", ".join(dirty[:20]) if dirty else "none"))
    if not body.strip():
        print("\n(no diff)")
        return 0
    print(f"\n--- diff{f' vs {ref}' if ref else ''} "
          f"({diff['bytes']} bytes{', truncated' if diff['truncated'] else ''}) ---")
    print(body if not diff["truncated"] else body + "\n... [truncated at 50KB]")
    # ── Phase C: critic verdict + explicit commit prompt ──
    verdict, flaws = _critic_verdict(context, body)
    print(f"\ncritic: {verdict}")
    for flaw in flaws:
        print(f"  - {flaw}")
    if dirty:
        print("\nCommit these changes?")
        print(f'  git -C "{root}" add -A && git -C "{root}" commit -m "<message>"')
        print("  (nothing is committed until you run it)")
    return 0


def _cmd_code_test(args: argparse.Namespace, context: Any) -> int:
    """Run the test suite through the pytest-aware runner (Phase B)."""
    from .tools.pytest_runner import format_test_result, run_tests

    root = str(Path(args.root).expanduser().resolve())
    result = run_tests(changed_only=bool(args.changed), repo=root)
    print(format_test_result(result))
    if result.get("failed"):
        return 1
    return 0 if result.get("ok") else 1


_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}


def _media_tools(context: Any) -> Any:
    return context.tools.register_builtins()


def _media_call(context: Any, name: str, **kwargs: Any) -> Any:
    """Call a media tool; return the value or raise a clean error."""
    outcome = _media_tools(context).call(name, actor="cli", **kwargs)
    if not outcome.ok:
        err = outcome.error
        msg = getattr(err, "message", None) or str(err)
        raise RuntimeError(msg)
    return outcome.value


def _cmd_media(args: argparse.Namespace, context: Any) -> int:
    """Route `nm media` to edit / probe / jobs / convert."""
    action = args.media_action
    if action == "edit":
        return _cmd_media_edit(args, context)
    if action == "probe":
        return _cmd_media_probe(args, context)
    if action == "jobs":
        return _cmd_media_jobs(args, context)
    if action == "convert":
        return _cmd_media_convert(args, context)
    print(f"unknown media action: {action}", file=sys.stderr)
    return 2


def _is_video_file(path: str, args: argparse.Namespace) -> bool:
    if getattr(args, "video", False):
        return True
    if getattr(args, "image", False):
        return False
    return Path(path).suffix.lower() in _VIDEO_EXTS


def _cmd_media_edit(args: argparse.Namespace, context: Any) -> int:
    as_json = getattr(args, "json", False)
    try:
        if _is_video_file(args.file, args):
            result = _media_call(context, "media_edit_video",
                                 video_path=args.file,
                                 instruction=args.instruction,
                                 dry_run=args.dry_run)
        else:
            result = _media_call(context, "media_edit",
                                 image_path=args.file,
                                 instruction=args.instruction,
                                 dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"media edit failed: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(result["plan"])
        return 0
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("job_id"):
        print(f"job {result['job_id']} queued: {result.get('summary', '')}")
        print(f"poll with: nm media jobs / media_job_status({result['job_id']})")
    else:
        print(f"wrote {result['output']}  ({result.get('summary', '')})")
        print(f"original untouched: {result['input']}")
    if result.get("job_id") and args.wait:
        return _cmd_media_wait(args, context, result["job_id"], as_json)
    return 0


def _cmd_media_wait(args: argparse.Namespace, context: Any,
                    job_id: str, as_json: bool) -> int:
    import time
    timeout = getattr(args, "timeout", 900.0)
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            info = _media_call(context, "media_job_status", job_id=job_id)
        except RuntimeError as exc:
            print(f"job poll failed: {exc}", file=sys.stderr)
            return 1
        status = info["status"]
        if status != last:
            prog = info.get("progress")
            extra = f" {prog:.0%}" if isinstance(prog, float) else ""
            print(f"job {job_id}: {status}{extra}")
            last = status
        if status in ("done", "failed"):
            if as_json:
                print(json.dumps(info, indent=2, default=str))
            elif status == "done":
                print(f"done → {info.get('output_ref')}")
            else:
                print(f"FAILED: {info.get('error')}", file=sys.stderr)
            return 0 if status == "done" else 1
        time.sleep(1.0)
    print(f"timed out after {timeout:.0f}s waiting for job {job_id}",
          file=sys.stderr)
    return 1


def _cmd_media_probe(args: argparse.Namespace, context: Any) -> int:
    try:
        info = _media_call(context, "media_edit_probe", path=args.file)
    except RuntimeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(info, indent=2, default=str))
        return 0
    if info.get("kind") == "video":
        print(f"{info['path']}: {info.get('width')}x{info.get('height')} "
              f"{info.get('video_codec')} {info.get('fps') or '?'}fps "
              f"{(info.get('duration') or 0):.1f}s "
              f"({info['bytes'] / 1e6:.1f}MB)")
    else:
        print(f"{info['path']}: {info.get('width')}x{info.get('height')} "
              f"{info.get('format')} {info.get('mode')} "
              f"({info['bytes'] / 1024:.0f}KB)")
    return 0


def _cmd_media_jobs(args: argparse.Namespace, context: Any) -> int:
    try:
        jobs = _media_call(context, "media_jobs", limit=args.limit)
    except RuntimeError as exc:
        print(f"jobs failed: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(jobs, indent=2, default=str))
        return 0
    if not jobs:
        print("no media jobs yet")
        return 0
    for j in jobs:
        prog = j.get("progress")
        extra = f" {prog:.0%}" if isinstance(prog, float) else ""
        print(f"{j['id'][:13]}  {j['status']}{extra}  {j['kind']}  "
              f"{j.get('label', '')}  → {j.get('output_ref') or '-'}")
    return 0


def _cmd_media_convert(args: argparse.Namespace, context: Any) -> int:
    as_json = getattr(args, "json", False)
    try:
        result = _media_call(context, "media_convert", path=args.file,
                             format=args.fmt)
    except RuntimeError as exc:
        print(f"convert failed: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("job_id"):
        print(f"job {result['job_id']} queued: {result.get('summary', '')}")
    else:
        print(f"wrote {result['output']}")
        print(f"original untouched: {result['input']}")
    if result.get("job_id") and args.wait:
        return _cmd_media_wait(args, context, result["job_id"], as_json)
    return 0


def _inbox_obj(context: Any) -> Any:
    """Build the drop-in Inbox for the CLI context's workspace."""
    from .workspace.inbox import Inbox

    root = Path(context.settings.workspace_dir)
    return Inbox(root, db=context.db)


def _cmd_improve(args: argparse.Namespace, context: Any) -> int:
    """Route `nm improve` to status / lessons / skill / rollback."""
    as_json = getattr(args, "json", False)
    action = args.improve_action
    try:
        if action == "status":
            from .agents.skill_evolution import SkillEvolutionLoop
            payload = SkillEvolutionLoop(context).status()
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                print(f"mode: {payload['mode']}")
                cands = payload.get("candidates", [])
                print(f"candidates: {len(cands)}")
                for c in cands[:10]:
                    print(f"  {c['skill']}: {c['count']} failures")
                print("recent edits:")
                for e in payload.get("recent_edits", [])[:10]:
                    print(f"  {e['id']} {e['skill_name']} [{e['status']}] "
                          f"(mode={e['mode']})")
                print(f"lessons: {payload.get('lessons', {})}")
            return 0
        if action == "lessons":
            from .agents.failure import FailureAnalyzer
            analyzer = FailureAnalyzer(context)
            if args.query:
                lessons = analyzer.rank(args.query, limit=args.limit)
            else:
                lessons = analyzer.recent(limit=args.limit)
            payload = [l.to_dict() for l in lessons]
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                if not payload:
                    print("no lessons yet")
                for l in payload:
                    print(f"{l['id']} [{l['category']}] "
                          f"usefulness={l['usefulness']} "
                          f"surfaced={l['times_surfaced']} "
                          f"{'DEMOTED ' if l['demoted'] else ''}"
                          f"{(l['prevention'] or l['lesson'])[:100]}")
            return 0
        if action == "skill":
            from .agents.skill_evolution import (
                SkillEvolutionLoop, SkillEvolutionError)
            loop = SkillEvolutionLoop(context)
            try:
                proposal = loop.propose(args.name)
            except SkillEvolutionError as exc:
                print(f"improve: {exc}", file=sys.stderr)
                return 1
            if args.propose:
                passed, results = loop.gate(proposal)
                payload = {
                    "skill_name": proposal["skill_name"],
                    "target": f"{proposal['target_kind']}:{proposal['target_ref']}",
                    "changed_lines": proposal["changed_lines"],
                    "before_hash": proposal["before_hash"],
                    "after_hash": proposal["after_hash"],
                    "fingerprint": proposal["fingerprint"],
                    "gate_passed": passed,
                    "gate": results,
                    "diff": proposal["diff"],
                }
                if as_json:
                    print(json.dumps(payload, indent=2, default=str))
                else:
                    print(f"proposal for {proposal['skill_name']}: "
                          f"{proposal['changed_lines']} changed lines, "
                          f"gate {'PASSED' if passed else 'FAILED'}")
                    for phase, r in results.items():
                        print(f"  {phase}: "
                              f"{'ok' if r.get('ok') else 'FAIL'} — "
                              f"{r.get('detail', '')[:120]}")
                    print("--- diff ---")
                    print(proposal["diff"][:3000])
                return 0
            print("nothing to do: pass --propose to draft a skill-edit "
                  "proposal", file=sys.stderr)
            return 2
        if action == "rollback":
            from .agents.skill_canary import CanaryRollout
            out = CanaryRollout(context).restore_version(args.skill,
                                                        args.hash)
            if as_json:
                print(json.dumps(out, indent=2, default=str))
            elif out.get("ok"):
                print(f"{args.skill} restored to version {args.hash}")
            else:
                print(f"improve: {out.get('error')}", file=sys.stderr)
                return 1
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"improve: {exc}", file=sys.stderr)
        return 1
    print(f"unknown improve action: {action}", file=sys.stderr)
    return 2


def _cmd_trade(args: argparse.Namespace, context: Any) -> int:
    """Route `nm trade` to the FinancialExpert / trading tool actions."""
    from .agents.financial_expert import FinancialExpert
    from .integrations import sentinel_bridge as bridge
    from .tools import trading as trading_tool

    as_json = getattr(args, "json", False)
    action = args.trade_action
    try:
        if action == "analyze":
            rep = FinancialExpert(context).analyze(
                args.symbol, market=args.market,
                timeframe=args.timeframe, bars=args.bars)
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0
        if action == "backtest":
            rep = FinancialExpert(context).backtest(
                args.symbol, market=args.market, profile=args.profile)
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0
        if action == "signal":
            rep = FinancialExpert(context).signal(args.symbol,
                                                  market=args.market)
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0
        if action == "compare":
            symbols = [s.strip() for s in args.symbols.split(",")
                       if s.strip()]
            rep = FinancialExpert(context).compare(symbols,
                                                   market=args.market)
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0
        if action == "strategies":
            names = bridge.list_strategies()[: args.limit]
            _emit(args, {"market": args.market, "count": len(names),
                         "strategies": names},
                  "\n".join(f"  {n}" for n in names) or "no strategies")
            return 0
        if action == "paper":
            paction = args.paper_action
            if paction == "start":
                out = trading_tool.paper_start(context, args.symbol,
                                               market=args.market,
                                               capital=args.capital)
                _emit(args, out,
                      f"paper session {out['session_id']} "
                      f"({'resumed' if out.get('resumed') else 'opened'}): "
                      f"{args.symbol} capital={out.get('capital')}")
                return 0
            if paction == "status":
                out = trading_tool.paper_status(
                    context, session_id=args.session, symbol=args.symbol)
                _emit(args, out,
                      f"{out['symbol']} equity={out['equity']:,.2f} "
                      f"pnl={out['pnl']:+,.2f} ({out['pnl_pct']:+.2%}) "
                      f"units={out['units']} orders={out['orders']} "
                      f"regime={out['last_action'].get('regime')}")
                return 0
            if paction == "stop":
                out = trading_tool.paper_stop(context,
                                              session_id=args.session)
                _emit(args, out,
                      f"paper session {out['session_id']} closed "
                      f"({out['symbol']})")
                return 0
        if action == "live":
            laction = args.live_action
            if laction == "unlock":
                out = trading_tool.live_unlock(context,
                                               confirm=args.confirm)
                _emit(args, out,
                      f"live unlocked for {out['expires_in_hours']}h "
                      f"(id {out['unlock_id']})")
                return 0
            if laction == "order":
                out = trading_tool.live_order(context, args.symbol,
                                              args.side, args.size,
                                              exchange=args.exchange)
                _emit(args, out,
                      f"LIVE {out['side']} {out['size']} {out['symbol']} "
                      f"order={out['order_id']}")
                return 0
        if action == "kill":
            out = trading_tool.kill_switch(context, "nm trade kill")
            _emit(args, out,
                  f"kill switch engaged: {out['reason']} "
                  f"(journal {out['journal_id']})")
            return 0
        if action == "doctor":
            rep = bridge.doctor()
            _emit(args, rep.to_dict(), rep.summary_text())
            return 0 if rep.ok else 1
    except bridge.LiveTradingDisabled as exc:
        print(f"trade: live trading disabled: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"trade: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown trade action: {action}", file=sys.stderr)
    return 2


def _cmd_swarm(args: argparse.Namespace, context: Any) -> int:
    """Route `nm swarm` to roles / run / debate."""
    from .agents.debate import Debate, Issue, WorkArtifact
    from .agents.fanout import fan_in, fan_out
    from .agents.role_specs import default_registry

    action = args.swarm_action
    try:
        if action == "roles":
            registry = default_registry()
            payload = {"roles": {
                name: registry.resolve(name).to_dict()
                for name in registry.names()}}
            lines = []
            for name in registry.names():
                spec = registry.resolve(name)
                lines.append(
                    f"- {name}: {spec.description}\n"
                    f"  read_only={spec.read_only} "
                    f"budget={spec.budget.wall_seconds:.0f}s/"
                    f"{spec.budget.tokens}tok\n"
                    f"  tools: {', '.join(spec.tool_allowlist)}")
            _emit(args, payload, "\n".join(lines))
            return 0
        if action == "debate":
            # Standalone debate: the model (when available) plays both
            # sides; without a model we report honestly instead of
            # fabricating a transcript.
            router = getattr(context, "router", None)
            if router is None:
                _emit(args, {"verdict": "no_model", "work": args.work},
                      "swarm debate: no model available; nothing debated.")
                return 2
            from .llm.base import Message, SamplingParams

            def coder_fn(brief, feedback, artifact):
                notes = ""
                if feedback:
                    notes = "\n".join(
                        f"- [{i.id}] {i.detail}" for i in feedback)
                resp = router.chat(
                    [Message.system(
                        "You are a coder. Produce the work, then revise it "
                        "against the critic's issues, quoting each issue id "
                        "you fix. Reply with JSON: "
                        '{"content": "...", "summary": "...", '
                        '"addresses": ["ISSUE-1", ...]}'),
                     Message.user(f"Work: {brief}\n\nCritic issues:\n{notes or '(none)'}")],
                    SamplingParams(temperature=0.4, max_tokens=2048,
                                   json_mode=True))
                return _work_from_json(resp.text, brief)

            def critic_fn(artifact, rubric):
                resp = router.chat(
                    [Message.system(
                        "You are an adversarial critic. Try to break the "
                        "work: counterexamples, edge cases, security holes. "
                        "Reply with JSON: {'verdict': "
                        "'approve|request_changes|reject', 'score': 0-100, "
                        "'issues': [{'id': 'ISSUE-n', 'severity': "
                        "'critical|major|minor', 'location': '...', "
                        "'detail': '...'}]}"),
                     Message.user(
                         f"Rubric: {', '.join(rubric)}\n\nWork:\n{artifact.content}")],
                    SamplingParams(temperature=0.3, max_tokens=2048,
                                   json_mode=True))
                return _critique_from_json(resp.text)

            debate = Debate(coder_fn=coder_fn, critic_fn=critic_fn,
                            max_rounds=args.rounds, context=context)
            result = debate.run(args.work)
            _emit(args, result.to_dict(),
                  f"swarm debate: {result.verdict} after {result.rounds} "
                  f"round(s) — {result.reason}")
            return 0 if result.approved else 1
        if action == "run":
            from .agents.blackboard import Blackboard
            from .agents.orchestrator import MasterOrchestrator, Plan, PlanStep

            roles = [r.strip() for r in args.roles.split(",") if r.strip()]
            registry = default_registry()
            board = Blackboard()
            angles = [f"angle {i + 1}: {r} perspective"
                      for i, r in enumerate(roles[:args.fanout])]
            fanout_res = fan_out(
                args.goal, angles or [f"research angle {i + 1}"
                                      for i in range(args.fanout)],
                role="researcher", blackboard=board, registry=registry,
                context=context)
            merged = fan_in(fanout_res.run_id, "concat_dedupe",
                            blackboard=board)
            steps = [PlanStep(name=f"research_{i}", goal=a, role="researcher")
                     for i, a in enumerate(fanout_res.angles)]
            steps.append(PlanStep(name="build", goal=args.goal, role="coder",
                                  depends_on=[s.name for s in steps]))
            steps.append(PlanStep(name="verify", goal=f"verify: {args.goal}",
                                  role="critic", depends_on=["build"]))
            plan = Plan(goal=args.goal, steps=steps,
                        rationale="swarm pipeline: fan-out research → "
                                  "coder builds → critic debates")
            orch = MasterOrchestrator(context, blackboard=board,
                                      roles=registry)
            result = orch.run(args.goal, plan=plan, reflect=False)
            payload = {"ok": result.ok, "answer": result.answer,
                       "research": merged.to_dict(),
                       "role_stats": orch.stats()["roles"]}
            _emit(args, payload,
                  f"swarm run: {'ok' if result.ok else 'failed'}\n"
                  f"{result.answer}")
            return 0 if result.ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"swarm: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown swarm action: {action}", file=sys.stderr)
    return 2


def _work_from_json(text: str, brief: str) -> "WorkArtifact":
    from .agents.debate import WorkArtifact

    import json as _json

    try:
        data = _json.loads(text)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return WorkArtifact(content=data.get("content", text[:2000]),
                        summary=str(data.get("summary", ""))[:500],
                        addresses=list(data.get("addresses", []) or []))


def _critique_from_json(text: str) -> "Critique":
    from .agents.debate import Critique, Issue

    import json as _json

    try:
        data = _json.loads(text)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    issues = []
    for i, raw in enumerate(data.get("issues", []) or []):
        if not isinstance(raw, dict):
            continue
        issues.append(Issue(
            id=str(raw.get("id", f"ISSUE-{i + 1}")),
            severity=str(raw.get("severity", "major")),
            location=str(raw.get("location", "")),
            detail=str(raw.get("detail", ""))[:1000]))
    return Critique(verdict=str(data.get("verdict", "request_changes")),
                    issues=issues,
                    score=float(data.get("score", 50) or 50))


def _cmd_inbox(args: argparse.Namespace, context: Any) -> int:
    """Route `nm inbox` to list / show / retry / release / add-link / sweep."""
    as_json = getattr(args, "json", False)
    try:
        inbox = _inbox_obj(context)
    except Exception as exc:  # noqa: BLE001
        print(f"inbox unavailable: {exc}", file=sys.stderr)
        return 1
    action = args.inbox_action
    try:
        if action == "list":
            items = inbox.list_items(
                status=args.status or None, room=args.room or None,
                limit=args.limit)
            payload = [i.to_dict() for i in items]
            if as_json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                if not payload:
                    print("inbox is empty")
                for i in payload:
                    print(f"{i['id']}  [{i['status']}] {i['kind']}: {i['name']}"
                          + (f"  → {i['intent']}" if i["intent"] else ""))
            return 0
        if action == "show":
            item = inbox.get_item(args.id).to_dict()
            if as_json:
                print(json.dumps(item, indent=2, default=str))
            else:
                for k, v in item.items():
                    print(f"{k}: {v}")
                hist = inbox.history(args.id, limit=10)
                if hist:
                    print("history:")
                    for h in hist:
                        print(f"  {h['intent']} → {h['outcome']}: "
                              f"{h['detail'][:100]}")
            return 0
        if action == "retry":
            item = inbox.retry(args.id)
            print(f"{item.id} re-queued (status={item.status})")
            return 0
        if action == "release":
            item = inbox.release(args.id)
            print(f"{item.id} released from quarantine (status={item.status})")
            return 0
        if action == "add-link":
            item = inbox.add_link(args.url, note=args.note,
                                  room=args.room or None)
            print(f"link queued: {item.id} → "
                  f"{'room ' + args.room if args.room else 'global inbox'}")
            return 0
        if action == "sweep":
            report = inbox.sweep()
            if as_json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print(f"swept: {report.get('pending_found', 0)} pending, "
                      f"{report.get('counts', {})}")
            return 0
    except (KeyError, ValueError) as exc:
        print(f"inbox: {exc}", file=sys.stderr)
        return 1
    print(f"unknown inbox action: {action}", file=sys.stderr)
    return 2


def _cmd_run(args: argparse.Namespace, context: Any) -> int:
    from .agents.orchestrator import MasterOrchestrator

    # Warn if using mock provider
    active = context.router.active_model
    is_mock = not active or active == "mock"
    if is_mock:
        print("⚠️  No model configured. Using offline mock provider (outputs will be empty).")
        print("   Run `nm models` to see available models, or `nm setup` for guided setup.\n")
    
    goal = " ".join(args.goal)
    orchestrator = MasterOrchestrator(context, max_steps=args.max_steps)
    result = orchestrator.run(goal, reflect=not args.no_reflect)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    print(f"plan ({len(result.plan.steps)} steps): {result.plan.rationale or 'n/a'}")
    for step in result.plan.steps:
        # Mark mock steps distinctly
        mock_tag = " [mock]" if is_mock else ""
        print(f"  - {step.name} [{step.role}/{step.kind.value}]{mock_tag} deps={step.depends_on}")
    print(f"\nreport: {result.report.to_dict()}")
    print(f"\n{result.answer}")
    if result.lessons:
        print("\nlessons:")
        for lesson in result.lessons:
            print(f"  - {lesson}")
    return 0 if result.ok else 1


def _cmd_ask(args: argparse.Namespace, context: Any) -> int:
    from .llm.base import Message, SamplingParams

    if args.model:
        context.router.set_active(args.model)
    
    # Warn if using mock provider
    active = context.router.active_model
    if not active or active == "mock":
        print("⚠️  No model configured. Using offline mock provider (outputs will be gibberish).")
        print("   Run `nm models` to see available models, or `nm setup` for guided setup.\n")
    
    messages = ([Message.system(args.system)] if args.system else []) + [
        Message.user(" ".join(args.prompt))
    ]
    response = context.router.chat(
        messages, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    )
    if args.json:
        print(json.dumps(response.to_dict(), indent=2, default=str))
    else:
        print(response.text or f"[no output: {response.error}]")
    return 0 if response.ok else 1


def _cmd_backup(args: argparse.Namespace, context: Any) -> int:
    from .storage.backup import BackupManager

    manager = BackupManager(
        context.db,
        context.settings.backup_dir,
        keep=context.settings.backup.keep,
        compress=context.settings.backup.compress,
        git_repo=context.settings.backup.git_repo,
    )
    if args.create:
        info = manager.create(label="cli")
        manager.rotate()
        _emit(args, info.to_dict(), f"created {info.name} ({info.size} bytes)")
        return 0
    if args.verify:
        problems = manager.verify()
        _emit(args, {"problems": problems}, "\n".join(problems) if problems else "backup verified clean")
        return 1 if problems else 0
    if args.restore:
        path = manager.restore(args.restore)
        _emit(args, {"restored": str(path)}, f"restored to {path}")
        return 0
    if args.push:
        result = manager.push_to_git()
        _emit(args, result, f"push: {result}")
        return 0 if result.get("pushed") else 1
    entries = manager.list()
    payload = [e.to_dict() for e in entries]
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if not entries:
        print("no backups")
        return 0
    for entry in entries:
        print(f"{entry.name}  {entry.size:>10} bytes  schema v{entry.schema_version}  {entry.label}")
    return 0


def _cmd_missions(args: argparse.Namespace, context: Any) -> int:
    from .missions import MissionRunner, MissionStore

    store = MissionStore(context.db)
    runner = MissionRunner(context, store=store)
    reflect = not args.no_reflect

    if args.start:
        result = runner.start(
            args.start,
            max_iterations=args.max_iterations,
            budget_wall=args.budget_wall,
            budget_tokens=args.budget_tokens,
            reflect=reflect,
        )
        _emit(args, result.to_dict(), _render_result(result))
        return 0 if result.ok else 1

    if args.resume:
        result = runner.resume(
            args.resume, max_iterations=args.max_iterations, reflect=reflect
        )
        _emit(args, result.to_dict(), _render_result(result))
        return 0 if result.ok else 1

    if args.resume_all:
        results = runner.resume_all(max_iterations=args.max_iterations)
        payload = [r.to_dict() for r in results]
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        elif not results:
            print("no interrupted missions")
        for result in results:
            print(_render_result(result))
        return 0

    if args.pause:
        try:
            mission = store.set_status(args.pause, "paused", "paused from the CLI")
        except (KeyError, ValueError) as exc:
            print(f"missions: {exc}", file=sys.stderr)
            return 2
        _emit(args, mission.to_dict(), f"paused {mission.id} — {mission.name}")
        return 0

    if args.cancel:
        try:
            mission = store.set_status(args.cancel, "cancelled", "cancelled from the CLI")
        except (KeyError, ValueError) as exc:
            print(f"missions: {exc}", file=sys.stderr)
            return 2
        _emit(args, mission.to_dict(), f"cancelled {mission.id} — {mission.name}")
        return 0

    if args.resume_status:
        try:
            mission = store.set_status(args.resume_status, "running", "")
        except (KeyError, ValueError) as exc:
            print(f"missions: {exc}", file=sys.stderr)
            return 2
        p = store.progress(mission.id)
        _emit(args, {"mission": mission.to_dict(), "progress": p},
              f"running {mission.id} — {mission.name}: "
              f"{p.get('steps_done', 0)}/{p.get('total_steps', 0)} steps "
              f"({p.get('percent', 0):.0f}%)")
        return 0

    if args.show:
        mission = store.get(args.show)
        history = store.checkpoint_history(mission.id, limit=10)
        payload = {
            "mission": mission.to_dict(),
            "checkpoints": [c.to_row() for c in history],
            "reflections": store.reflections(mission.id),
        }
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
            return 0
        prog = store.progress(mission.id)
        print(f"{mission.id}  [{mission.status}]")
        print(f"  goal:       {mission.goal}")
        print(f"  steps:      {prog['steps_done']}/{prog['total_steps']} "
              f"({prog['percent']:.0f}%) "
              + (f"— current: {prog['current_step']}" if prog["current_step"] else ""))
        print(f"  iterations: {mission.iterations}   success: {mission.success}")
        print(f"  spent:      {mission.spent_wall:.1f}s / {mission.spent_tokens} tokens")
        print(f"  completed:  {mission.state.get('completed_steps') or []}")
        print(f"  checkpoints: {[c.label for c in history]}")
        return 0

    rows = store.list(status=args.status, limit=50)
    payload = {"stats": store.stats(), "missions": [m.to_dict() for m in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    stats = store.stats()
    print(f"missions: {stats['total']} total, {stats['active']} active, "
          f"{stats['checkpoints']} checkpoints")
    for mission in rows:
        print(f" {mission.id}  [{mission.status:<9}] it={mission.iterations} "
              f"success={mission.success}  {mission.goal[:50]}")
    return 0


def _render_result(result: Any) -> str:
    lines = [
        f"mission {result.mission_id} -> {result.status} "
        f"(success={result.success}, {result.iterations} iterations, {result.seconds:.1f}s)"
    ]
    if result.resumed_from:
        lines.append(f"  resumed from checkpoint: {result.resumed_from}")
    for step in result.steps:
        mark = "ok " if step.ok else "ERR"
        lines.append(f"  [{mark}] {step.step} ({step.seconds:.2f}s) {step.detail[:80]}")
    if result.error:
        lines.append(f"  error: {result.error}")
    for lesson in result.lessons:
        lines.append(f"  lesson: {lesson}")
    return "\n".join(lines)


def _cmd_tui(args: argparse.Namespace, context: Any) -> int:
    """Interactive terminal UI. Commands are dispatched against the context."""
    from .tui import TuiState, run as run_tui
    from .tui.app import TuiApp

    state = TuiState(status=f"profile {context.settings.profile}")

    def on_submit(text: str) -> None:
        _dispatch_tui_command(context, state, text)

    app = TuiApp(context, state=state, on_submit=on_submit)
    import curses

    try:
        curses.wrapper(app.run)
    except curses.error as exc:
        print(f"could not start the TUI: {exc}", file=sys.stderr)
        return 1
    return 0


def _dispatch_tui_command(context: Any, state: Any, text: str) -> None:
    """Handle one line of TUI input. Slash-commands first, then chat."""
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        command = parts[0][1:].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if command in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if command == "help":
            state.say(
                "plain text chats with the active model\n"
                "/mem <text> remember   /recall <query> search memory\n"
                "/tools list tools   /models list models   /missions list missions\n"
                "/doctor environment   /clear clear   /quit exit",
                kind="info",
            )
        elif command == "tools":
            for name in context.tools.names():
                state.tool(name)
        elif command == "mem":
            state.say(f"remembered: {context.memory.remember(argument, source='tui')}", kind="tool")
        elif command == "recall":
            for record in context.memory.recall(argument, limit=5).records:
                state.say(f"{record.score:.3f} {record.content[:120]}", kind="assistant")
        elif command == "missions":
            from .missions import MissionStore

            for mission in MissionStore(context.db).list(limit=10):
                state.say(f"{mission.id}  [{mission.status}]  {mission.goal[:60]}", kind="info")
        elif command == "models":
            from .llm.registry import ModelRegistry

            for record in ModelRegistry(context.db).list(limit=10):
                mark = "*" if record.active else " "
                state.say(f" {mark} {record.name} ({record.kind})", kind="info")
        elif command == "doctor":
            state.say(f"profile={context.settings.profile} schema={context.db.scalar('SELECT COALESCE(MAX(version),0) FROM schema_migrations')}", kind="info")
        elif command == "clear":
            state.clear()
        else:
            state.error(f"unknown command /{command} — try /help")
        return

    from .llm.base import Message, SamplingParams

    response = context.router.chat(
        [Message.user(text)], SamplingParams(temperature=0.7, max_tokens=1024)
    )
    if response.ok:
        state.say(response.text, kind="assistant")
    else:
        state.error(response.error or "no response")


def _cmd_serve(args: argparse.Namespace, context: Any) -> int:
    from .api.server import serve

    return serve(
        context,
        host=context.settings.api.host,
        port=context.settings.api.port,
    )


def _cmd_queue(args: argparse.Namespace, context: Any) -> int:
    from .storage.queue import WorkQueue

    queue = WorkQueue(context.db)
    payload = {"pending": queue.pending(args.topic or None), "topics": queue.topics()}
    _emit(args, payload, json.dumps(payload, indent=2, default=str))
    return 0


# ── knowledge graph / cookies / structure CLI commands ─────────────────────


def _cmd_kg(args: argparse.Namespace, context: Any) -> int:
    """Knowledge graph maintenance from the operator's seat."""
    from .agents.kg import KnowledgeGraph

    g = KnowledgeGraph(context.db)
    action = getattr(args, "action", "stats") or "stats"
    try:
        limit = int(getattr(args, "limit", 10) or 10)
    except (TypeError, ValueError):
        limit = 10
    if action == "stats":
        st = g.stats()
        _emit(args, st, f"kg: {st['nodes']} nodes, {st['edges']} edges "
              f"({len(st.get('by_type', {}))} types)")
        return 0
    if action == "consolidate":
        out = g.consolidate()
        _emit(args, out, f"kg: consolidated — merged "
              f"{out.get('merged', 0)} duplicate node(s)")
        return 0
    if action == "communities":
        comms = g.communities(limit=limit)
        lines = [f"kg: {len(comms)} communities"]
        for i, c in enumerate(comms, 1):
            lines.append(f"  {i}. size {c.get('size', 0)} — "
                         f"{len(c.get('members') or [])} member ids, "
                         f"types {sorted((c.get('types') or {}).keys())}")
        _emit(args, comms, "\n".join(lines))
        return 0
    if action == "top":
        rows = g.top(n=limit)
        lines = [f"kg: top {len(rows)} hubs by degree"]
        lines.extend(f"  {r.get('label', '')} — {r.get('degree', 0)} "
                     f"({r.get('type', 'entity')})" for r in rows)
        _emit(args, rows, "\n".join(lines))
        return 0
    if action == "decay":
        out = g.decay()
        _emit(args, out, f"kg: decay — lowered confidence on "
              f"{out.get('updated', 0)} node(s); "
              f"{out.get('stale', 0)} now stale")
        return 0
    print(f"kg: unknown action {action}", file=sys.stderr)
    return 2


def _cmd_cookies(args: argparse.Namespace, context: Any) -> int:
    """Cookie lab CLI: parse/classify a Set-Cookie blob, or ingest it
    into the knowledge graph."""
    from .core.cookies import CookieLab

    lab = CookieLab()
    text = " ".join(getattr(args, "text", []) or []).strip()
    if not text:
        print('usage: nm cookies "<cookie text>"  |  '
              'nm cookies ingest "<cookie text>"', file=sys.stderr)
        return 2
    action = "parse"
    if text.lower().startswith(("ingest ", "ingest\n")):
        action = "ingest"
        text = text.split(None, 1)[1].strip()
    if action == "ingest":
        out = lab.ingest(context, text,
                        source=getattr(args, "source", "") or "cookies")
        _emit(args, out, f"cookies: ingested: {out.get('nodes', 0)} "
              f"graph node(s) — services: "
              f"{', '.join(out.get('services') or []) or 'none'}")
        return 0
    rep = lab.report(text)
    sec = rep.get("security", {})
    lines = [f"cookies: {rep['count']} parsed — "
             f"services: {', '.join(rep.get('services') or []) or 'none'}"]
    for c in rep["cookies"]:
        flags = ",".join(f if v in ("", True, None) else f"{f}={v}"
                         for f, v in (c.get("flags") or {}).items())
        via = f" [{c['decode_via']}]" if c.get("decode_via") else ""
        lines.append(f"  {c['name']} — {c.get('kind', 'unknown')} "
                     f"({c.get('service') or 'no service'})"
                     f"{(' flags: ' + flags) if flags else ''}{via}")
    lines.append(
        f"security: {len(sec.get('plaintext_auth') or [])} plaintext auth "
        f"cookie(s), {sec.get('with_httponly', 0)} HttpOnly, "
        f"{sec.get('with_secure', 0)} Secure")
    if sec.get("plaintext_auth"):
        lines.append("  plaintext: " + ", ".join(sec["plaintext_auth"]))
    _emit(args, rep, "\n".join(lines))
    return 0


def _cmd_structure(args: argparse.Namespace, context: Any) -> int:
    """Prompt-architect CLI: turn a raw objective into a structured brief."""
    from .agents.structuring import structure_text

    objective = (getattr(args, "objective", "") or "").strip()
    if not objective:
        print('usage: nm structure "<objective text>"', file=sys.stderr)
        return 2
    brief = structure_text(
        context, objective,
        for_=getattr(args, "for_", "mission") or "mission",
        polish=bool(getattr(args, "polish", False)))
    _emit(args, brief, brief.get("brief") or "")
    return 0


def _cmd_models_doctor(args: argparse.Namespace, context: Any) -> int:
    """Ping every provider on the router with a real round-trip.

    'Configured' is not 'working': each chat-capable provider gets a tiny
    live chat, vision floors get their health check. rc 0 only when a real
    model answers — a chain of dead providers or a misconfigured provider
    name exits 1 with the actionable line.
    """
    from .llm.base import Message, SamplingParams

    router = getattr(context, "router", None)
    if router is None:
        print("doctor: no model router — the context did not build", file=sys.stderr)
        return 1
    names = list(router.providers())
    rows: list[dict[str, Any]] = []
    chat_ok = False
    for name in names:
        provider = router.get(name)
        if provider is None:
            continue
        entry: dict[str, Any] = {"provider": name, "health": False,
                                 "chat": None, "error": ""}
        try:
            entry["health"] = bool(provider.health())
        except Exception as exc:  # noqa: BLE001 — a failing check is the answer
            entry["error"] = f"health: {type(exc).__name__}: {exc}"[:160]
        if "chat" in provider.capabilities:
            try:
                resp = provider.chat([Message.user("ping")],
                                     SamplingParams(max_tokens=8))
                entry["chat"] = bool(resp.ok)
                if not resp.ok:
                    entry["error"] = (resp.error or "chat failed")[:160]
            except Exception as exc:  # noqa: BLE001
                entry["chat"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"[:160]
            chat_ok = chat_ok or bool(entry["chat"])
        rows.append(entry)

    lines = ["models doctor:"]
    for e in rows:
        if e["chat"]:
            state = "chat ok"
        elif e["chat"] is None:
            state = "ready (no chat: capability-only)" if e["health"] \
                else f"unavailable — {e['error'] or 'health check failed'}"
        else:
            state = f"FAIL — {e['error'] or 'chat failed'}"
        lines.append(f"  {e['provider']}: {state}")

    llm_cfg = getattr(getattr(context, "settings", None), "llm", None)
    wanted = str(getattr(llm_cfg, "provider", "") or "")
    configured_ok = wanted in names
    if wanted and not configured_ok:
        lines.append(f"  configured provider {wanted!r} did not register — "
                     "check NM_LLM_PROVIDER (or run nm setup)")
    if not chat_ok:
        lines.append("  no provider answered a chat: NM_LLM_ALLOW_MOCK_FALLBACK=1 "
                     "boots a scripted mock for offline demos")
    rc = 0 if (chat_ok and configured_ok) else 1
    _emit(args, {"providers": rows, "ok": rc == 0,
                 "configured": wanted}, "\n".join(lines))
    return rc


def _cmd_setup(args: argparse.Namespace, context: Any) -> int:
    """Guided model setup wizard."""
    import os
    from pathlib import Path
    
    print("🧙 NoMorals AI - Guided Model Setup\n")
    print("This wizard will help you configure an LLM provider.\n")
    
    # Show current status
    active = context.router.active_model
    if active and active != "mock":
        print(f"Current active model: {active}")
        choice = input("Reconfigure? [y/N] ").strip().lower()
        if choice != "y":
            print("Keeping current configuration.")
            return 0
    else:
        print("No model configured (using mock provider).\n")
    
    # Provider selection
    print("Choose a provider:")
    print("  1. OpenRouter (free tier available, many models)")
    print("  2. Hugging Face (requires token, serverless inference)")
    print("  3. Local GGUF model (offline, requires download)")
    print("  4. OpenAI (requires paid API key)")
    print("  5. Skip (keep mock provider)\n")
    
    choice = input("Select [1-5]: ").strip()
    
    if choice == "5" or not choice:
        print("Skipping setup. You can run `nm setup` again later.")
        return 0
    
    env_file = Path.home() / ".nomorals/.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing .env
    env_vars = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    
    if choice == "1":
        # OpenRouter setup
        print("\n📡 OpenRouter Setup")
        print("Get your API key at: https://openrouter.ai/keys\n")
        api_key = input("Enter your OpenRouter API key: ").strip()
        
        if not api_key.startswith("sk-or-"):
            print("⚠️  Warning: OpenRouter keys should start with 'sk-or-'")
        
        env_vars["NM_LLM_PROVIDER"] = "openrouter"
        env_vars["NM_OPENAI_API_KEY"] = api_key
        env_vars["NM_OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        
        # Model selection
        print("\nRecommended free models:")
        print("  1. nvidia/nemotron-3-ultra-550b-a55b:free (1M context)")
        print("  2. openai/gpt-oss-120b:free (general + tool use)")
        print("  3. google/gemma-4-31b-it:free (vision)")
        print("  4. Enter custom model ID\n")
        
        model_choice = input("Select [1-4]: ").strip()
        model_map = {
            "1": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "2": "openai/gpt-oss-120b:free",
            "3": "google/gemma-4-31b-it:free",
        }
        
        if model_choice in model_map:
            model_id = model_map[model_choice]
        else:
            model_id = input("Enter model ID: ").strip()
        
        env_vars["NM_OPENAI_MODEL"] = model_id
        
    elif choice == "2":
        # Hugging Face setup
        print("\n🤗 Hugging Face Setup")
        print("Get your token at: https://huggingface.co/settings/tokens\n")
        token = input("Enter your HF token: ").strip()
        
        env_vars["HF_TOKEN"] = token
        env_vars["NM_LLM_PROVIDER"] = "hf_serverless"
        
        print("\nRecommended models:")
        print("  1. Sao10K/L3-8B-Stheno-v3.2")
        print("  2. Qwen/Qwen3-8B")
        print("  3. Enter custom model ID\n")
        
        model_choice = input("Select [1-3]: ").strip()
        model_map = {
            "1": "Sao10K/L3-8B-Stheno-v3.2",
            "2": "Qwen/Qwen3-8B",
        }
        
        if model_choice in model_map:
            model_id = model_map[model_choice]
        else:
            model_id = input("Enter model ID: ").strip()
        
        env_vars["NM_HF_MODEL"] = model_id
        
    elif choice == "3":
        # Local GGUF setup
        print("\n💻 Local GGUF Model Setup")
        print("You'll need to download a GGUF model file.\n")
        print("Recommended models:")
        print("  1. Phi-3.5-mini-instruct (3.8B, ~2.3GB Q4_K_M)")
        print("  2. Llama-3.2-3B-Instruct (3B, ~2GB Q4_K_M)")
        print("  3. Enter path to existing GGUF file\n")
        
        model_choice = input("Select [1-3]: ").strip()
        
        if model_choice == "3":
            model_path = input("Enter path to GGUF file: ").strip()
        else:
            print("\nTo download a model:")
            print("  1. Go to https://huggingface.co/models?search=gguf")
            print("  2. Download a Q4_K_M quantized model")
            print("  3. Place it in ~/.nomorals/models/")
            model_path = input("\nEnter path to downloaded GGUF file: ").strip()
        
        if not Path(model_path).exists():
            print(f"⚠️  File not found: {model_path}")
            return 1
        
        env_vars["NM_LLM_PROVIDER"] = "llama_cpp"
        env_vars["NM_LLAMA_CPP_MODEL"] = model_path
        
    elif choice == "4":
        # OpenAI setup
        print("\n🤖 OpenAI Setup")
        print("Get your API key at: https://platform.openai.com/api-keys\n")
        api_key = input("Enter your OpenAI API key: ").strip()
        
        if not api_key.startswith("sk-"):
            print("⚠️  Warning: OpenAI keys should start with 'sk-'")
        
        env_vars["NM_LLM_PROVIDER"] = "openai"
        env_vars["OPENAI_API_KEY"] = api_key
        env_vars["NM_OPENAI_MODEL"] = "gpt-4o-mini"
    
    # Write .env file
    with open(env_file, "w") as f:
        for key, value in sorted(env_vars.items()):
            f.write(f"{key}={value}\n")
    
    print(f"\n✅ Configuration written to {env_file}")
    print("\nTo activate, restart NoMorals AI or run:")
    print("  source ~/.nomorals/.env")
    
    # Offer to test
    print("\nTest the configuration now? [Y/n]")
    test_choice = input().strip().lower()
    
    if test_choice != "n":
        print("\n🧪 Testing configuration...")
        try:
            # Reload settings
            from .core.config import load_settings
            context.settings = load_settings()
            
            # Reinitialize router
            from .llm.router import LLMRouter
            context.router = LLMRouter(context.settings)
            
            # Test with a simple prompt
            from .llm.base import Message, SamplingParams
            response = context.router.chat(
                [Message.user("Say 'Hello, I'm working!' in one sentence.")],
                SamplingParams(max_tokens=50),
            )
            
            if response.ok and response.text:
                print(f"\n✅ Model responded: {response.text}")
                print(f"Active model: {context.router.active_model}")
            else:
                print(f"\n❌ Model failed: {response.error}")
                print("Check your API key and model configuration.")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            print("Check your configuration and try again.")
    
    return 0




def _cmd_reason(args, context):
    """Stub: reason command."""
    _emit(args, {"status": "ok"}, "Reasoning engine ready")
    return 0

def _cmd_workspace(args, context):
    """Stub: workspace command."""
    _emit(args, {"status": "ok"}, "Workspace ready")
    return 0

def _cmd_partner_ask(args, context) -> int:
    """One-shot partner question straight from the terminal.

    No gateway, no adapters, no platforms started: just her brain answering
    once. The exchange is journalled like any other DM so the memory and
    training pipelines see it too.
    """
    import time as _time

    from .core.ids import ulid_now
    from .llm.base import Message

    ask = (getattr(args, "ask", "") or "").strip()
    if not ask:
        print('usage: nm partner ask "<message>"', file=sys.stderr)
        return 2
    chat_key = getattr(args, "chat", "") or "local:console"
    try:
        from .partner.persona import default_persona

        persona = default_persona()
        override = getattr(getattr(context.settings, "partner", None),
                           "persona_name", "") or ""
        name = override or persona.name
        pronouns = persona.pronouns
    except Exception:  # noqa: BLE001 — the CLI works without the persona pack
        name, pronouns = "partner", "she/her"

    reply, model = "", ""
    router = getattr(context, "router", None)
    if router is not None:
        response = router.chat([
            Message.system(f"You are {name} ({pronouns}), answering your "
                           "person directly. Warm, brief, honest."),
            Message.user(ask),
        ])
        if not response.ok:
            print(f"partner: {response.error}", file=sys.stderr)
            return 1
        reply, model = response.text, response.model or ""

    db = context.db
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO conversations (id, title, agent, channel, "
                "created_at, updated_at) VALUES (?, ?, 'partner', 'local', "
                "0, ?) ON CONFLICT(id) DO UPDATE SET updated_at = "
                "excluded.updated_at",
                (chat_key, "you", _time.time()))
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, "
                "name, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ulid_now(), chat_key, "user", ask, "you", "", _time.time()))
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, "
                "name, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ulid_now(), chat_key, "assistant", reply, name, model,
                 _time.time()))
    except Exception as exc:  # noqa: BLE001 — journalling must not swallow the reply
        _log.warning("partner ask journal failed: %s", exc)

    _emit(args, {"reply": reply, "model": model, "chat": chat_key},
          f"{name} — {pronouns} [{model}]\n{reply}")
    return 0

def _cmd_train(args, context):
    """Training control: honest backend listing, or one real pipeline run."""
    import dataclasses

    from .training.backends import available_backends

    if getattr(args, "backends", False):
        rows = available_backends()
        configured = (context.settings.training.backend or "native").strip().lower()
        lines = [f"configured backend: {configured}"]
        for row in rows:
            mark = "ok   " if row["available"] else "miss "
            reason = "" if row["available"] else f" — {row['reason']}"
            lines.append(f"  {mark}{row['name']}{reason}")
        _emit(args, {"configured": configured, "backends": rows}, "\n".join(lines))
        return 0

    if not getattr(args, "run", False):
        _emit(args, {"started": False},
              "nothing to do — pass --backends to see engines or --run to train")
        return 0

    from .self_improvement import SelfImprovementJob

    job = SelfImprovementJob(context)
    result = job.run(
        force=True,
        backend=getattr(args, "backend", "") or "",
        base_model=getattr(args, "base_model", "") or "",
    )
    payload = dataclasses.asdict(result)
    if result.status in ("skipped",):
        _emit(args, payload, f"training skipped — {result.reason or 'policy'}")
        return 0
    if result.status in ("done", "completed", "succeeded"):
        _emit(args, payload,
              f"training {result.status} — run {result.run_id}"
              + (" PROMOTED" if result.promoted else " (gate did not promote)"))
        return 0
    _emit(args, payload, f"training {result.status} — {result.error or result.reason}")
    return 1

def _cmd_crack(args, context):
    """Offline hash cracking at the shell — one digest or a batch."""
    digests = [d for d in (list(getattr(args, "hashes", None) or [])
                           + [getattr(args, "hash", "") or ""]) if d]
    if not digests:
        print("usage: nm crack <digest> [digest…] [--algo md5|sha1|…] "
              "[--mode hybrid|dictionary|brute] [--words FILE]")
        return 0
    kw: dict = {}
    if getattr(args, "algo", ""):
        kw["algo"] = args.algo
    if getattr(args, "mode", ""):
        kw["mode"] = args.mode
    if getattr(args, "words", ""):
        kw["wordlist"] = args.words
    outcome = context.tools.call("hash_crack", target=",".join(digests), **kw)
    if not outcome.ok:
        print(f"crack failed: {getattr(outcome.error, 'message', outcome.error)}")
        return 1
    value = outcome.value or {}
    found = value.get("found") or {}
    _emit(args, value, "\n".join(f"{k} = {v}" for k, v in found.items())
          or f"no plaintext found for {len(digests)} digest(s)")
    return 0

def _cmd_decode(args, context):
    """Decode/identify: encodings, nested chains, known hashes (wave 71)."""
    from .core import decoder as D

    if getattr(args, "history", False):
        rows = D.decode_history(context.db, limit=15)
        if not rows:
            print("no decode history yet")
            return 0
        for row in rows:
            print(f"{row['id']}  {row.get('created_at', '')}  "
                  f"{row.get('source', '')}/{row.get('kind', '')}  "
                  f"{str(row.get('input_preview', row.get('input_text', '')))[:60]}")
        return 0
    if getattr(args, "show", ""):
        row = D.get_report(context.db, args.show)
        if not row:
            print(f"no report {args.show!r}")
            return 1
        print(row.get("report_json", ""))
        return 0
    if getattr(args, "mode", "") == "decoders":
        names = [d.name for d in D.DECODERS]
        _emit(args, {"count": len(names), "names": names},
              f"decoders ({len(names)}): " + ", ".join(names))
        return 0

    digest = getattr(args, "hash", "") or ""
    if digest:
        info = D.analyze(digest).hash or {}
        known = info.get("known")
        if known:
            _emit(args, info, f"known plaintext: {known.get('plaintext')}"
                  f"  (algorithm {known.get('algorithm')})")
        else:
            cands = ", ".join(info.get("algorithms") or []) or "no hash candidates"
            _emit(args, info, f"no known plaintext; candidates: {cands}")
        return 0

    text = getattr(args, "text", "") or ""
    if not text:
        print("usage: nm decode <text|file:path>  |  nm decode --hash <digest>  |  "
              "nm decode --mode decoders")
        return 2

    from .agents.decoder import DecoderAgent

    agent = DecoderAgent(context=context, name="cli")
    result = agent.run({"data": text, "explain": True})
    if not getattr(result, "ok", False):
        print(f"decode failed: {getattr(result, 'error', 'unknown')}")
        return 1
    out = dict(result.output)
    report = out.get("report") or {}
    best = report.get("best") or {}
    chain = "+".join(str(x) for x in (best.get("chain") or [])) \
        or str(best.get("decoder") or "?")
    best_out = str(best.get("output", ""))
    if len(best_out) > 300:
        best_out = best_out[:300] + " …"
    lines = [f"best: {chain} -> {best_out}"]
    for h in (report.get("hits") or [])[:6]:
        note = f"  {str(h.get('note'))[:70]}" if h.get("note") else ""
        lines.append(f"  {h.get('decoder')}: conf {h.get('confidence')}{note}")
    if out.get("explanation"):
        lines.append(f"explanation: {out['explanation']}")
    if out.get("saved_to"):
        lines.append(f"saved: {out['saved_to']}")
    _emit(args, out, "\n".join(lines))
    return 0

def _cmd_osint(args, context):
    """OSINT reports and the persistent identity graph, from the shell."""
    sub = getattr(args, "subcommand", "") or ""
    if sub in ("report", "domain", "ip", "url", "email"):
        tool = {"report": "osint_report", "domain": "osint_domain",
                "ip": "osint_ip", "url": "osint_url",
                "email": "osint_email"}[sub]
        outcome = context.tools.call(tool, target=args.target)
    elif sub == "campaign":
        outcome = context.tools.call("osint_campaign",
                                      seeds=",".join(args.seeds or []))
    elif sub == "decoder":
        from pathlib import Path as _P

        raw = _P(args.report).read_text()
        outcome = context.tools.call("osint_graph", action="ingest_decoder",
                                     report=raw, source=args.source)
    elif sub in ("stats", "clusters", "timeline", "clear", "node", "graph"):
        action = "stats" if sub == "graph" else sub
        extra: dict = {}
        if sub == "graph" and args.action:
            action = args.action
            if action == "node" and args.args:
                extra["node"] = args.args[0]
            if action == "merge" and len(args.args) >= 2:
                extra["a"], extra["b"] = args.args[0], args.args[1]
        if sub == "node":
            extra["node"] = args.ref
        outcome = context.tools.call("osint_graph", action=action, **extra)
    else:
        print("usage: nm osint report <target> | stats | clusters | node <ref>"
              " | decoder <report.json> [--source s] | campaign <seeds…>")
        return 0
    if not outcome.ok:
        print(f"error: {getattr(outcome.error, 'message', outcome.error)}")
        return 1
    import json as _json

    print(_json.dumps(outcome.value, indent=2, default=str)
          if getattr(args, "json", False) else str(outcome.value)[:3500])
    return 0


def _cmd_cipher(args, context):
    """nmc1 encryption at the shell: authenticated AES, classic ciphers, HMAC."""
    import base64 as _b64

    from .core import cipher as core
    from .tools.cipher import _key_bytes

    action = getattr(args, "action", "") or ""
    data = getattr(args, "data", "") or ""
    passphrase = getattr(args, "passphrase", "") or ""
    key_bytes = _key_bytes(getattr(args, "key", "") or "")

    if action in ("vault_export", "vault_import"):
        outcome = context.tools.call(
            "cipher", action=action, path=data, passphrase=passphrase,
            entry_pass=getattr(args, "entry_pass", "") or "")
        if not outcome.ok:
            print(f"error: {getattr(outcome.error, 'message', outcome.error)}")
            return 1
        result = outcome.value or {}
        if action == "vault_export":
            print(f"exported {result.get('count', '?')} entries to {data}")
        else:
            print(f"imported {result.get('count', 0)} entries from {data}")
        return 0

    if action in ("vault_put", "vault_get", "vault_list", "vault_rm"):
        # the sealed named-secrets vault lives in the cipher tool (agent
        # side); the CLI is a thin window onto it
        vault = getattr(args, "secret", "") or ""
        outcome = context.tools.call(
            "cipher", action=action, name=data, data=vault,
            passphrase=passphrase)
        if not outcome.ok:
            print(f"error: {getattr(outcome.error, 'message', outcome.error)}")
            return 1
        result = outcome.value or {}
        if action == "vault_put":
            print(f"stored {data} in the vault")
        elif action == "vault_get":
            print(result.get("data", ""))
        elif action == "vault_list":
            names = result.get("names", [])
            print("vault entries: " + (", ".join(names) if names else "none"))
        else:
            print(f"removed {data}" if result.get("removed", True)
                  else f"no vault entry {data}")
        return 0
    try:
        if action == "encrypt":
            if not data:
                print("usage: nm cipher encrypt <text> --passphrase <pw>")
                return 2
            if not passphrase and not key_bytes:
                print("error: encrypt needs --passphrase (or --key)")
                return 2
            print(core.aes_encrypt(
                data, passphrase=passphrase, key=key_bytes,
                mode=getattr(args, "mode", "") or "ctr"))
            return 0
        if action == "decrypt":
            blob = getattr(args, "blob", "") or data
            if not blob:
                print("usage: nm cipher decrypt --blob <nmc1:...> --passphrase <pw>")
                return 2
            plain = core.aes_decrypt(blob, passphrase=passphrase, key=key_bytes)
            try:
                print(plain.decode("utf-8"))
            except UnicodeDecodeError:
                print(f"<binary: {_b64.b64encode(plain).decode()}>")
            return 0
        if action == "classic":
            alg = (getattr(args, "algorithm", "") or "caesar").lower()
            do_dec = bool(getattr(args, "decrypt", False))
            if alg == "caesar":
                print(core.caesar(data, int(getattr(args, "shift", 1) or 1),
                                   decrypt=do_dec))
            elif alg == "vigenere":
                print(core.vigenere(data, getattr(args, "keyword", "") or "",
                                    decrypt=do_dec))
            elif alg == "atbash":
                print(core.atbash(data))
            elif alg == "b64":
                if do_dec:
                    print(core.b64_decode(data).decode("utf-8", "replace"))
                else:
                    print(core.b64_encode(data))
            else:
                print(f"unknown algorithm {alg!r} (caesar|vigenere|atbash|b64)")
                return 2
            return 0
        if action == "hmac":
            print(core.hmac_hex(getattr(args, "key", "") or "", data))
            return 0
        if action == "formats":
            _emit(args, {"algorithms": ["aes-ctr", "aes-cbc", "caesar",
                                        "vigenere", "atbash", "b64", "hmac-sha256"]},
                  "nmc1:v1 blobs - aes-ctr (default), aes-cbc; classic: caesar, "
                  "vigenere, atbash, b64; hmac-sha256")
            return 0
    except core.CipherError as exc:
        print(f"cipher error: {exc}")
        return 1
    print(f"unknown action: {action!r} (encrypt|decrypt|classic|hmac|formats)")
    return 2

def _cmd_monitor(args, context):
    """Watch files/URLs for changes — thin shell over agents.monitor."""
    from .agents.monitor import MonitorAgent

    agent = MonitorAgent(context)
    action = getattr(args, "action", "") or "status"
    ref = getattr(args, "ref", "") or ""

    if action == "add":
        if not ref:
            print("usage: nm monitor add <file-or-url> [--interval SECONDS] "
                  "[--webhook URL --secret S] [--watch content|size]")
            return 2
        try:
            _kw: dict = dict(
                interval=float(getattr(args, "interval", 300.0) or 300.0),
                webhook=getattr(args, "webhook", "") or "",
                secret=getattr(args, "secret", "") or "",
                watch=getattr(args, "watch", "") or "content")
            if getattr(args, "min_gap", None) is not None:
                _kw["min_gap"] = float(args.min_gap)
            row = agent.add(ref, **_kw)
        except ValueError as exc:
            print(f"monitor add failed: {exc}")
            return 1
        _emit(args, row,
              f"watching {ref} ({row.get('kind', 'file')}, "
              f"every {int(row.get('interval', 300) or 300)}s)")
        return 0
    if action == "alert":
        row = agent.set_alerting(
            ref,
            webhook=getattr(args, "webhook", None) or None,
            secret=getattr(args, "secret", None) or None,
            min_gap=(float(args.min_gap)
                     if getattr(args, "min_gap", None) is not None else None))
        if row is None:
            print(f"no such monitor: {ref}")
            return 1
        gap = int(row.get("min_alert_gap_s", 0) or 0)
        wh = row.get("webhook_url", "") or "off"
        _emit(args, {"monitor": row}, f"alerting updated: webhook={wh} gap={gap}s")
        return 0
    if action in ("webhook_test", "webhook-test"):
        res = agent.webhook_test(ref)
        if res is None:
            print(f"no such monitor (or no webhook set): {ref}")
            return 1
        ok = res.get("ok")
        _emit(args, res, f"webhook test: {'sent' if ok else 'failed'} "
                        f"({res.get('status', res.get('error', ''))})")
        return 0 if ok else 1
    if action == "tick":
        result = agent.tick()
        if isinstance(result, dict):
            text = "tick: " + (", ".join(f"{k}={v}" for k, v in result.items())
                               or "nothing due")
        else:
            text = f"tick: {result}"
        _emit(args, result if isinstance(result, dict) else {"result": result}, text)
        return 0
    if action == "list":
        rows = agent.list()
        if not rows:
            _emit(args, [], "no monitors - add one with `nm monitor add <file|url>`")
            return 0
        lines = []
        for r in rows:
            label = r.get("ref") or r.get("target") or ""
            lines.append(f" {label}  kind={r.get('kind', '')}  "
                         f"{'enabled' if r.get('enabled', True) else 'paused'}  "
                         f"every {int(r.get('interval', 300) or 300)}s")
        _emit(args, rows, "\n".join(lines))
        return 0
    if action == "status":
        st = agent.status()
        _emit(args, st,
              f"monitors: {st.get('enabled', 0)}/{st.get('total', 0)} enabled")
        return 0
    if action == "remove":
        if not ref:
            print("usage: nm monitor remove <ref>")
            return 2
        ok = bool(agent.remove(ref))
        print(f"removed {ref}" if ok else f"no monitor matching {ref}")
        return 0 if ok else 1
    if action in ("enable", "disable"):
        row = agent.set_enabled(ref, action == "enable")
        if row is None:
            print(f"no monitor matching {ref}")
            return 1
        _emit(args, row, f"{ref} {action}d")
        return 0
    print(f"unknown action: {action}")
    return 2

def _reply_path_report(settings_or_args, path: str = ""):
    """Return a reply path report for the given settings or args.
    
    Args:
        settings_or_args: Settings object or args namespace
        path: Optional path string
    
    Returns:
        A dict with provider and model status information
    """
    import os
    from nomorals.core.config import get_settings
    
    settings = settings_or_args
    if hasattr(settings_or_args, 'settings'):
        settings = settings_or_args.settings
    
    llm = getattr(settings, 'llm', None)
    if llm is None:
        llm = get_settings().llm
    
    provider = getattr(llm, 'provider', 'mock')
    local_model = getattr(llm, 'local_model', '')
    local_port = getattr(llm, 'local_port', 8080)
    
    # Check if model file is valid
    model_valid = bool(local_model) and os.path.exists(local_model)
    
    # Check local server state
    import socket
    server_state = "not running"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        result = s.connect_ex(('127.0.0.1', local_port))
        s.close()
        if result == 0:
            server_state = "running on port " + str(local_port)
    except Exception:
        pass
    
    # Build providers list
    providers = []
    if provider:
        providers.append({
            "name": provider,
            "responds": False
        })
    
    return {
        "active_provider": provider,
        "model_file": {
            "path": local_model,
            "valid": model_valid
        },
        "local_server": {
            "state": server_state,
            "port": local_port
        },
        "providers": providers
    }


def _configure_log_file(args: Any = None, settings: Any = None) -> None:
    """Configure file logging based on settings.
    
    Args:
        args: Namespace with log_level attribute (optional)
        settings: Settings object with log.file path (optional)
    """
    import logging
    from logging.handlers import RotatingFileHandler
    from pathlib import Path
    
    # Get log level from args or default
    level = "INFO"
    if args is not None:
        level = getattr(args, "log_level", "INFO")
    
    # Get log file path from settings
    log_file = ""
    if settings is not None:
        log_settings = getattr(settings, "log", None)
        if log_settings is not None:
            log_file = getattr(log_settings, "file", "")
    
    # If no log file configured, do nothing
    if not log_file:
        return
    
    # Resolve path against home if relative
    log_path = Path(log_file)
    if not log_path.is_absolute() and settings is not None:
        home = getattr(settings, "home", "~/.nomorals")
        log_path = Path(home).expanduser() / log_file
    
    # Create parent directories
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file handlers
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, RotatingFileHandler):
            root_logger.removeHandler(h)
    
    # Add new rotating file handler
    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    log_level = getattr(logging, level.upper(), logging.INFO)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root_logger.addHandler(handler)
    
    # Ensure root logger level allows messages through
    if root_logger.level > log_level or root_logger.level == logging.NOTSET:
        root_logger.setLevel(log_level)


def _cmd_commands(args: argparse.Namespace, context: Any) -> int:
    """List the command catalog — the same surface chat and the console share.

    An optional filter matches group names or command names (e.g. ``nm
    commands building`` → the building group only)."""
    from .social.chat.control import LIST_GROUPS, LIST_ONELINERS

    flt = (getattr(args, "filter", "") or "").strip().lower()
    groups: list[tuple[str, list[str]]] = []
    for name, kinds in LIST_GROUPS:
        if not flt:
            groups.append((name, list(kinds)))
            continue
        if flt in name.lower():
            groups.append((name, list(kinds)))
        else:
            hits = [k for k in kinds if flt in k
                    or flt in LIST_ONELINERS.get(k, "").lower()]
            if hits:
                groups.append((name, hits))
    payload = {"groups": [{"name": n, "commands": k} for n, k in groups]}
    lines = []
    for name, kinds in groups:
        lines.append(f"— {name} —")
        for k in kinds:
            lines.append(f"  /{k:<12} {LIST_ONELINERS.get(k, '')}")
    if not lines:
        lines = [f"no commands match {flt!r} — `nm commands` for the catalog"]
    _emit(args, payload, "\n".join(lines))
    return 0


def _cmd_zip(args: argparse.Namespace, context: Any) -> int:
    """Create and manage archives — delegated to the real Archivist system."""
    from .archives import Archivist

    a = Archivist(context)
    action = getattr(args, "action", "list") or "list"
    path = getattr(args, "path", "") or ""
    dest = getattr(args, "dest", "") or ""
    verbs = {"info", "list", "create", "extract", "compress", "digest"}
    if action not in verbs:
        # shorthand: `nm zip <file> --dest <archive>` == create
        action, path, dest = "create", action, dest

    try:
        if action == "info":
            out = a.info(path)
        elif action == "list":
            out = a.list(path)
        elif action == "create":
            if not dest:
                _emit(args, {"error": "--dest required"},
                      "usage: nm zip <path> --dest <archive.zip>")
                return 1
            out = a.create([path], dest)
            _emit(args, {"created": dest, **out},
                  f"created {dest} ({out.get('entries', 0)} entries, "
                  f"{out.get('bytes', 0)} bytes)")
            return 0
        elif action == "extract":
            out = a.extract(path)
            n = int(out.get("extracted") or 0)
            _emit(args, out, f"extracted {n} files to {out.get('dest', '')}")
            return 0
        elif action == "compress":
            out = a.compress(path)
        elif action == "digest" and path and Path(path).is_dir():
            out = a.digest_directory(path)
        else:
            out = a.digest(path)
    except Exception as exc:  # noqa: BLE001
        _emit(args, {"error": str(exc)}, f"zip: {exc}")
        return 1

    if action == "list":
        entries = out.get("entries") or []
        names = [e if isinstance(e, str) else str(e.get("name", ""))
                 for e in entries]
        _emit(args, out, "\n".join(names) if names
              else f"{out.get('format', 'archive')}: {out.get('path', path)} "
                   f"({out.get('count', len(names))} entries)")
    else:
        _emit(args, out, json.dumps(out, indent=2, default=str))
    return 0

def _cmd_connectors(args: argparse.Namespace, context: Any) -> int:
    """Manage external service connectors."""
    from .connectors.vault import CredentialVault
    
    action = getattr(args, "action", "list")
    name = getattr(args, "name", "")
    
    if action == "list":
        connectors = [
            {"name": "mono", "description": "Nigerian banks via Mono"},
            {"name": "plaid", "description": "US/EU banks via Plaid"},
            {"name": "privacy_cards", "description": "Virtual cards via Privacy.com"},
            {"name": "proxy_pool", "description": "Multi-source proxy pool"},
            {"name": "naija_commerce", "description": "Nigerian marketplaces (Jumia, Konga, Jiji)"},
        ]
        _emit(args, {"connectors": connectors, "count": len(connectors)},
              "\n".join(f"  {c['name']:20s} {c['description']}" for c in connectors))
    
    elif action == "status":
        if not name:
            _emit(args, {"error": "name required"}, "Usage: nm connectors status --name <connector>")
            return 1
        
        vault = CredentialVault()
        if name == "mono":
            from .connectors.finance import MonoConnector
            connector = MonoConnector(vault=vault)
        elif name == "plaid":
            from .connectors.finance import PlaidConnector
            connector = PlaidConnector(vault=vault)
        elif name == "privacy_cards":
            from .connectors.cards import PrivacyCardsConnector
            connector = PrivacyCardsConnector(vault=vault)
        elif name == "proxy_pool":
            from .connectors.proxies import ProxyPoolConnector
            connector = ProxyPoolConnector(vault=vault)
        elif name == "naija_commerce":
            from .connectors.commerce_ng import NaijaCommerceConnector
            connector = NaijaCommerceConnector(vault=vault)
        else:
            _emit(args, {"error": f"unknown connector: {name}"}, f"Unknown connector: {name}")
            return 1
        
        status = connector.status()
        _emit(args, status.to_dict(), f"Status: {'connected' if status.connected else 'disconnected'}")
    
    elif action == "connect":
        if not name:
            _emit(args, {"error": "name required"}, "Usage: nm connectors connect --name <connector>")
            return 1
        
        vault = CredentialVault()
        if name == "mono":
            from .connectors.finance import MonoConnector
            connector = MonoConnector(vault=vault)
        elif name == "plaid":
            from .connectors.finance import PlaidConnector
            connector = PlaidConnector(vault=vault)
        elif name == "privacy_cards":
            from .connectors.cards import PrivacyCardsConnector
            connector = PrivacyCardsConnector(vault=vault)
        else:
            _emit(args, {"error": f"connect not supported for: {name}"}, f"Connect not supported for: {name}")
            return 1
        
        url = connector.connect_url()
        if url:
            _emit(args, {"connect_url": url}, f"Open this URL to connect:\n{url}")
        else:
            _emit(args, {"error": "no connect URL available"}, "No connect URL available")
            return 1
    
    elif action == "disconnect":
        if not name:
            _emit(args, {"error": "name required"}, "Usage: nm connectors disconnect --name <connector>")
            return 1
        
        vault = CredentialVault()
        if name == "mono":
            from .connectors.finance import MonoConnector
            connector = MonoConnector(vault=vault)
        elif name == "plaid":
            from .connectors.finance import PlaidConnector
            connector = PlaidConnector(vault=vault)
        elif name == "privacy_cards":
            from .connectors.cards import PrivacyCardsConnector
            connector = PrivacyCardsConnector(vault=vault)
        elif name == "proxy_pool":
            from .connectors.proxies import ProxyPoolConnector
            connector = ProxyPoolConnector(vault=vault)
        elif name == "naija_commerce":
            from .connectors.commerce_ng import NaijaCommerceConnector
            connector = NaijaCommerceConnector(vault=vault)
        else:
            _emit(args, {"error": f"unknown connector: {name}"}, f"Unknown connector: {name}")
            return 1
        
        result = connector.disconnect()
        _emit(args, result, f"Disconnected: {result.get('disconnected', False)}")
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0


def _cmd_finance(args: argparse.Namespace, context: Any) -> int:
    """Bank account linking and transactions."""
    from .connectors.finance import Finance
    from .connectors.vault import CredentialVault
    
    action = getattr(args, "action", "status")
    provider = getattr(args, "provider", "auto")
    account_id = getattr(args, "account_id", "")
    days = getattr(args, "days", 30)
    
    vault = CredentialVault()
    finance = Finance(vault=vault)
    
    if action == "status":
        mono_status = finance.mono.status()
        plaid_status = finance.plaid.status()
        result = {
            "mono": mono_status.to_dict(),
            "plaid": plaid_status.to_dict(),
        }
        _emit(args, result, f"Mono: {'connected' if mono_status.connected else 'disconnected'}\n"
                           f"Plaid: {'connected' if plaid_status.connected else 'disconnected'}")
    
    elif action == "link":
        result = finance.link(provider)
        _emit(args, result, f"Connect URL: {result.get('connect_url', 'N/A')}")
    
    elif action == "accounts":
        accounts = finance.accounts()
        _emit(args, {"count": len(accounts), "accounts": accounts},
              f"Found {len(accounts)} account(s)")
    
    elif action == "transactions":
        if not account_id:
            _emit(args, {"error": "account_id required"}, "Usage: nm finance transactions --account-id <id>")
            return 1
        txs = finance.transactions(account_id, days)
        _emit(args, {"count": len(txs), "transactions": txs[:10]},
              f"Found {len(txs)} transaction(s)")
    
    elif action == "balance":
        if not account_id:
            _emit(args, {"error": "account_id required"}, "Usage: nm finance balance --account-id <id>")
            return 1
        accounts = finance.accounts()
        for acc in accounts:
            if acc["account_id"] == account_id:
                _emit(args, acc, f"Balance: {acc.get('balance_minor', 0) / 100:.2f} {acc.get('currency', 'NGN')}")
                return 0
        _emit(args, {"error": "account not found"}, f"Account {account_id} not found")
        return 1
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0


def _cmd_cards(args: argparse.Namespace, context: Any) -> int:
    """Virtual card management."""
    from .connectors.cards import PrivacyCardsConnector
    from .connectors.vault import CredentialVault
    
    action = getattr(args, "action", "list")
    token = getattr(args, "token", "")
    card_type = getattr(args, "type", "UNLOCKED")
    limit = getattr(args, "limit", 0)
    merchant = getattr(args, "merchant", "")
    
    vault = CredentialVault()
    connector = PrivacyCardsConnector(vault=vault)
    
    if action == "status":
        status = connector.status()
        _emit(args, status.to_dict(), f"Status: {'connected' if status.connected else 'disconnected'}")
    
    elif action == "list":
        cards = connector.list_cards()
        _emit(args, {"count": len(cards), "cards": [c.to_dict() for c in cards]},
              f"Found {len(cards)} card(s)")
    
    elif action == "create":
        if merchant:
            card = connector.create_for_purchase(merchant, limit or 10000)
        else:
            card = connector.create_card(card_type=card_type, spend_limit=limit)
        _emit(args, card.to_dict(), f"Created card: {card.token}")
    
    elif action == "pause":
        if not token:
            _emit(args, {"error": "token required"}, "Usage: nm cards pause --token <token>")
            return 1
        success = connector.pause_card(token)
        _emit(args, {"paused": success, "token": token}, f"Paused: {success}")
    
    elif action == "close":
        if not token:
            _emit(args, {"error": "token required"}, "Usage: nm cards close --token <token>")
            return 1
        success = connector.close_card(token)
        _emit(args, {"closed": success, "token": token}, f"Closed: {success}")
    
    else:
        _emit(args, {"error": f"unknown action: {action}"}, f"Unknown action: {action}")
        return 1
    
    return 0



def _cmd_autonomy(args: argparse.Namespace, context: Any) -> int:
    """Autonomy dial: status/on/off/tick/report/budget.

    ``on``/``off`` persist through the durable kv override (wave 51), so the
    next process — and the scheduler boot path — see the same dial state
    without touching config.toml.
    """
    from .agents.cognition import (CognitiveLoop, ModelBudget,
                                  autonomy_enabled, set_autonomy_enabled)

    action = getattr(args, "action", "status") or "status"
    if action == "status":
        enabled = autonomy_enabled(context)
        loop_status = CognitiveLoop(context).status()
        payload = {"enabled": enabled, **loop_status}
        _emit(args, payload,
              f"autonomy: {'on' if enabled else 'off'} "
              f"(interval {loop_status.get('interval_hours', 0)}h)")
        return 0
    if action in ("on", "enable"):
        set_autonomy_enabled(context, True)
        _emit(args, {"ok": True, "enabled": True, "persisted": True},
              "autonomy: enabled (persisted for the next boot)")
        return 0
    if action in ("off", "disable"):
        set_autonomy_enabled(context, False)
        _emit(args, {"ok": True, "enabled": False, "persisted": True},
              "autonomy: disabled (persisted for the next boot)")
        return 0
    if action == "tick":
        tick = CognitiveLoop(context).tick()
        stages = tick.get("stages", {})
        bits = ", ".join(
            f"{name}:{next(iter(st.keys()))}" if st else f"{name}:ok"
            for name, st in stages.items())
        _emit(args, tick,
              f"cognitive loop tick in {tick.get('seconds', 0)}s — {bits}")
        return 0
    if action == "report":
        report = CognitiveLoop(context).report()
        ticks = report.get("ticks", 0)
        lines = [f"cognitive loop report: {ticks} heartbeat(s), "
                 f"total {report.get('total_seconds', 0)}s "
                 f"(avg {report.get('avg_seconds', 0)}s)"]
        for name, counters in (report.get("stages") or {}).items():
            bits = " ".join(f"{k}={v}" for k, v in sorted(counters.items()))
            lines.append(f"{name}: {bits}")
        _emit(args, report, "\n".join(lines))
        return 0
    if action == "budget":
        cap_arg = str(getattr(args, "cap", "") or "")
        if getattr(args, "unlimited", False):
            _set_daily_call_cap(context, 0)
            _emit(args, {"ok": True, "cap": 0},
                  "model-call budget: set to 0 (unlimited) — persisted to the "
                  "home .env as NM_AUTONOMY_DAILY_MODEL_CALLS")
            return 0
        if cap_arg:
            try:
                n = int(cap_arg)
            except ValueError:
                print("autonomy budget: --cap needs an integer", file=sys.stderr)
                return 2
            if n < 0:
                print("autonomy budget: --cap must be >= 0 (0 = unlimited)",
                      file=sys.stderr)
                return 2
            _set_daily_call_cap(context, n)
            _emit(args, {"ok": True, "cap": n},
                  f"model-call budget: set to {n} calls/day — persisted to "
                  "the home .env as NM_AUTONOMY_DAILY_MODEL_CALLS")
            return 0
        budget = ModelBudget(context).report()
        if budget.get("unlimited"):
            text = (f"model budget: unlimited — used {budget.get('used', 0)} calls, "
                    f"{budget.get('tokens', 0)} tokens today, "
                    f"reserved {budget.get('reserved', 0)} in flight")
        else:
            text = (f"model budget: cap {budget.get('cap', 0)} calls/day — "
                    f"used {budget.get('used', 0)}, "
                    f"remaining {budget.get('remaining', 0)}, "
                    f"reserved {budget.get('reserved', 0)} "
                    f"(pressure {budget.get('pressure', 0.0):.0%}"
                    + (", throttled" if budget.get("throttled") else "")
                    + ") — lift with `nm autonomy budget --unlimited`")
        _emit(args, budget, text)
        return 0
    print(f"autonomy: unknown action {action}", file=sys.stderr)
    return 2


def _set_daily_call_cap(context: Any, calls: int) -> None:
    """Persist the daily model-call budget as NM_AUTONOMY_DAILY_MODEL_CALLS.

    Written to the home ``.env`` (the file users already maintain), replacing
    any existing line so one knob = exactly one line. Also applied to the
    in-memory settings so the current process agrees.
    """
    import os
    from pathlib import Path

    home = Path(os.path.expanduser(os.environ.get("NM_HOME")
                                    or getattr(context.settings, "home", "~/.nomorals")))
    key = "NM_AUTONOMY_DAILY_MODEL_CALLS"
    home.mkdir(parents=True, exist_ok=True)
    env_path = home / ".env"
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text().splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(key + "="):
            lines[i] = f"{key}={int(calls)}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={int(calls)}")
    env_path.write_text("\n".join(lines) + "\n")
    try:
        context.settings.autonomy.daily_model_calls = int(calls)
    except Exception:  # noqa: BLE001 — the file is the durable truth
        pass


def _cmd_music(args: argparse.Namespace, context: Any) -> int:
    """Compose songs / list styles through the music_writer tool."""
    tools = context.tools
    action = getattr(args, "action", "styles") or "styles"
    if action == "styles":
        out = tools.call("music_writer", action="styles")
        if not out.ok:
            print(f"music: {out.error}", file=sys.stderr)
            return 1
        styles = out.value.get("styles", {})
        _emit(args, out.value,
              "\n".join(f"  {k:<12} {v.get('label', k)} · {v.get('tempo', '')} "
                         f"bpm · {v.get('mode', '')}" for k, v in styles.items())
              or "no styles")
        return 0
    if action == "songs":
        out = tools.call("music_writer", action="song",
                         topic=getattr(args, "topic", "") or "")
        if not out.ok:
            print(f"music: {out.error}", file=sys.stderr)
            return 1
        _emit(args, out.value, json.dumps(out.value, indent=2, default=str))
        return 0
    topic = getattr(args, "topic", "") or ""
    if not topic:
        print("music compose needs a topic — nm music compose \"about what\"",
              file=sys.stderr)
        return 2
    out = tools.call("music_writer", action="compose", topic=topic,
                     style=getattr(args, "style", "pop") or "pop",
                     title=getattr(args, "title", "") or "",
                     key=getattr(args, "key", "") or "",
                     seed=int(getattr(args, "seed", "0") or 0))
    if not out.ok:
        print(f"music: {out.error}", file=sys.stderr)
        return 1
    song = out.value
    _emit(args, song,
          f"composed: {song.get('title', topic)} [{song.get('style', '')}]\n"
          f"  midi: {song.get('midi_path', '')}\n"
          f"  melody: {str(song.get('melody_description', ''))[:160]}")
    return 0


def _cmd_exec(args: argparse.Namespace, context: Any) -> int:
    """Run code through the sandbox (exec tool)."""
    tools = context.tools
    code = getattr(args, "code", "") or ""
    if code.strip().lower() in {"languages", "langs"}:
        from .execbox import CodeRunner

        langs = CodeRunner(context).languages()
        _emit(args, {k: v for k, v in langs.items()},
              "\n".join(f"  {k:<12} {'available' if getattr(v, 'available', True) else 'not installed'}"
                      for k, v in sorted(langs.items())))
        return 0
    if not code and not getattr(args, "file", ""):
        print("exec needs code — nm exec \"print(6*7)\" --lang python",
              file=sys.stderr)
        return 2
    out = tools.call("run_code", code=code,
                     language=getattr(args, "lang", "") or "",
                     timeout=float(getattr(args, "timeout", "30") or 30),
                     file=getattr(args, "file", "") or "")
    if not out.ok:
        print(f"exec: {out.error}", file=sys.stderr)
        return 1
    value = out.value
    _emit(args, value,
          (value.get("stdout") or "")
          + ("" if not value.get("stderr") else f"\n[stderr]\n{value['stderr']}"))
    return 0 if int(value.get("exit_code") or 0) == 0 else 1


def _cmd_apps(args: argparse.Namespace, context: Any) -> int:
    """Build / list / serve local apps through the build_app tool."""
    tools = context.tools
    action = getattr(args, "action", "list") or "list"
    name = getattr(args, "name", "") or ""
    kwargs: dict[str, Any] = {"action": action, "stack": getattr(args, "stack", "") or "static",
                              "features": getattr(args, "features", "") or "",
                              "title": getattr(args, "title", "") or ""}
    if name:
        kwargs["name"] = name
    if getattr(args, "port", ""):
        kwargs["port"] = int(args.port)
    out = tools.call("build_app", **kwargs)
    if not out.ok:
        print(f"apps: {out.error}", file=sys.stderr)
        return 1
    value = out.value
    if action == "build":
        v = value.get("validation") or {}
        _emit(args, value,
              f"built {name} [{kwargs['stack']}] — "
              f"{len(value.get('files') or [])} files, "
              f"validation: {'OK' if v.get('ok') else 'FAILED'}"
              + ("" if v.get("ok") else f" — {v.get('output', '')[:300]}"))
        return 0 if v.get("ok", True) else 1
    if action == "list":
        lines = [f"  {a.get('name')} [{a.get('stack')}] {a.get('path', '')}"
                 for a in (value.get("apps") or [])]
        _emit(args, value,
              "\n".join(lines) if lines else "no apps built yet — nm apps build <name> --stack flask")
        return 0
    _emit(args, value, json.dumps(value, indent=2, default=str))
    return 0


def _cmd_help(args: argparse.Namespace, context: Any) -> int:
    """`nm help [topic]` — the same pages the chat /help renders."""
    from .social.chat.control import detailed_help

    topic = getattr(args, "topic", "") or ""
    page = detailed_help(topic)
    _emit(args, {"help": page}, page)
    return 0


def _cmd_project(args: argparse.Namespace, context: Any) -> int:
    """Projects CLI: create/list/status/heal/replan/run."""
    from .agents.projects import ProjectManager

    pm = ProjectManager(context)
    action = getattr(args, "action", "list") or "list"
    pid = getattr(args, "id", "") or ""
    title = getattr(args, "title", "") or ""
    if action != "create" and not pid:
        pid = title  # positional reuse: `nm project heal <id>`
    if action == "create":
        objective = getattr(args, "description", "") or ""
        p = pm.create(title, objective=objective)
        _emit(args, p.to_dict(), f"created {p.id} — {p.title} [{p.status}]")
        return 0
    if action == "list":
        rows = pm.list_projects()
        lines = [f"{r.get('id', '')}  [{r.get('status', '')}]  "
                 f"{int(float(r.get('progress') or 0) * 100):3d}%  {r.get('title', '')}"
                 for r in rows]
        _emit(args, {"projects": rows},
              "\n".join(lines) if lines else "no projects")
        return 0
    if action == "heal":
        res = pm.heal(pid)
        n = int(res.get("healed_steps") or 0)
        text = (f"healed {n} step(s) on {pid}" if res.get("ok")
                else f"could not heal {pid}: {res.get('error', 'not healable')}")
        _emit(args, res, text)
        return 0
    if action == "replan":
        res = pm.replan(pid)
        _emit(args, res, f"replanned {pid}: {res.get('steps', '')}"
              if res.get("ok", True) else f"could not replan {pid}: "
              f"{res.get('error', '')}")
        return 0
    if action in ("status", "get"):
        st = pm.status(pid)
        if not st.get("ok", True):
            _emit(args, st, f"unknown project: {pid}")
            return 1
        human = (f"{pid} [{st.get('status', '?')}] "
                 f"{int(float(st.get('progress') or 0) * 100)}% — "
                 f"{st.get('title', '')}")
        _emit(args, st, human)
        return 0
    if action in ("plan", "run", "advance"):
        fn = getattr(pm, action if action != "advance" else "advance")
        res = fn(pid)
        payload = res.to_dict() if hasattr(res, "to_dict") else res
        _emit(args, payload, f"{action}: {pid}")
        return 0
    print(f"project: unknown action {action}", file=sys.stderr)
    return 2


def _cmd_mission(args: argparse.Namespace, context: Any) -> int:
    """Mission control CLI: the goal portfolio at a glance."""
    from .agents.mission import MissionControl

    mc = MissionControl(context)
    action = getattr(args, "action", "plan") or "plan"
    if action == "plan":
        plan = mc.plan()
        counts = plan.get("counts", {})
        cad = plan.get("cadence", {})
        eff = cad.get("effective_hours", cad.get("base_hours", "?"))
        lines = [
            f"mission plan — ready {counts.get('ready', 0)}, "
            f"blocked {counts.get('blocked', 0)}, "
            f"paused {counts.get('paused', 0)}, done {counts.get('done', 0)}",
            f"heartbeat: every {eff}h (base {cad.get('base_hours', '?')}h, "
            f"{'adaptive' if cad.get('adaptive') else 'fixed'})",
        ]
        nxt = plan.get("next")
        if nxt:
            lines.append(f"next: {nxt.get('id', '')} — {nxt.get('title', '')}")
        for g in plan.get("ready", [])[:10]:
            lines.append(f"  ready:   {g.get('id', '')}  {g.get('title', '')}")
        for b in plan.get("blocked", [])[:10]:
            goal = b.get("goal") or b
            lines.append(f"  blocked: {goal.get('id', '')}  "
                         f"{goal.get('title', '')}")
        ranking = plan.get("ranking") or []
        if ranking:
            try:
                from .agents.cognition import ModelBudget

                budget = ModelBudget(context).report()
                need = sum(int(r.get("est_calls") or 0) for r in ranking)
                if budget.get("unlimited"):
                    lines.append(f"budget: portfolio (~{need} calls) fits "
                                 "today — unlimited")
                elif need <= int(budget.get("remaining", 0)):
                    lines.append(f"budget: portfolio (~{need} calls) fits "
                                 "today — "
                                 f"{budget.get('remaining', 0)} calls left")
                else:
                    lines.append(f"budget: portfolio (~{need} calls) does "
                                 "NOT fit today — "
                                 f"{budget.get('remaining', 0)} left; cap it "
                                 "up with `nm autonomy budget --cap N`")
            except Exception:  # noqa: BLE001 — fit is a courtesy line
                pass
            lines.append("ranking (expected value):")
            for i, r in enumerate(ranking, 1):
                est = int(r.get("est_calls") or 0)
                lines.append(f"  {i}. {r.get('id', '')} {r.get('title', '')} "
                             f"— EV {r.get('expected_value', 0)} "
                             f"(risk {r.get('risk', '?')}, p{r.get('priority', 0)}, "
                             f"~{est} calls)")
        _emit(args, plan, "\n".join(lines))
        return 0
    if action == "next":
        out = mc.next()
        nxt = out.get("next")
        _emit(args, out,
              f"next: {nxt.get('id', '')} — {nxt.get('title', '')}" if nxt
              else "no goal ready")
        return 0
    if action == "status":
        plan = mc.plan()
        payload = {"counts": plan.get("counts", {}),
                   "cadence": plan.get("cadence", {})}
        _emit(args, payload, json.dumps(payload, indent=2, default=str))
        return 0
    if action == "health":
        # live-mission watchdog: a RUNNING mission whose newest checkpoint
        # went stale lost its worker — flag it for restart or cancel
        from .missions.runner import MissionRunner

        h = MissionRunner(context).health()
        lines = [f"mission health — {len(h['active'])} active, "
                 f"{len(h['stuck'])} stuck "
                 f"(stale after {int(h['stuck_after_seconds'])}s)"]
        for e in h["active"]:
            age = e.get("checkpoint_age_seconds")
            age_s = "never" if age is None else f"{age:g}s ago"
            lines.append(f"  {'STUCK' if e['stuck'] else 'live '}: "
                         f"{e['mission_id'][:8]} {e['status']} — {e['name']} "
                         f"(last checkpoint {age_s})")
        for mid in h["stuck"]:
            lines.append(f"  restart or cancel stuck mission: {mid}")
        _emit(args, h, "\n".join(lines))
        return 0
    print(f"mission: unknown action {action}", file=sys.stderr)
    return 2


def _cmd_goal(args: argparse.Namespace, context: Any) -> int:
    """Goals CLI: create/list/get/advance/tick + the project cascade."""
    from .agents.goals import GoalSystem

    gs = GoalSystem(context)
    action = getattr(args, "action", "list") or "list"
    arg = getattr(args, "arg", "") or ""
    arg2 = getattr(args, "arg2", "") or ""
    goal_id = getattr(args, "id", "") or (arg if action != "create" else "")
    title = getattr(args, "title", "") or (arg if action == "create" else "")

    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    if action == "create":
        priority = _int(getattr(args, "priority", ""), 0)
        g = gs.create(title, getattr(args, "description", "") or "",
                      priority=priority)
        payload: dict[str, Any] = {"goal": g.to_dict(), "id": g.id}
        lines = [f"created {g.id} — {g.title}"]
        if getattr(args, "project", False):
            sp = gs.spawn_project(g.id)
            payload["project"] = sp
            lines.append(f"project spawned: {sp.get('project_id', '')} "
                         f"(steps={sp.get('steps', 0)})")
        _emit(args, payload, "\n".join(lines))
        return 0
    if action in ("spawn", "spawn_project"):
        sp = gs.spawn_project(goal_id)
        _emit(args, sp, f"project spawned: {sp.get('project_id', '')}")
        return 0
    if action == "list":
        goals = gs.list(status=getattr(args, "filter_status", "") or "",
                        limit=_int(getattr(args, "limit", ""), 20) or 20)
        payload = {"goals": [g.to_dict() for g in goals]}
        lines = [f"{g.id}  [{g.status}]  p{g.priority}  "
                 f"{int(g.progress * 100):3d}%  {g.title}" for g in goals]
        _emit(args, payload, "\n".join(lines) if lines else "no goals yet")
        return 0
    if action == "get":
        g = gs.get(goal_id)
        _emit(args, {"goal": g.to_dict() if g else None},
              json.dumps(g.to_dict(), indent=2, default=str) if g
              else f"goal not found: {goal_id}")
        return 0 if g else 1
    if action == "next":
        g = gs.next_goal()
        if g is None:
            _emit(args, {"goal": None}, "no goal ready")
            return 0
        _emit(args, {"goal": g.to_dict()}, f"next: {g.id} — {g.title}")
        return 0
    if action == "priority":
        g = gs.set_priority(goal_id, _int(arg2))
        _emit(args, {"goal": g.to_dict() if g else None},
              f"priority set: {goal_id} -> {arg2}" if g
              else f"goal not found: {goal_id}")
        return 0 if g else 1
    if action == "depends":
        dep_id = arg2 or getattr(args, "description", "") or ""
        if getattr(args, "remove", False):
            g = gs.remove_dependency(goal_id, dep_id)
            verb = "dependency removed"
        else:
            g = gs.add_dependency(goal_id, dep_id)
            verb = "dependency added"
        _emit(args, {"goal": g.to_dict() if g else None},
              f"{verb}: {goal_id} waits for {dep_id}" if g
              else "goal not found")
        return 0 if g else 1
    if action == "advance":
        g = gs.advance(goal_id)
        _emit(args, {"goal": g.to_dict()}, f"advanced: {g.id} "
              f"({int(g.progress * 100)}%)")
        return 0
    if action == "adapt":
        g = gs.adapt(goal_id, reason=getattr(args, "reason", "") or "")
        _emit(args, {"goal": g.to_dict()}, f"adapted: {g.id}")
        return 0
    if action in ("complete", "pause", "resume"):
        g = getattr(gs, action)(goal_id)
        _emit(args, {"goal": g.to_dict() if g else None},
              f"{action}d: {goal_id}" if g else "goal not found")
        return 0 if g else 1
    if action == "reflect":
        from .agents.reflection import GoalReflector

        res = GoalReflector(context).reflect(goal_id, force=True)
        if res.get("ok"):
            kg_bits = res.get("kg", res.get("kg_nodes", "ok"))
            _emit(args, res, f"reflection for {goal_id} — kg: {kg_bits}, "
                            f"source: {res.get('source', 'heuristic')}")
        else:
            _emit(args, res, f"could not reflect {goal_id}: "
                  f"{res.get('error', 'unknown')}")
        return 0
    if action == "tick":
        advanced = gs.tick()
        payload = {"advanced": [g.to_dict() for g in advanced]}
        _emit(args, payload,
              f"ticked: {len(advanced)} goal(s) advanced" if advanced
              else "ticked: nothing to advance")
        return 0
    if action == "status":
        st = gs.status()
        _emit(args, st, json.dumps(st, indent=2, default=str))
        return 0
    print(f"goal: unknown action {action}", file=sys.stderr)
    return 2

def _cmd_stub(args: argparse.Namespace, context: Any, command: str) -> int:
    """Stub handler for commands not yet fully implemented."""
    _emit(args, {"command": command, "status": "stub"}, f"{command}: stub implementation")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
