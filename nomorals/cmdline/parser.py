"""``nm`` argument parser: the alias map and the argparse tree."""

from __future__ import annotations

import argparse

from ..version import __version__

# Canonical top-level command → short aliases. The single source of truth:
# each add_parser() below consumes CLI_ALIASES[name], and `nm help cli`
# renders this table. Aliases never collide with a canonical command name.
CLI_ALIASES: dict[str, list[str]] = {
    "status": ["st"],
    "session": ["sess"],
    "mind": ["m"],
    "doctor": ["dr"],
    "config": ["cfg"],
    "models": ["mod"],
    "memory": ["mem"],
    "power": ["pw"],
    "run": ["r"],
    "ask": ["a"],
    "backup": ["bak"],
    "missions": ["ms"],
    "mission": ["mi"],
    "queue": ["q"],
    "help": ["h"],
    "monitor": ["mon"],
    "watch": ["w"],
    "osint": ["os"],
    "finance": ["fin"],
    "project": ["proj"],
    "simulate": ["sim"],
    "briefing": ["br"],
    "train": ["tr"],
    "benchmark": ["bm"],
    "apps": ["ap"],
    "arena": ["ar"],
    "autonomy": ["auto"],
    "bet": ["bt"],
    "book": ["bk"],
    "build": ["bld"],
    "captcha": ["cap"],
    "cards": ["cd"],
    "cipher": ["cip"],
    "code": ["c"],
    "commands": ["cmd"],
    "connectors": ["conn"],
    "cookies": ["ck"],
    "crack": ["cr"],
    "data": ["d"],
    "datasci": ["ds"],
    "decode": ["dec"],
    "exec": ["e"],
    "goal": ["g"],
    "plugin": ["plug"],
    "hub": ["hb"],
    "improve": ["imp"],
    "inbox": ["ib"],
    "media": ["med"],
    "money": ["mn"],
    "music": ["mu"],
    "native": ["nat"],
    "owner": ["own"],
    "reason": ["rea"],
    "research-loop": ["rl"],
    "room": ["rm"],
    "serve": ["srv"],
    "setup": ["su"],
    "skill": ["sk"],
    "structure": ["struct"],
    "studio": ["stu"],
    "swarm": ["sw"],
    "tools": ["t"],
    "trade": ["td"],
    "trial": ["trl"],
    "tui": ["ui"],
    "timeline": ["tl"],
    "vision": ["v"],
    "voice": ["vc"],
    "weather": ["wx"],
    "workspace": ["ws"],
    "zip": ["z"],
    "deliver": ["dlv"],
    "doc": ["docs"],
    "wisdom": ["wis"],
    "trigger": ["trig"],
    "browse": ["brw"],
    "repo": ["rp"],
    "mesh": ["msh"],
    "sync": ["sy"],

    "search": ["s"],
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nm", description="NoMorals Core — self-hosted multi-agent AI substrate."
    )
    parser.add_argument("--version", action="version", version=f"nomorals {__version__}")
    parser.add_argument("--config", help="path to a TOML config file")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", aliases=CLI_ALIASES["doctor"],
                            help="show environment capabilities and health")
    doctor.add_argument("--build-native", action="store_true",
                        help="compile the native C++ accelerators before "
                             "reporting (same as: nm native --build)")
    sub.add_parser("config", aliases=CLI_ALIASES["config"],
                   help="print the effective configuration")
    sub.add_parser("setup", aliases=CLI_ALIASES["setup"], help="guided model setup wizard")

    models = sub.add_parser("models", aliases=CLI_ALIASES["models"],
                            help="inspect the model registry and catalog")
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
    # Broker actions (nomorals.cmdline.commands.models).  Positional and
    # optional so every pre-existing flag above keeps working unchanged.
    models.add_argument("model_action", nargs="?", default=None,
                        choices=["list", "add", "remove", "benchmark", "use", "select"],
                        help="broker action: list/add/remove/benchmark/use/select")
    models.add_argument("model_target", nargs="?", default="",
                        help="model id, HF repo id, or GGUF path for the action")
    models.add_argument("--capability", default="",
                        help="add: comma-separated capabilities (default: chat)")
    models.add_argument("--quant", default="Q4_K_M",
                        help="add: quantization label for a local GGUF")
    models.add_argument("--context-len", type=int, default=0,
                        help="add: context length for the model card")
    models.add_argument("--task-kind", default="",
                        help="select: task specialisation hint (e.g. code, judge)")
    models.add_argument("--rounds", type=int, default=5,
                        help="benchmark: probe rounds to measure (default: 5)")

    data = sub.add_parser("data", aliases=CLI_ALIASES["data"], help="fine-tune data: catalog, base models, persona mix")
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

    tools = sub.add_parser("tools", aliases=CLI_ALIASES["tools"], help="list available tools and their capabilities")
    tools.add_argument("--schema", action="store_true", help="emit full JSON schemas")

    memory = sub.add_parser("memory", aliases=CLI_ALIASES["memory"],
                            help="inspect and query memory")
    memory.add_argument("--query", default="")
    memory.add_argument("--limit", type=int, default=8)
    memory.add_argument("--consolidate", action="store_true")
    memory.add_argument("--stats", action="store_true")
    memory.add_argument("--remember", default="")
    # Prompt 11 subcommands — the flat flags above keep working for compat
    mem_sub = memory.add_subparsers(dest="memory_action")
    mem_sub.add_parser("show", help="show the user model Devon currently sees")
    m_list = mem_sub.add_parser("list", help="list raw memory records")
    m_list.add_argument("--kind", default="")
    m_list.add_argument("--query", default="")
    m_list.add_argument("--limit", type=int, default=20)
    m_list.add_argument("--include-private", action="store_true")
    m_edit = mem_sub.add_parser("edit", help="edit a memory record")
    m_edit.add_argument("record_id")
    m_edit.add_argument("text")
    m_forget = mem_sub.add_parser("forget", help="forget a memory record")
    m_forget.add_argument("record_id")
    m_fk = mem_sub.add_parser("forget-kind",
                              help="forget all records of a kind")
    m_fk.add_argument("kind")
    m_fk.add_argument("--yes", action="store_true",
                      help="skip confirmation")
    m_priv = mem_sub.add_parser("private",
                                help="mark a record private")
    m_priv.add_argument("record_id")
    m_pub = mem_sub.add_parser("public",
                               help="clear the private flag")
    m_pub.add_argument("record_id")
    mem_sub.add_parser("export", help="JSON dump of the user model + records")
    mem_sub.add_parser("rebuild", help="rebuild the user model now")
    mem_sub.add_parser("curate", help="run one memory-curation pass")

    owner = sub.add_parser("owner", aliases=CLI_ALIASES["owner"], help="owner identity seal (ingrained in code)")
    owner_sub = owner.add_subparsers(dest="owner_action")
    owner_sub.add_parser("seal",
                         help="bake your passphrase seal into the code "
                              "(prompts securely, min 20 chars)")
    owner_sub.add_parser("verify",
                         help="check an identity + passphrase against the seal")
    owner_sub.add_parser("whoami",
                         help="show the ingrained owner identities")

    power = sub.add_parser("power", aliases=CLI_ALIASES["power"], help="power mode")
    power_sub = power.add_subparsers(dest="power_action")
    power_sub.add_parser("unlock",
                         help="unlock power mode via owner seal "
                              "(prompts securely)")
    power_sub.add_parser("lock", help="lock power mode")
    power_sub.add_parser("status", help="power mode status")
    power_sub.add_parser("battery", help="battery/thermal status (power monitor)")

    bench = sub.add_parser("benchmark", aliases=CLI_ALIASES["benchmark"],
                           help="K3 scoreboard: benchmark the agent system")
    # not required: bare `nm benchmark` runs the whole scoreboard, and the
    # alias-map test parses every alias with no action
    bench_sub = bench.add_subparsers(dest="benchmark_action")
    b_run = bench_sub.add_parser("run", help="run a suite or all suites")
    b_run.add_argument("suite", nargs="?", default="all",
                       help="suite name or 'all' (swe_coding, research, "
                            "edits, builds, latency, reasoning, planning, "
                            "tool_use, self_correction)")
    b_run.add_argument("--limit", type=int, default=0,
                       help="max tasks per suite (0 = suite default)")
    b_run.add_argument("--export", default="",
                       help="write the full run report to this JSON path")
    b_list = bench_sub.add_parser("list", help="list saved benchmark runs")
    b_list.add_argument("--limit", type=int, default=20)
    b_list.add_argument("--suite", default="",
                        help="only runs of this suite selection")
    b_cmp = bench_sub.add_parser("compare",
                                 help="compare two runs (delta of B vs A)")
    b_cmp.add_argument("run_a", help="baseline run id")
    b_cmp.add_argument("run_b", help="challenger run id")

    agent = sub.add_parser("run", aliases=CLI_ALIASES["run"],
                           help="run a goal through the orchestrator")
    agent.add_argument("goal", nargs="+")
    agent.add_argument("--max-steps", type=int, default=8)
    agent.add_argument("--no-reflect", action="store_true")

    ask = sub.add_parser("ask", aliases=CLI_ALIASES["ask"],
                         help="single-turn chat with the active model")
    ask.add_argument("prompt", nargs="+")
    ask.add_argument("--system", default="")
    ask.add_argument("--max-tokens", type=int, default=1024)
    ask.add_argument("--temperature", type=float, default=0.7)
    ask.add_argument("--model", default="", help="hot-swap to this provider for this call")

    backup = sub.add_parser("backup", aliases=CLI_ALIASES["backup"],
                            help="create, list, verify, prune, or restore backups")
    backup.add_argument("verb", nargs="?", choices=["list", "verify", "prune"], default=None,
                        help="list backups, verify the latest, or prune old ones")
    backup.add_argument("--create", action="store_true")
    backup.add_argument("--list", action="store_true")
    backup.add_argument("--verify", action="store_true")
    backup.add_argument("--restore", default="")
    backup.add_argument("--push", action="store_true", help="push the latest backup to git")

    snapshot = sub.add_parser("snapshot",
                              help="point-in-time snapshots of live state")
    snapshot.add_argument("snapshot_action",
                          choices=["create", "list", "verify", "restore", "delete"],
                          nargs="?", default="list")
    snapshot.add_argument("snapshot_id", nargs="?", default="",
                          help="snapshot id (prefix ok); default: latest")
    snapshot.add_argument("--label", default="", help="label for create")
    snapshot.add_argument("--force", action="store_true",
                          help="restore over a running/dirty system")
    snapshot.add_argument("--json", action="store_true")

    recover = sub.add_parser("recover",
                             help="recover from the last good snapshot")
    recover.add_argument("snapshot_id", nargs="?", default="",
                         help="restore this snapshot instead of the last good one")
    recover.add_argument("--yes", action="store_true",
                         help="non-interactive: assume yes to all prompts")
    recover.add_argument("--json", action="store_true")

    update = sub.add_parser("update",
                            help="transactional self-update with auto-rollback")
    update.add_argument("--no-pull", action="store_true",
                        help="skip git pull (migrate + health-check only)")
    update.add_argument("--check-only", action="store_true",
                        help="run post-update health checks without updating")
    update.add_argument("--repo", default="",
                        help="repo dir override (default: auto-detected)")
    update.add_argument("--yes", action="store_true",
                        help="non-interactive: assume yes to all prompts")
    update.add_argument("--json", action="store_true")

    golden = sub.add_parser("golden",
                            help="golden end-to-end mission drills")
    golden.add_argument("golden_action",
                        choices=["list", "run", "resume"],
                        nargs="?", default="list")
    golden.add_argument("target", nargs="?", default="",
                        help="mission key (run) or mission id (resume)")
    golden.add_argument("--long", action="store_true",
                        help="real multi-minute drill instead of the unit version")
    golden.add_argument("--json", action="store_true")

    missions = sub.add_parser("missions", aliases=CLI_ALIASES["missions"],
                              help="list, run, resume, or inspect missions")
    missions.add_argument("--start", default="", help="start a new mission with this goal")
    missions.add_argument("--resume", default="", help="resume a mission by id")
    missions.add_argument("--resume-all", action="store_true", help="resume every interrupted mission")
    missions.add_argument("--status", default="", help="filter by status")
    missions.add_argument("--show", default="", help="show one mission's detail and checkpoints")
    missions.add_argument("--max-iterations", type=int, default=8)
    missions.add_argument("--budget-wall", type=float, default=0.0)
    missions.add_argument("--budget-tokens", type=int, default=0)
    missions.add_argument("--no-reflect", action="store_true")
    missions.add_argument("--accept", default="",
                          help="acceptance criteria as JSON for --start, e.g. "
                               "'{\"criteria\": [{\"name\": \"quality\", \"spec\": "
                               "{\"metric\": \"quality\", \"gte\": 0.8}}]}' — the "
                               "mission is verified through VERIFYING before it "
                               "may complete")
    missions.add_argument("--require-artifact", action="append", default=[],
                          metavar="TYPE",
                          help="require an artifact TYPE at acceptance "
                               "(repeatable; --start only)")
    missions.add_argument("--pause", default="", help="pause a mission by id (cross-process)")
    missions.add_argument("--cancel", default="", help="cancel a mission by id (terminal)")
    missions.add_argument("--resume-status", default="",
                          help="lift a pause and report the mission's status")

    sub.add_parser("serve", aliases=CLI_ALIASES["serve"], help="start the HTTP API server")
    sub.add_parser("tui", aliases=CLI_ALIASES["tui"], help="start the interactive terminal UI")
    queue = sub.add_parser("queue", aliases=CLI_ALIASES["queue"],
                           help="inspect the durable work queue")
    queue.add_argument("--topic", default="")

    # Additional subcommands expected by tests
    commands = sub.add_parser("commands", aliases=CLI_ALIASES["commands"], help="list available commands")
    commands.add_argument("filter", nargs="?", default="")

    status_p = sub.add_parser(
        "status",
        aliases=CLI_ALIASES["status"],
        help="system health snapshot — real numbers, fast, honest gaps",
        description=("nm status [--json]\n"
                     "One screen: version/profile, database, queue, missions,\n"
                     "memory, proactive delivery, power mode.\n"
                     "Every section is a bounded local query; a subsystem that\n"
                     "cannot be read prints 'unavailable' instead of fake zeros."),
    )
    status_p.add_argument("--json", action="store_true", help="Output as JSON")

    session_p = sub.add_parser(
        "session",
        aliases=CLI_ALIASES["session"],
        help="OS sessions: one session per chat, across all surfaces",
        description=("nm session list [--json]\n"
                     "nm session show <id> [--json]\n"
                     "nm session end <id>\n"
                     "Every surface (Telegram, WhatsApp, Discord, CLI) attaches\n"
                     "to one OS Session per chat — memory, persona, and gating\n"
                     "are keyed off the session, not the platform."),
    )
    session_p.add_argument("task", nargs="*", default=[],
                           help="verb and arguments (list|show|end)")
    session_p.add_argument("--json", action="store_true", help="Output as JSON")

    mind_p = sub.add_parser(
        "mind",
        aliases=CLI_ALIASES["mind"],
        help="the core mind: pending clarifications, recent jobs, router calls",
        description=("nm mind [status] [--json]\n"
                     "Inspect CoreMind's persisted state: open clarification\n"
                     "questions, recent routed jobs with outcomes, the last\n"
                     "objective. Router call counts and the last plan_error are\n"
                     "per-process / in-memory only — the CLI reports them as\n"
                     "unavailable rather than inventing numbers."),
    )
    mind_p.add_argument("action", nargs="?", default="status",
                        choices=["status"],
                        help="Action to perform")
    mind_p.add_argument("--json", action="store_true", help="Output as JSON")

    zip_cmd = sub.add_parser("zip", aliases=CLI_ALIASES["zip"], help="create and manage zip archives")
    zip_cmd.add_argument("action", nargs="?", default="list")
    zip_cmd.add_argument("path", nargs="?", default="")
    zip_cmd.add_argument("--dest", default="")

    deliver_cmd = sub.add_parser("deliver", aliases=CLI_ALIASES["deliver"],
                                 help="create-and-deliver flows: generate → zip → send")
    dsub = deliver_cmd.add_subparsers(dest="deliver_action")
    d_report = dsub.add_parser(
        "report",
        help="generate a styled report, zip it, optionally send it to a chat")
    d_report.add_argument("topic", help="what the report is about")
    d_report.add_argument(
        "--section", action="append", default=[], metavar="TITLE::BODY",
        help="one report section as 'Title::markdown body' (repeatable)")
    d_report.add_argument("--title", default="",
                          help="report title (defaults to the topic)")
    d_report.add_argument(
        "--to", default="", metavar="platform:chat",
        help="send the zip to this chat, e.g. telegram:123456 "
             "(omit to only build the report)")
    d_report.add_argument("--platform", default="",
                          help="platform when --to is a bare chat id")
    d_report.add_argument("--no-pdf", action="store_true",
                          help="skip the PDF artifact (HTML only)")
    d_report.add_argument("--json", action="store_true", help="Output as JSON")
    d_send = dsub.add_parser(
        "send",
        help="resend an already-built report zip to a chat")
    d_send.add_argument(
        "path", help="path to the report .zip from a previous build")
    d_send.add_argument(
        "--to", default="", metavar="platform:chat",
        help="send the zip to this chat, e.g. telegram:123456")
    d_send.add_argument("--platform", default="",
                        help="platform when --to is a bare chat id")
    d_send.add_argument("--caption", default="",
                        help="caption for the sent archive")
    d_send.add_argument("--json", action="store_true", help="Output as JSON")


    # Agent-tool subcommands
    autonomy = sub.add_parser("autonomy", aliases=CLI_ALIASES["autonomy"], help="Manage autonomous agent operations")
    autonomy.add_argument("action", nargs="?", default="status",
                         choices=["status", "tick", "report", "enable", "disable", "budget", "on", "off"],
                         help="Action to perform")
    autonomy.add_argument("--json", action="store_true", help="Output as JSON")
    autonomy.add_argument("--cap", default="",
                          help="set the daily model-call cap (budget action; persisted)")
    autonomy.add_argument("--unlimited", action="store_true",
                          help="lift the daily model-call cap (budget action)")

    goal = sub.add_parser("goal", aliases=CLI_ALIASES["goal"], help="Create and manage goals")
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

    mission = sub.add_parser("mission", aliases=CLI_ALIASES["mission"],
                             help="Mission control and planning")
    mission.add_argument("action", nargs="?", default="plan",
                        choices=["plan", "next", "status", "health",
                                 "replay", "redrive"],
                        help="Action to perform")
    mission.add_argument("--json", action="store_true", help="Output as JSON")
    mission.add_argument("--mission", default="",
                        help="mission id for the replay/redrive actions "
                             "(prefix ok when unambiguous)")
    mission.add_argument("--confirm", action="store_true",
                        help="required for redrive: actually create the new mission")

    skill = sub.add_parser("skill", aliases=CLI_ALIASES["skill"],
                           help="Executable skill packages: "
                                "list/install/enable/disable/run/benchmark "
                                "(library/create/show/delete/prune/restore/stats "
                                "= knowledge-skill library)")
    skill.add_argument("action", nargs="?", default="list",
                      choices=["list", "install", "enable", "disable", "run",
                               "benchmark", "library", "create", "show",
                               "delete", "prune", "restore", "stats"],
                      help="Action to perform")
    skill.add_argument("name", nargs="?", default="",
                       help="skill name (run/enable/disable/benchmark, "
                            "create/show/delete/restore)")
    skill.add_argument("--description", default="",
                       help="create: one-line description")
    skill.add_argument("--body", default="",
                       help="create: skill body text (prefix with @ to read from a file)")
    skill.add_argument("--kind", default="strategy",
                       help="create: skill kind")
    skill.add_argument("--tags", default="",
                       help="create: comma-separated tags")
    skill.add_argument("--pruned", action="store_true",
                       help="library list: include pruned (quarantined) skills")
    skill.add_argument("--manifest", default="",
                       help="install: manifest JSON file path, @path, or inline JSON")
    skill.add_argument("--input", default="",
                       help="run: JSON object of skill inputs")
    skill.add_argument("--version", default="",
                       help="run: execute a specific installed version")
    skill.add_argument("--json", action="store_true", help="Output as JSON")

    project = sub.add_parser("project", aliases=CLI_ALIASES["project"],
                             help="Manage projects")
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
    sim = sub.add_parser("simulate", aliases=CLI_ALIASES["simulate"],
                         help="Sandbox simulator: dry-run, run, compare, risk-classify commands")
    sim_sub = sim.add_subparsers(dest="simulate_action", required=True)
    sim_dry = sim_sub.add_parser("dry-run", help="preview what a command would do (executes nothing)")
    sim_dry.add_argument("cmd", help="shell command to preview")
    sim_run = sim_sub.add_parser("run", help="execute a command in the sandbox")
    sim_run.add_argument("cmd", help="shell command to run")
    sim_run.add_argument("--confirm", action="store_true",
                         help="allow high-risk commands")
    sim_run.add_argument("--timeout", type=float, default=120.0,
                         help="execution timeout in seconds")
    sim_cmp = sim_sub.add_parser("compare", help="run two commands in separate sandboxes and compare")
    sim_cmp.add_argument("command_a", help="first candidate command")
    sim_cmp.add_argument("command_b", help="second candidate command")
    sim_risk = sim_sub.add_parser("risk", help="classify a command's risk without running it")
    sim_risk.add_argument("cmd", help="shell command to classify")
    for _sp in (sim_dry, sim_run, sim_cmp, sim_risk):
        _sp.add_argument("--json", action="store_true", help="Output as JSON")
    rl = sub.add_parser("research-loop", aliases=CLI_ALIASES["research-loop"],
                        help="always-on research loop: status, tick, topics")
    rl.add_argument("action", nargs="?", default="status",
                    choices=["status", "tick", "run", "ensure", "enable",
                             "disable", "topics", "set_topics"],
                    help="Action to perform")
    rl.add_argument("topic", nargs="?", default="",
                    help="run: single topic to research right now")
    rl.add_argument("--topics", default="",
                    help="set_topics: comma-separated topic list")
    rl.add_argument("--max-topics", type=int, default=0,
                    help="tick: max topics per cycle (default: loop default)")
    rl.add_argument("--json", action="store_true", help="Output as JSON")
    code = sub.add_parser("code", aliases=CLI_ALIASES["code"],
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

    media = sub.add_parser("media", aliases=CLI_ALIASES["media"],
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

    studio = sub.add_parser("studio", aliases=CLI_ALIASES["studio"],
        help="Pro edit sessions: filters, grading, layers, AI edits, templates",
        description=("nm studio presets\n"
                     "nm studio filter <file> <preset> [--strength F]\n"
                     "nm studio grade <file> [--temperature N] [--tint N] "
                     "[--saturation F] [--contrast F] [--vignette F]\n"
                     "nm studio text <file> \"<text>\" [--position POS] [--size N]\n"
                     "nm studio ai <file> \"<instruction>\" [--strength F] "
                     "[--seed N] [--mask l,t,r,b]\n"
                     "nm studio template <name> [--param k=v ...] [--wait]\n"
                     "nm studio project save <file> <project> --ops JSON\n"
                     "nm studio project render <project> [--wait]\n"
                     "nm studio project describe <project>\n"
                     "nm studio batch <dir> <project> [--pattern GLOB]\n"
                     "nm studio compare <file> --ops JSON [--mode MODE]\n"
                     "nm studio gen-status"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    studio_sub = studio.add_subparsers(dest="studio_action", required=True)

    def _s_json(p):
        p.add_argument("--json", action="store_true", help="Output as JSON")

    s_presets = studio_sub.add_parser("presets", help="list studio presets")
    _s_json(s_presets)

    s_filter = studio_sub.add_parser("filter", help="apply a filter preset")
    s_filter.add_argument("file", help="image file (workspace-relative)")
    s_filter.add_argument("preset", help="filter preset name")
    s_filter.add_argument("--strength", type=float, default=1.0)
    _s_json(s_filter)

    s_grade = studio_sub.add_parser("grade", help="color grade an image")
    s_grade.add_argument("file", help="image file (workspace-relative)")
    s_grade.add_argument("--temperature", type=float, default=0.0)
    s_grade.add_argument("--tint", type=float, default=0.0)
    s_grade.add_argument("--saturation", type=float, default=1.0)
    s_grade.add_argument("--contrast", type=float, default=1.0)
    s_grade.add_argument("--vignette", type=float, default=0.0)
    _s_json(s_grade)

    s_text = studio_sub.add_parser("text", help="add pro text to an image")
    s_text.add_argument("file", help="image file (workspace-relative)")
    s_text.add_argument("text", help="the text to render")
    s_text.add_argument("--position", default="bottom")
    s_text.add_argument("--size", type=int, default=64)
    s_text.add_argument("--color", default="white")
    _s_json(s_text)

    s_ai = studio_sub.add_parser("ai", help="AI instruction edit "
                                 "('make it sunset')")
    s_ai.add_argument("file", help="image file (workspace-relative)")
    s_ai.add_argument("instruction", help="natural-language edit instruction")
    s_ai.add_argument("--strength", type=float, default=0.75)
    s_ai.add_argument("--seed", type=int, default=None)
    s_ai.add_argument("--mask", default=None,
                      help="region as l,t,r,b (or omit for full image)")
    _s_json(s_ai)

    s_tmpl = studio_sub.add_parser("template", help="build a template project")
    s_tmpl.add_argument("name", help="podcast-clip, quote-card, "
                                    "product-showcase, meme, slideshow")
    s_tmpl.add_argument("--param", action="append", default=[],
                        help="template param as k=v (repeatable)")
    s_tmpl.add_argument("--wait", action="store_true",
                        help="with video: block until the job finishes")
    _s_json(s_tmpl)

    s_proj = studio_sub.add_parser("project", help="save/load/render projects")
    s_proj.add_argument("action", choices=["save", "render", "describe"])
    s_proj.add_argument("file", nargs="?", default=None,
                        help="save: source file; render/describe: project file")
    s_proj.add_argument("project", nargs="?", default=None,
                        help="save: project file to write")
    s_proj.add_argument("--ops", default="[]",
                        help="save: op chain as JSON list")
    s_proj.add_argument("--wait", action="store_true")
    _s_json(s_proj)

    s_batch = studio_sub.add_parser("batch", help="apply a project to a folder")
    s_batch.add_argument("dir", help="source directory (workspace-relative)")
    s_batch.add_argument("project", help="saved studio project file")
    s_batch.add_argument("--pattern", default="*.jpg")
    s_batch.add_argument("--out-dir", default=None)
    _s_json(s_batch)

    s_cmp = studio_sub.add_parser("compare", help="before/after comparison")
    s_cmp.add_argument("file", help="image file (workspace-relative)")
    s_cmp.add_argument("--ops", default="[]",
                       help="op chain as JSON list")
    s_cmp.add_argument("--mode", default="side-by-side",
                       choices=["side-by-side", "split", "stacked", "html"])
    _s_json(s_cmp)

    s_gen = studio_sub.add_parser("gen-status",
                                  help="generative backend status")
    _s_json(s_gen)

    captcha = sub.add_parser("captcha", aliases=CLI_ALIASES["captcha"],
        help="captcha detection and solving for browser automation",
        description=("nm captcha detect --html FILE | --url URL\n"
                     "nm captcha status\n"
                     "nm captcha solve --kind KIND --sitekey KEY --url URL\n"
                     "  [--backend service|takeover|detect] [--image PATH|URL]\n"
                     "  [--v3-action NAME] [--min-score F]"),
    )
    captcha_sub = captcha.add_subparsers(dest="captcha_action", required=True)
    c_detect = captcha_sub.add_parser("detect", help="detect captchas in HTML")
    c_detect.add_argument("--html", default="",
                          help="HTML file to scan (default: fetch --url)")
    c_detect.add_argument("--url", default="",
                          help="page URL (fetched, or hints detection)")
    c_detect.add_argument("--json", action="store_true",
                          help="output as JSON")
    captcha_sub.add_parser("status", help="backend availability + audit path")
    c_solve = captcha_sub.add_parser("solve", help="solve one challenge")
    c_solve.add_argument("--kind", required=True,
                         help="recaptcha_v2|recaptcha_v3|recaptcha_enterprise|"
                              "hcaptcha|turnstile|image_captcha")
    c_solve.add_argument("--sitekey", default="",
                         help="the data-sitekey")
    c_solve.add_argument("--url", default="", help="page URL")
    c_solve.add_argument("--backend", default="auto",
                         help="service|takeover|detect|auto (default auto)")
    c_solve.add_argument("--image", default="",
                         help="image captcha: file path or URL")
    c_solve.add_argument("--v3-action", default="",
                         help="reCAPTCHA v3 action name")
    c_solve.add_argument("--min-score", type=float, default=0.3,
                         help="reCAPTCHA v3 score floor")
    c_solve.add_argument("--solver", dest="solver_enabled",
                         action="store_true", default=None,
                         help="use the solving service (default: on)")
    c_solve.add_argument("--no-solver", dest="solver_enabled",
                         action="store_false",
                         help="skip the service, go straight to takeover")
    c_solve.add_argument("--json", action="store_true",
                         help="output as JSON")

    bet = sub.add_parser("bet", aliases=CLI_ALIASES["bet"],
        help="sports bet analyst: ensemble ML, value + Kelly staking (analysis only)",
        description=("nm bet analyze --home H --away A [--league L] "
                     "[--odds H D A]\n"
                     "nm bet bankroll [--set AMOUNT]\n"
                     "nm bet backtest [--n N] [--seed S]\n"
                     "nm bet record --home H --away A --score HG-AG [--league L]"),
    )
    bet_sub = bet.add_subparsers(dest="bet_action", required=True)
    b_an = bet_sub.add_parser("analyze", help="ensemble analysis of a match")
    b_an.add_argument("--home", required=True)
    b_an.add_argument("--away", required=True)
    b_an.add_argument("--league", default="GEN")
    b_an.add_argument("--odds", nargs=3, type=float, metavar=("H", "D", "A"),
                      help="bookmaker odds: home draw away")
    b_an.add_argument("--bookmaker", default="cli")
    b_an.add_argument("--min-edge", type=float, default=0.04)
    b_br = bet_sub.add_parser("bankroll", help="show/set the paper bankroll")
    b_br.add_argument("--set", type=float, default=None)
    b_bt = bet_sub.add_parser("backtest", help="walk-forward backtest")
    b_bt.add_argument("--n", type=int, default=400)
    b_bt.add_argument("--seed", type=int, default=7)
    b_rc = bet_sub.add_parser("record", help="record a played result")
    b_rc.add_argument("--home", required=True)
    b_rc.add_argument("--away", required=True)
    b_rc.add_argument("--score", required=True, help="HG-AG, e.g. 2-1")
    b_rc.add_argument("--league", default="GEN")

    wx = sub.add_parser("weather", aliases=CLI_ALIASES["weather"],
        help="live weather + USA situations + timezone utilities (keyless)",
        description=("nm weather now [PLACE]\n"
                     "nm weather forecast [PLACE] [--days N]\n"
                     "nm weather alerts [PLACE]\n"
                     "nm weather usa\n"
                     "nm weather tz [YYYY-MM-DD HH:MM [FROM] [TO]]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    wx_sub = wx.add_subparsers(dest="weather_action", required=True)
    w_now = wx_sub.add_parser("now", help="current conditions + alerts")
    w_now.add_argument("place", nargs="?", default=None)
    w_fc = wx_sub.add_parser("forecast", help="daily forecast")
    w_fc.add_argument("place", nargs="?", default=None)
    w_fc.add_argument("--days", type=int, default=3)
    w_al = wx_sub.add_parser("alerts", help="active weather alerts")
    w_al.add_argument("place", nargs="?", default=None)
    wx_sub.add_parser("usa", help="USA national situations overview")
    w_tz = wx_sub.add_parser("tz", help="timezone conversion / owner clock")
    w_tz.add_argument("when", nargs="?", default=None,
                      help="YYYY-MM-DD HH:MM, ISO, or unix ts")
    w_tz.add_argument("from_zone", nargs="?", default=None)
    w_tz.add_argument("to_zone", nargs="?", default=None)

    voice = sub.add_parser("voice", aliases=CLI_ALIASES["voice"],
        help="Live voice loop: talk to Devon through your mic and speakers",
        description=("nm voice call [--turns N] [--profile P] [--device ID]\n"
                     "nm voice say \"text\" [--profile P] [--out PATH] [--perform] [--mood M]\n"
                     "nm voice fetch --backend cosyvoice|fish-s2-pro|fish-s1-mini|orpheus|dia\n"
                     "nm voice clone <name> <audio> --consent [--transcript T]\n"
                     "nm voice list | nm voice use <name> | nm voice current\n"
                     "nm voice listen [--secs N] [--out PATH]\n"
                     "nm voice transcribe <file>\n"
                     "nm voice stats [--json]\n"
                     "nm voice consent [--grant|--revoke] [--device ID]\n"
                     "nm voice purge\n"
                     "nm voice decrypt <file.enc> --key-file PATH"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    voice_sub = voice.add_subparsers(dest="voice_action", required=True)
    v_call = voice_sub.add_parser("call", help="live listen→think→speak loop")
    v_call.add_argument("--turns", type=int, default=0,
                        help="max turns (0 = until you say goodbye)")
    v_call.add_argument("--profile", default="",
                        help="TTS voice profile name")
    v_call.add_argument("--device", default="default",
                        help="device id (consent + conversation identity)")
    v_call.add_argument("--keep-audio", action="store_true",
                        help="retain raw utterances (encrypted with --audio-key-file)")
    v_call.add_argument("--audio-key-file", default="",
                        help="file holding the 32-byte audio key or a passphrase")
    v_call.add_argument("--no-tts", action="store_true",
                        help="skip speech output (text in terminal only)")
    v_call.add_argument("--json", action="store_true", help="Output as JSON")
    v_say = voice_sub.add_parser("say", help="speak one line via TTS")
    v_say.add_argument("text", help="text to speak")
    v_say.add_argument("--profile", default="", help="TTS voice profile name")
    v_say.add_argument("--voice", default="",
                       help="catalogue voice name (resolves backend + "
                            "profile; alias of --profile for raw profiles)")
    v_say.add_argument("--out", default="", help="wav output path")
    v_say.add_argument("--backend", default="",
                       help="TTS backend (default: settings or auto)")
    v_say.add_argument("--perform", action="store_true",
                       help="run the humanizing director: laughs, sighs, "
                            "breath, stutters, fillers, pauses, emphasis")
    v_say.add_argument("--mood", default="neutral",
                       help="performance mood (happy, sad, nervous, tired, …)")
    v_say.add_argument("--effect", default="",
                       help="named director house style "
                            "(dramatic_whisper, hype, bedtime_story, …)")
    v_say.add_argument("--intensity", type=int, default=3,
                       help="director intensity 0-5 (default: 3)")
    v_say.add_argument("--seed", type=int, default=None,
                       help="seed for reproducible performances")
    v_say.add_argument("--json", action="store_true", help="Output as JSON")
    v_listen = voice_sub.add_parser("listen",
                                    help="record one utterance to a wav file")
    v_listen.add_argument("--secs", type=float, default=10.0,
                          help="max seconds to record")
    v_listen.add_argument("--out", default="", help="wav output path")
    v_listen.add_argument("--device", default="default", help="device id")
    v_listen.add_argument("--json", action="store_true", help="Output as JSON")
    v_tr = voice_sub.add_parser("transcribe", help="transcribe an audio file")
    v_tr.add_argument("file", help="audio file path")
    v_tr.add_argument("--json", action="store_true", help="Output as JSON")
    v_stats = voice_sub.add_parser("stats", help="voice latency/session stats")
    v_stats.add_argument("--json", action="store_true", help="Output as JSON")
    v_consent = voice_sub.add_parser("consent",
                                     help="manage per-device recording consent")
    v_consent.add_argument("--device", default="default", help="device id")
    v_consent.add_argument("--grant", action="store_true",
                           help="grant consent non-interactively")
    v_consent.add_argument("--revoke", action="store_true",
                           help="revoke consent")
    v_consent.add_argument("--json", action="store_true", help="Output as JSON")
    v_purge = voice_sub.add_parser("purge",
                                   help="delete retained raw audio")
    v_purge.add_argument("--json", action="store_true", help="Output as JSON")
    v_dec = voice_sub.add_parser("decrypt",
                                 help="decrypt a retained .wav.enc utterance")
    v_dec.add_argument("file", help="encrypted utterance path")
    v_dec.add_argument("--key-file", required=True,
                       help="file holding the 32-byte audio key or passphrase")
    v_dec.add_argument("--out", default="",
                       help="output wav path (default: alongside, .wav)")
    v_fetch = voice_sub.add_parser(
        "fetch", help="download open TTS weights from HuggingFace")
    v_fetch.add_argument("--backend", default="cosyvoice",
                         help="model to fetch: cosyvoice (default), "
                              "fish-s2-pro, fish-s1-mini, orpheus, dia, "
                              "qwen3-tts")
    v_fetch.add_argument("--dest", default="",
                         help="destination dir (default: ~/.cache/nomorals/voice_models/<backend>)")
    v_fetch.add_argument("--repo", default="",
                         help="override the HuggingFace repo id")
    v_fetch.add_argument("--json", action="store_true", help="Output as JSON")
    # -- voice catalogue: named voices, switchable at runtime ---------------
    v_clone = voice_sub.add_parser(
        "clone", help="clone a voice from a reference clip into the catalogue")
    v_clone.add_argument("name", help="catalogue voice name")
    v_clone.add_argument("audio", help="reference audio file path")
    v_clone.add_argument("--transcript", default="",
                         help="words spoken in the reference clip "
                              "(improves zero-shot cloning)")
    v_clone.add_argument("--consent", action="store_true",
                         help="confirm this is your voice or you have "
                              "permission to clone it (required)")
    v_clone.add_argument("--backend", default="auto",
                         help="preferred backend for this voice")
    v_clone.add_argument("--describe", default="",
                         help="human description of the voice")
    v_clone.add_argument("--json", action="store_true", help="Output as JSON")
    v_vlist = voice_sub.add_parser("list", help="list catalogue voices")
    v_vlist.add_argument("--json", action="store_true", help="Output as JSON")
    v_use = voice_sub.add_parser("use",
                                 help="set the catalogue's active voice")
    v_use.add_argument("name", help="catalogue voice name")
    v_use.add_argument("--json", action="store_true", help="Output as JSON")
    v_cur = voice_sub.add_parser("current", help="show the active voice")
    v_cur.add_argument("--json", action="store_true", help="Output as JSON")
    v_rm = voice_sub.add_parser("rm", help="drop a catalogue voice")
    v_rm.add_argument("name", help="catalogue voice name")
    v_rm.add_argument("--json", action="store_true", help="Output as JSON")
    v_desc = voice_sub.add_parser("describe",
                                  help="describe a catalogue voice")
    v_desc.add_argument("name", help="catalogue voice name")
    v_desc.add_argument("text", help="description text")
    v_desc.add_argument("--json", action="store_true", help="Output as JSON")
    v_vtr = voice_sub.add_parser(
        "transcript", help="set a cloned voice's prompt transcript")
    v_vtr.add_argument("name", help="catalogue voice name")
    v_vtr.add_argument("text", help="words spoken in the reference clip")
    v_vtr.add_argument("--json", action="store_true", help="Output as JSON")

    inbox = sub.add_parser("inbox", aliases=CLI_ALIASES["inbox"],
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

    vision = sub.add_parser("vision", aliases=CLI_ALIASES["vision"],
        help="See images: describe, read text, locate UI elements",
        description=("nm vision describe <file> [\"question\"]\n"
                     "nm vision read-text <file>\n"
                     "nm vision locate <file> \"<target>\"\n"
                     "nm vision screenshot [--display N]  (privileged)"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    vision_sub = vision.add_subparsers(dest="vision_action", required=True)
    v_desc = vision_sub.add_parser("describe", help="describe an image")
    v_desc.add_argument("file", help="workspace-relative path, URL, or "
                                    "inbox:<id> / room:<slug>:<path> / attachment:<n>")
    v_desc.add_argument("question", nargs="?", default="",
                        help="optional question about the image")
    v_desc.add_argument("--json", action="store_true", help="Output as JSON")
    v_rt = vision_sub.add_parser("read-text", help="transcribe text from an image")
    v_rt.add_argument("file", help="workspace-relative path, URL, or reference")
    v_rt.add_argument("--json", action="store_true", help="Output as JSON")
    v_loc = vision_sub.add_parser("locate", help="locate a UI element or object")
    v_loc.add_argument("file", help="workspace-relative path, URL, or reference")
    v_loc.add_argument("target", help="what to find, e.g. 'the submit button'")
    v_loc.add_argument("--json", action="store_true", help="Output as JSON")
    v_shot = vision_sub.add_parser("screenshot",
                                   help="capture the local display (privileged: "
                                        "needs allow_screenshot + confirmation)")
    v_shot.add_argument("--display", type=int, default=0, help="display index")
    v_shot.add_argument("--prompt", default="",
                        help="optional question about the screenshot")
    v_shot.add_argument("--json", action="store_true", help="Output as JSON")

    room = sub.add_parser("room", aliases=CLI_ALIASES["room"],
        help="Project rooms: persistent per-goal workspaces",
        description=("nm room new \"<title>\" [--kind goal|project|ad_hoc] [--linked ID]\n"
                     "nm room list [--status active]\n"
                     "nm room enter <slug>\n"
                     "nm room status <slug>\n"
                     "nm room archive <slug> | nm room pause <slug> | nm room resume <slug>\n"
                     "nm room link <slug-a> <slug-b>\n"
                     "nm room search \"<query>\" [--deep]\n"
                     "nm room tick\n"
                     "nm room stale [--days 30]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    briefing = sub.add_parser(
        "briefing",
        aliases=CLI_ALIASES["briefing"],
        help="Morning briefing: the overnight digest",
        description=("nm briefing now            generate + deliver immediately\n"
                     "nm briefing today           print the last stored briefing\n"
                     "nm briefing retry           regenerate + deliver\n"
                     "nm briefing config          show time/timezone/topics/sections\n"
                     "nm briefing status          proactive switches + recent delivery states\n"
                     "nm briefing topics add|rm X  manage news topic filter\n"
                     "nm briefing symbols add|rm X  manage market symbols\n"
                     "nm briefing sections pin|unpin <name>\n"
                     "nm briefing followup <n>    expand item n from the briefing"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    briefing_sub = briefing.add_subparsers(dest="briefing_action",
                                           required=True)
    for _name in ("now", "today", "retry", "config", "status"):
        _p = briefing_sub.add_parser(_name, help=f"briefing {_name}")
        _p.add_argument("--json", action="store_true", help="Output as JSON")
    b_topics = briefing_sub.add_parser("topics", help="manage news topics")
    b_topics.add_argument("op", choices=["add", "rm"])
    b_topics.add_argument("topic", help="topic string")
    b_topics.add_argument("--json", action="store_true", help="Output as JSON")
    b_symbols = briefing_sub.add_parser("symbols", help="manage market symbols")
    b_symbols.add_argument("op", choices=["add", "rm"])
    b_symbols.add_argument("symbol", help="e.g. BTC")
    b_symbols.add_argument("--json", action="store_true",
                           help="Output as JSON")
    b_sections = briefing_sub.add_parser("sections",
                                         help="pin/unpin a section")
    b_sections.add_argument("op", choices=["pin", "unpin"])
    b_sections.add_argument("name", help="section name")
    b_sections.add_argument("--json", action="store_true",
                            help="Output as JSON")
    b_followup = briefing_sub.add_parser("followup",
                                         help="expand a briefing item")
    b_followup.add_argument("n", type=int, help="1-based item number")
    b_followup.add_argument("--json", action="store_true",
                            help="Output as JSON")
    timeline = sub.add_parser("timeline", aliases=CLI_ALIASES["timeline"],
                              help="event timeline: what happened, newest first")
    timeline.add_argument("--session", default="",
                          help="filter by session id")
    timeline.add_argument("--project", default="",
                          help="filter by project id")
    timeline.add_argument("--mission", default="",
                          help="filter by mission id")
    timeline.add_argument("--artifact", default="",
                          help="filter by artifact id")
    timeline.add_argument("--topic", default="",
                          help="filter by topic (glob, e.g. 'mission.*')")
    timeline.add_argument("--limit", type=int, default=50,
                          help="max rows (default 50)")
    timeline.add_argument("--json", action="store_true",
                          help="Output as JSON")
    room_sub = room.add_subparsers(dest="room_action", required=True)
    r_new = room_sub.add_parser("new", help="create a room")
    r_new.add_argument("title", help="room title")
    r_new.add_argument("--kind", default="ad_hoc",
                       choices=["goal", "project", "ad_hoc"])
    r_new.add_argument("--linked", default="",
                       help="linked goal/project id")
    r_new.add_argument("--json", action="store_true", help="Output as JSON")
    r_list = room_sub.add_parser("list", help="list rooms")
    r_list.add_argument("--status", default="",
                        help="filter: active|paused|archived")
    r_list.add_argument("--json", action="store_true", help="Output as JSON")
    r_enter = room_sub.add_parser("enter", help="enter a room (prints ROOM.md)")
    r_enter.add_argument("slug", help="room slug")
    r_enter.add_argument("--json", action="store_true", help="Output as JSON")
    r_status = room_sub.add_parser("status", help="room progress/blockers/decisions")
    r_status.add_argument("slug", help="room slug")
    r_status.add_argument("--json", action="store_true", help="Output as JSON")
    for _name in ("archive", "pause", "resume"):
        _p = room_sub.add_parser(_name, help=f"{_name} a room")
        _p.add_argument("slug", help="room slug")
        _p.add_argument("--json", action="store_true", help="Output as JSON")
    r_link = room_sub.add_parser("link", help="read-only cross reference")
    r_link.add_argument("slug_a", help="first room slug")
    r_link.add_argument("slug_b", help="second room slug")
    r_link.add_argument("--json", action="store_true", help="Output as JSON")
    r_search = room_sub.add_parser("search", help="search rooms")
    r_search.add_argument("query", help="search query")
    r_search.add_argument("--deep", action="store_true",
                          help="also search files/ contents")
    r_search.add_argument("--json", action="store_true", help="Output as JSON")
    r_tick = room_sub.add_parser("tick", help="advance active rooms now")
    r_tick.add_argument("--json", action="store_true", help="Output as JSON")
    r_stale = room_sub.add_parser("stale", help="rooms idle > N days")
    r_stale.add_argument("--days", type=float, default=30.0)
    r_stale.add_argument("--json", action="store_true", help="Output as JSON")


    improve = sub.add_parser("improve", aliases=CLI_ALIASES["improve"],
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

    trade = sub.add_parser("trade", aliases=CLI_ALIASES["trade"],
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

    swarm = sub.add_parser("swarm", aliases=CLI_ALIASES["swarm"],
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
    book = sub.add_parser("book", aliases=CLI_ALIASES["book"], help="AI-assisted book writing: create, write, build")
    book.add_argument("action", nargs="?", default="list",
                     choices=["list", "create", "run", "status", "build"],
                     help="Action to perform")
    book.add_argument("topic", nargs="?", default="", help="Book topic")
    book.add_argument("--chapters", type=int, default=5, help="Number of chapters")
    book.add_argument("--words", type=int, default=2000, help="Words per chapter")
    book.add_argument("--no-research", action="store_true", help="Skip research phase")
    book.add_argument("--slug", default="", help="Book slug")
    hub = sub.add_parser("hub", aliases=CLI_ALIASES["hub"], help="MediaHub: one-call media orchestrator (song/video/podcast)")
    hub.add_argument("mode", nargs="?", default="status",
                     choices=["song", "video", "podcast", "status", "styles"],
                     help="what to run")
    hub.add_argument("query", nargs="?", default="",
                     help="song topic, or video/podcast search query")
    hub.add_argument("--style", default="pop", help="song style")
    hub.add_argument("--platform", default="", help="video/podcast platform filter")
    hub.add_argument("--no-play", action="store_true",
                     help="don't queue/play the result")
    hub.add_argument("--json", action="store_true", help="Output as JSON")
    cipher = sub.add_parser("cipher", aliases=CLI_ALIASES["cipher"],
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
    osint_p = sub.add_parser("osint", aliases=CLI_ALIASES["osint"],
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
    structure_p = sub.add_parser("structure", aliases=CLI_ALIASES["structure"], help="Structure an objective into a brief")
    structure_p.add_argument("objective", nargs="?", default="",
                             help="task text to structure into a brief")
    structure_p.add_argument("--for", dest="for_", default="mission",
                             help="shape the brief for: mission|goal|chat")
    money_p = sub.add_parser("money", aliases=CLI_ALIASES["money"], help="money-making opportunities hunter")
    msub = money_p.add_subparsers(dest="money_action")
    _m = msub.add_parser("scan", help="run the opportunity finders (web + curated)")
    _m.add_argument("kind", nargs="?", default="",
                    help="optional kind: paid_task|referral|course|bounty|gig|arbitrage|content")
    _m.add_argument("--max", type=int, default=15, help="max results shown")
    _m = msub.add_parser("list", help="list stored opportunities")
    _m.add_argument("--new", action="store_true", help="only recent finds")
    _m.add_argument("--max", type=int, default=30, help="max results shown")
    _m = msub.add_parser("profile", help="show the earner profile")
    structure_p.add_argument("--polish", action="store_true",
                             help="let the model rewrite the brief")
    structure_p.add_argument("--json", action="store_true", help="Output as JSON")
    arena_parser = sub.add_parser("arena", aliases=CLI_ALIASES["arena"], help="Self-improvement arena: status + approve builds")
    arena_sub = arena_parser.add_subparsers(dest="arena_command", required=True)
    arena_sub.add_parser("status", help="Show arena status")
    arena_approve = arena_sub.add_parser("approve", help="Approve a pending arena build")
    arena_approve.add_argument("build_id", help="arena build id to approve")
    trial_parser = sub.add_parser("trial", aliases=CLI_ALIASES["trial"], help="Single-account trial flow: save/list credentials")
    trial_sub = trial_parser.add_subparsers(dest="trial_command", required=True)
    trial_save = trial_sub.add_parser("save", help="Save trial credentials to the encrypted vault")
    trial_save.add_argument("platform", help="platform name")
    trial_save.add_argument("login", help="account login / username")
    trial_save.add_argument("password", help="account password (stored encrypted in the vault)")
    trial_save.add_argument("--note", default="", help="optional note")
    trial_sub.add_parser("list", help="List stored trial accounts")
    train = sub.add_parser("train", aliases=CLI_ALIASES["train"],
                           help="model training: backends, runs")
    train.add_argument("--backends", action="store_true",
                       help="list training backends with honest machine availability")
    train.add_argument("--run", action="store_true",
                       help="kick off one pipeline run (collect→train→evaluate→promote)")
    train.add_argument("--backend", default="",
                       help="override the training backend for this run")
    train.add_argument("--base-model", default="",
                       help="HF base model id (required for external backends)")
    train.add_argument("--json", action="store_true", help="Output as JSON")
    help_p = sub.add_parser("help", aliases=CLI_ALIASES["help"],
                            help="Show help (chat catalog, CLI command help, or topic page)")
    help_p.add_argument("topic", nargs="?", default="",
                        help="command or topic page (e.g. code, budget)")
    help_p.add_argument("--json", action="store_true", help="Output as JSON")
    cookies_p = sub.add_parser("cookies", aliases=CLI_ALIASES["cookies"], help="Cookie lab: parse, classify, ingest")
    cookies_p.add_argument("text", nargs="*", default=[],
                           help="cookie/header text, or 'ingest <text>'")
    cookies_p.add_argument("--source", default="cookies",
                           help="source label for ingest")
    cookies_p.add_argument("--json", action="store_true", help="Output as JSON")
    sub.add_parser("reason", aliases=CLI_ALIASES["reason"], help="Reasoning engine")
    sub.add_parser("workspace", aliases=CLI_ALIASES["workspace"], help="Workspace management")
    monitor = sub.add_parser("monitor", aliases=CLI_ALIASES["monitor"],
                             help="watch files/URLs for changes")
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
    watch = sub.add_parser("watch", aliases=CLI_ALIASES["watch"],
                           help="background watchers with smart alerts")
    watch.add_argument("action", nargs="?", default="list",
                       choices=["add", "list", "tick", "pause", "resume", "rm",
                                "history", "alerts"])
    watch.add_argument("ref", nargs="?", default="",
                       help="plain-language spec (add) or watcher id/name")
    watch.add_argument("--severity", default="",
                       choices=["", "info", "important", "urgent"],
                       help="override parsed severity")
    watch.add_argument("--interval", type=float, default=None,
                       help="override check interval, seconds")
    watch.add_argument("--quiet", default="",
                       help="quiet hours, e.g. '22:00-07:00' (owner tz)")
    watch.add_argument("--channels", default="",
                       help="alert channels, e.g. 'telegram' or 'telegram,whatsapp'")
    watch.add_argument("--limit", type=int, default=50,
                       help="history/alert rows to show")
    watch.add_argument("--json", action="store_true", help="Output as JSON")
    crack = sub.add_parser("crack", aliases=CLI_ALIASES["crack"], help="Offline hash cracking (md5/sha1/sha256/…)")
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
    decode = sub.add_parser("decode", aliases=CLI_ALIASES["decode"],
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
    music = sub.add_parser("music", aliases=CLI_ALIASES["music"], help="Music generation")
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

    exec_cmd = sub.add_parser("exec", aliases=CLI_ALIASES["exec"], help="run code in the sandbox")
    exec_cmd.add_argument("code", nargs="?", default="",
                          help="source to run (or: languages)")
    exec_cmd.add_argument("--lang", default="", help="python|bash|node|… (auto by default)")
    exec_cmd.add_argument("--timeout", default="30")
    exec_cmd.add_argument("--file", default="", help="run a workspace file instead")
    exec_cmd.add_argument("--json", action="store_true", help="Output as JSON")

    apps = sub.add_parser("apps", aliases=CLI_ALIASES["apps"], help="build and manage local web apps")
    apps.add_argument("action", nargs="?", default="list",
                      choices=["build", "list", "info", "stacks", "serve",
                               "stop", "served"],
                      help="what to do")
    apps.add_argument("name", nargs="?", default="", help="app name")
    apps.add_argument("--stack", default="static")
    apps.add_argument("--features", default="", help="comma-separated feature list")
    apps.add_argument("--title", default="")
    apps.add_argument("--port", default="")
    apps.add_argument("--verify", dest="verify", action="store_true",
                      default=True,
                      help="runtime-verify the built app (default)")
    apps.add_argument("--no-verify", dest="verify", action="store_false",
                      help="skip runtime verification")
    apps.add_argument("--json", action="store_true", help="Output as JSON")

    build_cmd = sub.add_parser("build", aliases=CLI_ALIASES["build"],
                               help="scaffold -> verify (-> deliver) a builder template project")
    bsub = build_cmd.add_subparsers(dest="build_action")
    bsub.add_parser("kinds", help="list scaffold template kinds")
    b_verify = bsub.add_parser(
        "verify",
        help="scaffold a template and run the full verify lifecycle "
             "(install, tests, serve+smoke, export)")
    b_verify.add_argument("kind", help="template kind (see: nm build kinds)")
    b_verify.add_argument("name", help="project name")
    b_verify.add_argument("--dest", default=".",
                          help="directory to scaffold into (default: .)")
    b_verify.add_argument("--export-dir", default="",
                          help="where the export archive lands")
    b_verify.add_argument("--startup-timeout", type=float, default=10.0,
                          help="seconds to wait for the served app")
    b_verify.add_argument("--json", action="store_true",
                          help="Output as JSON")
    b_deliver = bsub.add_parser(
        "deliver",
        help="scaffold, verify, zip, and deliver a project to a chat")
    b_deliver.add_argument("kind", help="template kind (see: nm build kinds)")
    b_deliver.add_argument("name", help="project name")
    b_deliver.add_argument("--dest", default=".",
                           help="directory to scaffold into (default: .)")
    b_deliver.add_argument("--to", default="", metavar="platform:chat",
                           help="send the zip to this chat, e.g. telegram:123456")
    b_deliver.add_argument("--platform", default="",
                           help="platform when --to is a bare chat id")
    b_deliver.add_argument("--caption", default="",
                           help="caption for the sent archive")
    b_deliver.add_argument("--export-dir", default="",
                           help="where the zip archive lands")
    b_deliver.add_argument("--startup-timeout", type=float, default=10.0,
                           help="seconds to wait for the served app")
    b_deliver.add_argument("--json", action="store_true",
                           help="Output as JSON")

    # Connector commands
    connectors = sub.add_parser("connectors", aliases=CLI_ALIASES["connectors"], help="Manage external service connectors")
    connectors.add_argument("action", nargs="?", default="list",
                           choices=["list", "status", "connect", "disconnect",
                                    "provision", "checkpoint"],
                           help="Action to perform")
    connectors.add_argument("--name", help="Connector name")
    connectors.add_argument("--provider", help="Provider (mono, plaid, etc.)")
    connectors.add_argument("--kind", help="Provision kind (with 'provision' action)")
    connectors.add_argument("--params-json", default="{}",
                            help="JSON params for the provision kind")
    connectors.add_argument("--cop", default="list",
                            choices=["list", "resolve", "cancel"],
                            help="Checkpoint operation (with 'checkpoint' action)")
    connectors.add_argument("--id", help="Checkpoint id (with resolve/cancel)")
    connectors.add_argument("--note", default="",
                            help="Note recorded on resolve/cancel")

    # Finance commands
    finance = sub.add_parser("finance", aliases=CLI_ALIASES["finance"],
                             help="Bank account linking and transactions")
    finance.add_argument("action", nargs="?", default="status",
                        choices=["status", "link", "accounts", "transactions", "balance"],
                        help="Action to perform")
    finance.add_argument("--provider", default="auto", help="Provider (mono, plaid, auto)")
    finance.add_argument("--account-id", help="Account ID")
    finance.add_argument("--days", type=int, default=30, help="Transaction history days")

    # Native extensions command
    native = sub.add_parser("native", aliases=CLI_ALIASES["native"], help="native kernels: status and build")
    native.add_argument("--build", action="store_true",
                        help="compile the C++ kernels when a compiler is present")
    native.add_argument("--benchmark", action="store_true",
                        help="time the vector-search kernel native vs pure Python and check agreement")

        # Virtual cards commands
    cards = sub.add_parser("cards", aliases=CLI_ALIASES["cards"], help="Virtual card management")
    cards.add_argument("action", nargs="?", default="list",
                      choices=["list", "create", "pause", "close", "status"],
                      help="Action to perform")
    cards.add_argument("--token", help="Card token")
    cards.add_argument("--type", default="UNLOCKED", help="Card type")
    cards.add_argument("--limit", type=int, help="Spend limit in cents")
    cards.add_argument("--merchant", help="Merchant name")


    # Wave K commands: document engine, browser service, code workspace.
    doc = sub.add_parser("doc", aliases=CLI_ALIASES["doc"],
        help="document engine: parse, convert, search documents",
        description=("nm doc parse <file> [--json]\n"
                     "nm doc convert <file> --to md|html|txt|pdf|csv [--out PATH]\n"
                     "nm doc search <query> --dir DIR [--limit N]\n"
                     "nm doc show <file>\n"
                     "nm doc diff <file-a> <file-b> [--json]\n"
                     "nm doc summarize <file> [--sentences N] [--json]\n"
                     "nm doc ocr <scanned.pdf> [--lang eng] [--dpi 200] [--json]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doc.add_argument("task", nargs="*", default=[], help="verb and arguments")
    doc.add_argument("--to", default="", help="convert target: md|html|txt|pdf|csv")
    doc.add_argument("--out", default="", help="write converted output here")
    doc.add_argument("--dir", default="", help="search: directory of documents")
    doc.add_argument("--limit", type=int, default=10, help="search: max hits")
    doc.add_argument("--sentences", type=int, default=5,
                     help="summarize: sentences to keep")
    doc.add_argument("--lang", default="eng", help="ocr: tesseract language")
    doc.add_argument("--dpi", type=int, default=200, help="ocr: render DPI")
    doc.add_argument("--json", action="store_true", help="Output as JSON")

    browse = sub.add_parser("browse", aliases=CLI_ALIASES["browse"],
        help="browser service: sessions, tabs, forms, uploads, downloads, screenshots, proxies",
        description=("nm browse open <url> [--session S]\n"
                     "nm browse tabs|text|md|links|history [--session S]\n"
                     "nm browse fill <name> <value> | nm browse click <target>\n"
                     "nm browse submit [target] [--upload field=path ...]\n"
                     "nm browse extract [target] [--kind headings|tables|forms|meta|nav]\n"
                     "nm browse task --steps-file PATH | --steps-json '[...]'\n"
                     "nm browse cookies [export|import <path> [--format netscape|json]]\n"
                     "nm browse proxy status|rotate|set <url>|clear\n"
                     "nm browse downloads [--category C] [--limit N]\n"
                     "nm browse shot [--out PATH] | nm browse download <url> [--organize]\n"
                     "nm browse rtab open <url> [--proxy P] | rtab tabs|shot|fill|click|submit|wait|extract|download\n"
                     "nm browse close | nm browse sessions\n"
                     "nm browse daemon start|stop|status"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    browse.add_argument("task", nargs="*", default=[], help="verb and arguments")
    browse.add_argument("--session", default="",
                        help="browser session name (default: cli)")
    browse.add_argument("--out", default="", help="shot: save PNG here")
    browse.add_argument("--json", action="store_true", help="Output as JSON")
    browse.add_argument("--upload", action="append", default=[],
                        help="submit: file upload as field=path (repeatable)")
    browse.add_argument("--kind", default="",
                        help="extract: headings|tables|forms|meta|nav")
    browse.add_argument("--steps-file", default="",
                        help="task: JSON file with the step list")
    browse.add_argument("--steps-json", default="",
                        help="task: JSON step list inline")
    browse.add_argument("--format", default="",
                        help="cookies export/import: netscape|json")
    browse.add_argument("--category", default="",
                        help="downloads: filter by MIME category")
    browse.add_argument("--limit", type=int, default=100,
                        help="downloads: max rows (default 100)")
    browse.add_argument("--organize", action="store_true",
                        help="download: file into a MIME-category subfolder")
    browse.add_argument("--proxy", default="",
                        help="rtab open: proxy URL override for this tab")

    repo = sub.add_parser("repo", aliases=CLI_ALIASES["repo"],
        help="code workspace: branches, worktrees, patches, test/build",
        description=("nm repo status|branches|diff [ref]|log [n] [--root DIR]\\n"
                     "nm repo branch <name> | nm repo switch <name>\\n"
                     "nm repo worktree <add|list|remove> [path] [branch] [--force]\\n"
                     "nm repo patch review|apply|preview|record <file> [path] [--root DIR]\\n"
                     "nm repo test [selector] | nm repo build [target] [--root DIR]\\n"
                     "nm repo commit -m \"msg\" [paths...] | nm repo push|pull|fetch [remote] [branch]\\n"
                     "nm repo stash <push|pop|list> [-m \"msg\"]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    repo.add_argument("task", nargs="*", default=[], help="verb and arguments")
    repo.add_argument("--root", default=".", help="repository root")
    repo.add_argument("--yes", action="store_true",
                      help="patch apply: write files (default is dry-run)")
    repo.add_argument("--force", action="store_true",
                      help="worktree remove: discard dirty state")
    repo.add_argument("--message", "-m", default="",
                      help="commit message / stash message")
    repo.add_argument("--json", action="store_true", help="Output as JSON")

    wisdom = sub.add_parser("wisdom", aliases=CLI_ALIASES["wisdom"],
        help="wisdom keeper: corpus, history, practice sessions",
        description=("nm wisdom status [--json]\n"
                     "nm wisdom ask <query> [--limit N] [--json]\n"
                     "nm wisdom ingest <slug> | --all\n"
                     "nm wisdom search <query> [--limit N] [--json]\n"
                     "nm wisdom timeline [--tradition T] [--from Y] [--to Y] [--json]\n"
                     "nm wisdom compare <topic> [--json]\n"
                     "nm wisdom practice list [--json]\n"
                     "nm wisdom practice <session-id> [--rounds N] [--chat]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    wisdom.add_argument("task", nargs="*", default=[], help="verb and arguments")
    wisdom.add_argument("--all", action="store_true",
                        help="ingest: ingest every manifest entry")
    wisdom.add_argument("--limit", type=int, default=0, help="max results")
    wisdom.add_argument("--rounds", type=int, default=0, help="practice: repeat")
    wisdom.add_argument("--chat", action="store_true",
                        help="practice: deliver the session to chat via the "
                             "live gateway instead of pacing it in the terminal")
    wisdom.add_argument("--tradition", default="", help="timeline: filter tradition")
    wisdom.add_argument("--start", type=int, default=-3000,
                        help="timeline: start year")
    wisdom.add_argument("--end", type=int, default=2100,
                        help="timeline: end year")
    wisdom.add_argument("--json", action="store_true", help="Output as JSON")

    datasci = sub.add_parser("datasci", aliases=CLI_ALIASES["datasci"],
        help="data-science workspace: load, describe, query, plot",
        description=("nm datasci load <file> [--name N] [--overwrite]\n"
                     "nm datasci list [--json]\n"
                     "nm datasci describe <name> [--json]\n"
                     "nm datasci head <name> [--rows N] [--json]\n"
                     "nm datasci query <name> \"<pandas expr>\" [--as NEW] [--json]\n"
                     "nm datasci plot <name> --kind line|bar|scatter|hist "
                     "--x COL [--y COL] [--title T]\n"
                     "nm datasci drop <name>"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    datasci.add_argument("task", nargs="*", default=[], help="verb and arguments")
    datasci.add_argument("--name", default="", help="load: dataset name")
    datasci.add_argument("--overwrite", action="store_true",
                         help="load: replace existing dataset")
    datasci.add_argument("--rows", type=int, default=5, help="head: row count")
    datasci.add_argument("--as_name", default="",
                         help="query: save result as new dataset")
    datasci.add_argument("--kind", default="line",
                         choices=["line", "bar", "scatter", "hist"],
                         help="plot: chart kind")
    datasci.add_argument("--x", default="", help="plot: x column")
    datasci.add_argument("--y", default="", help="plot: y column")
    datasci.add_argument("--title", default="", help="plot: chart title")
    datasci.add_argument("--json", action="store_true", help="Output as JSON")

    plugin = sub.add_parser("plugin", aliases=CLI_ALIASES["plugin"],
        help="plugin packages: install, list, enable/disable, run",
        description=("nm plugin install <path|url|zip>\n"
                     "nm plugin list [--json]\n"
                     "nm plugin info <name> [--json]\n"
                     "nm plugin enable|disable <name>\n"
                     "nm plugin remove <name>\n"
                     "nm plugin run <name> [entry] [--json]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plugin.add_argument("task", nargs="*", default=[], help="verb and arguments")
    plugin.add_argument("--json", action="store_true", help="Output as JSON")
    mesh = sub.add_parser("mesh", aliases=CLI_ALIASES["mesh"],
        help="device mesh: nodes, presence, task dispatch",
        description=("nm mesh nodes [--json]\n"
                     "nm mesh register <name> [--platform P]\n"
                     "nm mesh heartbeat <node-id>\n"
                     "nm mesh dispatch <task-type> [--target NODE]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mesh.add_argument("task", nargs="*", default=[], help="verb and arguments")
    mesh.add_argument("--json", action="store_true", help="Output as JSON")

    sync = sub.add_parser("sync", aliases=CLI_ALIASES["sync"],
        help="multi-device sync: status, push, pull",
        description=("nm sync status [--json]\n"
                     "nm sync push --peer-db PATH [--json]\n"
                     "nm sync pull --peer-db PATH [--json]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sync.add_argument("task", nargs="*", default=[], help="verb and arguments")
    sync.add_argument("--json", action="store_true", help="Output as JSON")
    sync.add_argument("--peer-db", default="", help="peer database path")

    search = sub.add_parser("search", aliases=CLI_ALIASES["search"],
        help="universal federated search across memory, books, wisdom, docs, code, timeline",
        description=("nm search <query> [--source NAME]... [--type TYPE]...\n"
                     "            [--since DATE] [--before DATE] [--limit N]\n"
                     "            [--dir DIR] [--json]\n"
                     "sources: memory wisdom books docs code timeline\n"
                     "types:   memory passage book doc code event\n"
                     "dates: ISO-8601 or epoch seconds"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    search.add_argument("task", nargs="*", default=[], help="the search query")
    search.add_argument("--source", action="append", default=[],
                        help="source to search (repeatable; default: all)")
    search.add_argument("--type", action="append", default=[],
                        help="result type filter (repeatable)")
    search.add_argument("--since", default=None,
                        help="only hits on/after this date")
    search.add_argument("--before", default=None,
                        help="only hits on/before this date")
    search.add_argument("--limit", type=int, default=10, help="max hits")
    search.add_argument("--dir", default="",
                        help="docs: build an ad-hoc index from this directory")
    search.add_argument("--json", action="store_true", help="Output as JSON")


    trigger = sub.add_parser("trigger", aliases=CLI_ALIASES["trigger"],
        help="event-condition-action automation: triggers",
        description=("nm trigger add --name NAME --source SRC --action ACT [...]\n"
                     "nm trigger list [--json]\n"
                     "nm trigger remove <id>\n"
                     "nm trigger enable|disable <id>\n"
                     "nm trigger history [<id>] [--limit N] [--json]\n"
                     "nm trigger run <id>\n"
                     "sources: schedule|file|price|message|webhook\n"
                     "actions: notify|message|command|mission"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    trigger.add_argument("task", nargs="*", default=[], help="verb and arguments")
    trigger.add_argument("--name", default="", help="add: trigger name")
    trigger.add_argument("--source", default="", help="add: trigger source")
    trigger.add_argument("--action", default="", help="add: trigger action")
    trigger.add_argument("--condition", default="",
                         help="add: condition as JSON")
    trigger.add_argument("--params", default="",
                         help="add: action params as JSON")
    trigger.add_argument("--cooldown", type=float, default=0.0,
                         help="add: minimum seconds between fires")
    trigger.add_argument("--cron", default="",
                         help="schedule: cron expression")
    trigger.add_argument("--interval", default="",
                         help="schedule: interval like 30m, 2h, 1d")
    trigger.add_argument("--daily", default="",
                         help="schedule: daily time HH:MM")
    trigger.add_argument("--weekly", default="",
                         help="schedule: weekly 'DAY HH:MM'")
    trigger.add_argument("--once", default="",
                         help="schedule: one-time ISO datetime or epoch")
    trigger.add_argument("--path", default="", help="file: path to watch")
    trigger.add_argument("--on", default="",
                         help="file: change|create|delete")
    trigger.add_argument("--symbol", default="", help="price: symbol e.g. BTC")
    trigger.add_argument("--market", default="",
                         help="price: crypto|stocks|forex")
    trigger.add_argument("--op", default="",
                         help="price: lt|lte|gt|gte|eq|ne|changed|changed_by_pct")
    trigger.add_argument("--value", default="",
                         help="price: threshold value")
    trigger.add_argument("--pattern", default="",
                         help="message: regex pattern")
    trigger.add_argument("--chat", default="",
                         help="message source: only this chat key; "
                              "message action: destination chat key")
    trigger.add_argument("--sender", default="",
                         help="message: only this sender")
    trigger.add_argument("--secret", default="",
                         help="webhook: shared secret")
    trigger.add_argument("--title", default="", help="notify: title")
    trigger.add_argument("--body", default="", help="notify: body")
    trigger.add_argument("--text", default="", help="message action: text")
    trigger.add_argument("--command", dest="command_text", default="",
                         help="command action: nm command string")
    trigger.add_argument("--argv", nargs="*", default=None,
                         help="command action: nm argv words")
    trigger.add_argument("--goal", default="", help="mission action: goal")
    trigger.add_argument("--max-iterations", type=int, default=0,
                         help="mission action: max iterations")
    trigger.add_argument("--limit", type=int, default=0,
                         help="history: max rows")
    trigger.add_argument("--json", action="store_true", help="Output as JSON")

    return parser

    # Wave K commands: document engine, browser service, code workspace.
    doc = sub.add_parser("doc", aliases=CLI_ALIASES["doc"],
        help="document engine: parse, convert, search documents",
        description=("nm doc parse <file> [--json]\n"
                     "nm doc convert <file> --to md|html|txt|pdf|csv [--out PATH]\n"
                     "nm doc search <query> --dir DIR [--limit N]\n"
                     "nm doc show <file>\n"
                     "nm doc diff <file-a> <file-b> [--json]\n"
                     "nm doc summarize <file> [--sentences N] [--json]\n"
                     "nm doc ocr <scanned.pdf> [--lang eng] [--dpi 200] [--json]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doc.add_argument("task", nargs="*", default=[], help="verb and arguments")
    doc.add_argument("--to", default="", help="convert target: md|html|txt|pdf|csv")
    doc.add_argument("--out", default="", help="write converted output here")
    doc.add_argument("--dir", default="", help="search: directory of documents")
    doc.add_argument("--limit", type=int, default=10, help="search: max hits")
    doc.add_argument("--sentences", type=int, default=5,
                     help="summarize: sentences to keep")
    doc.add_argument("--lang", default="eng", help="ocr: tesseract language")
    doc.add_argument("--dpi", type=int, default=200, help="ocr: render DPI")
    doc.add_argument("--json", action="store_true", help="Output as JSON")

    browse = sub.add_parser("browse", aliases=CLI_ALIASES["browse"],
        help="browser service: sessions, tabs, downloads, screenshots",
        description=("nm browse open <url> [--session S]\n"
                     "nm browse tabs|text|md|links|history [--session S]\n"
                     "nm browse shot [--out PATH] | nm browse download <url>\n"
                     "nm browse close | nm browse sessions\n"
                     "nm browse daemon start|stop|status"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    browse.add_argument("task", nargs="*", default=[], help="verb and arguments")
    browse.add_argument("--session", default="",
                        help="browser session name (default: cli)")
    browse.add_argument("--out", default="", help="shot: save PNG here")
    browse.add_argument("--json", action="store_true", help="Output as JSON")

    repo = sub.add_parser("repo", aliases=CLI_ALIASES["repo"],
        help="code workspace: branches, worktrees, patches, test/build",
        description=("nm repo status|branches|diff [ref]|log [n] [--root DIR]\\n"
                     "nm repo branch <name> | nm repo switch <name>\\n"
                     "nm repo worktree <add|list|remove> [path] [branch] [--force]\\n"
                     "nm repo patch review|apply|preview|record <file> [path] [--root DIR]\\n"
                     "nm repo test [selector] | nm repo build [target] [--root DIR]\\n"
                     "nm repo commit -m \"msg\" [paths...] | nm repo push|pull|fetch [remote] [branch]\\n"
                     "nm repo stash <push|pop|list> [-m \"msg\"]"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    repo.add_argument("task", nargs="*", default=[], help="verb and arguments")
    repo.add_argument("--root", default=".", help="repository root")
    repo.add_argument("--yes", action="store_true",
                      help="patch apply: write files (default is dry-run)")
    repo.add_argument("--force", action="store_true",
                      help="worktree remove: discard dirty state")
    repo.add_argument("--message", "-m", default="",
                      help="commit message / stash message")
    repo.add_argument("--json", action="store_true", help="Output as JSON")

