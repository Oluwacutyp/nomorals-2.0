"""``nm completion`` — static shell completion generated from the live parser.

Best practice (shtab, edumatcher's completion design doc): generate *static*
completion scripts from the real ``argparse`` tree instead of running the
program on every Tab (argcomplete-style hooks pay the full import cost per
keystroke). The parser is the single source of truth, so completion can never
drift from the CLI — including aliases, options, and ``choices=`` values.

``nm completion bash`` prints the script to stdout (redirect it into your
shell's completion dir); install hints ship as comments inside the script.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _is_subparsers(action: Any) -> bool:
    return isinstance(action, argparse._SubParsersAction)


def _option_names(action: argparse.Action) -> list[str]:
    return list(getattr(action, "option_strings", []) or [])


def _choice_values(action: argparse.Action) -> list[str]:
    choices = getattr(action, "choices", None)
    if not choices or isinstance(choices, dict):
        return []
    try:
        return [str(c) for c in choices]
    except TypeError:
        return []


def _collect_options(parser: argparse.ArgumentParser) -> tuple[list[str], dict[str, list[str]], list[list[str]]]:
    """(option flags, flag → value-choices, positional choice lists)."""
    opts: list[str] = []
    opt_choices: dict[str, list[str]] = {}
    pos_choices: list[list[str]] = []
    for action in parser._actions:  # noqa: SLF001 - argparse has no public walker API
        if _is_subparsers(action):
            continue
        names = _option_names(action)
        if names:
            opts.extend(names)
            values = _choice_values(action)
            if values:
                for n in names:
                    opt_choices[n] = values
        elif _choice_values(action):
            # positional with choices= (e.g. `nm autonomy <action>`)
            pos_choices.append(_choice_values(action))
    # de-dup, keep order
    seen: set[str] = set()
    uniq = [o for o in opts if not (o in seen or seen.add(o))]
    return uniq, opt_choices, pos_choices


def _collect() -> dict[str, dict[str, Any]]:
    """canonical command → {aliases, opts, opt_choices, pos, subs}.

    ``subs`` maps a subcommand name → its own {opts, opt_choices, pos}.
    """
    from ..parser import _parser

    parser = _parser()
    top = None
    for action in parser._actions:  # noqa: SLF001
        if _is_subparsers(action):
            top = action
            break
    assert top is not None
    # global flags live on the top parser (before the subcommand)
    global_opts, _, _ = _collect_options(parser)

    commands: dict[str, dict[str, Any]] = {}
    seen: set[int] = set()
    for name, sub in top.choices.items():
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        canonical = sub.prog.split()[-1]
        aliases = sorted(k for k, v in top.choices.items()
                         if v is sub and k != canonical)
        opts, opt_choices, pos = _collect_options(sub)
        subs: dict[str, dict[str, Any]] = {}
        for action in sub._actions:  # noqa: SLF001
            if not _is_subparsers(action):
                continue
            sub_seen: set[int] = set()
            for sname, ssub in action.choices.items():
                if id(ssub) in sub_seen:
                    continue
                sub_seen.add(id(ssub))
                s_opts, s_choices, s_pos = _collect_options(ssub)
                subs[sname] = {"opts": s_opts, "opt_choices": s_choices,
                               "pos": s_pos}
        commands[canonical] = {
            "aliases": aliases,
            "opts": opts,
            "opt_choices": opt_choices,
            "pos": pos,
            "subs": subs,
        }
    return {"_global": {"opts": global_opts}, "commands": commands}


def _all_names(commands: dict[str, dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for canonical, spec in commands.items():
        names.append(canonical)
        names.extend(spec["aliases"])
    return sorted(names)


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------

def _bash(spec: dict[str, Any]) -> str:
    commands = spec["commands"]
    glob = " ".join(spec["_global"]["opts"])
    lines = [
        "# nm completion for bash.",
        "# install: nm completion bash > ~/.local/share/bash-completion/completions/nm",
        "#     or: echo 'eval \"$(nm completion bash)\"' >> ~/.bashrc",
        "_nm() {",
        '    local cur prev cmd sub words',
        "    COMPREPLY=()",
        '    cur="${COMP_WORDS[COMP_CWORD]}"',
        '    prev="${COMP_WORDS[COMP_CWORD-1]}"',
        '    cmd="${COMP_WORDS[1]}"',
        "    # resolve aliases to canonical names",
        "    case \"$cmd\" in",
    ]
    for canonical, cspec in sorted(commands.items()):
        for alias in cspec["aliases"]:
            lines.append(f"        {alias}) cmd={canonical} ;;")
    lines += [
        "    esac",
        '    if [[ $COMP_CWORD -eq 1 ]]; then',
        f'        COMPREPLY=( $(compgen -W "{" ".join(_all_names(commands))} {glob}" -- "$cur") )',
        "        return 0",
        "    fi",
        '    case "$cmd" in',
    ]
    for canonical, cspec in sorted(commands.items()):
        words: list[str] = []
        for p in cspec["pos"]:
            words.extend(p)
        words.extend(cspec["opts"])
        if cspec["subs"]:
            words.extend(sorted(cspec["subs"]))
        var = f"words_{canonical}".replace("-", "_")  # bash ids: no dashes
        lines.append(f"        {canonical})")
        lines.append(f'            local {var}="{" ".join(words)}"')
        lines.append(f'            if [[ $COMP_CWORD -eq 2 ]]; then')
        lines.append(f'                COMPREPLY=( $(compgen -W "${var}" -- "$cur") )')
        if cspec["subs"]:
            lines.append("            else")
            lines.append('                sub="${COMP_WORDS[2]}"')
            lines.append('                case "$sub" in')
            for sname, sspec in sorted(cspec["subs"].items()):
                lines.append(f"                    {sname})")
                lines.append(f'                        COMPREPLY=( $(compgen -W "{" ".join(sspec["opts"])}" -- "$cur") )')
                lines.append("                        ;;")
            lines.append("                esac")
        # option value choices, e.g. `nm autonomy --system <TAB>`
        if cspec["opt_choices"]:
            lines.append("            fi")
            lines.append('            case "$prev" in')
            for opt, values in sorted(cspec["opt_choices"].items()):
                lines.append(f"                {opt})")
                lines.append(f'                    COMPREPLY=( $(compgen -W "{" ".join(values)}" -- "$cur") )')
                lines.append("                    return 0 ;;")
            lines.append("            esac")
            lines.append("            return 0 ;;")
        else:
            lines.append("            fi")
            lines.append("            return 0 ;;")
    lines += [
        "    esac",
        "}",
        "complete -F _nm nm",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# zsh
# ---------------------------------------------------------------------------

def _zsh(spec: dict[str, Any]) -> str:
    commands = spec["commands"]
    lines = [
        "#compdef nm",
        "# nm completion for zsh.",
        "# install: nm completion zsh > ~/.zsh/completions/_nm   (add ~/.zsh/completions to $fpath)",
        "_nm() {",
        "    local -a cmds",
        "    cmds=(",
    ]
    for canonical, cspec in sorted(commands.items()):
        help_txt = ""
        lines.append(f"        '{canonical}:{help_txt}'")
    lines += [
        "    )",
        "    _arguments -C \\",
        "        '--config[path to a TOML config file]:config:_files' \\",
        "        '--log-level[log level]:level:(DEBUG INFO WARNING ERROR)' \\",
        "        '--json[emit JSON instead of prose]' \\",
        "        '--no-color[disable colored output]' \\",
        "        '1:command:->cmd' \\",
        "        '*:: :->args' && return 0",
        "    case $state in",
        "        cmd)",
        "            _describe -t commands 'nm command' cmds && return 0",
        "            ;;",
        "        args)",
        "            case $words[1] in",
    ]
    for canonical, cspec in sorted(commands.items()):
        words: list[str] = []
        for p in cspec["pos"]:
            words.extend(p)
        lines.append(f"                {canonical})")
        if words or cspec["subs"]:
            lines.append(f"                    _values 'subcommand' {' '.join(words)} {' '.join(sorted(cspec['subs']))} && return 0")
        else:
            lines.append("                    _arguments \\")
            for opt in cspec["opts"]:
                lines.append(f"                        '{opt}[{opt}]' \\")
            lines.append("                    && return 0")
        lines.append("                    ;;")
    lines += [
        "            esac",
        "            ;;",
        "    esac",
        "}",
        "_nm \"$@\"",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# fish
# ---------------------------------------------------------------------------

def _fish(spec: dict[str, Any]) -> str:
    commands = spec["commands"]
    lines = [
        "# nm completion for fish.",
        "# install: nm completion fish > ~/.config/fish/completions/nm.fish",
    ]
    for canonical in sorted(commands):
        lines.append(f"complete -c nm -f -n '__fish_use_subcommand' -a '{canonical}'")
    for canonical, cspec in sorted(commands.items()):
        base = f"__fish_seen_subcommand_from {canonical}"
        for opt in cspec["opts"]:
            if opt.startswith("--"):
                lines.append(
                    f"complete -c nm -f -n '{base}' -l '{opt[2:]}'")
            elif opt.startswith("-") and not opt.startswith("--"):
                lines.append(
                    f"complete -c nm -f -n '{base}' -s '{opt[1:]}'")
        for p in cspec["pos"]:
            lines.append(
                f"complete -c nm -f -n '{base}; and not __fish_seen_subcommand_from {' '.join(p)}' "
                f"-a '{' '.join(p)}'")
        for sname, sspec in sorted(cspec["subs"].items()):
            lines.append(
                f"complete -c nm -f -n '{base}' -a '{sname}'")
            sub_base = f"{base}; and __fish_seen_subcommand_from {sname}"
            for opt in sspec["opts"]:
                if opt.startswith("--"):
                    lines.append(
                        f"complete -c nm -f -n '{sub_base}' -l '{opt[2:]}'")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# powershell
# ---------------------------------------------------------------------------

def _powershell(spec: dict[str, Any]) -> str:
    commands = spec["commands"]
    cmds = " ".join(sorted(commands))
    lines = [
        "# nm completion for PowerShell.",
        "# install: nm completion powershell | Out-String | Invoke-Expression",
        "#     (add that line to your $PROFILE)",
        "Register-ArgumentCompleter -Native -CommandName nm -ScriptBlock {",
        "    param($wordToComplete, $commandAst, $cursorPosition)",
        f"    $cmds = @({', '.join(repr(c) for c in sorted(commands))})",
        "    $words = $commandAst.ToString() -split '\\s+'",
        "    $idx = $words.Count - 1",
        "    if ($wordToComplete -ne '') { $idx -= 1 }",
        "    if ($idx -le 1) {",
        "        $cmds | Where-Object { $_ -like \"$wordToComplete*\" } | ForEach-Object {",
        "            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


_GENERATORS = {
    "bash": _bash,
    "zsh": _zsh,
    "fish": _fish,
    "powershell": _powershell,
}


def _cmd_completion(args: argparse.Namespace, settings: Any = None) -> int:
    """``nm completion {bash,zsh,fish,powershell}`` — print the script."""
    shell = (getattr(args, "shell", "") or "").strip().lower()
    gen = _GENERATORS.get(shell)
    if gen is None:
        print(f"unknown shell: {shell or '(none)'} "
              f"— choose from: {', '.join(sorted(_GENERATORS))}",
              file=sys.stderr)
        return 2
    # The script must be byte-exact (users redirect it into a file), so it
    # goes to stdout raw — no styling, no _emit prose.
    print(gen(_collect()), end="")
    return 0
