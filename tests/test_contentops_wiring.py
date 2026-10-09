"""Wiring tests for the shorts empire: CLI dispatch, chat dispatch, registry."""

import argparse
import importlib


def test_cli_shorts_registered():
    from nomorals.cmdline import parser as parser_mod

    assert "shorts" in parser_mod.CLI_ALIASES
    assert parser_mod.CLI_ALIASES["shorts"] == ["sh"]


def test_cli_shorts_parses():
    from nomorals.cmdline import parser as parser_mod

    # find the build function
    build = getattr(parser_mod, "_parser", None)
    assert build is not None, "no parser builder found"
    p = build()
    args = p.parse_args(["shorts", "make", "motivation", "discipline"])
    assert args.command == "shorts"
    assert args.shorts_action == "make"
    assert args.niche == "motivation"
    args = p.parse_args(["sh", "niches"])
    assert args.shorts_action == "niches"


def test_cmd_shorts_importable():
    from nomorals.cmdline.commands.shorts import cmd_shorts, HANDLERS

    assert callable(cmd_shorts)
    assert {"make", "queue", "status", "resume", "post",
            "niches", "niche", "calendar", "ledger", "estimate"} <= set(HANDLERS)


def test_chat_registry_has_shorts():
    from nomorals.social.chat import control as control_mod

    found = False
    for name in dir(control_mod):
        obj = getattr(control_mod, name)
        if isinstance(obj, dict) and "shorts" in obj:
            found = True
            break
    assert found, "shorts missing from control.py registries"


def test_runtime_dispatch_mentions_shorts():
    import pathlib

    src = pathlib.Path("nomorals/agents/partner/runtime.py").read_text()
    assert '"shorts"' in src and "_control_shorts" in src
    src2 = pathlib.Path("nomorals/agents/partner/runtime_media.py").read_text()
    assert "def _control_shorts" in src2


def test_contentops_package_imports():
    import nomorals.media.contentops as co

    assert co is not None
    from nomorals.media.contentops.niches import list_niches

    assert len(list_niches()) >= 7
