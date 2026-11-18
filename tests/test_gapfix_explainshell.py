"""ExplainShell lane (52-websites #43): the shell-command explainer.

Paste a shell command; get each fragment annotated with the exact
help text for that fragment — grounded in the man pages of the
binaries actually installed on the host, never improvised. A
fragment with no grounded doc is marked unknown, never filled in
silently. Devon explains commands *before* running them in the
sandbox (explain_and_run), which is the workflow ExplainShell
can't join. The organ itself never executes anything.

Failing-first contract: every test here imports the new names lazily
(inside the test), so on the pristine base each fails with an
ImportError/AttributeError for the right reason, and after the build
each passes on real behaviour. No test touches the network: the
--help and tldr rungs always use injected fake fetchers, and the
man corpus is always built from fixture roff pages in a temp man
root — except one real-page test, skipped when the host has no tar
man page.
"""

from __future__ import annotations

import argparse
import gzip
import time

# ── fixture roff pages ────────────────────────────────────────────────

TAR_ROFF = """\\
.TH TAR 1 "July 11, 2022" "TAR" "GNU TAR Manual"
.SH NAME
tar \\- an archiving utility
.SH SYNOPSIS
\\fBtar\\fR {\\fBc\\fR|\\fBx\\fR} [\\fB\\-f\\fR \\fIARCHIVE\\fR]
.SH OPTIONS
.TP
\\fB\\-c\\fR, \\fB\\-\\-create\\fR
Create a new archive.
.TP
\\fB\\-x\\fR, \\fB\\-\\-extract\\fR, \\fB\\-\\-get\\fR
Extract files from an archive.
.TP
\\fB\\-z\\fR, \\fB\\-\\-gzip\\fR, \\fB\\-\\-gunzip\\fR, \\fB\\-\\-ungzip\\fR
Filter the archive through gzip.
.TP
\\fB\\-f\\fR, \\fB\\-\\-file\\fR=\\fIARCHIVE\\fR
Use archive file or device ARCHIVE.
.TP
\\fB\\-v\\fR, \\fB\\-\\-verbose\\fR
Verbosely list files processed.
.TP
\\fB\\-\\-force\\-local\\fR
Do not interpret file names starting with a colon as remote.
.SH DESCRIPTION
GNU tar is an archiving program.
"""

# Real find.1 shapes: single-dash multi-letter options under
# .SH EXPRESSION, .IP terms with lowercase placeholders.
FIND_ROFF = """\\
.TH FIND 1
.SH NAME
find \\- search for files in a directory hierarchy
.SH OPTIONS
.TP
.B \\-H
Do not follow symbolic links.
.SH EXPRESSION
.IP "\\-name \\fIpattern\\fR"
Base of file name matches shell pattern.
.IP "\\-mtime \\fIn\\fR"
File's data was last modified n*24 hours ago.
.IP "\\-type \\fIc\\fR"
File is of type c.
"""

FROB_ROFF = """\\
.TH FROB 1 "January 1, 2024" "FROB" "Frob Manual"
.SH NAME
frob \\- frobnicate a widget
.SH DESCRIPTION
The frob command frobs widgets. It has no OPTIONS section at all.
.SH OPTIONS
.IP "\\fB\\-\\-wiggle\\fR"
Wiggle the widget while frobnicating.
"""

IPTABLES_ROFF = """\\
.TH IPTABLES 8
.SH NAME
iptables \\- administration tool for IPv4/IPv6 packet filtering and NAT
.SH OPTIONS
.TP
.B \\-A, \\-\\-append \\fIchain\\fR
Append one or more rules to the end of the selected chain.
.TP
.B \\-p, \\-\\-protocol \\fIprotocol\\fR
The protocol of the rule or of the packet to check.
.TP
.B \\-j, \\-\\-jump \\fItarget\\fR
This specifies the target of the rule.
"""

# Real iptables shape: the tcp match options live on the sibling
# page iptables-extensions(8), not iptables(8).
IPTABLES_EXT_ROFF = """\\
.TH IPTABLES-EXTENSIONS 8
.SH NAME
iptables-extensions \\- list of extensions in the standard iptables distribution
.SH DESCRIPTION
.SS tcp
.TP
.B \\-\\-dport, \\-\\-destination\\-port \\fIport\\fR
Destination port or port range specification.
.TP
.B \\-\\-sport, \\-\\-source\\-port \\fIport\\fR
Source port or port range specification.
"""

GREP_ROFF = """\\
.TH GREP 1 "2023-01-01" "GREP" "User Commands"
.SH NAME
grep \\- print lines that match patterns
.SH OPTIONS
.TP
\\fB\\-i\\fR, \\fB\\-\\-ignore\\-case\\fR
Ignore case distinctions in patterns and input data.
.TP
\\fB\\-r\\fR, \\fB\\-\\-recursive\\fR
Read all files under each directory, recursively.
.TP
\\fB\\-e\\fR \\fIPATTERNS\\fR, \\fB\\-\\-regexp\\fR=\\fIPATTERNS\\fR
Use PATTERNS as the patterns.
.TP
.B \\-\\^--help
Output a usage message and exit.
.TP
.BR \\-l ", " \\-\\^--files-with-matches
Suppress prefixing of file names on output.
.TP
.BR \\-q ", " \\-\\^--quiet ", " \\-\\^--silent
Quiet; do not write anything to standard output.
"""

HELP_TEXT = """\\
Usage: frobnicate [OPTION]... [FILE]...
Frobnicate widgets.

  -w, --wiggle      wiggle the widget while frobnicating
  -q, --quiet       suppress all normal output
      --help        display this help and exit
"""

TLDR_MD = """\\
# frobnicate

> Frobnicate a widget.
> More information: <https://example.invalid/frobnicate>.

- Wiggle the widget:

`frobnicate --wiggle {{widget}}`

- Suppress all normal output:

`frobnicate --quiet {{widget}}`
"""


def _write_man(root, name, section, text):
    """Write one gzipped roff page into a fixture man root."""
    from pathlib import Path

    d = Path(root) / f"man{section}"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"{name}.{section}.gz", "wt",
                   encoding="utf-8") as fh:
        fh.write(text)


def _fixture_root(tmp_path):
    _write_man(tmp_path, "tar", 1, TAR_ROFF)
    _write_man(tmp_path, "frob", 1, FROB_ROFF)
    _write_man(tmp_path, "find", 1, FIND_ROFF)
    _write_man(tmp_path, "grep", 1, GREP_ROFF)
    _write_man(tmp_path, "iptables", 8, IPTABLES_ROFF)
    _write_man(tmp_path, "iptables-extensions", 8, IPTABLES_EXT_ROFF)
    _write_man(tmp_path, "grep", 1, GREP_ROFF)
    return str(tmp_path)


class _FakeCtx:
    """Minimal context: settings with a home_path, nothing else."""

    def __init__(self, home):
        class _Settings:
            pass

        self.settings = _Settings()
        self.settings.home_path = str(home)
        self.settings.home = str(home)


class _FakeRegistry:
    """Records the tool specs ``register()`` declares."""

    context = None

    def __init__(self):
        self.tools = {}
        self.calls = []

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco

    def call(self, name, **kwargs):
        self.calls.append((name, kwargs))
        from nomorals.core.result import Err
        return Err(Exception(f"unknown tool {name}"))


class _FakeOk:
    def __init__(self, value):
        self.ok = True
        self.value = value


# ── the bash parser ───────────────────────────────────────────────────

class TestParser:
    def test_simple_command_roles(self):
        from nomorals.codews.explain import parse_bash

        frags = parse_bash("tar -czf a.tgz dir")
        by_token = {f.token: f for f in frags}
        assert by_token["tar"].role == "command"
        assert by_token["-czf"].role == "option"
        assert by_token["a.tgz"].role == "argument"
        assert by_token["dir"].role == "argument"
        # Spans are real character offsets into the command.
        assert by_token["tar"].span == (0, 3)
        assert by_token["a.tgz"].span == (9, 14)

    def test_pipe_and_redirect(self):
        from nomorals.codews.explain import parse_bash

        frags = parse_bash("find . -name '*.py' | grep -i test > out.txt")
        roles = [(f.token, f.role) for f in frags]
        assert ("find", "command") in roles
        assert ("grep", "command") in roles
        assert ("|", "pipe") in roles
        assert (">", "redirect") in roles
        assert ("out.txt", "argument") in roles
        # The quoted pattern stays one token.
        assert any(f.token == "*.py" for f in frags)

    def test_separators_and_prefix(self):
        from nomorals.codews.explain import parse_bash

        frags = parse_bash("cd /tmp && sudo apt-get update; ls")
        roles = [(f.token, f.role) for f in frags]
        assert ("&&", "separator") in roles
        assert (";", "separator") in roles
        assert ("sudo", "prefix") in roles
        assert ("apt-get", "command") in roles
        assert ("cd", "command") in roles

    def test_substitution_is_a_fragment(self):
        from nomorals.codews.explain import parse_bash

        frags = parse_bash("echo $(date +%Y)")
        roles = [(f.token, f.role) for f in frags]
        assert any(r == "substitution" for _, r in roles)
        assert ("date", "command") in roles

    def test_unterminated_quote_is_unparsed_not_fatal(self):
        from nomorals.codews.explain import parse_bash

        frags = parse_bash("grep 'foo bar")
        assert any(f.role == "unparsed" for f in frags)

    def test_empty_command_refused(self):
        from nomorals.codews.explain import explain_command

        out = explain_command("/nonexistent-home", "   ")
        assert out["ok"] is False
        assert out["fragments"] == []


# ── the roff man-page parser ─────────────────────────────────────────

class TestRoff:
    def test_tar_page_parses(self, tmp_path):
        from nomorals.codews.man_corpus import parse_man_page_text

        page = parse_man_page_text(TAR_ROFF, name="tar", section=1)
        assert "archiving utility" in page.summary
        assert "--create" in page.options
        assert "-c" in page.options
        assert "Create a new archive." in page.options["--create"]["excerpt"]
        assert page.options["--file"]["takes_value"] is True
        assert page.options["--create"]["takes_value"] is False
        assert "-z" in page.options
        assert "gzip" in page.options["--gzip"]["excerpt"].lower()

    def test_ip_style_options_parse(self):
        from nomorals.codews.man_corpus import parse_man_page_text

        page = parse_man_page_text(FROB_ROFF, name="frob", section=1)
        assert "frobnicate a widget" in page.summary
        assert "--wiggle" in page.options
        assert "Wiggle the widget" in page.options["--wiggle"]["excerpt"]

    def test_caret_escape_recovers_long_flags(self):
        # Real grep.1 writes long flags as \-\^--long (groff
        # zero-width break). The \^ must not eat the flag.
        from nomorals.codews.man_corpus import parse_man_page_text

        page = parse_man_page_text(GREP_ROFF, name="grep", section=1)
        assert "--help" in page.options
        assert "--files-with-matches" in page.options
        assert "-l" in page.options
        assert "--quiet" in page.options
        assert "--silent" in page.options
        assert "-q" in page.options
        assert "-l" in page.options["--files-with-matches"]["aliases"]

    def test_br_term_and_continuation(self):
        # .BR is a font macro, not a paragraph break: a .BR term
        # line after .TP is the term, and a .BR line inside a
        # description is continuation text (real tar.1 splits
        # "Filter the archive through gzip." across such lines).
        from nomorals.codews.man_corpus import parse_man_page_text

        text = (".TH GZIP 1\n.SH NAME\ngzip \\- compress files\n"
                ".SH OPTIONS\n.TP\n.B \\-z, \\-\\-gzip\n"
                "Filter the archive through\n.BR gzip (1).\n")
        page = parse_man_page_text(text, name="gzip", section=1)
        assert "-z" in page.options
        assert "gzip (1)" in page.options["-z"]["excerpt"]

    def test_single_dash_long_options_parse(self):
        from nomorals.codews.man_corpus import parse_man_page_text

        page = parse_man_page_text(FIND_ROFF, name="find", section=1)
        assert "-name" in page.options
        assert page.options["-name"]["takes_value"] is True
        assert page.options["-name"]["value_name"] == "pattern"
        assert "-mtime" in page.options
        assert page.options["-mtime"]["takes_value"] is True
        assert "-type" in page.options
        assert "-H" in page.options

    def test_page_without_options_has_description(self):
        from nomorals.codews.man_corpus import parse_man_page_text

        text = (".TH NOTHING 1\n.SH NAME\nnothing \\- does nothing\n"
                ".SH DESCRIPTION\nIt does nothing at all.\n")
        page = parse_man_page_text(text, name="nothing", section=1)
        assert page.options == {}
        assert "does nothing" in page.description

    def test_malformed_roff_never_raises(self):
        from nomorals.codews.man_corpus import parse_man_page_text

        page = parse_man_page_text("garbage with no macros at all\n"
                                   ".SH\n.TP\n\\fB", name="junk",
                                   section=1)
        assert page.name == "junk"
        assert isinstance(page.options, dict)


# ── the man corpus ────────────────────────────────────────────────────

class TestCorpus:
    def test_ingest_and_lookup(self, tmp_path):
        from nomorals.codews.man_corpus import ManCorpus

        corpus = ManCorpus(str(tmp_path / "corpus.sqlite"))
        stats = corpus.ingest([_fixture_root(tmp_path / "man")])
        assert stats["pages"] >= 3
        assert stats["options"] >= 10
        page = corpus.get("tar")
        assert page is not None
        assert "--create" in page.options
        assert corpus.get("no-such-command") is None
        corpus.close()

    def test_fts_search_finds_by_description(self, tmp_path):
        from nomorals.codews.man_corpus import ManCorpus

        corpus = ManCorpus(str(tmp_path / "corpus.sqlite"))
        corpus.ingest([_fixture_root(tmp_path / "man")])
        hits = corpus.search("archiving utility")
        assert any(h["name"] == "tar" for h in hits)
        corpus.close()

    def test_ingest_is_incremental(self, tmp_path):
        from nomorals.codews.man_corpus import ManCorpus

        root = _fixture_root(tmp_path / "man")
        corpus = ManCorpus(str(tmp_path / "corpus.sqlite"))
        first = corpus.ingest([root])
        second = corpus.ingest([root])
        assert second["pages"] == 0  # nothing changed: nothing re-done
        assert first["pages"] == 6
        corpus.close()

    def test_status_reports_corpus(self, tmp_path):
        from nomorals.codews.man_corpus import ManCorpus

        corpus = ManCorpus(str(tmp_path / "corpus.sqlite"))
        before = corpus.status()
        assert before["pages"] == 0
        corpus.ingest([_fixture_root(tmp_path / "man")])
        after = corpus.status()
        assert after["pages"] == 6
        assert after["options"] >= 10
        assert after["roots"]
        corpus.close()

    def test_missing_root_is_honest(self, tmp_path):
        from nomorals.codews.man_corpus import ManCorpus

        corpus = ManCorpus(str(tmp_path / "corpus.sqlite"))
        stats = corpus.ingest(["/nonexistent/man/root"])
        assert stats["pages"] == 0
        assert stats["errors"] == []
        corpus.close()

    def test_real_tar_page_parses(self):
        import os

        import pytest

        path = "/usr/share/man/man1/tar.1.gz"
        if not os.path.exists(path):
            pytest.skip("no tar man page on this host")
        from nomorals.codews.man_corpus import parse_man_page_file

        page = parse_man_page_file(path)
        assert page is not None
        assert page.name == "tar"
        for flag in ("-c", "--create", "-z", "--gzip", "-f", "--file",
                     "-x", "--extract"):
            assert flag in page.options, f"real tar page lacks {flag}"


# ── the explainer ─────────────────────────────────────────────────────

def _explain(tmp_path, command, store=None, **kw):
    """Explain against the fixture corpus, pre-ingested."""
    from nomorals.codews.explain import explain_command
    from nomorals.codews.man_corpus import ManCorpus

    corpus = ManCorpus(str(tmp_path / "corpus.sqlite"))
    corpus.ingest([_fixture_root(tmp_path / "man")])
    try:
        return explain_command(str(tmp_path), command, corpus=corpus,
                               store=store, **kw)
    finally:
        corpus.close()


class TestExplain:
    def test_tar_czf_fully_grounded(self, tmp_path):
        out = _explain(tmp_path, "tar -czf a.tgz dir")
        assert out["ok"] is True
        assert out["unknown_count"] == 0
        by_token = {f["token"]: f for f in out["fragments"]}
        assert "Create a new archive" in by_token["-c"]["doc"]["excerpt"]
        assert "gzip" in by_token["-z"]["doc"]["excerpt"].lower()
        assert "archive file" in by_token["-f"]["doc"]["excerpt"].lower()
        assert by_token["-f"]["doc"]["source"] == "man"
        assert by_token["a.tgz"]["detail"] == "value of -f"
        assert by_token["tar"]["doc"]["source"] == "man"

    def test_mutation_changes_the_explanation(self, tmp_path):
        create = _explain(tmp_path, "tar -czf a.tgz dir")
        extract = _explain(tmp_path, "tar -xzf a.tgz")
        c_text = [f for f in create["fragments"]
                  if f["token"] == "-c"][0]["doc"]["excerpt"]
        x_text = [f for f in extract["fragments"]
                  if f["token"] == "-x"][0]["doc"]["excerpt"]
        assert c_text != x_text
        assert "Create" in c_text
        assert "Extract" in x_text

    def test_unknown_command_is_marked_never_invented(self, tmp_path):
        out = _explain(tmp_path, "zqxwv --blorp file")
        assert out["ok"] is True
        by_token = {f["token"]: f for f in out["fragments"]}
        assert by_token["zqxwv"]["unknown"] is True
        assert by_token["zqxwv"]["doc"] is None
        assert out["unknown_count"] >= 1
        assert "no man page" in out["summary"].lower() or \
            "unknown" in out["summary"].lower()

    def test_unknown_option_suggests(self, tmp_path):
        out = _explain(tmp_path, "tar --crat a.tgz")
        frag = [f for f in out["fragments"]
                if f["token"] == "--crat"][0]
        assert frag["unknown"] is True
        assert frag.get("suggestion") == "--create"

    def test_unambiguous_abbreviation_is_named(self, tmp_path):
        # GNU tools accept an unambiguous long-flag prefix; the match
        # is grounded but named as an abbreviation, never silent.
        out = _explain(tmp_path, "tar --creat a.tgz")
        frag = [f for f in out["fragments"]
                if f["token"] == "--creat"][0]
        assert frag["unknown"] is False
        assert "abbreviation of --create" in frag["doc"].get("note", "")
        assert "Create a new archive" in frag["doc"]["excerpt"]

    def test_builtin_has_a_doc(self, tmp_path):
        out = _explain(tmp_path, "cd /tmp")
        frag = [f for f in out["fragments"] if f["token"] == "cd"][0]
        assert frag["unknown"] is False
        assert frag["doc"]["source"] == "builtin"

    def test_help_rung_grounds_unknown_command(self, tmp_path):
        out = _explain(tmp_path, "frobnicate --wiggle w",
                       help_fetcher=lambda cmd: HELP_TEXT)
        by_token = {f["token"]: f for f in out["fragments"]}
        assert by_token["--wiggle"]["doc"]["source"] == "help"
        assert "wiggle" in by_token["--wiggle"]["doc"]["excerpt"].lower()

    def test_tldr_rung_grounds_unknown_command(self, tmp_path):
        out = _explain(tmp_path, "frobnicate --quiet w",
                       tldr_fetcher=lambda cmd: TLDR_MD)
        frag = [f for f in out["fragments"]
                if f["token"] == "--quiet"][0]
        assert frag["doc"]["source"] == "tldr"
        assert "Suppress" in frag["doc"]["excerpt"]

    def test_single_dash_long_option_matches_whole(self, tmp_path):
        # find -name is ONE option, not the bundle -n -a -m -e;
        # -mtime -7: the -7 is the value of -mtime, not an option.
        out = _explain(tmp_path, "find /x -name '*.log' -mtime -7")
        by_token = {f["token"]: f for f in out["fragments"]}
        assert by_token["-name"]["unknown"] is False
        assert by_token["-name"]["doc"]["source"] == "man"
        assert by_token["*.log"]["detail"] == "value of -name"
        assert by_token["-mtime"]["unknown"] is False
        assert by_token["-7"]["detail"] == "value of -mtime"
        assert "-n" not in by_token and "-a" not in by_token

    def test_iptables_dport_grounded_from_sibling_page(self, tmp_path):
        # --dport is documented on iptables-extensions(8), not
        # iptables(8) — the explainer must find it there, not mark
        # it unknown.
        from nomorals.codews.man_corpus import ManCorpus

        corpus = ManCorpus(str(tmp_path / "corpus.sqlite"))
        corpus.ingest([_fixture_root(tmp_path / "man")])
        try:
            page, rec = corpus.find_option("--dport", "iptables")
            assert page == "iptables-extensions(8)"
            assert "Destination port" in rec["excerpt"]
            assert corpus.find_option("--no-such-flag",
                                      "iptables") == (None, None)
        finally:
            corpus.close()
        out = _explain(tmp_path,
                       "iptables -A INPUT -p tcp --dport 22 -j ACCEPT")
        by_token = {f["token"]: f for f in out["fragments"]}
        assert by_token["--dport"]["unknown"] is False
        assert by_token["--dport"]["doc"]["source"] == "man"
        assert by_token["--dport"]["doc"]["page"] == \
            "iptables-extensions(8)"
        assert by_token["22"]["detail"] == "value of --dport"

    def test_grep_long_flags_grounded(self, tmp_path):
        out = _explain(tmp_path, "grep -l pattern file")
        by_token = {f["token"]: f for f in out["fragments"]}
        assert by_token["-l"]["unknown"] is False
        assert by_token["-l"]["doc"]["source"] == "man"
        out = _explain(tmp_path, "grep --files-with-matches pattern file")
        by_token = {f["token"]: f for f in out["fragments"]}
        assert by_token["--files-with-matches"]["unknown"] is False

    def test_ip_option_grounded_from_man(self, tmp_path):
        out = _explain(tmp_path, "frob --wiggle w")
        frag = [f for f in out["fragments"]
                if f["token"] == "--wiggle"][0]
        # The fixture frob page has --wiggle via .IP, so this must be
        # grounded from the man corpus, not invented.
        assert frag["unknown"] is False
        assert frag["doc"]["source"] == "man"

    def test_store_round_trip(self, tmp_path):
        from nomorals.codews.explain import open_explanation_store

        store = open_explanation_store(str(tmp_path))
        first = _explain(tmp_path, "tar -czf a.tgz dir", store=store)
        second = _explain(tmp_path, "grep -ri foo .", store=store)
        assert len(store.list()) == 2
        got = store.get(first["explanation_id"])
        assert got["command"] == "tar -czf a.tgz dir"
        assert store.get(first["explanation_id"][:8])["command"] == \
            "tar -czf a.tgz dir"
        assert store.get("no-such-id") is None
        assert second["explanation_id"] != first["explanation_id"]

    def test_reply_names_excerpts_and_unknowns(self, tmp_path):
        from nomorals.codews.explain import format_explanation_reply

        out = _explain(tmp_path, "tar -czf a.tgz dir && zqxwv")
        reply = format_explanation_reply(out)
        assert "tar -czf a.tgz dir && zqxwv" in reply
        assert "Create a new archive" in reply
        assert "unknown" in reply.lower()

    def test_long_command_refused(self, tmp_path):
        from nomorals.codews.explain import explain_command

        out = explain_command(str(tmp_path), "x" * 5000)
        assert out["ok"] is False
        assert "too long" in out["error"].lower()

    def test_deep_nesting_finishes(self, tmp_path):
        cmd = "echo " + "$(echo " * 25 + "x" + ")" * 25
        started = time.monotonic()
        out = _explain(tmp_path, cmd)
        assert time.monotonic() - started < 5.0
        assert out["ok"] is True


# ── the brain tool ────────────────────────────────────────────────────

class TestTool:
    def test_tool_registered_and_explains(self, tmp_path):
        from nomorals.tools import explain_command as explain_tool

        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))
        explain_tool.register(registry)
        assert "explain_command" in registry.tools
        fn = registry.tools["explain_command"]
        ingested = fn(action="ingest",
                      man_roots=_fixture_root(tmp_path / "man"))
        assert ingested["ok"] is True
        assert ingested["pages"] >= 3
        out = fn(command="tar -czf a.tgz dir")
        assert out["ok"] is True
        assert out["unknown_count"] == 0
        assert "Create a new archive" in out["reply"]

    def test_tool_status_and_list(self, tmp_path):
        from nomorals.tools import explain_command as explain_tool

        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))
        explain_tool.register(registry)
        fn = registry.tools["explain_command"]
        assert fn(action="status")["ok"] is True
        fn(action="ingest", man_roots=_fixture_root(tmp_path / "man"))
        fn(command="tar -czf a.tgz dir")
        assert len(fn(action="list")["explanations"]) == 1

    def test_explain_and_run_runs_after_explaining(self, tmp_path):
        from nomorals.tools import explain_command as explain_tool

        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))

        def _call(name, **kwargs):
            registry.calls.append((name, kwargs))
            return _FakeOk({"exit_code": 0, "stdout": "ran\n",
                            "stderr": "", "backend": "rlimit"})

        registry.call = _call
        explain_tool.register(registry)
        fn = registry.tools["explain_command"]
        fn(action="ingest", man_roots=_fixture_root(tmp_path / "man"))
        out = fn(command="tar -czf a.tgz dir", action="explain_and_run")
        assert out["ok"] is True
        assert "Create a new archive" in out["reply"]
        assert out["run"]["exit_code"] == 0
        assert any(c[0] == "shell_run" for c in registry.calls)

    def test_explain_and_run_never_runs_a_bad_command(self, tmp_path):
        from nomorals.tools import explain_command as explain_tool

        registry = _FakeRegistry()
        registry.context = _FakeCtx(str(tmp_path))
        registry.call = lambda name, **kw: registry.calls.append(
            (name, kw)) or _FakeOk({})
        explain_tool.register(registry)
        fn = registry.tools["explain_command"]
        out = fn(command="   ", action="explain_and_run")
        assert out["ok"] is False
        assert not any(c[0] == "shell_run" for c in registry.calls)


# ── the CLI ───────────────────────────────────────────────────────────

class TestCli:
    def _context(self, tmp_path):
        return _FakeCtx(str(tmp_path))

    def test_nm_explain_round_trip(self, tmp_path, capsys):
        from nomorals.cmdline.commands.explain import _cmd_explain

        root = _fixture_root(tmp_path / "man")
        args = argparse.Namespace(explain_query=["ingest"], json=False,
                                  man_root=root)
        assert _cmd_explain(args, self._context(tmp_path)) == 0
        args = argparse.Namespace(
            explain_query=["tar", "-czf", "a.tgz", "dir"], json=False,
            man_root=None)
        assert _cmd_explain(args, self._context(tmp_path)) == 0
        assert "Create a new archive" in capsys.readouterr().out

    def test_nm_explain_no_args_is_usage(self, tmp_path):
        from nomorals.cmdline.commands.explain import _cmd_explain

        args = argparse.Namespace(explain_query=[], json=False,
                                  man_root=None)
        assert _cmd_explain(args, self._context(tmp_path)) == 2

    def test_nm_explain_list(self, tmp_path, capsys):
        from nomorals.cmdline.commands.explain import _cmd_explain

        root = _fixture_root(tmp_path / "man")
        ctx = self._context(tmp_path)
        _cmd_explain(argparse.Namespace(explain_query=["ingest"],
                                        json=False, man_root=root), ctx)
        _cmd_explain(argparse.Namespace(
            explain_query=["tar", "-czf", "a.tgz", "dir"], json=False,
            man_root=None), ctx)
        args = argparse.Namespace(explain_query=["list"], json=False,
                                  man_root=None)
        assert _cmd_explain(args, ctx) == 0
        assert "tar -czf" in capsys.readouterr().out


# ── wiring pins ───────────────────────────────────────────────────────

class TestWiring:
    def test_layering_pin(self):
        from tests.test_layering import MODULE_PINS

        assert MODULE_PINS["nomorals.tools.explain_command"] == 5

    def test_public_command_count(self):
        from nomorals.social.chat.control import PUBLIC_COMMANDS

        assert "explain" in PUBLIC_COMMANDS
        assert len(PUBLIC_COMMANDS) == 277
