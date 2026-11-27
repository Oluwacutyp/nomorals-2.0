"""CodeBeautify family: parse -> validate -> transform with located errors.

Golden corpus per format pair — round-trip invariants
(beautify∘minify = identity on data, convert∘convert-back = equal
data), error-position pins on malformed fixtures.
"""

from __future__ import annotations

import pytest

JSON_DOC = '{"name": "devon", "tags": ["a", "b"], "nested": {"x": 1, "y": [true, null]}}'
JSON_PRETTY = """{
  "name": "devon",
  "tags": [
    "a",
    "b"
  ],
  "nested": {
    "x": 1,
    "y": [
      true,
      null
    ]
  }
}"""

YAML_DOC = "name: devon\ntags:\n  - a\n  - b\nnested:\n  x: 1\n"
TOML_DOC = 'name = "devon"\ncount = 3\n[owner]\nname = "death"\n'
XML_DOC = "<root><item id=\"1\">one</item><item id=\"2\">two</item></root>"
CSV_DOC = "name,age\nada,36\ngrace,28\n"
SQL_DOC = "select a, b from t where x = 1 and y = 2 order by a"
CSS_DOC = "a{color:red;margin:0}b .c{font-weight:bold}"
HTML_DOC = "<div><p>hello <b>world</b></p><br><img src=\"x.png\"></div>"


# ── JSON ──────────────────────────────────────────────────────────────

class TestJson:
    def test_beautify_exact(self):
        from nomorals.codews.textfmt import beautify

        out = beautify(JSON_DOC, "json")
        assert out.ok is True
        assert out.text == JSON_PRETTY + "\n"
        assert out.format_in == "json" and out.format_out == "json"
        assert out.diagnostics == []

    def test_minify_then_beautify_is_identity_on_data(self):
        import json

        from nomorals.codews.textfmt import beautify, minify

        mini = minify(JSON_DOC, "json")
        assert mini.ok is True
        assert mini.text == json.dumps(json.loads(JSON_DOC), separators=(",", ":"))
        back = beautify(mini.text, "json")
        assert json.loads(back.text) == json.loads(JSON_DOC)

    def test_validate_malformed_names_line_and_column(self):
        from nomorals.codews.textfmt import validate

        bad = '{\n  "a": 1,\n  "b": \n}'
        out = validate(bad, "json")
        assert out.ok is False
        assert len(out.diagnostics) == 1
        d = out.diagnostics[0]
        assert d.kind == "parse"
        assert d.line == 4  # json's own position: the } on line 4
        assert d.col == 1   # is where the missing value is discovered
        assert "Expecting value" in d.message

    def test_validate_valid_is_ok(self):
        from nomorals.codews.textfmt import validate

        out = validate(JSON_DOC, "json")
        assert out.ok is True
        assert out.diagnostics == []

    def test_empty_input_is_empty_not_parse(self):
        from nomorals.codews.textfmt import validate

        out = validate("   ", "json")
        assert out.ok is False
        assert out.diagnostics[0].kind == "empty"


class TestJsonl:
    def test_beautify_each_line(self):
        from nomorals.codews.textfmt import beautify

        out = beautify('{"a":1}\n{"b":2}', "jsonl")
        assert out.ok is True
        assert out.text == '{\n  "a": 1\n}\n{\n  "b": 2\n}\n'

    def test_bad_line_names_the_line(self):
        from nomorals.codews.textfmt import validate

        out = validate('{"a":1}\n{"b":}\n{"c":3}', "jsonl")
        assert out.ok is False
        assert out.diagnostics[0].line == 2
        assert out.diagnostics[0].kind == "parse"


# ── YAML / TOML ───────────────────────────────────────────────────────

class TestYaml:
    def test_beautify_and_round_trip(self):
        import json

        from nomorals.codews.textfmt import beautify, convert

        out = beautify(YAML_DOC, "yaml")
        assert out.ok is True
        assert "name: devon" in out.text
        back = convert(out.text, "yaml", "json")
        assert back.ok is True
        assert json.loads(back.text)["nested"]["x"] == 1

    def test_malformed_yaml_names_line(self):
        from nomorals.codews.textfmt import validate

        out = validate("a: 1\n b: [unclosed\n", "yaml")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"
        assert out.diagnostics[0].line >= 1

    def test_missing_yaml_is_honest(self, monkeypatch):
        import sys

        from nomorals.codews.textfmt import validate

        monkeypatch.setitem(sys.modules, "yaml", None)
        out = validate(YAML_DOC, "yaml")
        assert out.ok is False
        assert out.diagnostics[0].kind == "missing_dependency"
        assert "yaml" in out.diagnostics[0].message.lower()


class TestToml:
    def test_beautify_and_round_trip(self):
        from nomorals.codews.textfmt import beautify, convert

        out = beautify(TOML_DOC, "toml")
        assert out.ok is True
        assert 'name = "devon"' in out.text
        back = convert(out.text, "toml", "json")
        assert back.ok is True
        assert '"owner"' in back.text

    def test_malformed_toml_names_line(self):
        from nomorals.codews.textfmt import validate

        out = validate('name = \n', "toml")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"
        assert out.diagnostics[0].line == 1

    def test_null_cannot_be_toml(self):
        from nomorals.codews.textfmt import convert

        out = convert('{"a": null}', "json", "toml")
        assert out.ok is False
        assert "null" in out.diagnostics[0].message.lower()


# ── XML / CSV ─────────────────────────────────────────────────────────

class TestXml:
    def test_beautify_nests(self):
        from nomorals.codews.textfmt import beautify

        out = beautify(XML_DOC, "xml")
        assert out.ok is True
        assert out.text == (
            "<root>\n"
            "  <item id=\"1\">one</item>\n"
            "  <item id=\"2\">two</item>\n"
            "</root>\n"
        )

    def test_mismatched_tag_names_line(self):
        from nomorals.codews.textfmt import validate

        out = validate("<root>\n  <a>x</b>\n</root>", "xml")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"
        assert out.diagnostics[0].line == 2

    def test_minify_strips_indentation(self):
        from nomorals.codews.textfmt import minify

        out = minify("<root>\n  <a>x</a>\n</root>", "xml")
        assert out.ok is True
        assert out.text == "<root><a>x</a></root>"


class TestCsv:
    def test_parse_and_convert_to_json(self):
        import json

        from nomorals.codews.textfmt import convert

        out = convert(CSV_DOC, "csv", "json")
        assert out.ok is True
        rows = json.loads(out.text)
        assert rows == [{"name": "ada", "age": "36"}, {"name": "grace", "age": "28"}]

    def test_ragged_row_names_the_row(self):
        from nomorals.codews.textfmt import validate

        out = validate("a,b\n1,2,3\n4,5\n", "csv")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"
        assert out.diagnostics[0].line == 2


# ── SQL ───────────────────────────────────────────────────────────────

class TestSql:
    def test_beautify_clauses_per_line(self):
        from nomorals.codews.textfmt import beautify

        out = beautify(SQL_DOC, "sql")
        assert out.ok is True
        assert out.text == (
            "SELECT a, b\nFROM t\nWHERE x = 1 AND y = 2\nORDER BY a"
        )

    def test_unbalanced_paren_is_diagnostic(self):
        from nomorals.codews.textfmt import validate

        out = validate("select * from t where (x = 1", "sql")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"
        assert "paren" in out.diagnostics[0].message.lower()

    def test_unterminated_string_is_diagnostic(self):
        from nomorals.codews.textfmt import validate

        out = validate("select * from t where name = 'ada", "sql")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"

    def test_minify_collapses(self):
        from nomorals.codews.textfmt import minify

        out = minify("SELECT  a,\n  b\nFROM t", "sql")
        assert out.ok is True
        assert out.text == "SELECT a, b FROM t"


# ── CSS ───────────────────────────────────────────────────────────────

class TestCss:
    def test_beautify_rules(self):
        from nomorals.codews.textfmt import beautify

        out = beautify(CSS_DOC, "css")
        assert out.ok is True
        assert out.text == (
            "a {\n  color: red;\n  margin: 0;\n}\nb .c {\n  font-weight: bold;\n}"
        )

    def test_unbalanced_brace_is_diagnostic(self):
        from nomorals.codews.textfmt import validate

        out = validate("a { color: red;", "css")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"
        assert "brace" in out.diagnostics[0].message.lower()

    def test_minify(self):
        from nomorals.codews.textfmt import minify

        out = minify("a {\n  color: red;\n}", "css")
        assert out.ok is True
        assert out.text == "a{color:red}"


# ── HTML ──────────────────────────────────────────────────────────────

class TestHtml:
    def test_beautify_nests_and_voids(self):
        from nomorals.codews.textfmt import beautify

        out = beautify(HTML_DOC, "html")
        assert out.ok is True
        assert out.text == (
            "<div>\n"
            "  <p>\n"
            "    hello\n"
            "    <b>world</b>\n"
            "  </p>\n"
            "  <br>\n"
            "  <img src=\"x.png\">\n"
            "</div>"
        )

    def test_mismatched_close_names_line(self):
        from nomorals.codews.textfmt import validate

        out = validate("<div>\n  <p>x</div>\n", "html")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"
        assert out.diagnostics[0].line == 2

    def test_unclosed_tag_at_eof(self):
        from nomorals.codews.textfmt import validate

        out = validate("<div><p>x", "html")
        assert out.ok is False
        assert "p" in out.diagnostics[0].message


# ── JS (prettier ladder) ──────────────────────────────────────────────

class TestJs:
    def test_no_prettier_is_honest_refusal(self, monkeypatch):
        from nomorals.codews import textfmt

        monkeypatch.setattr(textfmt.shutil, "which", lambda name: None)
        out = textfmt.beautify("const x={a:1}", "js")
        assert out.ok is False
        assert out.diagnostics[0].kind == "missing_dependency"
        assert "prettier" in out.diagnostics[0].message.lower()

    def test_injected_prettier_rung_runs(self, tmp_path, monkeypatch):
        import stat

        from nomorals.codews import textfmt

        fake = tmp_path / "prettier"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "sys.stdout.write(sys.stdin.read().replace('{a:1}', '{ a: 1 }'))\n"
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(textfmt.shutil, "which",
                            lambda name: str(fake) if name == "prettier" else None)
        out = textfmt.beautify("const x={a:1}", "js")
        assert out.ok is True
        assert "{ a: 1 }" in out.text


# ── converters ───────────────────────────────────────────────────────

class TestConvert:
    def test_json_to_yaml_and_back_is_equal_data(self):
        import json

        from nomorals.codews.textfmt import convert

        there = convert(JSON_DOC, "json", "yaml")
        assert there.ok is True
        back = convert(there.text, "yaml", "json")
        assert back.ok is True
        assert json.loads(back.text) == json.loads(JSON_DOC)

    def test_json_to_toml_and_back(self):
        import json

        from nomorals.codews.textfmt import convert

        doc = '{"name": "devon", "count": 3, "owner": {"name": "death"}}'
        there = convert(doc, "json", "toml")
        assert there.ok is True
        back = convert(there.text, "toml", "json")
        assert json.loads(back.text) == json.loads(doc)

    def test_json_to_xml_and_back(self):
        import json

        from nomorals.codews.textfmt import convert

        there = convert('{"root": {"item": ["one", "two"]}}', "json", "xml")
        assert there.ok is True
        back = convert(there.text, "xml", "json")
        assert json.loads(back.text) == {"root": {"item": ["one", "two"]}}

    def test_json_to_jsonl_and_back(self):
        import json

        from nomorals.codews.textfmt import convert

        doc = '[{"a": 1}, {"a": 2}]'
        there = convert(doc, "json", "jsonl")
        assert there.ok is True
        back = convert(there.text, "jsonl", "json")
        assert json.loads(back.text) == json.loads(doc)

    def test_json_to_typescript(self):
        from nomorals.codews.textfmt import convert

        out = convert(
            '{"user": {"name": "ada", "age": 36, "tags": ["x"]}}',
            "json", "typescript")
        assert out.ok is True
        assert "interface User" in out.text
        assert "name: string" in out.text
        assert "age: number" in out.text
        assert "tags: string[]" in out.text

    def test_json_to_php(self):
        from nomorals.codews.textfmt import convert

        out = convert('{"name": "ada", "age": 36}', "json", "php")
        assert out.ok is True
        assert "class" in out.text
        assert "public string $name" in out.text
        assert "public int $age" in out.text

    def test_hex_utf8_both_ways(self):
        from nomorals.codews.textfmt import convert

        there = convert("hello", "utf8", "hex")
        assert there.ok is True
        assert there.text == "68656c6c6f"
        back = convert(there.text, "hex", "utf8")
        assert back.ok is True
        assert back.text == "hello"

    def test_bad_hex_names_position(self):
        from nomorals.codews.textfmt import convert

        out = convert("zz", "hex", "utf8")
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"

    def test_html_to_jsx_renames(self):
        from nomorals.codews.textfmt import convert

        out = convert(
            '<label for="n" class="big">name</label>', "html", "jsx")
        assert out.ok is True
        assert 'htmlFor="n"' in out.text
        assert 'className="big"' in out.text

    def test_unsupported_pair_is_refused_by_name(self):
        from nomorals.codews.textfmt import convert

        out = convert("select 1", "sql", "yaml")
        assert out.ok is False
        assert out.diagnostics[0].kind == "unsupported_format"
        assert "sql" in out.diagnostics[0].message
        assert "yaml" in out.diagnostics[0].message


# ── JSON fixer ────────────────────────────────────────────────────────

class TestFixJson:
    def test_repairs_trailing_comma_single_quotes_comments(self):
        import json

        from nomorals.codews.textfmt import fix_json

        broken = "{'a': 1, // note\n'b': [1, 2,], /* c */\n}"
        out = fix_json(broken)
        assert out.ok is True
        assert json.loads(out.text) == {"a": 1, "b": [1, 2]}
        kinds = {d.kind for d in out.diagnostics}
        assert "fixed" in kinds
        assert any("trailing comma" in d.message for d in out.diagnostics)

    def test_unfixable_stays_an_error_with_position(self):
        from nomorals.codews.textfmt import fix_json

        out = fix_json('{"a": ')
        assert out.ok is False
        assert out.diagnostics[0].kind == "parse"
        assert out.diagnostics[0].line == 1

    def test_already_valid_is_untouched(self):
        from nomorals.codews.textfmt import fix_json

        out = fix_json(JSON_DOC)
        assert out.ok is True
        assert out.diagnostics == []


# ── detection + result shape ──────────────────────────────────────────

class TestDetect:
    @pytest.mark.parametrize("text,fmt", [
        ('{"a": 1}', "json"),
        ('<root><a>x</a></root>', "xml"),
        ('name = "devon"\n', "toml"),
        ("name: devon\n", "yaml"),
        ("select a from t", "sql"),
        ("not a format at all {{{", "unknown"),
    ])
    def test_detect_format(self, text, fmt):
        from nomorals.codews.textfmt import detect_format

        assert detect_format(text) == fmt

    def test_result_round_trips(self):
        from nomorals.codews.textfmt import TransformResult, beautify

        out = beautify(JSON_DOC, "json")
        again = TransformResult.from_dict(out.to_dict())
        assert again == out


# ── brain tools ───────────────────────────────────────────────────────

class _FakeCtx:
    def __init__(self, home):
        class _Settings:
            home_path = home

        self.settings = _Settings()
        self.router = None
        self.tools = None


class _FakeRegistry:
    """Records the tool specs ``register()`` declares."""

    context = None

    def __init__(self):
        self.tools = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


class TestTools:
    def test_format_text_beautifies(self):
        from nomorals.tools import datatools

        registry = _FakeRegistry()
        registry.context = _FakeCtx("/tmp")
        datatools.register(registry)
        assert {"format_text", "validate_text", "convert_data"} <= set(
            registry.tools)
        out = registry.tools["format_text"](text='{"a":1}', format="json")
        assert out["ok"] is True
        assert out["text"] == '{\n  "a": 1\n}\n'

    def test_validate_text_reports_position(self):
        from nomorals.tools import datatools

        registry = _FakeRegistry()
        registry.context = _FakeCtx("/tmp")
        datatools.register(registry)
        out = registry.tools["validate_text"](text="<a><b></a>", format="xml")
        assert out["ok"] is False
        assert out["diagnostics"][0]["kind"] == "parse"
        assert out["diagnostics"][0]["line"] == 1

    def test_convert_data_yaml_to_toml(self):
        from nomorals.tools import datatools

        registry = _FakeRegistry()
        registry.context = _FakeCtx("/tmp")
        datatools.register(registry)
        out = registry.tools["convert_data"](
            text=YAML_DOC, from_format="yaml", to_format="toml")
        assert out["ok"] is True
        assert 'name = "devon"' in out["text"]

    def test_unknown_format_refused(self):
        from nomorals.tools import datatools

        registry = _FakeRegistry()
        registry.context = _FakeCtx("/tmp")
        datatools.register(registry)
        out = registry.tools["format_text"](text="x", format="frobnicate")
        assert out["ok"] is False
        assert "frobnicate" in out["diagnostics"][0]["message"]


# ── chat surface + CLI ────────────────────────────────────────────────

class TestChatAndCli:
    @staticmethod
    def _runtime(tmp_path):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        rt.settings = SimpleNamespace(home_path=str(tmp_path))
        return rt

    def test_control_fmt_auto_detects(self, tmp_path):
        rt = self._runtime(tmp_path)
        out = rt._control_fmt('{"a":1}')
        assert '"a": 1' in out

    def test_control_fmt_validate_bad_xml(self, tmp_path):
        rt = self._runtime(tmp_path)
        out = rt._control_fmt("validate xml <a><b></a>")
        assert "line 1" in out.lower()

    def test_control_fmt_convert(self, tmp_path):
        rt = self._runtime(tmp_path)
        out = rt._control_fmt("convert json yaml {\"a\": 1}")
        assert "a: 1" in out

    def test_control_fmt_fix(self, tmp_path):
        rt = self._runtime(tmp_path)
        out = rt._control_fmt("fix {'a': 1,}")
        assert '"a": 1' in out

    def test_control_fmt_empty_is_usage(self, tmp_path):
        rt = self._runtime(tmp_path)
        assert rt._control_fmt("").startswith("usage:")

    @staticmethod
    def _context(tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)))

    def test_cli_beautify_exits_zero(self, tmp_path, capsys):
        import argparse

        from nomorals.cmdline.commands.fmt import _cmd_fmt

        args = argparse.Namespace(
            fmt_query=['{"a":1}'], fmt_format=None, to_format=None,
            operation="beautify", json=False)
        assert _cmd_fmt(args, self._context(tmp_path)) == 0
        assert '"a": 1' in capsys.readouterr().out

    def test_cli_bad_json_exits_one(self, tmp_path, capsys):
        import argparse

        from nomorals.cmdline.commands.fmt import _cmd_fmt

        args = argparse.Namespace(
            fmt_query=['{"a": }'], fmt_format="json", to_format=None,
            operation="beautify", json=False)
        assert _cmd_fmt(args, self._context(tmp_path)) == 1

    def test_public_command_registered(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            PUBLIC_COMMANDS,
        )

        assert "fmt" in PUBLIC_COMMANDS
        assert CONTROL_COMMANDS["fmt"] == (1, None)
        assert "fmt" in COMMAND_DETAILS
        assert len(PUBLIC_COMMANDS) == 287
