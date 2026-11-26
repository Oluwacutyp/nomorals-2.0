"""English Arpabet prediction — the learned GRU rung (§3C.2).

Ports echelon/grapheme_to_phoneme.rs (BSD-4-Clause, Brandon Thomas
2020; Kyubyong g2p.py lineage) into Devon's English G2P:
``nomorals/media/g2p_learned/`` runs the shipped seq2seq GRU over
the bundled ``model.npz`` weights, and ``nomorals/media/g2p_en.py``
answers out-of-lexicon words from it — lexicon → learned → rules.

What is pinned here:
- the Rust test vectors, exactly (lib.rs ``#[cfg(test)]``):
  test → T EH1 S T; zelda → Z EH1 L D AH0;
  symphonia → S IH0 M F OW1 N IY0 AH0;
- determinism: repeated eval gives the same phonemes;
- the core keeps Rust semantics: no lowercasing (an uppercase
  letter maps to <unk>), unknown characters map to <unk>;
- the rung order: a lexicon word stays "lexicon", an OOV word is
  "learned", and the letter-to-sound rules still fire — with the
  old outputs — when the learned rung is unavailable;
- the weights on disk are the shapes the Rust model trained with
  (hidden 256 throughout; the Rust source comments saying 64/128
  are stale — the .npy headers say 256);
- wiring: the ``tonal_g2p`` tool is registered (786 tools), its
  new ``arpabet`` action answers with the source named, ``nm g2p``
  works from the terminal, and ``/g2p`` answers in chat.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.media.g2p_en import g2p_en, learned_status, phonemize
from nomorals.media.g2p_learned import (
    G2PLearnedError,
    get_model,
    model_status,
    predict_word,
)
from nomorals.media.g2p_learned.model import (
    GRAPHEMES,
    PHONEME_TABLE,
    Model,
)

# The Rust test vectors (src/lib.rs #[cfg(test)] predict_phonemes).
RUST_VECTORS = {
    "test": ["T", "EH1", "S", "T"],
    "zelda": ["Z", "EH1", "L", "D", "AH0"],
    "symphonia": ["S", "IH0", "M", "F", "OW1", "N", "IY0", "AH0"],
}


class TestRustVectors:
    def test_rust_test_vectors_exact(self):
        model = get_model()
        for word, expected in RUST_VECTORS.items():
            assert model.predict(word) == expected

    def test_repeated_eval_is_deterministic(self):
        model = get_model()
        for _ in range(10):
            assert model.predict("test") == RUST_VECTORS["test"]

    def test_vocab_tables_match_the_rust_source(self):
        assert GRAPHEMES[:3] == ("<pad>", "<unk>", "</s>")
        assert len(GRAPHEMES) == 29
        assert GRAPHEMES[3:] == tuple(chr(c) for c in range(ord("a"),
                                                        ord("z") + 1))
        assert len(PHONEME_TABLE) == 74
        assert PHONEME_TABLE[:4] == ("<pad>", "<unk>", "<s>", "</s>")
        assert PHONEME_TABLE[4] == "AA0"
        assert PHONEME_TABLE[-1] == "ZH"
        assert "UW" in PHONEME_TABLE  # the bare UW entry (g2p.py quirk)

    def test_model_shapes_are_256_hidden(self):
        status = model_status()
        assert status["available"] is True
        assert status["shapes"]["enc_emb"] == [29, 256]
        assert status["shapes"]["enc_w_ih"] == [768, 256]
        assert status["shapes"]["dec_emb"] == [74, 256]
        assert status["shapes"]["fc_w"] == [74, 256]
        assert status["shapes"]["fc_b"] == [74]


class TestCoreSemantics:
    def test_uppercase_maps_to_unk_like_the_rust(self):
        # The Rust core does not lowercase: "Test" encodes T as <unk>,
        # so no T phoneme comes out — the model reads E S T.
        assert get_model().predict("Test") == ["EH1", "S", "T"]

    def test_wrapper_normalises_then_predicts(self):
        assert predict_word("Test") == RUST_VECTORS["test"]
        assert predict_word("  TEST! ") == RUST_VECTORS["test"]

    def test_wrapper_refuses_nothing_pronounceable(self):
        assert predict_word("") == []
        assert predict_word("12345") == []

    def test_missing_model_file_refuses_by_name(self):
        with pytest.raises(G2PLearnedError) as exc:
            Model.load("/nonexistent/model.npz")
        assert exc.value.kind == "model_unavailable"


class TestRungOrder:
    def test_lexicon_word_stays_lexicon(self):
        result = g2p_en("hello")
        assert result.source == "lexicon"
        assert result.phonemes == ["HH", "AH0", "L", "OW1"]

    def test_oov_word_is_learned_now(self):
        result = g2p_en("blorp")
        assert result.source == "learned"
        assert result.phonemes == ["B", "L", "AO1", "R", "P"]

    def test_oov_digraph_word_is_learned(self):
        result = g2p_en("shplank")
        assert result.source == "learned"
        assert result.phonemes[0] == "S"  # the model hears S, not SH
        assert "NG" in result.phonemes

    def test_sentence_sources_per_word(self):
        words = phonemize("hello blorp world")
        assert [w.source for w in words] == ["lexicon", "learned",
                                              "lexicon"]

    def test_rules_still_fire_when_learned_is_down(self, monkeypatch):
        import nomorals.media.g2p_learned as learned

        def _down(word):
            raise G2PLearnedError("model_unavailable",
                                  "learned rung disabled for this test")

        monkeypatch.setattr(learned, "predict_word", _down)
        result = g2p_en("blorp")
        assert result.source == "rules"
        assert result.phonemes == ["B", "L", "AO1", "R", "P"]
        status = learned_status()
        assert status["available"] is True  # the model file is fine;
        # only this call path was disabled — status reports the file.

    def test_learned_status_reports_the_bundle(self):
        status = learned_status()
        assert status["available"] is True
        assert status["reason"] == "ok"
        assert status["arrays"] == 12


class TestWiring:
    def test_tool_count_is_786_after_g2p_learned(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # 781 after this lane (B.27) added vtuber. 782 after this
        # lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet).
        assert len(registry._tools) == 800
        assert "tonal_g2p" in registry._tools

    def test_tool_tonal_g2p_arpabet_action(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        outcome = registry.call("tonal_g2p", action="arpabet",
                                word="blorp")
        assert getattr(outcome, "ok", False), outcome
        value = outcome.value
        assert value["phonemes"] == ["B", "L", "AO1", "R", "P"]
        assert value["source"] == "learned"
        assert value["learned"]["available"] is True

    def test_tool_tonal_g2p_unknown_action(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        outcome = registry.call("tonal_g2p", action="bogus")
        assert not getattr(outcome, "ok", True)

    def test_cli_g2p_alias(self, capsys):
        from nomorals.cmdline.dispatch import main

        assert main(["g2p", "pronounce", "blorp"]) == 0
        out = capsys.readouterr().out
        assert "B L AO1 R P" in out
        assert "learned" in out
        assert main(["gp", "status"]) == 0
        assert "learned" in capsys.readouterr().out

    def test_chat_g2p(self):
        from nomorals.agents.partner.runtime_voice import (
            RuntimeVoiceMixin,
        )

        mixin = RuntimeVoiceMixin()
        reply = mixin._control_g2p("blorp", "test")
        assert "B L AO1 R P" in reply
        assert "learned" in reply
        assert "g2p status:" in mixin._control_g2p("status", "test")
        assert "/g2p <word>" in mixin._control_g2p("", "test")


if __name__ == "__main__":
    import unittest

    unittest.main()
