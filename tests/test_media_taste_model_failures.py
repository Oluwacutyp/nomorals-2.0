"""Pass tail-115 pins for nomorals/media/taste_model.py — 0 converts / 3 exclusions.

64 / 68 (candidate_features): a non-numeric bpm / energy omits just
that one feature; every other feature is still extracted and nothing
is invented in its place. Typed per-feature guards. Flagged-accepted
(no noqa; both stay in the flagged scan).

150 (TasteModel.load): ANY load failure — missing file, corrupt JSON,
one non-numeric weight value — returns a fresh empty model. A missing
file is the designed cold start ("scores everything 0.5 honestly").
BUT the registered tool handler runs load -> feedback -> save in one
call, so on a corrupt file the next feedback save rewrites the file
with only the new feedback: byte-proven below, one bad weight value
destroys every healthy learned weight and the accept/reject counts.
That is the load-then-rewrite data-loss shape — NOT a batch convert;
it is pinned here and ranked on the targeted-fix shelf (repair:
distinguish "unreadable" from "absent" before any rewrite proceeds).

No except handlers appear in this file (AST-checked).
"""
import json

from nomorals.media.taste_model import TasteModel, candidate_features


def test_64_bad_bpm_omits_only_the_bpm_feature():
    feats = candidate_features({"bpm": "fast", "genre": "jazz"})
    assert not any(f.startswith("bpm:") for f in feats)
    assert "genre:jazz" in feats


def test_64_healthy_bpm_control():
    assert "bpm:fast" in candidate_features({"bpm": 120})


def test_68_bad_energy_omits_only_the_energy_feature():
    feats = candidate_features({"energy": "high"})
    assert not any(f.startswith("energy:") for f in feats)


def test_68_healthy_energy_control():
    assert "energy:high" in candidate_features({"energy": 0.9})


def test_150_missing_file_is_the_designed_cold_start(tmp_path):
    m = TasteModel.load(str(tmp_path / "none.json"))
    assert m.weights == {} and m.accepts == 0 and m.rejects == 0


def test_150_healthy_file_loads(tmp_path):
    p = tmp_path / "tm.json"
    p.write_text(json.dumps({"weights": {"genre:jazz": 1.5},
                             "accepts": 5, "rejects": 2}))
    m = TasteModel.load(str(p))
    assert m.accepts == 5 and m.rejects == 2
    assert m.weights.get("genre:jazz") == 1.5


def test_150_corrupt_json_starts_fresh(tmp_path):
    p = tmp_path / "tm.json"
    p.write_text("{corrupt json")
    m = TasteModel.load(str(p))
    assert m.weights == {} and m.accepts == 0


def test_150_one_bad_weight_value_poisons_the_whole_load(tmp_path):
    p = tmp_path / "tm.json"
    p.write_text(json.dumps({"weights": {"genre:jazz": 2.0,
                                         "genre:bad": "xyz"},
                             "accepts": 5, "rejects": 1}))
    m = TasteModel.load(str(p))
    assert m.weights == {} and m.accepts == 0 and m.rejects == 0


def test_150_wipe_proof_feedback_save_rewrites_over_corruption(tmp_path):
    # the tool handler's exact shape: load -> feedback -> save
    p = tmp_path / "tm.json"
    p.write_text(json.dumps({"weights": {"genre:jazz": 2.0,
                                         "genre:bad": "xyz"},
                             "accepts": 5, "rejects": 1}))
    m = TasteModel.load(str(p))
    m.feedback({"genre": "jazz"}, True)
    m.save(str(p))
    after = json.loads(p.read_text())
    # the learned 2.0 weight and the 5/1 counts are gone — the file
    # holds only the single new feedback, learned from scratch (0.35)
    assert after["accepts"] == 1 and after["rejects"] == 0
    assert after["weights"].get("genre:jazz") == 0.35
