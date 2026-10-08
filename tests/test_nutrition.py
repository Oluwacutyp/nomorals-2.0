"""Photo meal logging + 2 questions (build-map #54). All offline."""
import pytest

from nomorals.health import nutrition as N


class FakeSeer:
    def __init__(self, text: str = "", fail: bool = False):
        self.text = text
        self.fail = fail
        self.last_prompt = ""

    def see(self, path, question=""):
        self.last_prompt = question
        if self.fail:
            raise RuntimeError("no vision model available")
        return self.text


def _draft(seer_text="jollof rice | one plate\ndodo | a few pieces"):
    return N.log_meal("/tmp/meal.jpg", seer=FakeSeer(seer_text))


# ── identification ──────────────────────────────────────────────────────────

def test_identifies_nigerian_foods():
    d = _draft()
    assert d.ok
    names = [i.name for i in d.items]
    assert "jollof rice" in names
    assert "dodo" in names


def test_food_ranges_not_point_estimates():
    d = _draft("jollof rice | one plate")
    item = d.items[0]
    assert item.cal_low < item.cal_high
    assert item.cal_low > 0


def test_unknown_food_honest_estimate():
    d = _draft("mystery stew | one bowl")
    assert d.items[0].known is False
    assert d.items[0].cal_low < d.items[0].cal_high


def test_no_food_visible():
    d = _draft("NONE")
    assert not d.ok
    assert "no food" in d.error


def test_vision_failure_honest():
    d = N.log_meal("/tmp/meal.jpg", seer=FakeSeer(fail=True))
    assert not d.ok
    assert "vision unavailable" in d.error
    assert d.items == []


def test_seer_prompt_asks_structured():
    seer = FakeSeer("jollof rice | one plate")
    N.log_meal("/tmp/x.jpg", seer=seer)
    assert "one per line" in seer.last_prompt.lower()


# ── the 2 questions ─────────────────────────────────────────────────────────

def test_asks_about_oil_not_visible_rice():
    d = _draft("jollof rice | one plate")
    qs = N.follow_up_questions(d)
    assert len(qs) <= 2
    assert any("oil" in q for q in qs)
    # never asks what the rice IS — it's clearly visible
    assert not any("what" in q.lower() and "rice" in q.lower() for q in qs)


def test_max_two_questions():
    d = _draft("jollof rice | one plate\ndodo | some\nakara | few")
    assert len(N.follow_up_questions(d)) <= 2


def test_non_oily_meal_asks_portion_and_drink():
    d = _draft("white rice | one plate\nfish | one piece")
    qs = N.follow_up_questions(d)
    assert len(qs) == 2
    assert any("portion" in q for q in qs)
    assert any("drink" in q for q in qs)


def test_no_questions_when_failed():
    d = _draft("NONE")
    assert N.follow_up_questions(d) == []


def test_draft_message_format():
    d = _draft()
    msg = N.format_draft_message(d)
    assert "jollof rice" in msg
    assert "approximately" in msg
    assert "1." in msg and "2." in msg


# ── completion ──────────────────────────────────────────────────────────────

def test_complete_meal_oil_adjusts():
    d = _draft("jollof rice | one plate")
    base_low = d.items[0].cal_low
    log = N.complete_meal(d, {"oil": "yes", "portion": "medium"})
    assert log.cal_low == base_low + 100
    assert "approximately" in log.range_text


def test_complete_meal_portion_scaling():
    d = _draft("jollof rice | one plate")
    small = N.complete_meal(d, {"portion": "small"})
    large = N.complete_meal(d, {"portion": "large"})
    assert small.cal_high < large.cal_high


def test_complete_meal_drink_added():
    d = _draft("jollof rice | one plate")
    log = N.complete_meal(d, {"portion": "medium", "drink": "zobo"})
    assert any(i.name == "zobo" for i in log.items)


def test_complete_meal_logs_timeline(tmp_path):
    from nomorals.health.timeline import HealthTimeline
    tl = HealthTimeline(db_path=str(tmp_path / "h.db"))
    d = _draft("jollof rice | one plate")
    log = N.complete_meal(d, {"portion": "medium"}, timeline=tl)
    assert log.timeline_event_id
    events = tl.timeline(event_type="note")
    assert any("jollof rice" in e.text for e in events)
    assert any("approximately" in e.text for e in events)


def test_expense_pairing(tmp_path):
    from nomorals.finance.ledger import Ledger
    from nomorals.health.timeline import HealthTimeline
    lg = Ledger(path=str(tmp_path / "f.jsonl"))
    tl = HealthTimeline(db_path=str(tmp_path / "h.db"))
    d = _draft("jollof rice | one plate")
    log = N.complete_meal(d, {"portion": "medium"}, cost_kobo=250000,
                          timeline=tl, ledger=lg)
    assert log.cost_kobo == 250000
    assert "₦2,500" in log.summary() or "2,500" in log.summary()


def test_no_medical_advice_in_outputs():
    d = _draft()
    log = N.complete_meal(d, {"portion": "medium"})
    blob = (N.format_draft_message(d) + log.summary()).lower()
    for banned in ("you should eat", "diagnos", "prescrib", "sounds like"):
        assert banned not in blob


# ── answer parsing ──────────────────────────────────────────────────────────

def test_parse_answers_oil():
    qs = N.follow_up_questions(_draft())
    a = N.parse_meal_answers("yes, quite oily", qs)
    assert a["oil"] == "yes"
    a2 = N.parse_meal_answers("no oil at all", qs)
    assert a2["oil"] == "no"


def test_parse_answers_portion():
    d = _draft("white rice | one plate")
    qs = N.follow_up_questions(d)
    a = N.parse_meal_answers("it was a large portion", qs)
    assert a["portion"] == "large"


# ── chat glue ───────────────────────────────────────────────────────────────

def test_meal_intent_detection():
    assert N.meal_intent_in_text("log this meal")
    assert N.meal_intent_in_text("what calories did I eat?")
    assert not N.meal_intent_in_text("what's the weather like")


def test_pending_draft_registry():
    d = _draft()
    N.arm_meal("chat1", d)
    assert N.pending_meal("chat1") is d
    assert N.consume_meal("chat1") is d
    assert N.pending_meal("chat1") is None


def test_nigerian_foods_db_coverage():
    for staple in ("jollof rice", "pounded yam", "egusi soup", "beans",
                   "dodo", "suya", "amala", "moi moi"):
        assert staple in N.NIGERIAN_FOODS, staple
        e = N.NIGERIAN_FOODS[staple]
        assert e.cal_low < e.cal_high
