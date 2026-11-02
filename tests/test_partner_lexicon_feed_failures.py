"""Pass tail-117 pins: nomorals/partner/lexicon_feed.py — 3 silent handlers.

- 215 E104 ``terms_by_signal``: the usage-ranking pass crashes ->
  the base score-ordered terms (the ranking is garnish, per the
  in-code comment — never load-bearing); healthy ranking pinned.
- 265 E104 ``usage``: the durable-usage query crashes -> the
  in-memory count, the rung the docstring's "Never raises" contract
  promises; the healthy DB value wins when it answers.
- 489 E104 ``seed_partner_lexicon``: the post-seed version probe
  crashes -> "version": 0 in the report dict, while "seeded" and
  "added" stay truthful (the seeding itself really happened);
  healthy version pinned at its real value.

All exclusions: degradation to the module's own documented rung or
presentational metadata; nothing fabricated.
"""

from nomorals.partner import lexicon_feed as lf


class FakeDB:
    def __init__(self, mode):
        self.mode = mode

    def execute(self, *args, **kwargs):
        return None

    def query_one(self, *args, **kwargs):
        if self.mode == "raise":
            raise RuntimeError("db boom")
        return {"uses": 5}


def make_feed(db):
    feed = lf.LexiconFeed.__new__(lf.LexiconFeed)
    feed._uses = {("tone", "yeah"): 2}
    feed.db = db
    return feed


# ── 265: usage ───────────────────────────────────────────────────────

def test_usage_healthy_db_value_wins():
    assert make_feed(FakeDB("ok")).usage("yeah", "tone") == 5


def test_usage_db_crash_returns_memory_count():
    assert make_feed(FakeDB("raise")).usage("yeah", "tone") == 2


# ── 215: terms_by_signal ─────────────────────────────────────────────

def make_rank_feed(usage_fn):
    feed = lf.LexiconFeed.__new__(lf.LexiconFeed)
    feed.terms = lambda category, limit=20: (
        "alpha", "beta", "gamma")[:limit]
    feed.usage = usage_fn
    return feed


def test_terms_by_signal_ranking_crash_returns_base_order():
    def boom(term, category):
        raise RuntimeError("usage boom")

    feed = make_rank_feed(boom)
    assert feed.terms_by_signal("tone", limit=2) == ("alpha", "beta")


def test_terms_by_signal_healthy_ranks_used_first():
    feed = make_rank_feed(
        lambda t, c: {"alpha": 0, "beta": 10, "gamma": 0}[t])
    assert feed.terms_by_signal("tone", limit=3)[0] == "beta"


# ── 489: seed version probe ──────────────────────────────────────────

class FakeStore:
    calls = 0
    mode = "ok"

    def __init__(self, ns):
        pass

    def version(self, module):
        FakeStore.calls += 1
        if FakeStore.calls == 1:
            return 0  # precheck: not yet seeded
        if FakeStore.mode == "raise":
            raise RuntimeError("version boom")
        return 7

    def acquire(self, candidates, **kwargs):
        return {"added": list(candidates[:1]), "skipped": []}


class SeedDB:
    def query(self, *args, **kwargs):
        return []


def test_seed_version_crash_reports_zero_but_seeds(monkeypatch):
    FakeStore.calls = 0
    FakeStore.mode = "raise"
    monkeypatch.setattr(lf, "LexiconStore", FakeStore)
    res = lf.seed_partner_lexicon(SeedDB())
    assert res["seeded"] is True
    assert res["version"] == 0
    assert len(res["added"]) > 0


def test_seed_healthy_reports_real_version(monkeypatch):
    FakeStore.calls = 0
    FakeStore.mode = "ok"
    monkeypatch.setattr(lf, "LexiconStore", FakeStore)
    res = lf.seed_partner_lexicon(SeedDB())
    assert res["seeded"] is True and res["version"] == 7
