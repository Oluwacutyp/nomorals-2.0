"""Expose dormant organs — a verified registry, NOT a rebuild.

MORE_TO_ADD gap: expose-dormant. This module does not re-implement any
organ. It is a registry of existing organs that work but have no
command/client wired to them: each entry names the real module, says
what it does (taken from that module's own docstring/header — verified
by reading the file), and shows how to invoke it (import path + example).

Verification: :func:`verify_all` checks every entry's file exists on
disk. Entries whose file is missing are NEVER listed — they land in
:attr:`SKIPPED` with the reason. Lesson encoded here: always read the
real tree, never trust screenshots alone.

This registry never imports the target modules (no side effects, no
import cycles) — it only records paths and invocation strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Entry", "REGISTRY", "SKIPPED", "entries", "verify_all", "find"]

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (nomorals/meta/ -> repo)


@dataclass(frozen=True)
class Entry:
    """One dormant-but-working organ."""

    name: str
    module: str  # dotted import path, e.g. "nomorals.health.coach"
    path: str  # repo-relative file path
    what: str  # from the module's own docstring/header
    invoke: str  # import path + minimal example
    symbols: tuple[str, ...] = ()


def _e(name: str, module: str, what: str, invoke: str, symbols: tuple[str, ...] = ()) -> Entry:
    rel = module.replace(".", "/") + ".py"
    # ``nomorals.goals`` lives in goals/__init__.py, not goals.py.
    if module == "nomorals.goals":
        rel = "nomorals/goals/__init__.py"
    return Entry(name=name, module=module, path=rel, what=what, invoke=invoke, symbols=symbols)


# Every entry below was verified by reading the module's docstring/header
# and confirming the path exists. See verify_all() / SKIPPED.
_REGISTRY_CANDIDATES: list[Entry] = [
    # -- health/ ---------------------------------------------------------
    _e("health coach", "nomorals.health.coach",
       "Ask-your-data coaching over HealthKit / Health Connect biometrics: sleep, recovery/readiness, activity. Coaching, NOT diagnosis.",
       "from nomorals.health.coach import HealthCoach  # HealthCoach(source).ask('how did I sleep?')",
       ("HealthCoach", "guard_coaching")),
    _e("health sleep-need", "nomorals.health.coach",
       "Sleep-need estimation + sleep score vs need (functions inside coach.py, not a separate module).",
       "from nomorals.health.coach import sleep_need, sleep_score_vs_need  # sleep_need(recent_avg_hours=6.5)",
       ("sleep_need", "sleep_score_vs_need")),
    _e("health readiness", "nomorals.health.coach",
       "Readiness band advice + ReadinessPoint history (functions inside coach.py).",
       "from nomorals.health.coach import readiness_band_advice  # readiness_band_advice('low')",
       ("readiness_band_advice", "ReadinessPoint")),
    _e("health form", "nomorals.health.form",
       "Seer as form coach — video movement analysis: lift form, rep counting, gait.",
       "from nomorals.health.form import analyze_form  # analyze_form(video_path, 'squat')",
       ("analyze_form", "analyze_gait", "FormStore")),
    _e("health training", "nomorals.health.training",
       "Recovery-gated training plans + voice coaching; biometrics gate the plan.",
       "from nomorals.health.training import generate_plan, compute_readiness  # generate_plan('12-week strength')",
       ("generate_plan", "compute_readiness", "Plan")),
    _e("health drift", "nomorals.health.drift",
       "Drift detection + recovery actions — 3-day rolling watch on sleep/mood/HRV/activity, intervene before burnout.",
       "from nomorals.health.drift import DriftReport, recovery_plan  # recovery_plan(report)",
       ("DriftSignal", "DriftReport", "recovery_plan")),
    _e("health nutrition", "nomorals.health.nutrition",
       "Photo meal logging with 2-question completion; fixes photo-only undercounting.",
       "from nomorals.health.nutrition import MealLog  # MealLog().add(...)",
       ("FoodEntry", "MealDraft", "MealLog")),
    _e("health previsit", "nomorals.health.previsit",
       "Pre-visit layer — navigation, NOT diagnosis: triage routing, Nigeria price transparency.",
       "from nomorals.health.previsit import triage_route  # triage_route(symptoms)",
       ("triage_route", "Route")),
    _e("health patterns", "nomorals.health.patterns",
       "Mood-pattern detection from memory. Wellness companionship — never therapy.",
       "from nomorals.health.patterns import detect_lag_patterns  # detect_lag_patterns(timeline)",
       ("detect_lag_patterns", "Pattern")),
    _e("health challenges", "nomorals.health.challenges",
       "Photo-proof challenges + streak/squad mechanics.",
       "from nomorals.health.challenges import ChallengeStore, verify_proof_photo  # ChallengeStore().create(...)",
       ("Challenge", "Squad", "ChallengeStore")),
    _e("health timeline", "nomorals.health.timeline",
       "Patient-side health timeline — tracking only, Devon is not a doctor.",
       "from nomorals.health.timeline import HealthTimeline  # HealthTimeline().add(HealthEvent(...))",
       ("HealthTimeline", "HealthEvent")),
    # -- legal/ ----------------------------------------------------------
    _e("legal contracts", "nomorals.legal.contracts",
       "Contract risk review — letter grades + Nigerian playbooks. Legal information, never legal advice.",
       "from nomorals.legal.contracts import REVIEW_STYLES, CLAUSE_CATEGORIES, ReviewDiff  # ReviewDiff(old, new)",
       ("REVIEW_STYLES", "CLAUSE_CATEGORIES", "ReviewDiff", "Review")),
    _e("legal portfolio", "nomorals.legal.portfolio",
       "Contract portfolio — all contracts as one obligation tracker: obligations, key dates, renewals, SLA terms.",
       "from nomorals.legal.portfolio import Contract  # Contract(...).obligations",
       ("Contract", "Obligation")),
    _e("legal regulatory", "nomorals.legal.regulatory",
       "Regulatory change monitoring — CBN/SEC/NDPA/FIRS from official regulator domains; AI flags, human confirms.",
       "from nomorals.legal.regulatory import RegulatoryItem  # RegulatoryItem(...)",
       ("RegulatoryItem", "Regulator")),
    _e("legal aid", "nomorals.legal.aid",
       "Consumer legal aid — plain-language legal information in Nigerian languages, grounded in the corpus.",
       "from nomorals.legal.aid import corpus_search  # corpus_search('tenancy', language='pidgin')",
       ("corpus_search", "Answer")),
    _e("legal research", "nomorals.legal.research",
       "RAG + confidence + citations — the anti-hallucination stack over the verified legal corpus.",
       "from nomorals.legal.research import Citation  # research(query) with citations",
       ("Citation",)),
    # -- learn/ ----------------------------------------------------------
    _e("learn Socratic tutor", "nomorals.learn.tutor",
       "Socratic tutoring engine — 'guide me' withholds answers (EMT pump→hint→prompt→assertion), 'direct' teaches straight.",
       "from nomorals.learn.tutor import SocraticEngine  # SocraticEngine().turn(...)",
       ("SocraticEngine", "TutorTurn")),
    _e("learn BKT", "nomorals.learn.tutor",
       "Bayesian Knowledge Tracing posterior per skill (Corbett & Anderson 1995, pyBKT equations) + half-life forgetting.",
       "from nomorals.learn.tutor import BKT_DEFAULTS  # BKT_DEFAULTS; MasteryModel.bkt_update(...)",
       ("BKT_DEFAULTS", "MasteryModel")),
    _e("learn FSRS flashcards", "nomorals.learn.flashcards",
       "Mistake notebook — wrong answers become spaced-repetition flashcards on FSRS-lite (DSR core of FSRS-4.5).",
       "from nomorals.learn.flashcards import MistakeNotebook  # MistakeNotebook().add(mistake, correction)",
       ("MistakeNotebook", "MistakeCard", "FSRS")),
    _e("learn Ekiti tutor", "nomorals.learn.dialect",
       "Ekiti / Ilawe Ekiti dialect tutoring — flagship Yoruba dialect: conversation, debate, tonal minimal pairs.",
       "from nomorals.learn.dialect import EkitiTutor  # EkitiTutor().practice(...)",
       ("EkitiTutor", "detect_dialect_detail")),
    _e("learn WAEC/JAMB", "nomorals.learn.curriculum",
       "WAEC/JAMB curriculum alignment — syllabus codes, topic order, mapped to real WAEC syllabus structure.",
       "from nomorals.learn.curriculum import subject_topics, curriculum_order  # subject_topics('physics')",
       ("subject_topics", "curriculum_order", "SyllabusTopic")),
    _e("learn course generation", "nomorals.learn.curriculum",
       "Course generation that never starts from a bare prompt: clarify -> storyboard -> build.",
       "from nomorals.learn.curriculum import *  # course builders in nomorals.learn.curriculum",
       ("SyllabusTopic",)),
    _e("learn misconception bank", "nomorals.learn.tutor",
       "Anticipated misconceptions per question script (AutoTutor EMT expectations + misconceptions).",
       "from nomorals.learn.tutor import Misconception, Expectation  # QuestionScript(expectations=[...], misconceptions=[...])",
       ("Misconception", "Expectation", "QuestionScript")),
    # -- integrations/ ---------------------------------------------------
    _e("integrations email", "nomorals.integrations.email_integration",
       "Email integration, three backends: Gmail API (OAuth), IMAP, browser automation.",
       "from nomorals.integrations.email_integration import EmailIntegration  # EmailIntegration().inbox()",
       ("EmailIntegration", "EmailMessage")),
    _e("integrations calendar", "nomorals.integrations.calendar_integration",
       "Calendar integration: Google Calendar API (OAuth), CalDAV, browser automation fallback.",
       "from nomorals.integrations.calendar_integration import CalendarIntegration  # CalendarIntegration().agenda()",
       ("CalendarIntegration", "CalendarEvent")),
    _e("integrations payments", "nomorals.integrations.payment_integration",
       "Payments: crypto wallets, Stripe/PayPal APIs, virtual card generation.",
       "from nomorals.integrations.payment_integration import PaymentApproval  # approval-gated payments",
       ("PaymentApproval", "Wallet", "VirtualCard")),
    _e("integrations smart home", "nomorals.integrations.smarthome_integration",
       "Smart home: Home Assistant API (2000+ devices), direct device APIs (Hue, ...).",
       "from nomorals.integrations.smarthome_integration import SmartHomeIntegration  # SmartHomeIntegration().devices()",
       ("SmartHomeIntegration", "Device")),
    _e("integrations MQTT", "nomorals.integrations.mqtt_client",
       "Native MQTT client — one client, thousands of devices; Zigbee2MQTT topic grammar as device abstraction.",
       "from nomorals.integrations.mqtt_client import *  # state_topic(device), set_topic(device)",
       ("state_topic", "set_topic")),
    _e("integrations Naija shopping", "nomorals.integrations.naija_shopping",
       "Naija Shopping Engine — price tracking and deal detection across Nigerian marketplaces.",
       "from nomorals.integrations.naija_shopping import get_fx_rate  # get_fx_rate('USD')",
       ("get_fx_rate",)),
    _e("integrations digital twin", "nomorals.integrations.digital_twin",
       "Home Digital Twin — persistent queryable model of the home from entity state changes.",
       "from nomorals.integrations.digital_twin import HomeTwin  # HomeTwin().state(...)",
       ("HomeTwin", "StateEvent", "Anomaly")),
    # -- audio / documents / characters / goals / matching ---------------
    _e("audio fingerprinting", "nomorals.audio.fingerprint",
       "Devon's own audio fingerprinting + acoustic analysis — Shazam-style spectral-constellation recognition, native-first.",
       "from nomorals.audio.fingerprint import fingerprint  # fingerprint(wav_path)",
       ("fingerprint", "read_mono")),
    _e("audio pipeline", "nomorals.audio.pipeline",
       "Unified voice pipeline — capture → cleanup → transcribe → edit → multi-speaker TTS → master.",
       "from nomorals.audio.pipeline import EngineRegistry  # EngineRegistry().pipeline()",
       ("EngineRegistry",)),
    _e("documents index", "nomorals.documents.index",
       "Full-text search over parsed documents, ranked with BM25 over SQLite FTS5.",
       "from nomorals.documents.index import DocumentIndex  # DocumentIndex().add(doc); .search(q)",
       ("DocumentIndex",)),
    _e("documents summarize", "nomorals.documents.summarize",
       "Extractive summarization and keyword hooks for parsed documents.",
       "from nomorals.documents.summarize import summarize  # summarize(parsed_doc)",
       ("summarize",)),
    _e("characters", "nomorals.characters.character",
       "Character: a persistent agent entity — identity, memory, mood, goals, relationships. Not a prompt template.",
       "from nomorals.characters.character import Character  # Character(...)",
       ("Character",)),
    _e("characters ensemble", "nomorals.characters.ensemble",
       "Ensemble scenes: multiple characters in one living conversation — podcasts, roundtables.",
       "from nomorals.characters.ensemble import Scene, DramaDirector  # Scene(characters=[...])",
       ("Scene", "DramaDirector")),
    _e("goals", "nomorals.goals",
       "Goals and ideas tracking — durable per-user goals with subgoals, progress history, deadlines, reminders, templates, briefings.",
       "from nomorals.goals import GoalTracker  # GoalTracker().add(Goal(...))",
       ("GoalTracker", "Goal", "IdeaTracker")),
    _e("matching stable", "nomorals.matching.stable",
       "Stable matching mechanisms — two-sided, one-pool, allocation (Gale-Shapley). One best pairing per day.",
       "from nomorals.matching.stable import stable_match  # stable_match(prefs_a, prefs_b)",
       ("stable_match", "MatchResult")),
    _e("matching chat", "nomorals.matching.chat",
       "Chat control for /match — daily batches + stable matching. Never raises.",
       "from nomorals.matching.chat import format_batch  # format_batch(batch)",
       ("format_batch",)),
    # -- mesh / hub / native / api / context ------------------------------
    _e("mesh node", "nomorals.mesh.node",
       "Mesh node registry: identity, heartbeat, presence across phone / cloud worker / laptop.",
       "from nomorals.mesh.node import NodeRegistry  # NodeRegistry().heartbeat(node)",
       ("NodeRegistry", "MeshNode")),
    _e("mesh tasks", "nomorals.mesh.tasks",
       "Mesh task dispatch over the durable work queue; topic-namespace targeting.",
       "from nomorals.mesh.tasks import MeshTasks  # MeshTasks().dispatch(task)",
       ("MeshTasks", "MeshTask")),
    _e("hub server", "nomorals.hub.server",
       "Hub HTTP server: mesh + sync over one stdlib listener, thread-per-connection.",
       "from nomorals.hub.server import *  # serve(host, port)",
       ()),
    _e("native biometric", "nomorals.native.biometric",
       "Phone-native biometric approval (fingerprint) for sensitive actions via Termux:API.",
       "from nomorals.native.biometric import *  # approve_with_biometric(reason)",
       ("BiometricResult", "BiometricStatus")),
    _e("api server", "nomorals.api.server",
       "HTTP API — stdlib only, so the service runs anywhere Python does. Deliberately not FastAPI.",
       "from nomorals.api.server import *  # serve(...)",
       ("Principal",)),
    _e("api MCP server", "nomorals.api.mcp_server",
       "Model Context Protocol (MCP) server — Devon as infrastructure: memory + tool registry to external agents.",
       "from nomorals.api.mcp_server import MCPServer  # MCPServer().serve()",
       ("MCPServer",)),
    _e("api ACP", "nomorals.api.acp",
       "ACP (Agent Communication Protocol) surface for Devon.",
       "from nomorals.api.acp import *  # ACP handlers",
       ()),
    _e("context engine", "nomorals.context.engine",
       "ContextEngine: token-budgeted assembly of model-call context (system, profile, project, mission, artifacts).",
       "from nomorals.context.engine import ContextEngine  # ContextEngine().build(...)",
       ("ContextEngine", "BuiltContext")),
    _e("context budget", "nomorals.context.budget",
       "Token budgets for context assembly — split a total budget across named sections, trim to fit.",
       "from nomorals.context.budget import ContextBudget  # ContextBudget(total=8000).fit(section, text)",
       ("ContextBudget",)),
]

#: Verified registry — entries whose file exists on disk. Built at import.
REGISTRY: list[Entry] = []

#: Entries whose file did NOT exist on disk — skipped, never listed as live.
SKIPPED: list[tuple[Entry, str]] = []

for _candidate in _REGISTRY_CANDIDATES:
    _p = _PACKAGE_ROOT / _candidate.path
    if _p.is_file():
        REGISTRY.append(_candidate)
    else:
        SKIPPED.append((_candidate, f"missing on disk: {_candidate.path}"))


def entries() -> list[Entry]:
    """All verified dormant-organ entries."""
    return list(REGISTRY)


def verify_all() -> tuple[list[Entry], list[tuple[Entry, str]]]:
    """Re-check every entry against disk. Returns (ok, missing)."""
    ok: list[Entry] = []
    missing: list[tuple[Entry, str]] = []
    for entry in _REGISTRY_CANDIDATES:
        if (_PACKAGE_ROOT / entry.path).is_file():
            ok.append(entry)
        else:
            missing.append((entry, f"missing on disk: {entry.path}"))
    return ok, missing


def find(keyword: str) -> list[Entry]:
    """Case-insensitive search over name / module / description."""
    needle = keyword.strip().lower()
    return [
        e for e in REGISTRY
        if needle in e.name.lower() or needle in e.module.lower() or needle in e.what.lower()
    ]
