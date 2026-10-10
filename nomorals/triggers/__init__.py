"""Event-condition-action automation for Devon.

A trigger is ``when <source> says so, do <action>``:

* **sources** — ``schedule`` (cron/interval, via the existing scheduler),
  ``file`` (path change), ``price`` (keyless market-data threshold),
  ``message`` (regex on inbound chat), ``webhook`` (HTTP POST, plain /
  GitHub-HMAC / Stripe-HMAC + replay dedup), ``entity_state`` (Home
  Assistant state_changed, event-driven), ``bus`` (event-bus glob),
  ``url`` (website change watch, changedetection.io-style).
* **actions** — ``notify`` (owner-only channel), ``message`` (send to a
  chat), ``command`` (run an ``nm`` command), ``mission`` (start a mission).
  ``notify``/``message``/``mission`` params support ``{{evidence}}``
  templates; ``notify``/``message`` support ``digest: true`` batching.
* **gates** — HA-style ``conditions`` (``time_window`` / ``rate`` /
  ``evidence``) evaluated after a source match, before the action.
* **modes** — ``parallel`` (default) | ``single`` | ``queued`` overlap control.
* **templates** — built-in blueprints via ``add_from_template``.

Definitions persist in SQLite (``storage.db.Database``, the same pattern
the scheduler uses) and survive restarts.  Every evaluation outcome —
fired, no-match, skipped, error — lands in the history table; nothing is
dropped silently.  Invalid definitions fail fast at creation time.

Layer: L5, peers with scheduler/missions.  Only imports downward (L4 and
below) and sideways (L5 peers: agents, integrations, missions, scheduler).
"""

from __future__ import annotations

from . import display
from .engine import TriggerEngine, attach, message_hook, verify_webhook_signature
from .models import (
    ACTION_MESSAGE,
    ACTION_COMMAND,
    ACTION_MISSION,
    ACTION_NOTIFY,
    ACTIONS,
    CONDITION_EVIDENCE,
    CONDITION_RATE,
    CONDITION_TIME_WINDOW,
    CONDITION_TYPES,
    MODE_PARALLEL,
    MODE_QUEUED,
    MODE_SINGLE,
    MODES,
    OUTCOME_ERROR,
    OUTCOME_FIRED,
    OUTCOME_NO_MATCH,
    OUTCOME_SKIPPED,
    SKIP_ALREADY_RUNNING,
    SKIP_CONDITION,
    SKIP_COOLDOWN,
    SKIP_DISABLED,
    SOURCE_BUS,
    SOURCE_ENTITY_STATE,
    SOURCE_FILE,
    SOURCE_MESSAGE,
    SOURCE_PRICE,
    SOURCE_SCHEDULE,
    SOURCE_URL,
    SOURCE_WEBHOOK,
    SOURCES,
    WEBHOOK_SCHEME_GITHUB,
    WEBHOOK_SCHEME_PLAIN,
    WEBHOOK_SCHEME_STRIPE,
    WEBHOOK_SCHEMES,
    Trigger,
    TriggerError,
    TriggerSpec,
    validate_definition,
    validate_trigger_spec,
)
from .sources import to_cloudevent
from .store import TriggerStore
from .templates import list_templates, render_template

__all__ = [
    "TriggerEngine",
    "TriggerStore",
    "Trigger",
    "TriggerError",
    "TriggerSpec",
    "attach",
    "message_hook",
    "validate_definition",
    "validate_trigger_spec",
    "verify_webhook_signature",
    "to_cloudevent",
    "list_templates",
    "render_template",
    "display",
    "SOURCES",
    "ACTIONS",
    "SOURCE_SCHEDULE",
    "SOURCE_FILE",
    "SOURCE_PRICE",
    "SOURCE_MESSAGE",
    "SOURCE_WEBHOOK",
    "SOURCE_ENTITY_STATE",
    "SOURCE_BUS",
    "SOURCE_URL",
    "ACTION_NOTIFY",
    "ACTION_MESSAGE",
    "ACTION_COMMAND",
    "ACTION_MISSION",
    "MODE_PARALLEL",
    "MODE_SINGLE",
    "MODE_QUEUED",
    "MODES",
    "CONDITION_TIME_WINDOW",
    "CONDITION_RATE",
    "CONDITION_EVIDENCE",
    "CONDITION_TYPES",
    "OUTCOME_FIRED",
    "OUTCOME_NO_MATCH",
    "OUTCOME_SKIPPED",
    "OUTCOME_ERROR",
    "SKIP_DISABLED",
    "SKIP_COOLDOWN",
    "SKIP_CONDITION",
    "SKIP_ALREADY_RUNNING",
    "WEBHOOK_SCHEME_PLAIN",
    "WEBHOOK_SCHEME_GITHUB",
    "WEBHOOK_SCHEME_STRIPE",
    "WEBHOOK_SCHEMES",
]
