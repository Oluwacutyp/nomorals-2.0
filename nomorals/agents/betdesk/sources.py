"""Desk sources: betting joins the shared source registry, and the
desk's API-key needs join the real self-provisioning registry.

Two existing organs, extended — never paralleled:

* ``media/source_registry.SourceRegistry`` is the ONE source of truth
  every class grows through (downloads #86, music #88, books #89).
  The desk registers its seed sources under kind ``betting`` the
  same way: seeds are the floor, outcomes move them, candidates
  enter untrusted.  Book domains coincide with the collector
  providers in :mod:`.collectors`, so a fetch outcome and a registry
  health record are about the same host.
* ``accounts/provisioning.PROVISION_REGISTRY`` is how Devon gets
  herself API keys (drive signup/login on the owner's own account,
  validate, vault, bridge to env).  The desk's data keys — the Odds
  API (``ODDS_API_KEY``, already read by
  ``agents/sports_bet.TheOddsApiFetcher``) and football-data.org
  (``FOOTBALL_DATA_API_KEY``) — are registered here as services with
  real validators, so ``provision odds-api`` works through the same
  driver as every other service instead of the desk growing its own
  key plumbing.

Pass 8 (directed) completes the rungs the desk build named:

* **Page flows** — both services now carry a real page flow
  (``odds_api_key_v1`` / ``football_data_key_v1``, registered into
  the provisioning module's ``_PAGE_FLOWS`` by
  :func:`ensure_provisioning`).  Neither service mints per-token
  keys the way HF does: each account holds ONE key, displayed on
  its account page (the Odds API account page; the football-data
  client home).  The flow's job is to open that page on the logged-in
  session, clear any CAPTCHA through the injected solver, and read
  the key off the page — the driver's live validation immediately
  after is what proves the read was right (a wrong read fails
  validation and is never stored).
* **Registry resolution** — the desk resolves its registry through
  :func:`betting_registry_for_context` (the #94 wisdom resolver's
  shape): the shared ``DEFAULT_REGISTRY`` BOUND to the context's KV
  store, so the desk's seeds, outcomes and health persist into the
  same state the 6h health loop verifies.  Before this, the desk
  held the unbound singleton: seeds registered in memory only, the
  health loop never saw them, and every source sat at ``unknown``.

Betting BOOK accounts are deliberately absent: books are the owner's
own accounts (the desk studies them through his sessions); they are
never provisioned, never opened under generated identities.
"""

from __future__ import annotations

import re
from typing import Any
from .failures import record_failure

__all__ = [
    "BETTING_SEEDS", "ensure_betting_sources", "betting_status",
    "betting_registry_for_context",
    "ensure_provisioning", "ODDS_API_SERVICE", "FOOTBALL_DATA_SERVICE",
    "odds_api_key_flow", "football_data_key_flow",
]


# ── source-registry seeds ──────────────────────────────────────────────

#: (nation, domain, name, auth_needed, notes)
BETTING_SEEDS: tuple = (
    ("US", "stake.com", "Stake", True,
     "Stake Originals crash family + sportsbook; history behind the "
     "owner's session."),
    ("NG", "sportybet.com", "SportyBet", True,
     "Spribe Aviator (NG). Public site bot-walls datacenter networks."),
    ("NG", "msport.com", "MSport", True,
     "Spribe Aviator (NG); session-shaped history."),
    ("NG", "football.com", "Football.com", True,
     "Aviator + crash family (NG); session-shaped history."),
    ("GLOBAL", "bustabit.com", "Bustabit", False,
     "The original crash game; provably-fair hash chain, published "
     "salts. History rides its app — exported logs import directly."),
    ("GLOBAL", "the-odds-api.com", "The Odds API", True,
     "Cross-book odds API (keyed, free tier) — the desk's odds feed; "
     "self-provisioned via accounts/provisioning."),
    ("GLOBAL", "football-data.org", "football-data.org", True,
     "Fixtures/results API (keyed, free tier) — the analyst's "
     "history feed; self-provisioned via accounts/provisioning."),
    ("GLOBAL", "site.api.espn.com", "ESPN scoreboard", False,
     "Keyless fixtures+odds feed the sports analyst already reads."),
)


#: The ESPN scoreboard endpoint the sports analyst already reads —
#: the health loop probes the ESPN source through it (a root page on
#: an API host proves nothing; this endpoint answering IS the source
#: being alive).
ESPN_PROBE_URL = ("https://site.api.espn.com/apis/site/v2/sports/"
                  "soccer/eng.1/scoreboard")

#: Per-domain descriptor extras beyond the seed tuple's fields: the
#: two keyed data APIs carry the extra kind ``api`` (the health loop
#: treats kind ``betting`` + ``api`` + auth like the #90 keyed search
#: APIs — root reachability only, key truth judged by real fetches
#: and provisioning validation); ESPN carries its real probe URL as
#: its search template so the loop probes the endpoint, not the root.
_SEED_EXTRAS: dict[str, dict[str, Any]] = {
    "the-odds-api.com": {"kinds": ("betting", "api")},
    "football-data.org": {"kinds": ("betting", "api")},
    "site.api.espn.com": {"search_url_template": ESPN_PROBE_URL},
}


def _seed_descriptor(nation: str, domain: str, name: str, auth: bool,
                     notes: str) -> Any:
    from ...media.source_registry import (
        Licence, SourceDescriptor, TrustTier)
    extras = _SEED_EXTRAS.get(domain, {})
    return SourceDescriptor(
        nation=nation, language="en", domain=domain, name=name,
        kinds=tuple(extras.get("kinds", ("betting",))),
        licence=Licence.UNKNOWN,
        trust_tier=TrustTier.UNVERIFIED, auth_needed=auth,
        search_url_template=str(extras.get("search_url_template", "")),
        notes=notes)


def ensure_betting_sources(registry: Any) -> list[dict[str, Any]]:
    """Register the betting seed sources on the shared registry.

    Idempotent: a seed already present (by domain) with the current
    shape is reported, not duplicated.  A seed persisted in an older
    shape (kinds/probe template from before Pass 8) is updated in
    place — the registry is the one source of truth, and a stale
    descriptor would silently keep the old probe behaviour.
    Returns one status dict per seed.
    """
    out: list[dict[str, Any]] = []
    for nation, domain, name, auth, notes in BETTING_SEEDS:
        desc = _seed_descriptor(nation, domain, name, auth, notes)
        existing = registry.for_domain(domain)
        if existing is not None:
            if (tuple(existing.kinds) == tuple(desc.kinds)
                    and existing.search_url_template
                    == desc.search_url_template
                    and existing.auth_needed == desc.auth_needed):
                out.append({"domain": domain, "already_registered": True})
                continue
            registry._by_key[desc.key] = desc
            try:
                registry._persist()
            except Exception as e:  # persistence is best-effort
                record_failure("sources.ensure_betting_sources.persist_update", e)
                pass
            out.append({"domain": domain, "updated": True})
            continue
        added = registry.register(desc)
        if added:
            try:
                registry._persist()
            except Exception as e:  # persistence is best-effort
                record_failure("sources.ensure_betting_sources.persist_register", e)
                pass
        out.append({"domain": domain, "registered": bool(added)})
    return out


def betting_registry_for_context(context: Any = None,
                                 injected: Any = None) -> Any:
    """The source registry the desk's outcomes land in.

    Mirrors the #94 wisdom resolver: an injected registry wins;
    otherwise the shared ``DEFAULT_REGISTRY`` bound to the context's
    KV store — the same persisted state the download health loop
    verifies — so desk-registered seeds and collector outcomes are
    visible to the loop (and survive restarts).  A db-less context
    (CLI, tests) keeps the unbound singleton: a working registry,
    memory-only.  Never raises.
    """
    if injected is not None:
        return injected
    from ...media.source_registry import DEFAULT_REGISTRY
    reg = DEFAULT_REGISTRY
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is not None:
            from ...storage.kv import KVStore
            reg.bind_store(KVStore(db))
    except Exception as e:  # persistence is an enhancement;
        record_failure("sources.betting_registry_for_context", e)
        pass  # its absence never sinks the desk
    return reg


def betting_status(registry: Any) -> dict[str, Any]:
    """Registry view of the betting class: sources + health + origin."""
    rows = []
    for desc in registry.by_kind("betting"):
        rows.append({
            "domain": desc.domain, "name": desc.name,
            "health": registry.health_for(desc),
            "origin": registry.origin_for(desc),
            "active": registry.is_active(desc),
            "stats": registry.stats_for(desc),
        })
    return {"sources": rows, "count": len(rows)}


# ── self-provisioning: the desk's API keys ─────────────────────────────

ODDS_API_SERVICE = "odds-api"
FOOTBALL_DATA_SERVICE = "football-data"


def _validate_odds_api(token: str, http_get: Any) -> Any:
    """The Odds API: 200 for a live key, 401 for a dead one."""
    from ...accounts.provisioning import ValidationResult
    try:
        status, _body = http_get(
            f"https://api.the-odds-api.com/v4/sports/?apiKey={token}", {})
    except Exception as exc:  # noqa: BLE001 - transport is inconclusive
        return ValidationResult("inconclusive",
                                f"validation request failed: {exc}")
    if status == 200:
        return ValidationResult("ok", "odds api key accepted")
    if status in (401, 403):
        return ValidationResult("rejected", f"odds api answered HTTP {status}")
    return ValidationResult("inconclusive",
                            f"odds api answered HTTP {status}")


def _validate_football_data(token: str, http_get: Any) -> Any:
    """football-data.org: 200 for a live key, 403 for a dead one."""
    from ...accounts.provisioning import ValidationResult
    try:
        status, _body = http_get(
            "https://api.football-data.org/v4/competitions/",
            {"X-Auth-Token": token})
    except Exception as exc:  # noqa: BLE001 - transport is inconclusive
        return ValidationResult("inconclusive",
                                f"validation request failed: {exc}")
    if status == 200:
        return ValidationResult("ok", "football-data key accepted")
    if status in (401, 403):
        return ValidationResult("rejected",
                                f"football-data answered HTTP {status}")
    return ValidationResult("inconclusive",
                            f"football-data answered HTTP {status}")


# ── page flows (Pass 8) ──────────────────────────────────────────────
#
# Both services keep ONE key per account and display it on the
# account page — there is no create-a-token dialog to walk (the HF
# shape).  The flows therefore read the key off the page the driver
# lands on after login, with the hf_token_flow discipline: every
# step fails loudly by name, CAPTCHAs go through the injected
# solver first, and the value read is proven by the driver's live
# validation before anything is stored — a wrong read is a
# KEY_REJECTED outcome, never a vaulted mistake.

#: Both services issue 32-char hex keys/tokens.
_API_KEY_RE = re.compile(r"\b[0-9a-f]{32}\b")


def _extract_hex_api_key(tab: Any) -> str:
    """Read a 32-hex API key off the current page.  Input/code values
    first (the account pages render the key into a readonly field),
    then the first key-shaped value after an "api key / api token"
    label in the page text, then any key-shaped value in the text.
    Never logged; the caller treats the return as a secret."""
    from ...accounts.provisioning import _tab_text
    evaluate = getattr(tab, "evaluate", None)
    if callable(evaluate):
        js = (
            "(() => { const vals = [...document.querySelectorAll"
            "('input, code, textarea')].map(e => e.value || "
            "e.textContent || ''); const hit = vals.find(v => "
            "/[0-9a-f]{32}/.test(v || '')); return hit ? (hit.match"
            "(/[0-9a-f]{32}/) || [null])[0] : null; })()")
        try:
            found = evaluate(js)
            if found:
                return str(found)
        except Exception:  # noqa: BLE001 - fall through to text scan
            pass
    text = _tab_text(tab, max_chars=40000)
    lowered = text.lower()
    for label in ("api key", "api token", "your token", "your key"):
        idx = lowered.find(label)
        if idx >= 0:
            match = _API_KEY_RE.search(text, idx)
            if match:
                return match.group(0)
    match = _API_KEY_RE.search(text)
    return match.group(0) if match else ""


def _open_key_page(tab: Any, service: Any, *, step: str,
                   captcha_solver: Any = None) -> None:
    """Navigate to the service's key page and clear the walls that
    can be cleared: a login wall means the session restore failed
    (loud step error, the hf_token_flow contract); a CAPTCHA goes
    through the solver, or blocks the flow for a human."""
    from ...accounts.provisioning import (
        FlowBlocked, FlowStepError, _detect_captchas, _tab_text)
    tab.navigate(service.token_url)
    text = _tab_text(tab)
    url = str(getattr(tab, "url", "") or "")
    if "/login" in url or ("log in" in text.lower()
                           and "password" in text.lower()):
        raise FlowStepError(
            step, "the key page answered with a login wall — the "
                  "browser session was not logged in")
    for ch in _detect_captchas(tab):
        if captcha_solver is None:
            raise FlowBlocked(
                f"CAPTCHA ({ch.get('kind')}) on the key page")
        result = captcha_solver(ch)
        if not result.get("ok"):
            raise FlowBlocked(
                f"CAPTCHA ({ch.get('kind')}) on the key page and the "
                f"solver could not clear it: {result.get('detail', '')}")


def _read_key_after_reveal(tab: Any, reveal_labels: tuple) -> str:
    """Extract the key; if the page hides it behind a reveal control,
    click it and extract again.  "" when no key materializes."""
    from ...accounts.provisioning import FlowStepError, _click_text
    key = _extract_hex_api_key(tab)
    if key:
        return key
    try:
        _click_text(tab, *reveal_labels)
    except FlowStepError:
        return ""
    return _extract_hex_api_key(tab)


def odds_api_key_flow(tab: Any, service: Any, *,
                      token_name: str = "",
                      captcha_solver: Any = None) -> str:
    """Fetch the account's Odds API key from its account page.

    The Odds API shows the account's one key on
    ``the-odds-api.com/account/`` — the flow reads it (revealing it
    first when the page asks).  ``token_name`` is accepted for the
    page-flow contract and unused: this service has no named tokens.
    """
    from ...accounts.provisioning import FlowStepError
    _open_key_page(tab, service, step="open_account_page",
                   captcha_solver=captcha_solver)
    key = _read_key_after_reveal(
        tab, ("Show API Key", "Reveal API Key", "View API Key", "Reveal"))
    if not key:
        raise FlowStepError(
            "read_key_value",
            "no API key value found on the Odds API account page — "
            "the page shape has likely changed; nothing was stored")
    return key


def football_data_key_flow(tab: Any, service: Any, *,
                           token_name: str = "",
                           captcha_solver: Any = None) -> str:
    """Fetch the account's football-data.org API token from the
    client home page, where the service displays it ("Your API
    Token").  Same read-and-prove shape as the Odds API flow."""
    from ...accounts.provisioning import FlowStepError
    _open_key_page(tab, service, step="open_client_home",
                   captcha_solver=captcha_solver)
    key = _read_key_after_reveal(
        tab, ("Show API Token", "Show token", "Reveal token", "Reveal"))
    if not key:
        raise FlowStepError(
            "read_token_value",
            "no API token value found on the football-data client "
            "home — the page shape has likely changed; nothing was "
            "stored")
    return key


#: Post-submit markers specific to a subscribe contract: the page
#: acknowledging the subscription IS the success state (the key is
#: emailed), on top of the shared verify/success marker sets.
_ODDS_SUBSCRIBE_MARKERS = (
    "you're subscribed", "you are subscribed",
    "subscription confirmed", "thanks for subscribing",
    "thank you for subscribing", "subscribed successfully",
)


def _click_free_plan_start(tab: Any) -> bool:
    """Click the FREE (Starter) plan's START control on the Odds API
    get-access page.

    Every plan carries its own START button, so a text-only click is
    not enough: it would take whichever plan the page lists first.
    The plan-aware path (JS, the ``_click_text`` discipline) walks up
    from each START control to the plan block around it and clicks
    the one whose block says FREE / Starter. Only when no plan
    context is readable at all does it fall back to a plain START
    click — on the recorded page shape the free Starter plan is the
    first listed, so the fallback lands on the same button.
    """
    from ...accounts.provisioning import FlowStepError, _click_text

    evaluate = getattr(tab, "evaluate", None)
    if callable(evaluate):
        js = (
            "(() => { const btns = [...document.querySelectorAll("
            "'button, a, [role=button], input[type=submit]')]"
            ".filter(e => (e.innerText || e.value || '')"
            ".trim().toUpperCase() === 'START');"
            " if (!btns.length) return '';"
            " for (const b of btns) { let n = b;"
            " for (let i = 0; i < 6 && n; i++) {"
            " n = n.parentElement;"
            " if (n && /free|starter/i.test(n.innerText || ''))"
            " { b.click(); return 'free'; } } }"
            " btns[0].click(); return 'first'; })()")
        try:
            if evaluate(js):
                return True
        except Exception as exc:  # noqa: BLE001 - try the label click
            # The JS rung failed; the label click below still gets
            # its chance — but the failure is on the desk's record.
            record_failure("sources._click_free_plan_start", exc)
    try:
        _click_text(tab, "START")
        return True
    except FlowStepError:
        return False


def _subscribe_get_access(tab: Any) -> bool:
    """Perform the get-access subscribe action, best mechanism first.

    The page's recorded shapes, in the order they should be tried:

    1. the per-plan START control (the contract the live page
       states today — plan cards, Starter free);
    2. a labelled subscribe control found by the text-click
       discipline (JS first, the ``_click_text`` way);
    3. a DIRECT click on the labelled controls — ``_click_text``
       skips direct clicks whenever the tab has a JS engine, even
       one that finds nothing, and the service's own recorded form
       shape answers only to a direct "Get Access" click;
    4. a plain form submit — the shape the desk's earlier drive
       model recorded (an email-only form that submits).

    On the live page the generic drive died because NONE of its
    candidates (Sign Up / Create Account / … / raw submit) existed;
    the ladder above tries the actions this page actually offers
    before the raw submit, so the recorded failure cannot recur
    while the page keeps any of its known shapes.
    """
    from ...accounts.provisioning import FlowStepError, _click_text

    if _click_free_plan_start(tab):
        return True
    try:
        _click_text(tab, "Subscribe", "Get Access", "Continue",
                    "Submit", "Sign Up")
        return True
    except FlowStepError as exc:
        record_failure("sources._subscribe_get_access", exc)
    for label in ("Subscribe", "Get Access"):
        try:
            tab.click(label)
            return True
        except Exception:  # noqa: BLE001 - try the next shape
            continue
    try:
        tab.submit("")
        return True
    except Exception:  # noqa: BLE001 - no shape matched
        return False


def odds_api_signup_flow(tab: Any, service: Any, *,
                         name: str, email: str, username: str,
                         password: str,
                         captcha_solver: Any = None) -> Any:
    """Drive the Odds API's REAL signup contract: a subscribe form.

    The service's own page states the contract — "Subscribe to
    receive your API key via email" — and its Get Access section
    offers per-plan START buttons (the Starter plan is free). There
    is no username/password account form for the generic form drive
    to fill, and "Get Access" is the section HEADING, not a control:
    the generic drive's submit step could never match (the owner's
    2026-10-26 drive died at step ``submit_signup`` for exactly that
    reason).

    This flow fills the one field the page has — the email — and
    performs the subscribe action through the shape ladder in
    ``_subscribe_get_access`` (the plan START first). The outcome is
    read off the page that answers: an email acknowledgement is
    ``verification_pending`` (the key arrives in the subscribed
    inbox, where the driver's existing verification path takes
    over); an in-page acceptance is ``created``. ``name`` /
    ``username`` / ``password`` are accepted for the signup-flow
    contract and unused — the service never asks for them.
    """
    from ...accounts.provisioning import (
        _SIGNUP_EXISTS_MARKERS,
        _SIGNUP_SUCCESS_MARKERS,
        _SIGNUP_VERIFY_MARKERS,
        FlowStepError,
        SignupAccountExists,
        SignupFlowResult,
        _click_text,
        _fill_first,
        _solve_or_block,
        _tab_text,
        _wall_block,
    )
    from ...accounts.signup_driver import WallKind, classify_wall

    if not service.signup_url:
        raise FlowStepError(
            "open_signup_page",
            f"no signup URL registered for {service.display}")
    tab.navigate(service.signup_url)
    _solve_or_block(tab, service, captcha_solver=captcha_solver,
                    submitted=False)
    email_fields = ("email", "user_email", "signup_email",
                    "your-email", "mail", "user[email]")
    filled = bool(_fill_first(tab, email_fields, email,
                              step="fill_email", required=False))
    if not _subscribe_get_access(tab):
        raise FlowStepError(
            "submit_signup",
            "no subscribe action matched on the Odds API "
            "get-access page (no plan START control, no labelled "
            "subscribe control, no submittable form) — the page's "
            "shape has changed, so the submit step cannot be "
            "automated against the page as it stands")
    if not filled:
        # The email field only appears once a plan is started.
        _fill_first(tab, email_fields, email, step="fill_email",
                    required=True)
        try:
            _click_text(tab, "Subscribe", "Get Access", "Continue",
                        "Submit", "Sign Up", "START")
        except FlowStepError as exc:
            raise FlowStepError(
                "submit_signup",
                "the post-plan subscribe form offered no confirm "
                "control") from exc
    _solve_or_block(tab, service, captcha_solver=captcha_solver,
                    submitted=True)
    text = _tab_text(tab, max_chars=40000).lower()
    wall, evidence = classify_wall(page_text=text)
    if wall is WallKind.CAPTCHA:
        _solve_or_block(tab, service, captcha_solver=captcha_solver,
                        submitted=True)
        text = _tab_text(tab, max_chars=40000).lower()
        wall, evidence = classify_wall(page_text=text)
    if wall is not WallKind.NONE and wall is not WallKind.CAPTCHA:
        raise _wall_block(service, wall, evidence)
    if any(m in text for m in _SIGNUP_EXISTS_MARKERS):
        raise SignupAccountExists(
            f"{service.display} says an account already exists for "
            f"{email}")
    if any(m in text for m in _SIGNUP_VERIFY_MARKERS) \
            or any(m in text for m in _ODDS_SUBSCRIBE_MARKERS):
        return SignupFlowResult(
            "verification_pending",
            "the Odds API emails the API key to the subscribed "
            "address — the signup completes from that inbox")
    if any(m in text for m in _SIGNUP_SUCCESS_MARKERS):
        return SignupFlowResult(
            "created",
            "the subscribe step was accepted in-page; the account "
            "page carries the API key")
    raise FlowStepError(
        "confirm_signup",
        "after subscribing, the page matched no known outcome (no "
        "email acknowledgement, no success signal, no wall) — the "
        "post-submit step of this signup does not match the page, "
        "so it is the step that cannot be automated as-is")


def ensure_provisioning() -> list[str]:
    """Register the desk's keyed data services with the REAL
    self-provisioning registry (idempotent — later registrations win,
    and re-registering the same entry is a no-op in effect).

    Returns the service ids ensured.  The driver can then provision
    them like any other service: signup/login on the owner's own
    account, key page flow (above), validation BEFORE the vault write
    counts, vault->env bridge (``ODDS_API_KEY`` /
    ``FOOTBALL_DATA_API_KEY`` are read per-call by the analyst's
    fetchers, so a provisioned key takes effect with no restart).
    """
    from ...accounts.provisioning import (
        ProvisionService, _PAGE_FLOWS, _SIGNUP_FLOWS, _VALIDATORS,
        register_provision_service)

    _VALIDATORS.setdefault("odds_api_key", _validate_odds_api)
    _VALIDATORS.setdefault("football_data_key", _validate_football_data)
    _PAGE_FLOWS.setdefault("odds_api_key_v1", odds_api_key_flow)
    _PAGE_FLOWS.setdefault("football_data_key_v1",
                           football_data_key_flow)
    _SIGNUP_FLOWS.setdefault("odds_api_signup_v1",
                             odds_api_signup_flow)

    register_provision_service(ProvisionService(
        id=ODDS_API_SERVICE,
        display="The Odds API",
        aliases=("oddsapi", "the-odds-api"),
        vault_service="the-odds-api",
        env_keys=("ODDS_API_KEY",),
        login_url="https://the-odds-api.com/login/",
        token_url="https://the-odds-api.com/account/",
        signup_url="https://the-odds-api.com/#get-access",
        validator="odds_api_key",
        flow="odds_api_key_v1",
        signup_flow="odds_api_signup_v1",
    ))
    register_provision_service(ProvisionService(
        id=FOOTBALL_DATA_SERVICE,
        display="football-data.org",
        aliases=("footballdata",),
        vault_service="football-data",
        env_keys=("FOOTBALL_DATA_API_KEY",),
        login_url="https://www.football-data.org/client/login",
        token_url="https://www.football-data.org/client/home",
        signup_url="https://www.football-data.org/client/register",
        validator="football_data_key",
        flow="football_data_key_v1",
        signup_flow="form_signup_v1",
    ))
    # ``/provision desk`` — the desk word drives both keyed services
    # in turn (owner's usage, 2026-10-23: the bare word hard-failed
    # before the bundle existed).
    from ...accounts.provisioning import register_provision_bundle

    register_provision_bundle(
        "desk", (ODDS_API_SERVICE, FOOTBALL_DATA_SERVICE))
    return [ODDS_API_SERVICE, FOOTBALL_DATA_SERVICE]
