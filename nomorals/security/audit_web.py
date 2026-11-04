"""ExposureScanner: audit a live site's PUBLIC surface.

The Lovable-study failure shapes, checked deterministically:

- exposed files: ``/.env`` and friends, ``/.git`` — validated
  by content shape, so a catch-all SPA that answers 200 with
  HTML for every path produces NO findings (a status code is
  not evidence; content is);
- unauthenticated data backends: PostgREST/Supabase-style REST
  roots (tables listed, a bounded sample read, the source's
  own total taken from its Content-Range), Firebase RTDB
  ``/.json`` reads, public storage-bucket listings;
- client-side secret mining: the landing page + its JS bundles
  run through the shared pattern battery — a service_role JWT
  in a bundle is a master key shipped to every visitor;
- API surface discovery: the endpoints the app itself calls
  (fetch/axios literals in its own code), each probed for
  "answers with no credential at all".

Discipline, implemented in the fetch layer (not in prompts):
the scanner sends NO credentials (the one exception is a
client-shipped anon key found in the target's own bundle —
public by design, and the finding says which door was used);
it attempts no auth bypass; per-host pacing keeps request
volume at ordinary-use levels under a hard per-scan budget;
and third-party collection reads are row-capped in this layer
(:data:`audit_evidence.DEFAULT_SAMPLE_CAP`) — the pack proves
the exposure, it does not copy the database.  The owner's own
assets (registry-declared) read at the own-asset cap.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .audit_evidence import (
    DEFAULT_SAMPLE_CAP, OWN_ASSET_SAMPLE_CAP, build_pack,
    field_sensitivity, schema_of)
from .audit_patterns import jwt_payload, scan_text_for_secrets

__all__ = [
    "Fetcher", "FetchResult", "ScanBudgetExceeded",
    "ExposureScanner", "scan_url",
    "EXPOSED_FILE_CHECKS", "MAX_REQUESTS_PER_SCAN",
]

#: Hard per-scan request budget: an audit is a visitor, not a
#: crawler.  Every check shares this budget.
MAX_REQUESTS_PER_SCAN = 80

_UA = "Mozilla/5.0 (compatible; DevonSecurityAudit/1.0)"


class ScanBudgetExceeded(Exception):
    """The scan's request budget ran out mid-battery."""


@dataclass
class FetchResult:
    status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    final_url: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status is not None and self.error == ""

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except (ValueError, TypeError):
            return None


def _host_is_private(host: str) -> bool:
    """True when the host is, or resolves to, a non-public
    address.  Any private answer blocks (no rebinding hole)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # unresolvable: fail closed for the gate
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local \
                or ip.is_reserved or ip.is_multicast \
                or ip.is_unspecified:
            return True
    return False


class Fetcher:
    """The scanner's only hands.  Pacing, budget, byte caps and
    the private-target gate all live here so no check can
    accidentally out-run them."""

    def __init__(self, *, min_interval_s: float = 0.35,
                 timeout_s: float = 10.0,
                 max_requests: int = MAX_REQUESTS_PER_SCAN,
                 allow_private: bool = False) -> None:
        self.min_interval_s = float(min_interval_s)
        self.timeout_s = float(timeout_s)
        self.max_requests = int(max_requests)
        self.allow_private = bool(allow_private)
        self.requests_made = 0
        self._last_by_host: dict[str, float] = {}
        self._private_cache: dict[str, bool] = {}

    def _gate(self, url: str) -> str:
        """'' when the fetch may proceed, else the refusal."""
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return f"scheme {parts.scheme!r} is not http(s)"
        host = parts.hostname or ""
        if not host:
            return "no host in URL"
        if not self.allow_private:
            cached = self._private_cache.get(host)
            if cached is None:
                cached = _host_is_private(host)
                self._private_cache[host] = cached
            if cached:
                return (f"host {host} resolves to a private/"
                        "loopback address — refused for a "
                        "non-operator scan")
        return ""

    def get(self, url: str, *, headers: dict[str, str] | None = None,
            max_bytes: int = 2_000_000) -> FetchResult:
        if self.requests_made >= self.max_requests:
            raise ScanBudgetExceeded(
                f"request budget ({self.max_requests}) exhausted")
        refusal = self._gate(url)
        if refusal:
            return FetchResult(final_url=url, error=refusal)
        host = urllib.parse.urlsplit(url).netloc
        last = self._last_by_host.get(host, 0.0)
        wait = self.min_interval_s - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_by_host[host] = time.time()
        self.requests_made += 1
        req_headers = {"User-Agent": _UA, "Accept": "*/*"}
        req_headers.update(headers or {})
        req = urllib.request.Request(url, headers=req_headers,
                                     method="GET")
        started = time.time()
        try:
            with urllib.request.urlopen(
                    req, timeout=self.timeout_s) as resp:
                body = resp.read(max_bytes + 1)
                return FetchResult(
                    status=resp.status,
                    headers={k.lower(): v for k, v in
                             resp.headers.items()},
                    body=body[:max_bytes],
                    final_url=resp.geturl(),
                    elapsed_s=time.time() - started,
                    truncated=len(body) > max_bytes)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read(max_bytes + 1)
            except Exception:  # noqa: BLE001 - error body is a
                body = b""        # courtesy, not a requirement
            return FetchResult(
                status=exc.code,
                headers={k.lower(): v for k, v in
                         (exc.headers.items() if exc.headers
                          else [])},
                body=body[:max_bytes], final_url=url,
                elapsed_s=time.time() - started,
                truncated=len(body) > max_bytes)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return FetchResult(final_url=url,
                               error=f"{type(exc).__name__}: {exc}",
                               elapsed_s=time.time() - started)


# ── finding helpers ──────────────────────────────────────────
def _finding(check: str, kind: str, title: str, location: str,
             detail: str, pack: dict[str, Any] | None = None,
             **meta: Any) -> dict[str, Any]:
    return {"source": "web", "check": check, "kind": kind,
            "title": title, "location": location,
            "detail": detail, "evidence": pack, "meta": meta}


def _rows_from_json(data: Any) -> list[dict[str, Any]] | None:
    """Rows when the payload is a collection (a bare list, or a
    list under a data/results/items/records key)."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in ("data", "results", "items", "records", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return None


def _total_from_headers(headers: dict[str, str],
                        data: Any) -> int | None:
    """The source's own count: PostgREST's Content-Range
    (``0-4/137``), an X-Total-Count, or a count/total field."""
    cr = headers.get("content-range", "")
    if "/" in cr:
        tail = cr.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    for name in ("x-total-count", "x-total", "total-count"):
        value = headers.get(name, "").strip()
        if value.isdigit():
            return int(value)
    if isinstance(data, dict):
        for key in ("total", "count", "total_count", "totalCount"):
            value = data.get(key)
            if isinstance(value, int) and value >= 0:
                return value
    return None


# ── exposed-file battery ─────────────────────────────────────
def _looks_like_env(text: str) -> bool:
    keys = 0
    sensitive = 0
    for line in text.splitlines()[:60]:
        m = re.match(r"\s*(?:export\s+)?([A-Z][A-Z0-9_]{2,})\s*=",
                     line)
        if m:
            keys += 1
            if re.search(r"SECRET|KEY|PASSWORD|TOKEN|DSN|DATABASE|"
                         r"PRIVATE|CREDENTIAL", m.group(1)):
                sensitive += 1
    return keys >= 2 and sensitive >= 1


#: (path, kind, validator-name) — a 200 is never enough; the
#: validator must recognise the file's own shape.
EXPOSED_FILE_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("/.env", "exposed_file", "env"),
    ("/.env.local", "exposed_file", "env"),
    ("/.env.production", "exposed_file", "env"),
    ("/.env.bak", "exposed_file", "env"),
    ("/.git/HEAD", "exposed_git", "git_head"),
    ("/.git/config", "exposed_git", "git_config"),
    ("/backup.sql", "exposed_file", "sql_dump"),
    ("/dump.sql", "exposed_file", "sql_dump"),
    ("/db.sql", "exposed_file", "sql_dump"),
    ("/wp-config.php.bak", "exposed_file", "wp_config"),
    ("/config.php.bak", "exposed_file", "php_config"),
    ("/.aws/credentials", "exposed_file", "aws_creds"),
    ("/actuator/env", "exposed_file", "actuator"),
    ("/server-status", "exposed_file", "server_status"),
)


def _validate_exposed(validator: str, text: str) -> bool:
    head = text[:4000]
    if validator == "env":
        return _looks_like_env(head)
    if validator == "git_head":
        return head.startswith("ref:") or bool(
            re.match(r"^[0-9a-f]{40}\s*$", head.strip()))
    if validator == "git_config":
        return "[core]" in head and "repositoryformatversion" in head
    if validator == "sql_dump":
        return "INSERT INTO" in head or "CREATE TABLE" in head
    if validator == "wp_config":
        return "DB_PASSWORD" in head or "table_prefix" in head
    if validator == "php_config":
        return "<?php" in head and ("password" in head.lower()
                                    or "db_" in head.lower())
    if validator == "aws_creds":
        return "[default]" in head or "aws_access_key_id" in head
    if validator == "actuator":
        return '"propertySources"' in head or '"systemEnvironment"' in head
    if validator == "server_status":
        return "Apache Server Status" in head or "Total accesses" in head
    return False


_SUPABASE_RX = re.compile(r"https://[a-z0-9-]+\.supabase\.co")
_FIREBASE_RX = re.compile(
    r"https://[a-z0-9-]+(?:-default-rtdb)?\.[a-z0-9-]*"
    r"firebasedatabase\.app|https://[a-z0-9-]+\.firebaseio\.com")
_BUCKET_RX = re.compile(
    r"https://[a-z0-9.-]+\.s3[.-][a-z0-9-]*\.amazonaws\.com|"
    r"https://storage\.googleapis\.com/[a-z0-9._-]+|"
    r"https://[a-z0-9._-]+\.storage\.googleapis\.com")
_FETCH_RX = re.compile(
    r"(?:fetch|axios\.(?:get|post|put|delete|request)|"
    r"\.open\(\s*[\"'](?:GET|POST|PUT|DELETE)[\"']\s*,)"
    r"\s*\(?\s*[\"'`]([^\"'`]{3,200})[\"'`]")
_URL_RX = re.compile(r"[\"'`](https?://[^\"'`\s]{5,200})[\"'`]")
_API_PATH_RX = re.compile(r"[\"'`](/(?:api|rest|v\d|graphql)[^\"'`\s]{2,160})[\"'`]")
_SCRIPT_SRC_RX = re.compile(r"<script[^>]+src=[\"']([^\"']+)[\"']",
                            re.IGNORECASE)
_SCRIPT_BODY_RX = re.compile(r"<script[^>]*>(.*?)</script>",
                             re.IGNORECASE | re.DOTALL)


class ExposureScanner:
    """One target, one battery.  Construct with a Fetcher to
    share pacing/budget across the checks."""

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self.fetcher = fetcher or Fetcher()

    # ── collection reading (row caps live HERE) ──────────────
    def read_collection(self, url: str, *,
                        headers: dict[str, str] | None = None,
                        sample_cap: int) -> dict[str, Any] | None:
        """GET a JSON collection.  Returns {schema, total,
        sample, result} with the sample cut to ``sample_cap``
        in this layer — or None when the URL does not answer
        with a readable collection."""
        res = self.fetcher.get(url, headers=headers)
        if res.status not in (200, 206):  # 206: PostgREST's
            return None                   # ranged-read answer
        data = res.json()
        rows = _rows_from_json(data)
        if rows is None:
            return None
        total = _total_from_headers(res.headers, data)
        if total is None and isinstance(data, list):
            total = len(data)
        return {"schema": schema_of(rows[:max(sample_cap, 1)]),
                "total": total,
                "sample": rows[:sample_cap],
                "result": res}

    # ── individual checks ────────────────────────────────────
    def check_exposed_files(self, base: str, *,
                            own_asset: bool) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for path, kind, validator in EXPOSED_FILE_CHECKS:
            url = base.rstrip("/") + path
            res = self.fetcher.get(url, max_bytes=200_000)
            if res.status != 200 or not _validate_exposed(
                    validator, res.text):
                continue
            hits = scan_text_for_secrets(res.text[:50_000])
            pack = build_pack(
                check="exposed_files", target=url,
                request={"method": "GET", "url": url},
                response={"status": res.status,
                          "headers": res.headers,
                          "content_type":
                              res.headers.get("content-type", "")},
                own_asset=own_asset,
                notes=f"content validated as {validator}; "
                      f"{len(hits)} secret-shaped value(s) inside")
            title = f"exposed file {path} served to anyone"
            if hits:
                title += " — contains secret-shaped values"
            findings.append(_finding(
                "exposed_files", kind, title, url,
                f"{path} answers 200 with content that validates "
                f"as {validator}. Secret-shaped values inside: "
                + (", ".join(f"{h.pattern} {h.redacted}"
                             for h in hits[:5]) or "none"),
                pack, secret_patterns=[h.pattern for h in hits]))
        return findings

    def check_client_secrets(self, base: str, html: str,
                             bundles: list[tuple[str, str]], *,
                             own_asset: bool) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        sources = [(base, html)] + bundles
        for url, text in sources:
            for hit in scan_text_for_secrets(text):
                role = ""
                if hit.pattern == "jwt":
                    claims = jwt_payload(hit.value)
                    role = str(claims.get("role") or "")
                kind = "client_secret"
                title = (f"{hit.pattern} shipped to every visitor "
                         f"in {urllib.parse.urlsplit(url).path or url}")
                if role == "service_role":
                    kind = "client_master_key"
                    title = ("service_role key shipped in client "
                             "code — a master key in the bundle")
                elif hit.pattern in ("firebase_api_key",) \
                        or (hit.pattern == "jwt" and role == "anon"):
                    kind = "client_config_key"
                    title = ("client config key in shipped code "
                             "(public by design — noted, not a leak)")
                pack = build_pack(
                    check="client_secrets", target=url,
                    request={"method": "GET", "url": url},
                    response={"status": 200},
                    own_asset=own_asset,
                    notes=f"pattern {hit.pattern}"
                          + (f", jwt role={role}" if role else ""))
                findings.append(_finding(
                    "client_secrets", kind, title, url,
                    f"pattern {hit.pattern}: {hit.redacted}"
                    + (f" (JWT role: {role})" if role else ""),
                    pack, pattern=hit.pattern, jwt_role=role))
        return findings

    def discover_endpoints(self, base: str, html: str,
                           bundles: list[tuple[str, str]]) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()

        def add(raw: str) -> None:
            raw = raw.strip()
            if not raw or raw.startswith(("data:", "blob:", "#")):
                return
            url = urllib.parse.urljoin(base, raw)
            if url not in seen:
                seen.add(url)
                found.append(url)

        for _url, text in [(base, html)] + bundles:
            for m in _FETCH_RX.finditer(text):
                add(m.group(1))
            for m in _API_PATH_RX.finditer(text):
                add(m.group(1))
        return found[:24]

    def check_api_probes(self, endpoints: list[str], *,
                         own_asset: bool,
                         sample_cap: int) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for url in endpoints[:12]:
            got = self.read_collection(url, sample_cap=sample_cap)
            if got is None:
                # a single-record JSON object with personal shape
                # is still an open door
                res = self.fetcher.get(url)
                if res.status != 200:
                    continue
                data = res.json()
                if not isinstance(data, dict) or _rows_from_json(data) is not None:
                    continue
                schema = schema_of([data])
                tier, _fields = field_sensitivity(schema)
                if tier < 1:
                    continue
                got = {"schema": schema, "total": None,
                       "sample": [data], "result": res}
            schema = got["schema"]
            tier, fields = field_sensitivity(schema)
            pack = build_pack(
                check="api_probe", target=url,
                request={"method": "GET", "url": url},
                response={"status": 200,
                          "headers": got["result"].headers,
                          "content_type": got["result"].headers.get(
                              "content-type", "")},
                schema=schema, total_count=got["total"],
                sample=got["sample"], own_asset=own_asset,
                notes="answered with NO credential of any kind")
            findings.append(_finding(
                "api_probe", "unauthenticated_api",
                f"API route answers with no auth: "
                f"{urllib.parse.urlsplit(url).path}",
                url,
                f"{url} returned data to an unauthenticated GET. "
                f"Fields: {', '.join(list(schema)[:12]) or '—'}"
                + (f"; sensitive: {', '.join(fields)}" if fields
                   else "")
                + (f"; source states {got['total']} records"
                   if got["total"] is not None else ""),
                pack, sensitivity_tier=tier,
                total_count=got["total"]))
        return findings

    def check_postgrest(self, roots: list[str], *,
                        anon_key: str = "", own_asset: bool,
                        sample_cap: int) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for root in roots[:3]:
            root = root.rstrip("/")
            spec = self.fetcher.get(
                root + "/", headers={
                    "Accept": "application/openapi+json"})
            if spec.status != 200:
                continue
            data = spec.json()
            if not isinstance(data, dict) \
                    or not isinstance(data.get("paths"), dict):
                continue
            tables = [p.lstrip("/") for p in data["paths"]
                      if p.startswith("/") and "{" not in p][:6]
            for table in tables:
                url = f"{root}/{table}"
                got = self._postgrest_table(
                    url, anon_key="", sample_cap=sample_cap)
                door = "no key at all"
                if got is None and anon_key:
                    got = self._postgrest_table(
                        url, anon_key=anon_key,
                        sample_cap=sample_cap)
                    door = "the anon key shipped in the bundle"
                if got is None:
                    continue
                schema = got["schema"]
                tier, fields = field_sensitivity(schema)
                pack = build_pack(
                    check="postgrest", target=url,
                    request={"method": "GET", "url": url,
                             "headers": {"Prefer": "count=exact"}},
                    response={"status": 200,
                              "headers": got["result"].headers,
                              "content_type": "application/json"},
                    schema=schema, total_count=got["total"],
                    sample=got["sample"], own_asset=own_asset,
                    notes=f"table readable via {door}")
                findings.append(_finding(
                    "postgrest", "public_backend",
                    f"database table '{table}' readable via {door}",
                    url,
                    f"PostgREST table {table}: "
                    f"{got['total'] if got['total'] is not None else '?'} "
                    f"records per the source's own Content-Range. "
                    f"Fields: {', '.join(list(schema)[:12])}"
                    + (f"; sensitive: {', '.join(fields)}" if fields
                       else ""),
                    pack, sensitivity_tier=tier,
                    total_count=got["total"], table=table,
                    door=door))
        return findings

    def _postgrest_table(self, url: str, *, anon_key: str,
                         sample_cap: int) -> dict[str, Any] | None:
        headers = {"Prefer": "count=exact",
                   "Range-Unit": "items",
                   "Range": f"0-{max(sample_cap - 1, 0)}"}
        if anon_key:
            headers["apikey"] = anon_key
            headers["Authorization"] = f"Bearer {anon_key}"
        return self.read_collection(url, headers=headers,
                                    sample_cap=sample_cap)

    def check_firebase(self, bases: list[str], *, own_asset: bool,
                       sample_cap: int) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for db in bases[:3]:
            db = db.rstrip("/")
            res = self.fetcher.get(db + "/.json?shallow=true")
            if res.status != 200:
                continue
            data = res.json()
            if not isinstance(data, dict) or not data:
                continue
            sample_res = self.fetcher.get(
                db + "/.json?orderBy=%22$key%22"
                f"&limitToFirst={sample_cap}")
            rows: list[dict[str, Any]] = []
            sample_data = sample_res.json()
            if isinstance(sample_data, dict):
                rows = [{"_key": k,
                         **(v if isinstance(v, dict) else
                            {"_value_type": type(v).__name__})}
                        for k, v in list(sample_data.items())]
            schema = schema_of(rows)
            tier, fields = field_sensitivity(schema)
            pack = build_pack(
                check="firebase_rtdb", target=db + "/.json",
                request={"method": "GET", "url": db + "/.json"},
                response={"status": 200, "headers": res.headers,
                          "content_type": "application/json"},
                schema=schema, total_count=None, sample=rows,
                own_asset=own_asset,
                notes=f"{len(data)} top-level keys listed via "
                      "shallow read, no credential sent")
            findings.append(_finding(
                "firebase_rtdb", "public_backend",
                "Firebase database readable with no auth",
                db + "/.json",
                f"{db} answered /.json unauthenticated: "
                f"{len(data)} top-level keys "
                f"({', '.join(list(data)[:8])})"
                + (f"; sensitive fields: {', '.join(fields)}"
                   if fields else ""),
                pack, sensitivity_tier=tier,
                top_level_keys=len(data)))
        return findings

    def check_buckets(self, urls: list[str], *, own_asset: bool,
                      ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for url in urls[:3]:
            res = self.fetcher.get(url)
            if res.status != 200 \
                    or "<ListBucketResult" not in res.text[:2000]:
                continue
            keys = re.findall(r"<Key>([^<]{1,200})</Key>",
                              res.text)[:20]
            pack = build_pack(
                check="storage_bucket", target=url,
                request={"method": "GET", "url": url},
                response={"status": 200, "headers": res.headers,
                          "content_type":
                              res.headers.get("content-type", "")},
                own_asset=own_asset,
                notes="bucket listing is public; object keys "
                      "named, objects NOT downloaded")
            findings.append(_finding(
                "storage_bucket", "public_bucket",
                "storage bucket lists its objects publicly", url,
                f"{url} returns a public listing; first keys: "
                + (", ".join(keys[:8]) or "—"),
                pack, listed_keys=len(keys)))
        return findings

    # ── the battery ──────────────────────────────────────────
    def scan(self, url: str, *, own_asset: bool = False) -> dict[str, Any]:
        started = time.time()
        notes: list[str] = []
        checks_run: list[str] = []
        findings: list[dict[str, Any]] = []
        sample_cap = OWN_ASSET_SAMPLE_CAP if own_asset \
            else DEFAULT_SAMPLE_CAP
        target = url.strip()
        if "://" not in target:
            target = "https://" + target
        base = target.rstrip("/")
        origin = urllib.parse.urlsplit(target)
        site_root = f"{origin.scheme}://{origin.netloc}"

        landing = self.fetcher.get(target)
        if landing.error or landing.status is None:
            return {"target": target, "own_asset": own_asset,
                    "checks_run": [], "findings": [],
                    "notes": [f"target unreachable: "
                              f"{landing.error or 'no response'}"],
                    "requests_made": self.fetcher.requests_made,
                    "started_at": started,
                    "finished_at": time.time()}
        if landing.status >= 400:
            notes.append(
                f"landing answered {landing.status} — the battery "
                "still ran (exposed files do not need a homepage)")
        html = landing.text if landing.status == 200 else ""

        # JS bundles (bounded): srcs from the landing page
        bundles: list[tuple[str, str]] = []
        for src in _SCRIPT_SRC_RX.findall(html)[:8]:
            burl = urllib.parse.urljoin(target, src)
            bres = self.fetcher.get(burl, max_bytes=1_500_000)
            if bres.status == 200:
                bundles.append((burl, bres.text))
        inline = "\n".join(_SCRIPT_BODY_RX.findall(html))
        if inline.strip():
            bundles.append((target + "#inline", inline))

        try:
            checks_run.append("exposed_files")
            findings += self.check_exposed_files(
                site_root, own_asset=own_asset)

            checks_run.append("client_secrets")
            findings += self.check_client_secrets(
                target, html, bundles, own_asset=own_asset)

            checks_run.append("api_probe")
            endpoints = self.discover_endpoints(target, html, bundles)
            findings += self.check_api_probes(
                endpoints, own_asset=own_asset,
                sample_cap=sample_cap)

            checks_run.append("postgrest")
            roots: list[str] = []
            corpus = html + "\n" + "\n".join(t for _u, t in bundles)
            for m in _SUPABASE_RX.finditer(corpus):
                roots.append(m.group(0) + "/rest/v1")
            anon_key = ""
            for hit in scan_text_for_secrets(corpus):
                if hit.pattern == "jwt" and jwt_payload(
                        hit.value).get("role") == "anon":
                    anon_key = hit.value
                    break
            findings += self.check_postgrest(
                roots, anon_key=anon_key, own_asset=own_asset,
                sample_cap=sample_cap)

            checks_run.append("firebase_rtdb")
            fb_bases = list({m.group(0)
                             for m in _FIREBASE_RX.finditer(corpus)})
            findings += self.check_firebase(
                fb_bases, own_asset=own_asset,
                sample_cap=sample_cap)

            checks_run.append("storage_bucket")
            bucket_urls = list({m.group(0)
                                for m in _BUCKET_RX.finditer(corpus)})
            findings += self.check_buckets(
                bucket_urls, own_asset=own_asset)
        except ScanBudgetExceeded as exc:
            notes.append(f"battery stopped early: {exc}")

        return {"target": target, "own_asset": own_asset,
                "checks_run": checks_run, "findings": findings,
                "notes": notes,
                "requests_made": self.fetcher.requests_made,
                "started_at": started, "finished_at": time.time()}


def scan_url(url: str, *, own_asset: bool = False,
             allow_private: bool = False,
             min_interval_s: float = 0.35) -> dict[str, Any]:
    """One-shot convenience: build a disciplined fetcher and
    run the whole battery."""
    fetcher = Fetcher(allow_private=allow_private,
                      min_interval_s=min_interval_s)
    return ExposureScanner(fetcher).scan(url, own_asset=own_asset)
