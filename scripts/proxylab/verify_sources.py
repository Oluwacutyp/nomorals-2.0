"""Live-verify candidate proxy sources: fetch the list, parse it with the
lab's real parsers, then TEST A SAMPLE OF PROXIES FOR REAL (connect
through them, confirm they relay). Only sources with >=1 working proxy
should be added to the catalog.

Usage: python3 scripts/proxylab/verify_sources.py
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/hatch/workspace/devon")
from nomorals.tools.proxylab import ProxyScraper, ProxyTester

UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 "
      "Firefox/128.0")

# (name, url, kind, sample_per_scheme) — new candidates + the existing
# proxydb-socks5 (must be re-verified after the parser fix)
CANDIDATES = [
    ("proxydb-http", "https://proxydb.net/?protocol=http", "html", 8),
    ("proxydb-socks5", "https://proxydb.net/?protocol=socks5", "html", 8),
    ("freeproxyworld-socks5",
     "https://www.freeproxy.world/?type=socks5", "html", 8),
    ("freeproxyworld-socks4",
     "https://www.freeproxy.world/?type=socks4", "html", 4),
]


def fetch(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> None:
    scraper = ProxyScraper()
    tester = ProxyTester(timeout=6.0, max_workers=12, detect_country=False)
    report: dict = {"sources": {}, "ts": time.time()}
    for name, url, kind, sample in CANDIDATES:
        print(f"--- {name}", flush=True)
        entry: dict = {"url": url, "kind": kind}
        try:
            text = fetch(url).decode("utf-8", "replace")
            parsed = (scraper.parse_list(text, "http") if kind == "list"
                      else scraper.parse_json(text) if kind == "json"
                      else scraper.parse_html(text))
        except Exception as exc:  # noqa: BLE001
            entry["fetch_error"] = f"{type(exc).__name__}: {exc}"[:160]
            print(f"    FETCH FAILED: {entry['fetch_error']}")
            report["sources"][name] = entry
            continue
        entry["parsed"] = len(parsed)
        by_scheme: dict[str, int] = {}
        for p in parsed:
            by_scheme[p.scheme] = by_scheme.get(p.scheme, 0) + 1
        entry["by_scheme"] = by_scheme
        print(f"    parsed {len(parsed)} proxies: {by_scheme}", flush=True)
        # sample per scheme, test for real
        sampled = []
        for scheme in sorted(by_scheme):
            sampled.extend([p for p in parsed if p.scheme == scheme][:sample])
        t0 = time.time()
        tester.test_many(sampled, limit=len(sampled))
        working = [p for p in sampled if p.alive]
        entry["sampled"] = len(sampled)
        entry["working"] = len(working)
        entry["seconds"] = round(time.time() - t0, 1)
        entry["working_proxies"] = [
            {"url": p.url, "latency_ms": p.latency_ms,
             "anonymity": p.anonymity, "egress_ip": p.egress_ip,
             "claimed_anonymity": p.claimed_anonymity}
            for p in working]
        print(f"    LIVE: {len(working)}/{len(sampled)} working "
              f"({entry['seconds']}s)", flush=True)
        for p in working:
            print(f"      OK {p.url} {p.latency_ms}ms "
                  f"{p.anonymity} egress={p.egress_ip}", flush=True)
        report["sources"][name] = entry
    out = "/home/hatch/workspace/devon/scripts/proxylab/verification.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
