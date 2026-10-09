"""Broader live sweep: bigger samples across strong sources (built-in +
new candidates) to find which sources yield WORKING proxies right now."""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/hatch/workspace/devon")
from nomorals.tools.proxylab import ProxyScraper, ProxyTester

UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 "
      "Firefox/128.0")

CANDIDATES = [
    ("monosans-http",
     "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
     "list", 15),
    ("monosans-socks5",
     "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt",
     "list", 15),
    ("spysme-http", "https://spys.me/proxy.txt", "list", 15),
    ("openproxylist-http", "https://api.openproxylist.xyz/http.txt",
     "list", 15),
    ("thespeedx-socks5",
     "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
     "list", 15),
    ("geonode", "https://proxylist.geonode.com/api/proxy-list"
     "?limit=100&page=1&sort_by=lastChecked&sort_type=desc", "json", 15),
    ("proxydb-http", "https://proxydb.net/?protocol=http", "html", 15),
    ("proxydb-socks5", "https://proxydb.net/?protocol=socks5", "html", 15),
    ("freeproxyworld-socks5",
     "https://www.freeproxy.world/?type=socks5", "html", 15),
    ("advancedname", "https://advanced.name/freeproxy", "html", 15),
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
        # spread the sample across the list (front/middle/back), not
        # just the first N — dead heads are common
        n = len(parsed)
        idxs = sorted({0, n // 3, n // 2, 2 * n // 3, n - 1} |
                      {i * n // sample for i in range(sample) if n})
        sampled = [parsed[i] for i in idxs if i < n][:sample]
        t0 = time.time()
        tester.test_many(sampled, limit=len(sampled))
        working = [p for p in sampled if p.alive]
        entry.update(sampled=len(sampled), working=len(working),
                     seconds=round(time.time() - t0, 1),
                     working_proxies=[
                         {"url": p.url, "latency_ms": p.latency_ms,
                          "anonymity": p.anonymity, "egress_ip": p.egress_ip}
                         for p in working])
        print(f"    parsed {len(parsed)} | LIVE {len(working)}/"
              f"{len(sampled)} working ({entry['seconds']}s)", flush=True)
        for p in working:
            print(f"      OK {p.url} {p.latency_ms}ms {p.anonymity} "
                  f"egress={p.egress_ip}", flush=True)
        report["sources"][name] = entry
    out = "/home/hatch/workspace/devon/scripts/proxylab/sweep.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
