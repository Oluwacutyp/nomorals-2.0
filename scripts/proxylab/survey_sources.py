"""Live survey: fetch candidate proxy-list endpoints, parse them with the
lab's real parsers, report what each yields RIGHT NOW. No guesses."""
import sys
import time
import urllib.request

sys.path.insert(0, "/home/hatch/workspace/devon")
from nomorals.tools.proxylab import ProxyScraper

UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 "
      "Firefox/128.0")

CANDIDATES = [
    # (name, url, kind) — new candidates not in the built-in catalog
    ("proxydb-http", "https://proxydb.net/?protocol=http", "html"),
    ("proxydb-socks4", "https://proxydb.net/?protocol=socks4", "html"),
    ("proxydb-socks5-ng", "https://proxydb.net/?protocol=socks5&country=NG", "html"),
    ("proxydb-socks5-eu-de", "https://proxydb.net/?protocol=socks5&country=DE", "html"),
    ("proxynova-http", "https://www.proxynova.com/proxy-server-list/", "html"),
    ("proxynova-socks5", "https://www.proxynova.com/proxy-server-list/?port=1080", "html"),
    ("my-proxy-http", "https://free-proxy-list.net/en/", "html"),
    ("openproxy-spy", "https://openproxy.space/", "html"),
    # existing catalog entries to re-verify (spot check)
    ("CAT-adv", "https://advanced.name/freeproxy", "html"),
    ("CAT-fpl", "https://free-proxy-list.net/", "html"),
    ("CAT-proxydb-s5", "https://proxydb.net/?protocol=socks5", "html"),
    ("CAT-spysme-http", "https://spys.me/proxy.txt", "list"),
    ("CAT-spysme-socks", "https://spys.me/socks.txt", "list"),
    ("CAT-geonode", "https://proxylist.geonode.com/api/proxy-list?limit=200&page=1&sort_by=lastChecked&sort_type=desc", "json"),
    ("CAT-openproxylist", "https://api.openproxylist.xyz/http.txt", "list"),
    ("CAT-thespeedx-socks5", "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt", "list"),
    ("CAT-monosans-json", "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies.json", "json"),
    ("CAT-proxyscrape-v4-socks5", "https://api.proxyscrape.com/v4/free-proxy-list/get?request=displayproxies&proxy_format=ipport&format=text&country=all&proxy_type=socks5", "list"),
]


def fetch(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> None:
    scraper = ProxyScraper()
    for name, url, kind in CANDIDATES:
        t0 = time.time()
        try:
            raw = fetch(url)
            text = raw.decode("utf-8", "replace")
            if kind == "list":
                parsed = scraper.parse_list(text, "http")
            elif kind == "json":
                parsed = scraper.parse_json(text)
            else:
                parsed = scraper.parse_html(text)
            schemes = {}
            for p in parsed:
                schemes[p.scheme] = schemes.get(p.scheme, 0) + 1
            print(f"OK   {name:28s} {len(parsed):5d} proxies schemes={schemes} "
                  f"({time.time()-t0:.1f}s, {len(raw)//1024}KB)")
            # sample 3 for eyeball validation
            for p in parsed[:3]:
                print(f"     -> {p.url} country={p.country!r}")
        except Exception as exc:
            print(f"FAIL {name:28s} {type(exc).__name__}: {str(exc)[:110]}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
