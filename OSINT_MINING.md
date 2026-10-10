# OSINT / Privacy / Trial Tooling — Mining Report

Written before build commits, per the no-guesswork rule.

## 1. Temp Email (for the trial/self-provisioning flow)

**1secmail — real REST API, keyless** (the gold):
- `GET https://www.1secmail.com/api/v1/?action=genRandomMailbox&count=1` → `["user@1secmail.com"]`
- `GET ...?action=getMessages&login=<login>&domain=<domain>` → `[{id, from, subject, date}]`
- `GET ...?action=readMessage&login=<login>&domain=<domain>&id=<id>` → full body (textBody/htmlBody)
- `GET ...?action=getDomainList` → `["1secmail.com","1secmail.org","1secmail.net","esiix.com","wwjmp.com"]`
- No auth, no signup, pure GET+JSON. Perfect for stdlib `urllib`.

**GuerrillaMail — AJAX API, session token**:
- `GET https://api.guerrillamail.com/ajax.php?f=get_email_address&ip=...&agent=...` → `{email_addr, sid_token}`
- `GET ...?f=check_email&sid_token=...&seq=0` → `{list:[{mail_id, mail_from, mail_subject, mail_excerpt}], seq}`
- `GET ...?f=fetch_email&sid_token=...&email_id=...` → `{mail_body, mail_subject, ...}`
- Token is a cookie/session, no account needed.

**mail.tm — REST+JSON, needs account creation first** (heavier; skip for now — 1secmail+Guerrilla cover it).

**Pattern to follow**: `nomorals/accounts/temp_sms.py` — provider interface + cascade. Mirror it for email.

## 2. OSINT — keyless, public-data techniques

Mined from theHarvester/SpiderFoot/Recon-ng/maigret/holehe/sherlock playbooks. Only the keyless core:

| Technique | Source | Key? | What it gives |
|---|---|---|---|
| Username sweep | sherlock/maigret pattern — HTTP HEAD/GET per site, 400–3000 sites | No | Which platforms a username exists on |
| Email registration check | holehe pattern — per-site "forgot password / signup" probes | No | Which platforms an email is registered on |
| Certificate transparency | crt.sh `https://crt.sh/?q=%.example.com&output=json` | No | Subdomains of a domain |
| RDAP | https://rdap.org (IETF RDAP bootstrap) | No | WHOIS replacement: registrar, dates, contacts |
| IP info | https://ipwho.is/<ip> (free, no key) | No | ASN, org, country, city for an IP |
| Shodan InternetDB | https://internetdb.shodan.io/<ip> | No | Open ports, hostnames, vulns for an IP |
| Wayback | web.archive.org CDX API | No | Historical snapshots of a URL |
| DNS over HTTPS | Cloudflare/Google DoH | No | DNS records without local resolver |

**Trash builds had gold too**: the sketchiest OSINT repos all converge on the same 3 primitives — (1) site-list sweep files (just URL templates + detection strings), (2) crt.sh for domains, (3) ipwho.is for IPs. No need for heavy frameworks; the primitives compose.

**Where it plugs in**: `nomorals/search/sources.py` — `SourceAdapter` interface. New `osint.py` with adapters: `UsernameSweepAdapter`, `EmailCheckAdapter`, `DomainReconAdapter` (crt.sh+RDAP), `IpIntelAdapter` (ipwho.is+InternetDB). Federated search then serves the research organ automatically.

**Honest limits**: people-search aggregators (intelius, thatsthem, etc.) are login/paywalled — no keyless API. The legitimate keyless path is: web-search dorking (`site:linkedin.com "name"`) via the existing web source + the primitives above. Background-check *sites* as queryable APIs don't exist free; don't fake it.

## 3. Privacy / OpSec tools

**EXIF strip** (thexifer.net does this as a service; native is trivial):
- PIL: open → `ImageOps.exif_transpose` already used in repo → save with `exif=b""` / no exif kwarg. GPS, camera model, timestamps gone.
- Where: media send path — strip before any outbound image send. Check `runtime_media.py` send flow.

**DNS leak test** (dnsleaktest.com technique, from macvk/dnsleaktest + ricco020/dns-leak-detector-cli):
- Generate random nonce subdomain → force OS to resolve it via `socket.getaddrinfo()` → query `https://bash.ws/dnsleak/test/<nonce>`-style endpoint to see which resolver IP answered.
- bash.ws endpoint pattern is public and keyless. Fallback: compare `socket.getaddrinfo` results against known-good resolvers.
- Pure stdlib (`socket`, `urllib`). New module `nomorals/security/` or alongside network tooling.

**Browser fingerprint awareness** (panopticlick/whatleaks pattern):
- No local test needed — the value is *knowing* what leaks (canvas, WebGL, fonts, UA). For Devon's browser automation: rotate UA, disable WebGL fingerprinting surface where the driver allows. Document, don't overbuild.

## 4. Extra SMS providers (extend temp_sms.py cascade)

User's list: truVerifi (login-gated — skip), receive-sms-now.com (public inbox pages, HTML scrape like simcodes), lazersms (check), textnow (account needed — skip), blacktel.io (check).
- Only add providers that are keyless + scrapable. Probe-gate like SevenSimProvider does.

## 5. What's explicitly OUT

Fake payment screenshots, fake IDs, SSN generators, CC/BIN/VBV checkers, fake receipts, fake tickets. Not mined, not built.

## Build plan

1. `nomorals/accounts/temp_mail.py` — 1secmail + GuerrillaMail providers, cascade, wait-for-code (mirrors temp_sms.py)
2. `nomorals/search/osint.py` — 4 SourceAdapters (username, email, domain, IP)
3. `nomorals/security/dnsleak.py` — stdlib DNS leak check (new `nomorals/security/` package)
4. EXIF strip — find the media send path, wire `strip_exif()` in
5. temp_sms.py — add receive-sms-now + any other keyless scrapable providers
6. Spine tools for: `osint_lookup`, `temp_email`, `dns_leak_check`, `strip_exif`
