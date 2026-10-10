# SECURITY_SWEEP_MINING.md — security module (nomorals/security/)

Module scope: DNS leak detection (`dnsleak.py`), image metadata stripping
(`exif.py`). Mining the best external implementations before building.

## 1. DNS leak detection

### macvk/dnsleaktest (github.com/macvk/dnsleaktest) — reference implementation
- Protocol (from `dnsleaktest.py` source): `GET https://bash.ws/id` returns a
  leak-test id; then **30 hosts** `1..30.<id>.bash.ws` are resolved **in
  parallel** (`ThreadPoolExecutor(max_workers=30)`); then
  `GET https://bash.ws/dnsleak/test/<id>?json`.
- Response is a JSON **array of typed items**: `{"type": "ip"|"dns"|
  "conclusion", "ip": ..., "country_name": ..., "asn": ...}`. The server
  already supplies ASN/country — no third-party geo-IP call needed.
- **BUG THIS EXPOSES IN OUR CODE**: we generate our own `uuid` nonce and
  query `<nonce>.bash.ws`, but bash.ws keys results by the id *it* issues via
  `/id`. A self-made nonce returns nothing. Fix: follow the real protocol.
- Multiple probes matter: resolvers that round-robin only show up if you
  issue many queries. Our single `getaddrinfo` can miss resolvers.
- Also ships `.sh`/`.bat`/Go variants and an `-i` interface option.

### ricco020/dns-leak-detector-cli (github.com/ricco020/dns-leak-detector-cli)
- `api64.ipify.org` for **IPv6 egress detection** (IPv4-only VPN + live IPv6
  = leak surface).
- `ip-api.com` as ASN/geo fallback when the leak endpoint doesn't enrich.
- **Windows SMHNR heuristic**: Smart Multi-Homed Name Resolution sends DNS
  to all adapters in parallel → inherent leak risk; flag it.
- **bash.ws is rate-limited**: empty resolver lists after repeated runs → add
  retry-with-backoff and say so honestly.

### Tugzrida/dnsleaktest-cli (gist.github.com/Tugzrida/6fe83682157ead89875a76d065874973)
- Alternative backend: `POST https://www.dnsleaktest.com/api/v1/identifiers`
  with UUID identifiers, then fetch results. 6 lookups default, 36 extended.
  (Kept as documented alternative; we standardize on bash.ws.)

### Windows SMHNR — registry facts (ovpn.com, ValdikSS, ghacks.net, pavellizunov/vpnrouter)
- `HKLM\SOFTWARE\Policies\Microsoft\Windows NT\DNSClient\
  DisableSmartNameResolution = 1` and
  `HKLM\SYSTEM\CurrentControlSet\Services\Dnscache\Parameters\
  DisableParallelAandAAAA = 1` disable the parallel-to-all-adapters behavior.
- Detectable from Python via `winreg` → reportable risk factor, no admin needed
  to *detect*.

### Transparent DNS proxy / hijack detection (dnsleaktest.com methodology)
- Compare *configured* resolvers (OS settings) vs *detected* resolvers
  (bash.ws). Configured `1.1.1.1` but detected ISP resolver ⇒ the network is
  intercepting DNS. Real technique, missing from our code.

### Per-OS configured-resolver sources (standard sysadmin tooling)
- Linux: `resolvectl status` / `nmcli dev show` / `/etc/resolv.conf`
  (systemd-resolved stub `127.0.0.53` must not be mistaken for a real resolver).
- macOS: `scutil --dns`. Windows: `Get-DnsClientServerAddress` / registry /
  `ipconfig /all`. Best-effort, never raise.

## 2. Image metadata stripping

### mat2 — the gold standard (0xacab/mat2, paper arxiv.org/pdf/1212.3648v3)
- Policy: *"any piece of the file that (1) is not data, and (2) can be
  removed, is considered a threat and deleted"* — whitelist approach per
  format; unknown fields are removed, not kept.
- UX: never cleans in place by default → writes `<name>.cleaned.<ext>`;
  `--inplace` is opt-in; `-s/--show` lists harmful metadata without removing;
  `-L/--lightweight` removes only some. We adopt: `clean_copy()` (mat2-style),
  in-place `strip_exif()` kept for backward compat, `--show` ≙ `exif_summary()`.
- Honesty rule from the man page: absence of detectable metadata ≠ proof of
  cleanliness for complex formats — say so.

### BatchPurifier LITE JPEG taxonomy (betanews.com review)
Complete JPEG metadata checklist a scrubber must handle:
**EXIF incl. thumbnail + geotag (APP1)**, **XMP (APP1, coexists with EXIF)**,
**Photoshop resources / IPTC (APP13)**, **ICC profile (APP2)**,
**Adobe App14 (APP14)**, **COM segments**, **JFIF header (APP0)**,
"other hidden data". Our Pillow-only path misses XMP/IPTC/APP14/COM/JFIF —
Pillow re-encode drops EXIF but **not** all of these, and re-encoding a JPEG
**loses quality**. Fix: lossless JPEG *segment surgery* in pure stdlib —
drop metadata markers, keep image data byte-identical (no re-encode).

### PNG metadata (Wipefey, libavif issue #1333, PNG spec)
- `tEXt`/`iTXt`/`zTXt` text chunks, `tIME` modification time, `eXIf`,
  `iCCP` (ICC profile can name the device) all survive naive handling.
- Pillow preserves `im.info`/`im.text` across save unless rebuilt — chunk
  surgery (parse per PNG spec, recompute CRCs) is lossless and exact.

### WebP (C2PA spec embedding rules)
- VP8X container: FourCC chunks; `EXIF `, `XMP `, `ICCP` chunks are the
  metadata surface; droppable with padding-aware parsing. C2PA credentials
  live in a dedicated JUMBF box.

### AI-generation / provenance signals (Seizmann/wipefey, lars-1987/metastrip, C2PA spec)
- PNG `tEXt` `"parameters"` (Stable Diffusion/A1111), `Software` stamps
  (Midjourney/DALL·E), C2PA JUMBF (`jumb`/`c2pa` markers, JPEG APP11),
  `trainedAlgorithmicMedia` digital source type, `claim_generator`/
  `claim_generator_info` naming model + signer cert subject naming the
  company. `exif_summary()` should surface these — a modern gap in our code.
- Caveat (versely.studio): platforms like X strip C2PA on upload — absence
  is not proof the source was never signed.

### ExifTool / piexif (exiftool.org)
- `exiftool -all= -overwrite_original` remains the CLI reference;
  `piexif.remove()` for EXIF-only. Our stdlib surgery covers the no-dependency
  path (Termux-friendly, per standing build rule: stdlib first).

### Orientation trap (libjxl format overview discussion)
- Dropping EXIF *without* applying orientation leaves the image visually
  rotated. Correct order: read orientation (IFD0 tag `0x0112`) → if ≠ 1,
  transpose pixels (needs re-encode) → then strip. Lossless surgery only
  when orientation is already normal.

## 3. WebRTC / IPv6 leak surface (the standard leak-test trio)

ipleak.net, expressvpn, mullvad, browserleaks all test three things: **DNS,
WebRTC, IPv6**. We only do DNS.

### WebRTC methodology (swellequation/webrtc-stun-candidate-detection, openreplay ICE test, dev.to writeups)
- Browser gathers ICE candidates: **host** (local interface IPs — high leak
  risk), **srflx** (server-reflexive public IP via STUN), **relay** (TURN —
  low risk). Verdict: srflx IP vs HTTP public IP; differ ⇒ leak.
- **IPv6 is commonly exposed even with an active VPN** — test it explicitly.
- Key insight (feder-cr/invisible_playwright): *STUN uses UDP while SOCKS
  carries TCP, so a STUN request leaves via the real interface* — a Python
  STUN probe replicates the srflx half of the browser test faithfully.
- Firefox masks host candidates as mDNS `.local` — note browser variance.

### STUN binary protocol (RFC 5389 / RFC 8489)
- Binding Request: type `0x0001`, length `0x0000`, magic cookie `0x2112A442`
  (network order), 96-bit transaction id (20 bytes total).
- Binding Success Response: type `0x0101`; `XOR-MAPPED-ADDRESS` (attr
  `0x0020`): family byte, X-Port = port ⊕ (cookie ≫ 16), X-Address = IPv4
  bytes ⊕ cookie. Legacy `MAPPED-ADDRESS` (`0x0001`) as fallback.
- Public servers: `stun.l.google.com:19302` (used by the mined tools).

## 4. What the module SHOULD have (gaps → build list)

**dnsleak.py**
- [ ] Real bash.ws protocol (`/id` → N parallel probes → `/dnsleak/test/<id>?json`
      with typed items; server gives ASN/country directly)
- [ ] Multi-probe aggregation (catch round-robin resolvers), ThreadPool
- [ ] Retry-with-backoff on rate-limited empty results
- [ ] Per-OS configured resolvers (resolvectl/nmcli/scutil/registry) +
      transparent-proxy detection (configured ≠ detected)
- [ ] Windows SMHNR risk detection via winreg
- [ ] IPv6 egress capture; server `conclusion` item surfaced
- [ ] Human-readable `format_report()` for chat/terminal

**exif.py**
- [ ] Lossless JPEG segment surgery (APP0/APP1/APP2/APP13/APP14/COM dropped;
      no re-encode, no quality loss) — primary path
- [ ] PNG chunk surgery (tEXt/iTXt/zTXt/tIME/eXIf/iCCP dropped, CRCs recomputed)
- [ ] WebP chunk surgery (EXIF/XMP/ICCP dropped)
- [ ] Orientation-aware: parse IFD0 0x0112; transpose+re-encode only when ≠ 1
- [ ] `exif_summary()`: GPS decoded to decimal lat/lon, XMP/IPTC/ICC/COM
      presence, PNG text keys, C2PA + AI-generation signals
- [ ] `verify_clean()` post-strip verification (wipefey pattern)
- [ ] `clean_copy()` mat2-style `<name>.cleaned.<ext>` (no in-place by default
      for the new path), `secure_delete()` (overwrite+unlink) for backups
- [ ] `format_summary()` human-readable output

**netleak.py (NEW — completes the trio)**
- [ ] stdlib RFC 5389 STUN client (UDP, retransmit) → srflx IP
- [ ] Local interface IP enumeration (host-candidate equivalent) +
      RFC1918/loopback classification
- [ ] IPv6 egress detection (api64.ipify.org + interface scan)
- [ ] Verdicts: WebRTC-style leak (srflx ≠ HTTP IP), IPv6 egress advisory,
      local-exposure note; honest about browser-vs-Python limits
- [ ] `NetLeakReport` + `format_report()`, never raises

**__init__.py**: export the public surface.
