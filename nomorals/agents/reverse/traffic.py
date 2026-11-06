"""The traffic rung (Pass 3) — capture files as evidence.

Three native readers, one probed second voice:

* **pcap / pcapng** — a built-in dissector (Ethernet / ARP /
  IPv4 / IPv6 / TCP / UDP / ICMP / DNS / TLS handshake / cleartext
  HTTP) producing the protocol census, the conversation
  inventory, DNS questions and answers, the TLS facts the
  capture itself carries (SNI, negotiated versions, certificate
  subjects — from cleartext handshakes), and HTTP request lines.
* **HAR** — the browser-export format, parsed natively.
* **mitmproxy flow files** — parsed natively (the format is
  tnetstring-serialized flow state; bodies are hashed for the
  record, never copied), plus the JSONL written by Devon's own
  mitmproxy addon (:mod:`.mitm_scripts` — a file the owner runs
  where mitmproxy already lives; nothing here starts, proxies or
  intercepts anything).

**TShark is probe-only**, with the reason on the record:
Wireshark publishes no standalone Linux binary — its Linux
channel is the OS distributions' package managers, whose signing
is the vendor's, not a publisher pin Devon can record (the
readelf / binwalk precedent). Where a machine has tshark, the
rung runs it as a second engine voice and folds its summary
beside the native one; where it does not, the native readers
carry the analysis alone and say so.

Honesty rules, enforced by construction: an encrypted payload is
recorded as opaque bytes and a residual unknown — never as a
guess about its content; TLS 1.3 certificates travel encrypted
and are reported as such; every record names the capture file
as its authority and carries the analysis limits on itself.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import struct
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .failures import record_failure

__all__ = [
    "PROVIDER_ID", "PROVIDER_NAME", "TSHARK_PROVIDER_ID",
    "FLOWS_PROVIDER_ID", "TrafficError", "sniff", "sniff_addon",
    "parse_tnetstring", "parse_flows",
    "read_packets", "dissect", "parse_har", "parse_addon_jsonl",
    "probe_tshark", "run_tshark", "parse_tshark_stats",
    "parse_tshark_fields", "parse_capinfos",
    "fold_evidence", "fold_addon", "LIMITATIONS",
]

PROVIDER_ID = "devon.traffic-native"
PROVIDER_NAME = "Devon built-in capture readers"
PROVIDER_VERSION = "1"
TSHARK_PROVIDER_ID = "devon.tshark"
TSHARK_PROVIDER_NAME = "TShark (probed system engine)"
FLOWS_PROVIDER_ID = "devon.mitmproxy-flows"
FLOWS_PROVIDER_NAME = "mitmproxy flow file (native reader)"
ADDON_PROVIDER_ID = "devon.mitm-capture-addon"

RUN_TIMEOUT_S = 120.0
_STREAM_CAP = 256 * 1024
_MAX_PACKETS = 2_000_000

LIMITATIONS = (
    "the capture file is the whole authority — facts are what "
    "the capture contains, and a conversation the capture missed "
    "is invisible here, not disproven",
    "encrypted payloads are opaque: TLS application data and "
    "TLS 1.3 certificate contents are counted and named as "
    "unknowns, never guessed at",
    "stream reassembly is bounded per direction; facts drawn "
    "from very large streams may be incomplete and say so",
)

Runner = Callable[..., tuple[int, str]]


class TrafficError(RuntimeError):
    """The traffic rung cannot answer; the message says why."""


def _run(argv: list[str], *, timeout: float = RUN_TIMEOUT_S,
         env: dict[str, str] | None = None) -> tuple[int, str]:
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout, env=env)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


# ── sniffing ────────────────────────────────────────────────────────

_PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", False), b"\xa1\xb2\xc3\xd4": (">", False),
    b"\x4d\x3c\xb2\xa1": ("<", True), b"\xa1\xb2\x3c\x4d": (">", True),
}


def sniff(data: bytes) -> str | None:
    """Content-sniff a capture family: pcap | pcapng | har |
    mitm-flows | None. By content, never by file name."""
    if data[:4] in _PCAP_MAGICS:
        return "pcap"
    if data[:4] == b"\x0a\x0d\x0d\x0a":
        return "pcapng"
    head = data[:65536].lstrip()
    if head[:1] == b"{":
        try:
            doc = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            doc = None
        if isinstance(doc, dict) and isinstance(
                (doc.get("log") or {}).get("entries"), list):
            return "har"
    try:
        value, _next = parse_tnetstring(data, 0)
    except (ValueError, IndexError):
        value = None
    if isinstance(value, dict):
        keys = {_text_key(k) for k in value}
        if "request" in keys or ("type" in keys
                                 and "client_conn" in keys):
            return "mitm-flows"
    return None


def sniff_addon(data: bytes) -> bool:
    """True when the bytes are a Devon capture-addon JSONL file —
    the first record carries her addon id."""
    for line in data[:8192].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line.decode("utf-8", "replace"))
        except ValueError:
            return False
        return isinstance(row, dict) and str(
            row.get("addon") or "").startswith("devon.mitm")
    return False


def _text_key(key: Any) -> str:
    if isinstance(key, bytes):
        return key.decode("utf-8", "replace")
    return str(key)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return "" if value is None else str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {_text_key(k): v for k, v in value.items()}


# ── tnetstring (the mitmproxy flow-file encoding) ───────────────────

def parse_tnetstring(data: bytes, pos: int = 0) -> tuple[Any, int]:
    """Parse one tnetstring value at ``pos`` → (value, next pos).

    Grammar (mitmproxy.utils.tnetstring): ``<len>:<payload><tag>``
    with tag ',' bytes, '#' int, '^' float, '!' bool, '~' null,
    ']' list, '}' dict. Dict keys carry the ';' tag (a string key,
    returned here as bytes like ',' values). Raises ValueError on
    malformed input.
    """
    colon = data.find(b":", pos, pos + 16)
    if colon < 0:
        raise ValueError("tnetstring: no length prefix")
    try:
        length = int(data[pos:colon])
    except ValueError as exc:
        raise ValueError("tnetstring: bad length") from exc
    start = colon + 1
    end = start + length
    if end >= len(data) + 0 and end > len(data):
        raise ValueError("tnetstring: truncated payload")
    payload = data[start:end]
    if end >= len(data):
        raise ValueError("tnetstring: missing type tag")
    tag = chr(data[end])
    nxt = end + 1
    if tag in (",", ";"):
        return payload, nxt
    if tag == "#":
        return int(payload), nxt
    if tag == "^":
        return float(payload), nxt
    if tag == "!":
        return payload == b"true", nxt
    if tag == "~":
        return None, nxt
    if tag == "]":
        items = []
        inner = 0
        while inner < len(payload):
            item, inner = parse_tnetstring(payload, inner)
            items.append(item)
        return items, nxt
    if tag == "}":
        out: dict[Any, Any] = {}
        inner = 0
        while inner < len(payload):
            key, inner = parse_tnetstring(payload, inner)
            val, inner = parse_tnetstring(payload, inner)
            out[key] = val
        return out, nxt
    raise ValueError(f"tnetstring: unknown tag {tag!r}")


def parse_flows(data: bytes) -> list[dict[str, Any]]:
    """All flow dicts in a mitmproxy dump file."""
    flows: list[dict[str, Any]] = []
    pos = 0
    while pos < len(data):
        if data[pos:pos + 1].isspace():
            pos += 1
            continue
        value, pos = parse_tnetstring(data, pos)
        if isinstance(value, dict):
            flows.append(_as_dict(value))
    return flows


# ── pcap / pcapng readers ───────────────────────────────────────────

def _read_pcap(data: bytes) -> tuple[list[tuple[float, int, bytes]], str]:
    en, ns = _PCAP_MAGICS[data[:4]]
    _vmaj, _vmin, _tz, _sig, _snap, linktype = struct.unpack_from(
        en + "HHIIII", data, 4)
    packets: list[tuple[float, int, bytes]] = []
    pos = 24
    scale = 1e-9 if ns else 1e-6
    while pos + 16 <= len(data) and len(packets) < _MAX_PACKETS:
        sec, frac, incl, _orig = struct.unpack_from(en + "IIII",
                                                    data, pos)
        pos += 16
        packets.append((sec + frac * scale, linktype,
                        data[pos:pos + incl]))
        pos += incl
    return packets, "pcap"


def _read_pcapng(data: bytes) -> tuple[list[tuple[float, int, bytes]], str]:
    packets: list[tuple[float, int, bytes]] = []
    interfaces: dict[int, dict[str, Any]] = {}
    en = "<"
    pos = 0
    while pos + 12 <= len(data) and len(packets) < _MAX_PACKETS:
        if data[pos:pos + 4] == b"\x0a\x0d\x0d\x0a":
            # Section header: the byte-order magic inside decides the
            # section endianness BEFORE total_length can be trusted.
            magic = data[pos + 8:pos + 12]
            if magic == b"\x4d\x3c\x2b\x1a":
                en = "<"
            elif magic == b"\x1a\x2b\x3c\x4d":
                en = ">"
        block_type, total = struct.unpack_from(en + "II", data, pos)
        if total < 12 or pos + total > len(data):
            break
        body = data[pos + 8:pos + total - 4]
        if block_type == 0x0A0D0D0A:
            interfaces = {}
        elif block_type == 0x00000001 and len(body) >= 8:
            linktype, _reserved, snaplen = struct.unpack_from(
                en + "HHI", body, 0)
            resolution = 1e-6
            opts = body[8:]
            opos = 0
            while opos + 4 <= len(opts):
                code, olen = struct.unpack_from(en + "HH", opts, opos)
                if code == 0:
                    break
                if code == 9 and olen >= 1:
                    val = opts[opos + 4]
                    resolution = (2.0 ** -(val & 0x7F)) if val & 0x80 \
                        else 10.0 ** -val
                opos += 4 + ((olen + 3) & ~3)
            interfaces[len(interfaces)] = {
                "linktype": linktype, "snaplen": snaplen,
                "resolution": resolution}
        elif block_type == 0x00000006 and len(body) >= 20:
            if_id, ts_hi, ts_lo, caplen, _plen = struct.unpack_from(
                en + "IIIII", body, 0)
            iface = interfaces.get(if_id) or {"linktype": 1,
                                              "resolution": 1e-6}
            ts = ((ts_hi << 32) | ts_lo) * iface["resolution"]
            packets.append((ts, iface["linktype"],
                            body[20:20 + caplen]))
        elif block_type == 0x00000003 and len(body) >= 4:
            _plen, = struct.unpack_from(en + "I", body, 0)
            iface = interfaces.get(0) or {"linktype": 1}
            packets.append((0.0, iface["linktype"], body[4:]))
        pos += total
    return packets, "pcapng"


def read_packets(data: bytes, kind: str
                 ) -> tuple[list[tuple[float, int, bytes]], str]:
    """Normalize a pcap/pcapng into (ts, linktype, frame) rows."""
    if kind == "pcap":
        return _read_pcap(data)
    if kind == "pcapng":
        return _read_pcapng(data)
    raise TrafficError(f"not a packet capture kind: {kind!r}")


# ── the dissector ───────────────────────────────────────────────────

def _ip_text(raw: bytes) -> str:
    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        return raw.hex()


def _dns_name(msg: bytes, pos: int) -> tuple[str, int]:
    labels: list[str] = []
    jumped = False
    next_pos = pos
    guard = 0
    while pos < len(msg) and guard < 128:
        guard += 1
        length = msg[pos]
        if length == 0:
            if not jumped:
                next_pos = pos + 1
            break
        if length & 0xC0 == 0xC0:
            if pos + 1 >= len(msg):
                break
            pointer = ((length & 0x3F) << 8) | msg[pos + 1]
            if not jumped:
                next_pos = pos + 2
            jumped = True
            pos = pointer
            continue
        pos += 1
        labels.append(msg[pos:pos + length].decode("utf-8", "replace"))
        pos += length
    return ".".join(labels), next_pos


_DNS_TYPES = {1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 12: "PTR",
              15: "MX", 16: "TXT", 28: "AAAA", 33: "SRV", 65: "HTTPS"}


def _parse_dns(msg: bytes) -> dict[str, Any] | None:
    if len(msg) < 12:
        return None
    _ident, flags, qd, an, _ns, _ar = struct.unpack_from(">HHHHHH",
                                                         msg, 0)
    out: dict[str, Any] = {"is_response": bool(flags & 0x8000),
                           "questions": [], "answers": []}
    pos = 12
    try:
        for _ in range(qd):
            name, pos = _dns_name(msg, pos)
            qtype, _qclass = struct.unpack_from(">HH", msg, pos)
            pos += 4
            out["questions"].append({
                "name": name,
                "type": _DNS_TYPES.get(qtype, f"TYPE{qtype}")})
        for _ in range(an):
            name, pos = _dns_name(msg, pos)
            rtype, _rclass, _ttl, rdlen = struct.unpack_from(
                ">HHIH", msg, pos)
            pos += 10
            rdata = msg[pos:pos + rdlen]
            value = ""
            if (rtype == 1 and rdlen == 4) \
                    or (rtype == 28 and rdlen == 16):
                value = _ip_text(rdata)
            elif rtype in (2, 5, 12):
                value, _ = _dns_name(msg, pos)
            out["answers"].append({
                "name": name,
                "type": _DNS_TYPES.get(rtype, f"TYPE{rtype}"),
                "value": value})
            pos += rdlen
    except (struct.error, IndexError):
        return out if out["questions"] else None
    return out


_TLS_VERSIONS = {0x0300: "SSL 3.0", 0x0301: "TLS 1.0",
                 0x0302: "TLS 1.1", 0x0303: "TLS 1.2",
                 0x0304: "TLS 1.3"}
_TLS_CIPHERS = {
    0x1301: "TLS_AES_128_GCM_SHA256",
    0x1302: "TLS_AES_256_GCM_SHA384",
    0x1303: "TLS_CHACHA20_POLY1305_SHA256",
    0xC02F: "ECDHE-RSA-AES128-GCM-SHA256",
    0xC030: "ECDHE-RSA-AES256-GCM-SHA384",
    0xC02B: "ECDHE-ECDSA-AES128-GCM-SHA256",
    0xC02C: "ECDHE-ECDSA-AES256-GCM-SHA384",
    0x009C: "RSA-AES128-GCM-SHA256",
    0x009D: "RSA-AES256-GCM-SHA384",
    0x002F: "RSA-AES128-SHA",
    0x0035: "RSA-AES256-SHA",
}


def _der_read(buf: bytes, pos: int) -> tuple[int, int, int]:
    """One DER TLV → (tag, content_start, content_end)."""
    tag = buf[pos]
    pos += 1
    if tag & 0x1F == 0x1F:  # multi-byte tag number
        while buf[pos] & 0x80:
            pos += 1
        pos += 1
    first = buf[pos]
    pos += 1
    if first & 0x80:
        nbytes = first & 0x7F
        length = int.from_bytes(buf[pos:pos + nbytes], "big")
        pos += nbytes
    else:
        length = first
    return tag, pos, pos + length


_OID_NAMES = {"2.5.4.3": "CN", "2.5.4.10": "O", "2.5.4.11": "OU",
              "2.5.4.6": "C", "2.5.4.7": "L", "2.5.4.8": "ST"}


def _der_oid(buf: bytes) -> str:
    if not buf:
        return ""
    parts = [buf[0] // 40, buf[0] % 40]
    value = 0
    for byte in buf[1:]:
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            parts.append(value)
            value = 0
    return ".".join(str(p) for p in parts)


def _der_name(buf: bytes) -> dict[str, str]:
    """An X.501 Name (SEQUENCE OF SET OF SEQUENCE{OID, value})."""
    out: dict[str, str] = {}
    try:
        _tag, start, end = _der_read(buf, 0)
        pos = start
        while pos < end:
            _t, set_start, set_end = _der_read(buf, pos)
            spos = set_start
            while spos < set_end:
                _t2, seq_start, seq_end = _der_read(buf, spos)
                _t3, oid_start, oid_end = _der_read(buf, seq_start)
                vtag, val_start, val_end = _der_read(buf, oid_end)
                oid = _der_oid(buf[oid_start:oid_end])
                raw = buf[val_start:val_end]
                if vtag == 0x1E:
                    value = raw.decode("utf-16-be", "replace")
                elif vtag == 0x14:
                    value = raw.decode("latin-1", "replace")
                else:
                    value = raw.decode("utf-8", "replace")
                name = _OID_NAMES.get(oid, oid)
                if name not in out:
                    out[name] = value
                spos = seq_end
            pos = set_end
    except (IndexError, struct.error):
        return out
    return out


def _cert_facts(der: bytes) -> dict[str, Any]:
    """Subject + issuer of one DER certificate (minimal walk)."""
    facts: dict[str, Any] = {}
    try:
        _t, cert_start, _e = _der_read(der, 0)
        _t, tbs_start, tbs_end = _der_read(der, cert_start)
        pos = tbs_start
        tag, _s, after = _der_read(der, pos)
        if tag == 0xA0:  # [0] explicit version
            pos = after
            _t, _s, pos = _der_read(der, pos)  # serial
        else:
            pos = after  # serial already consumed positionally
        _t, _s, pos = _der_read(der, pos)      # signature algorithm
        _t, iss_start, iss_end = _der_read(der, pos)
        facts["issuer"] = _der_name(der[pos:iss_end])
        _t, _s, pos = _der_read(der, iss_end)  # validity
        _t, sub_start, sub_end = _der_read(der, pos)
        facts["subject"] = _der_name(der[pos:sub_end])
    except (IndexError, struct.error) as exc:
        facts["parse_error"] = str(exc)
    return facts


def _parse_tls_stream(buf: bytes, session: dict[str, Any],
                      direction: str) -> None:
    """Walk TLS records in one reassembled stream direction."""
    pos = 0
    pending = session.setdefault(
        "_hs_pending", {"client": b"", "server": b""})
    while pos + 5 <= len(buf):
        rtype, _ver, rlen = struct.unpack_from(">BHH", buf, pos)
        pos += 5
        payload = buf[pos:pos + rlen]
        pos += rlen
        if rtype == 23:
            session["opaque_bytes"] = session.get("opaque_bytes", 0) \
                + len(payload)
        elif rtype == 22:
            data = pending[direction] + payload
            used = _parse_handshakes(data, session, direction)
            pending[direction] = data[used:]
    # a leftover partial handshake is simply unparseable — the
    # session record says what WAS established, nothing more


def _parse_handshakes(buf: bytes, session: dict[str, Any],
                      direction: str) -> int:
    """Parse complete handshake messages; returns bytes consumed."""
    pos = 0
    while pos + 4 <= len(buf):
        htype = buf[pos]
        hlen = int.from_bytes(buf[pos + 1:pos + 4], "big")
        if pos + 4 + hlen > len(buf):
            break
        body = buf[pos + 4:pos + 4 + hlen]
        try:
            if htype == 1:
                _client_hello(body, session)
            elif htype == 2:
                _server_hello(body, session)
            elif htype == 11:
                _certificate(body, session)
        except (IndexError, struct.error) as exc:
            record_failure("reverse.traffic.tls_parse", exc)
        pos += 4 + hlen
    return pos


def _extensions(body: bytes, pos: int) -> dict[int, bytes]:
    exts: dict[int, bytes] = {}
    if pos + 2 > len(body):
        return exts
    total, = struct.unpack_from(">H", body, pos)
    pos += 2
    end = min(len(body), pos + total)
    while pos + 4 <= end:
        etype, elen = struct.unpack_from(">HH", body, pos)
        exts[etype] = body[pos + 4:pos + 4 + elen]
        pos += 4 + elen
    return exts


def _client_hello(body: bytes, session: dict[str, Any]) -> None:
    legacy, = struct.unpack_from(">H", body, 0)
    pos = 2 + 32
    sid_len = body[pos]
    pos += 1 + sid_len
    cs_len, = struct.unpack_from(">H", body, pos)
    pos += 2 + cs_len
    comp_len = body[pos]
    pos += 1 + comp_len
    exts = _extensions(body, pos)
    offered = [legacy]
    if 43 in exts:
        data = exts[43]
        n = data[0] if data else 0
        offered = [struct.unpack_from(">H", data, 1 + i * 2)[0]
                   for i in range(n // 2) if 1 + i * 2 + 2 <= len(data)]
    session["client_versions"] = [
        _TLS_VERSIONS.get(v, hex(v)) for v in offered]
    if 0 in exts:
        data = exts[0]
        list_len, = struct.unpack_from(">H", data, 0)
        epos = 2
        end = min(len(data), 2 + list_len)
        while epos + 3 <= end:
            name_type = data[epos]
            name_len, = struct.unpack_from(">H", data, epos + 1)
            name = data[epos + 3:epos + 3 + name_len].decode(
                "utf-8", "replace")
            if name_type == 0:
                session["sni"] = name
            epos += 3 + name_len
    if 16 in exts:
        data = exts[16]
        list_len, = struct.unpack_from(">H", data, 0)
        epos = 2
        alpn = []
        while epos < 2 + list_len and epos < len(data):
            n = data[epos]
            alpn.append(data[epos + 1:epos + 1 + n].decode(
                "utf-8", "replace"))
            epos += 1 + n
        session["alpn_offered"] = alpn


def _server_hello(body: bytes, session: dict[str, Any]) -> None:
    legacy, = struct.unpack_from(">H", body, 0)
    pos = 2 + 32
    sid_len = body[pos]
    pos += 1 + sid_len
    cipher, = struct.unpack_from(">H", body, pos)
    pos += 2
    pos += 1  # compression
    exts = _extensions(body, pos)
    version = legacy
    if 43 in exts and len(exts[43]) >= 2:
        version, = struct.unpack_from(">H", exts[43], 0)
    session["version"] = _TLS_VERSIONS.get(version, hex(version))
    session["cipher"] = _TLS_CIPHERS.get(cipher, hex(cipher))
    if 16 in exts and exts[16]:
        data = exts[16]
        n = data[2] if len(data) > 2 else 0
        session["alpn"] = data[3:3 + n].decode("utf-8", "replace")


def _certificate(body: bytes, session: dict[str, Any]) -> None:
    total = int.from_bytes(body[:3], "big")
    pos = 3
    certs = []
    end = min(len(body), 3 + total)
    while pos + 3 <= end:
        clen = int.from_bytes(body[pos:pos + 3], "big")
        pos += 3
        der = body[pos:pos + clen]
        pos += clen
        certs.append(_cert_facts(der))
    session.setdefault("certificates", []).extend(certs)


_HTTP_METHODS = (b"GET", b"POST", b"PUT", b"DELETE", b"HEAD",
                 b"OPTIONS", b"PATCH", b"TRACE", b"CONNECT")


def _parse_http_stream(buf: bytes, session: dict[str, Any],
                       direction: str) -> None:
    text = buf
    if direction == "request":
        for match in re.finditer(
                rb"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|TRACE|"
                rb"CONNECT) ([^\r\n ]+) HTTP/(1\.[01])", text):
            start = match.start()
            line_end = text.find(b"\r\n\r\n", start)
            headers_blob = text[match.end():line_end] \
                if line_end > 0 else b""
            headers = {}
            for line in headers_blob.split(b"\r\n"):
                if b":" in line:
                    key, _, value = line.partition(b":")
                    headers[key.strip().lower().decode(
                        "utf-8", "replace")] = value.strip().decode(
                            "utf-8", "replace")
            session.setdefault("requests", []).append({
                "method": match.group(1).decode(),
                "uri": match.group(2).decode("utf-8", "replace"),
                "http_version": "HTTP/" + match.group(3).decode(),
                "host": headers.get("host", ""),
                "user_agent": headers.get("user-agent", ""),
                "content_type": headers.get("content-type", ""),
            })
    else:
        for match in re.finditer(
                rb"HTTP/(1\.[01]) (\d{3}) ([^\r\n]*)", text):
            line_end = text.find(b"\r\n\r\n", match.start())
            headers_blob = text[match.end():line_end] \
                if line_end > 0 else b""
            headers = {}
            for line in headers_blob.split(b"\r\n"):
                if b":" in line:
                    key, _, value = line.partition(b":")
                    headers[key.strip().lower().decode(
                        "utf-8", "replace")] = value.strip().decode(
                            "utf-8", "replace")
            session.setdefault("responses", []).append({
                "status": int(match.group(2)),
                "reason": match.group(3).decode("utf-8", "replace"),
                "content_type": headers.get("content-type", ""),
            })


def _reassemble(segments: list[tuple[int, bytes]]) -> bytes:
    out = bytearray()
    segments.sort(key=lambda s: s[0])
    for _seq, payload in segments:
        if not payload:
            continue
        out.extend(payload)
        if len(out) >= _STREAM_CAP:
            break
    return bytes(out[:_STREAM_CAP])


def dissect(packets: list[tuple[float, int, bytes]], fmt: str
            ) -> dict[str, Any]:
    """Dissect normalized packets into the traffic facts dict."""
    protocols: dict[str, dict[str, int]] = {}
    conversations: dict[tuple, dict[str, Any]] = {}
    tcp_segments: dict[tuple, dict[str, list]] = {}
    dns_queries: list[dict[str, Any]] = []
    dns_responses: list[dict[str, Any]] = []
    opaque_total = 0

    def count(name: str, size: int) -> None:
        row = protocols.setdefault(name, {"frames": 0, "bytes": 0})
        row["frames"] += 1
        row["bytes"] += size

    for _ts, linktype, frame in packets:
        size = len(frame)
        payload = frame
        ethertype = None
        if linktype == 1 and len(frame) >= 14:
            count("eth", size)
            ethertype, = struct.unpack_from(">H", frame, 12)
            payload = frame[14:]
            if ethertype == 0x8100 and len(frame) >= 18:
                ethertype, = struct.unpack_from(">H", frame, 16)
                payload = frame[18:]
        elif linktype in (101, 12) and frame:
            ethertype = 0x0800 if frame[0] >> 4 == 4 else 0x86DD
        if ethertype == 0x0806:
            count("arp", size)
            continue
        if ethertype not in (0x0800, 0x86DD) or not payload:
            count("other", size)
            continue
        version = payload[0] >> 4
        if version == 4 and len(payload) >= 20:
            ihl = (payload[0] & 0x0F) * 4
            proto = payload[9]
            src = _ip_text(payload[12:16])
            dst = _ip_text(payload[16:20])
            l4 = payload[ihl:]
            count("ip", size)
        elif version == 6 and len(payload) >= 40:
            proto = payload[6]
            src = _ip_text(payload[8:24])
            dst = _ip_text(payload[24:40])
            l4 = payload[40:]
            count("ipv6", size)
        else:
            count("ip", size)
            continue
        if proto == 1:
            count("icmp", size)
            key = ("icmp", src, dst)
            conv = conversations.setdefault(key, {
                "protocol": "icmp", "a": src, "b": dst,
                "packets_ab": 0, "packets_ba": 0,
                "bytes_ab": 0, "bytes_ba": 0})
            conv["packets_ab"] += 1
            conv["bytes_ab"] += size
            continue
        if proto not in (6, 17) or len(l4) < 4:
            count(f"ip-proto-{proto}", size)
            continue
        sport, dport = struct.unpack_from(">HH", l4, 0)
        proto_name = "tcp" if proto == 6 else "udp"
        count(proto_name, size)
        a_ep, b_ep = f"{src}:{sport}", f"{dst}:{dport}"
        key = (proto_name, min(a_ep, b_ep), max(a_ep, b_ep))
        conv = conversations.setdefault(key, {
            "protocol": proto_name, "a": key[1], "b": key[2],
            "packets_ab": 0, "packets_ba": 0,
            "bytes_ab": 0, "bytes_ba": 0})
        if a_ep == key[1]:
            conv["packets_ab"] += 1
            conv["bytes_ab"] += size
            direction = "ab"
        else:
            conv["packets_ba"] += 1
            conv["bytes_ba"] += size
            direction = "ba"
        if proto == 17:
            udp_payload = l4[8:]
            if sport == 53 or dport == 53:
                parsed = _parse_dns(udp_payload)
                if parsed:
                    count("dns", size)
                    target = dns_responses if parsed["is_response"] \
                        else dns_queries
                    target.append(parsed)
        else:
            if len(l4) < 20:
                continue
            seq, _ack, _off_flags = struct.unpack_from(">IIH", l4, 4)
            header_len = ((l4[12] >> 4) & 0x0F) * 4
            segment = l4[header_len:]
            if segment:
                # application-layer census at packet depth: a
                # segment opening with a TLS record type or an
                # HTTP start-line counts for its layer, the same
                # way the hierarchy counters see it
                if len(segment) >= 3 and segment[0] in (
                        0x14, 0x15, 0x16, 0x17) \
                        and segment[1] == 0x03:
                    count("tls", size)
                elif segment.startswith(_HTTP_METHODS) \
                        or segment.startswith(b"HTTP/"):
                    count("http", size)
            streams = tcp_segments.setdefault(
                key, {"ab": [], "ba": []})
            if segment:
                streams[direction].append((seq, segment))

    tls_sessions: list[dict[str, Any]] = []
    http_exchanges: list[dict[str, Any]] = []
    for key, streams in tcp_segments.items():
        session: dict[str, Any] = {"opaque_bytes": 0}
        for direction in ("ab", "ba"):
            buf = _reassemble(streams[direction])
            if not buf:
                continue
            if buf[:1] in (b"\x16", b"\x14", b"\x17"):
                _parse_tls_stream(buf, session,
                                  "client" if direction == "ab"
                                  else "server")
            _parse_http_stream(buf, session,
                               "request" if direction == "ab"
                               else "response")
        if session.get("requests") or session.get("responses"):
            requests = session.pop("requests", [])
            responses = session.pop("responses", [])
            for i, req in enumerate(requests):
                resp = responses[i] if i < len(responses) else {}
                http_exchanges.append({
                    **req,
                    "status": resp.get("status"),
                    "response_content_type":
                        resp.get("content_type", ""),
                    "conversation": f"{key[1]} <-> {key[2]}",
                })
        if any(k in session for k in ("sni", "version",
                                      "client_versions",
                                      "certificates")) \
                or session.get("opaque_bytes"):
            session.pop("_hs_pending", None)
            session["conversation"] = f"{key[1]} <-> {key[2]}"
            tls_sessions.append(session)
            opaque_total += session.get("opaque_bytes", 0)

    conv_rows = sorted(
        conversations.values(),
        key=lambda c: -(c["bytes_ab"] + c["bytes_ba"]))[:64]
    stamps = [ts for ts, _lt, _f in packets if ts]
    return {
        "format": fmt,
        "packets": len(packets),
        "bytes": sum(len(f) for _t, _l, f in packets),
        "first_ts": min(stamps) if stamps else None,
        "last_ts": max(stamps) if stamps else None,
        "protocols": protocols,
        "conversations": conv_rows,
        "dns": {"queries": dns_queries, "responses": dns_responses},
        "tls": {"sessions": tls_sessions,
                "opaque_bytes": opaque_total},
        "http": {"exchanges": http_exchanges},
    }


# ── HAR ─────────────────────────────────────────────────────────────

def parse_har(doc: dict[str, Any]) -> dict[str, Any]:
    """Facts from one HAR document (already JSON-parsed)."""
    entries = (doc.get("log") or {}).get("entries") or []
    exchanges: list[dict[str, Any]] = []
    hosts: Counter = Counter()
    statuses: Counter = Counter()
    methods: Counter = Counter()
    for entry in entries:
        req = entry.get("request") or {}
        resp = entry.get("response") or {}
        url = str(req.get("url") or "")
        split = urlsplit(url)
        host = split.hostname or ""
        hosts[host] += 1
        methods[str(req.get("method") or "")] += 1
        statuses[str(resp.get("status"))] += 1
        content = resp.get("content") or {}
        exchanges.append({
            "method": req.get("method"),
            "url": url,
            "host": host,
            "path": split.path,
            "status": resp.get("status"),
            "request_body_size": req.get("bodySize"),
            "response_mime": content.get("mimeType"),
            "response_size": content.get("size"),
            "started": entry.get("startedDateTime"),
        })
    return {"format": "har", "entries": len(entries),
            "hosts": dict(hosts), "methods": dict(methods),
            "statuses": dict(statuses), "exchanges": exchanges}


# ── mitmproxy flows + the Devon addon JSONL ─────────────────────────

def flows_facts(flows: list[dict[str, Any]]) -> dict[str, Any]:
    """Facts from parsed mitmproxy flow dicts. Bodies are sized
    and hashed — never copied into the record."""
    exchanges: list[dict[str, Any]] = []
    hosts: Counter = Counter()
    for flow in flows:
        req = _as_dict(flow.get("request"))
        if not req:
            continue
        resp = _as_dict(flow.get("response"))
        req_body = req.get("content") or b""
        resp_body = resp.get("content") or b""
        headers = {_text(k).lower(): _text(v)
                   for pair in (resp.get("headers") or [])
                   if isinstance(pair, (list, tuple)) and len(pair) == 2
                   for k, v in [pair]}
        host = _text(req.get("host"))
        hosts[host] += 1
        exchanges.append({
            "method": _text(req.get("method")),
            "host": host,
            "port": req.get("port"),
            "path": _text(req.get("path")),
            "status": resp.get("status_code"),
            "request_body_size": len(req_body),
            "request_body_sha256":
                hashlib.sha256(req_body).hexdigest() if req_body
                else "",
            "response_body_size": len(resp_body),
            "response_body_sha256":
                hashlib.sha256(resp_body).hexdigest() if resp_body
                else "",
            "response_content_type": headers.get("content-type", ""),
        })
    return {"format": "mitm-flows", "flows": len(flows),
            "hosts": dict(hosts), "exchanges": exchanges}


def parse_addon_jsonl(text: str) -> dict[str, Any]:
    """Facts from a Devon capture-addon JSONL file. Lines whose
    ``addon`` id is not Devon's are counted and skipped — the
    fold only vouches for her own instrument's records."""
    records: list[dict[str, Any]] = []
    foreign = 0
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            foreign += 1
            continue
        if not isinstance(row, dict) or not str(
                row.get("addon") or "").startswith("devon.mitm"):
            foreign += 1
            continue
        records.append(row)
    exchanges = [{
        "method": r.get("method"), "host": r.get("host"),
        "port": r.get("port"), "path": r.get("path"),
        "status": r.get("status"),
        "request_body_size": (r.get("request_body") or {}).get("size"),
        "request_body_sha256":
            (r.get("request_body") or {}).get("sha256", ""),
        "response_body_size":
            (r.get("response_body") or {}).get("size"),
        "response_body_sha256":
            (r.get("response_body") or {}).get("sha256", ""),
        "client": r.get("client"), "server": r.get("server"),
        "duration_s": r.get("duration_s"),
    } for r in records]
    return {"format": "mitm-addon-jsonl", "records": len(records),
            "foreign_lines_skipped": foreign, "exchanges": exchanges}


# ── tshark: the probed second voice ─────────────────────────────────

def probe_tshark(runner: Runner | None = None) -> dict[str, Any]:
    """Probe tshark. Probe-only posture, stated on the row:
    Wireshark publishes no standalone Linux binary to pin — the
    distro package channel is the OS vendor's, not a publisher
    pin — so Devon probes and drives a system tshark where one
    exists and never provisions it."""
    reason = ("tshark is probe-only: Wireshark publishes no "
              "standalone Linux binary channel (source + distro "
              "packages only), so there is no publisher pin for "
              "Devon to provision against; the built-in readers "
              "carry the analysis")
    path = shutil.which("tshark")
    if not path:
        return {"available": False, "path": "", "version": "",
                "detail": "no system tshark on PATH. " + reason}
    run = runner or _run
    try:
        _rc, out = run([path, "--version"], timeout=10.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "path": path, "version": "",
                "detail": f"tshark at {path} did not answer: {exc}"}
    first = (out or "").strip().splitlines()[0] if out else ""
    match = re.search(r"TShark \(Wireshark\) ([\d.]+)", first)
    if not match:
        return {"available": False, "path": path, "version": "",
                "detail": f"tshark at {path} answered its probe "
                          f"with something else: {first}"}
    return {"available": True, "path": path,
            "version": match.group(1),
            "detail": f"tshark probed at {path} "
                      f"({match.group(1)}); driven as the second "
                      "engine voice, never provisioned — " + reason}


_CONV_ROW = re.compile(
    r"^(\S+)\s+<->\s+(\S+)\s+(\d+)\s+(\d+) bytes\s+(\d+)\s+(\d+)"
    r" bytes\s+(\d+)\s+(\d+) bytes")


def parse_tshark_stats(text: str) -> dict[str, Any]:
    """Parse ``tshark -q -z io,phs -z conv,*`` output. Pure — the
    shapes pinned here are TShark 4.2's real output, recorded
    from a real run."""
    protocols: dict[str, dict[str, int]] = {}
    conversations: list[dict[str, Any]] = []
    in_phs = False
    for line in (text or "").splitlines():
        if "Protocol Hierarchy Statistics" in line:
            in_phs = True
            continue
        if line.startswith("=") and "Conversations" in line:
            in_phs = False
            continue
        if in_phs:
            match = re.match(r"\s*(\S+)\s+frames:(\d+) bytes:(\d+)",
                             line)
            if match:
                protocols[match.group(1)] = {
                    "frames": int(match.group(2)),
                    "bytes": int(match.group(3))}
            continue
        match = _CONV_ROW.match(line.strip())
        if match:
            conversations.append({
                "a": match.group(1), "b": match.group(2),
                "packets_ba": int(match.group(3)),
                "bytes_ba": int(match.group(4)),
                "packets_ab": int(match.group(5)),
                "bytes_ab": int(match.group(6)),
                "packets": int(match.group(7)),
                "bytes": int(match.group(8)),
            })
    return {"protocols": protocols, "conversations": conversations}


_FIELD_COLUMNS = ("frame.number", "ip.proto", "dns.qry.name",
                  "dns.flags.response", "dns.a",
                  "tls.handshake.extensions_server_name",
                  "tls.handshake.version", "http.request.method",
                  "http.request.uri", "http.host",
                  "http.response.code")


def parse_tshark_fields(text: str) -> dict[str, Any]:
    """Parse the ``-T fields`` corroboration pass (tab-separated,
    the column set of :data:`_FIELD_COLUMNS`)."""
    dns_names: list[str] = []
    dns_answers: list[str] = []
    sni: list[str] = []
    http: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) < len(_FIELD_COLUMNS):
            parts += [""] * (len(_FIELD_COLUMNS) - len(parts))
        row = dict(zip(_FIELD_COLUMNS, parts, strict=False))
        if row["dns.qry.name"]:
            dns_names.append(row["dns.qry.name"])
        if row["dns.a"]:
            dns_answers.append(row["dns.a"])
        if row["tls.handshake.extensions_server_name"]:
            sni.append(row["tls.handshake.extensions_server_name"])
        if row["http.request.method"]:
            http.append({"method": row["http.request.method"],
                         "uri": row["http.request.uri"],
                         "host": row["http.host"]})
        if row["http.response.code"]:
            http.append({"status": int(row["http.response.code"])
                         if row["http.response.code"].isdigit()
                         else row["http.response.code"]})
    return {"dns_names": dns_names, "dns_answers": dns_answers,
            "sni": sni, "http": http}


def parse_capinfos(text: str) -> dict[str, Any]:
    wanted = {"File type": "file_type",
              "File encapsulation": "encapsulation",
              "Number of packets": "packets",
              "Capture duration": "duration",
              "Data size": "data_size"}
    out: dict[str, Any] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        mapped = wanted.get(key.strip())
        if mapped:
            out[mapped] = value.strip()
    return out


def run_tshark(target: str | Path, *,
               runner: Runner | None = None) -> dict[str, Any]:
    """Run the probed tshark over one capture → parsed facts.
    Raises :class:`TrafficError` when unavailable or refused."""
    state = probe_tshark(runner)
    if not state["available"]:
        raise TrafficError("tshark rung unavailable: "
                           + state["detail"])
    exe = state["path"]
    run = runner or _run
    try:
        _rc, stats_out = run(
            [exe, "-r", str(target), "-q", "-z", "io,phs",
             "-z", "conv,ip", "-z", "conv,tcp", "-z", "conv,udp"],
            timeout=RUN_TIMEOUT_S)
        _rc2, fields_out = run(
            [exe, "-r", str(target), "-T", "fields",
             *[arg for col in _FIELD_COLUMNS
               for arg in ("-e", col)]],
            timeout=RUN_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrafficError(f"tshark run failed: {exc}") from exc
    facts = parse_tshark_stats(stats_out)
    facts["fields"] = parse_tshark_fields(fields_out)
    facts["version"] = state["version"]
    capinfos = shutil.which("capinfos")
    if capinfos:
        try:
            _rc3, cap_out = run([capinfos, str(target)],
                                timeout=30.0)
            facts["capinfos"] = parse_capinfos(cap_out)
        except (OSError, subprocess.TimeoutExpired) as exc:
            record_failure("reverse.traffic.capinfos", exc)
    return facts


# ── the fold ────────────────────────────────────────────────────────

def fold_evidence(store: Any, *, subject_sha256: str,
                  subject_name: str, subject_size: int,
                  subject_path: str, kind: str,
                  facts: dict[str, Any],
                  tshark_facts: dict[str, Any] | None = None,
                  link_to: str = "") -> dict[str, Any]:
    """Fold one capture analysis into Evidence + unknowns."""
    evidence_ids: list[str] = []
    unknowns: list[dict[str, Any]] = []
    common = {"subject_sha256": subject_sha256,
              "subject_name": subject_name,
              "subject_size": subject_size}
    locations = [{"type": "artifact-path", "value": subject_path}]
    links = [link_to] if link_to else []
    is_flows = kind == "mitm-flows"
    provider = {
        "provider_id": FLOWS_PROVIDER_ID if is_flows else PROVIDER_ID,
        "provider_name": FLOWS_PROVIDER_NAME if is_flows
        else PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
    }

    summary_rec = store.record(
        predicate_type="traffic.summary", operation="capture.analyze",
        parameters={"kind": kind},
        result={k: facts.get(k) for k in (
            "format", "packets", "bytes", "first_ts", "last_ts",
            "protocols", "entries", "hosts", "flows") if k in facts},
        limitations=list(LIMITATIONS), locations=locations,
        links=links, **provider, **common)
    evidence_ids.append(summary_rec.evidence_id)
    anchor = summary_rec.evidence_id

    if facts.get("conversations"):
        rec = store.record(
            predicate_type="traffic.conversations",
            operation="capture.conversations",
            parameters={"kind": kind},
            result={"conversations": facts["conversations"],
                    "count": len(facts["conversations"])},
            limitations=list(LIMITATIONS), locations=locations,
            links=[anchor], **provider, **common)
        evidence_ids.append(rec.evidence_id)

    dns = facts.get("dns") or {}
    if dns.get("queries") or dns.get("responses"):
        rec = store.record(
            predicate_type="traffic.dns", operation="capture.dns",
            parameters={"kind": kind},
            result={"queries": dns.get("queries") or [],
                    "responses": dns.get("responses") or []},
            limitations=list(LIMITATIONS), locations=locations,
            links=[anchor], **provider, **common)
        evidence_ids.append(rec.evidence_id)

    tls = facts.get("tls") or {}
    if tls.get("sessions"):
        rec = store.record(
            predicate_type="traffic.tls", operation="capture.tls",
            parameters={"kind": kind},
            result={"sessions": [
                        {k: v for k, v in s.items()
                         if not k.startswith("_")}
                        for s in tls["sessions"]],
                    "opaque_bytes": tls.get("opaque_bytes", 0)},
            limitations=list(LIMITATIONS), locations=locations,
            links=[anchor], **provider, **common)
        evidence_ids.append(rec.evidence_id)
    if tls.get("opaque_bytes"):
        unknowns.append(store.open_unknown(
            f"{tls['opaque_bytes']} bytes of TLS application data "
            "in this capture are encrypted — their content is "
            "unestablished by any static read of the capture",
            subject_sha256=subject_sha256, raised_by=anchor,
            supporting=[anchor],
            required_authority="controlled-replay",
            required_confidence="observed").to_dict())

    http = facts.get("http") or {}
    exchanges = http.get("exchanges") or facts.get("exchanges") or []
    if exchanges:
        rec = store.record(
            predicate_type="traffic.http", operation="capture.http",
            parameters={"kind": kind},
            result={"exchanges": exchanges,
                    "count": len(exchanges)},
            limitations=list(LIMITATIONS) + ([
                "this capture came through a proxy the owner ran "
                "— inside that proxy TLS is plaintext by design, "
                "which is why these files are sensitive; bodies "
                "are hashed, never copied"] if is_flows else []),
            locations=locations, links=[anchor], **provider,
            **common)
        evidence_ids.append(rec.evidence_id)

    if tshark_facts is not None:
        rec = store.record(
            predicate_type="traffic.tshark_summary",
            operation="tshark.analyze",
            provider_id=TSHARK_PROVIDER_ID,
            provider_name=TSHARK_PROVIDER_NAME,
            provider_version=str(tshark_facts.get("version") or ""),
            parameters={"engine": "tshark"},
            result={"protocols": tshark_facts.get("protocols") or {},
                    "conversations":
                        tshark_facts.get("conversations") or [],
                    "fields": tshark_facts.get("fields") or {},
                    "capinfos": tshark_facts.get("capinfos") or {}},
            limitations=list(LIMITATIONS), locations=locations,
            links=[anchor], **common)
        evidence_ids.append(rec.evidence_id)

    return {"evidence_ids": evidence_ids, "unknowns": unknowns}


def fold_addon(store: Any, *, subject_sha256: str,
               subject_name: str, subject_size: int,
               subject_path: str, facts: dict[str, Any],
               addon_version: str = "",
               link_to: str = "") -> dict[str, Any]:
    """Fold a Devon capture-addon JSONL into Evidence. The
    subject is the JSONL file itself — the artifact of record the
    owner's own proxy run produced."""
    evidence_ids: list[str] = []
    links = [link_to] if link_to else []
    rec = store.record(
        predicate_type="traffic.http", operation="addon.flows",
        provider_id=ADDON_PROVIDER_ID,
        provider_name="Devon mitmproxy capture addon "
                      "(owner-invoked)",
        provider_version=addon_version or PROVIDER_VERSION,
        parameters={"source": "devon-capture-addon"},
        result={"exchanges": facts.get("exchanges") or [],
                "records": facts.get("records", 0),
                "foreign_lines_skipped":
                    facts.get("foreign_lines_skipped", 0)},
        subject_sha256=subject_sha256, subject_name=subject_name,
        subject_size=subject_size,
        limitations=[
            "recorded by Devon's mitmproxy addon during a proxy "
            "run the owner invoked on their own traffic — inside "
            "that proxy TLS is plaintext by design, which is why "
            "the capture is sensitive; bodies are hashed, never "
            "copied into evidence",
            *LIMITATIONS],
        locations=[{"type": "artifact-path", "value": subject_path}],
        links=links)
    evidence_ids.append(rec.evidence_id)
    return {"evidence_ids": evidence_ids, "unknowns": []}
