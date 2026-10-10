"""Sweep tests for nomorals/security: dnsleak, netleak, exif.

No live network: socket/urllib entry points are monkeypatched.
"""

import binascii
import io
import struct
from pathlib import Path
from unittest import mock

import pytest

# ── fixtures: synthetic binary images ────────────────────────────────────

from nomorals.security import exif as E
from nomorals.security import dnsleak as D
from nomorals.security import netleak as N


def _seg(marker, payload):
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


def _tiff(bo="<", orientation=1, gps=None):
    order = b"II" if bo == "<" else b"MM"

    def rat(n, d):
        return struct.pack(bo + "II", n, d)

    entries = [struct.pack(bo + "HHI", 0x0112, 3, 1)
               + struct.pack(bo + "H", orientation) + b"\x00\x00"]
    extra = b""
    if gps:
        (lat, lat_ref, lon, lon_ref) = gps
        ifd0_len = 2 + 2 * 12 + 4
        gps_off = 8 + ifd0_len
        gps_data_off = gps_off + (2 + 4 * 12 + 4)

        def dms(v):
            d = int(v)
            m = int((v - d) * 60)
            s = ((v - d) * 60 - m) * 60
            return d, m, s

        la_d, la_m, la_s = dms(abs(lat))
        lo_d, lo_m, lo_s = dms(abs(lon))
        g = []
        g.append(struct.pack(bo + "HHI", 0x0001, 2, 2) + lat_ref.encode() + b"\x00\x00\x00")
        g.append(struct.pack(bo + "HHI", 0x0002, 5, 3) + struct.pack(bo + "I", gps_data_off))
        g.append(struct.pack(bo + "HHI", 0x0003, 2, 2) + lon_ref.encode() + b"\x00\x00\x00")
        g.append(struct.pack(bo + "HHI", 0x0004, 5, 3) + struct.pack(bo + "I", gps_data_off + 24))
        gps_ifd = struct.pack(bo + "H", 4) + b"".join(g) + struct.pack(bo + "I", 0)
        gps_data = (rat(la_d, 1) + rat(la_m, 1) + rat(int(la_s * 100), 100)
                    + rat(lo_d, 1) + rat(lo_m, 1) + rat(int(lo_s * 100), 100))
        entries.append(struct.pack(bo + "HHI", 0x8825, 4, 1)
                       + struct.pack(bo + "I", gps_off))
        extra = gps_ifd + gps_data
    ifd0 = struct.pack(bo + "H", len(entries)) + b"".join(entries) \
        + struct.pack(bo + "I", 0)
    return order + struct.pack(bo + "H", 42) + struct.pack(bo + "I", 8) \
        + ifd0 + extra


def make_jpeg(orientation=1, gps=None, with_xmp=True):
    exif = b"Exif\x00\x00" + _tiff("<", orientation, gps)
    parts = [b"\xff\xd8",
             _seg(0xE0, b"JFIF\x00" + b"\x00" * 9),
             _seg(0xE1, exif),
             _seg(0xED, b"Photoshop 3.0\x00iptc-data"),
             _seg(0xFE, b"comment-leak")]
    if with_xmp:
        parts.append(_seg(0xE1, b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta/>"))
    parts += [_seg(0xDA, b"\x01\x02"),
              b"\x11\x22\xff\x00\x33",  # fake scan data (stuffed FF)
              b"\xff\xd9"]
    return b"".join(parts)


def _chunk(typ, data):
    return (struct.pack(">I", len(data)) + typ + data
            + struct.pack(">I", binascii.crc32(typ + data) & 0xFFFFFFFF))


def make_png():
    ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00" * 12 for _ in range(4))  # 4x4 RGB rows
    import zlib
    idat = zlib.compress(raw)
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"tEXt", b"Software\x00Midjourney v6")
            + _chunk(b"tEXt", b"parameters\x00steps: 30, sampler: Euler")
            + _chunk(b"tIME", b"\x07\xea\x01\x02\x03\x04\x05")
            + _chunk(b"iCCP", b"profile\x00\x00" + b"\x00" * 8)
            + _chunk(b"eXIf", b"II*\x00")
            + _chunk(b"IDAT", idat)
            + _chunk(b"IEND", b""))


def make_webp():
    vp8x = b"VP8X" + struct.pack("<I", 10) + b"\x00" * 10
    exifc = b"EXIF" + struct.pack("<I", 6) + b"II*\x00\x00\x00"
    xmpc = b"XMP " + struct.pack("<I", 5) + b"xxxxx\x00"  # odd → pad
    vp8 = b"VP8 " + struct.pack("<I", 4) + b"\x9d\x01\x2a\x00"
    body = vp8x + exifc + xmpc + vp8
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body


# ── EXIF orientation / GPS parsing ───────────────────────────────────────

def test_exif_orientation_both_endians():
    for bo in ("<", ">"):
        info = E._parse_exif_app1(b"Exif\x00\x00" + _tiff(bo, orientation=6))
        assert info["orientation"] == 6, bo


def test_exif_gps_decimal_decode():
    # Lagos-ish: 6.5244N, 3.3792E
    info = E._parse_exif_app1(
        b"Exif\x00\x00" + _tiff("<", gps=(6.5244, "N", 3.3792, "E")))
    gps = info["gps"]
    assert gps is not None
    assert abs(gps["lat"] - 6.5244) < 0.001
    assert abs(gps["lon"] - 3.3792) < 0.001


def test_exif_gps_south_west_negative():
    info = E._parse_exif_app1(
        b"Exif\x00\x00" + _tiff(">", gps=(33.86, "S", 151.20, "E")))
    assert info["gps"]["lat"] < 0
    assert info["gps"]["lon"] > 0


# ── JPEG surgery ─────────────────────────────────────────────────────────

def test_jpeg_surgery_drops_all_metadata_keeps_image():
    raw = make_jpeg()
    clean, meta = E._jpeg_surgery(raw)
    assert sorted(meta["removed"]) == ["comment", "exif", "jfif",
                                      "photoshop/iptc", "xmp"]
    assert clean[:2] == b"\xff\xd8" and clean[-2:] == b"\xff\xd9"
    assert b"\x11\x22\xff\x00\x33" in clean  # scan data intact
    assert b"Photoshop" not in clean and b"xmpmeta" not in clean
    assert meta["orientation"] == 1


def test_jpeg_surgery_is_lossless_for_normal_orientation():
    from PIL import Image
    img = Image.new("RGB", (32, 32), color="green")
    ex = img.getexif()
    ex[271] = "TestCamera"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=ex)
    raw = buf.getvalue()
    clean = E.strip_metadata_bytes(raw)
    # pixel data identical → decode both and compare
    a = Image.open(io.BytesIO(raw)); a.load()
    b = Image.open(io.BytesIO(clean)); b.load()
    assert list(a.getdata()) == list(b.getdata())
    assert len(clean) < len(raw)


def test_jpeg_orientation_transposed_not_rotated():
    from PIL import Image
    img = Image.new("RGB", (40, 20), color="red")
    ex = img.getexif()
    ex[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=ex)
    clean = E.strip_metadata_bytes(buf.getvalue())
    out = Image.open(io.BytesIO(clean)); out.load()
    assert out.size == (20, 40)  # transposed, not left rotated
    ok, _ = E.verify_clean(clean)
    assert ok


def test_scan_metadata_jpeg_flags():
    s = E.scan_metadata(make_jpeg(gps=(6.5, "N", 3.3, "E")))
    assert s["format"] == "jpeg"
    assert s["has_exif"] and s["has_xmp"] and s["has_iptc"] and s["has_comment"]
    assert s["has_gps"] and abs(s["gps"]["lat"] - 6.5) < 0.01


# ── PNG surgery ──────────────────────────────────────────────────────────

def test_png_surgery_drops_text_time_icc_exif():
    raw = make_png()
    clean, meta = E._png_surgery(raw)
    assert "tEXt:Software" in meta["removed"]
    assert "tEXt:parameters" in meta["removed"]
    assert "tIME" in meta["removed"] and "iCCP" in meta["removed"]
    # structural chunks survive, CRCs valid
    from PIL import Image
    im = Image.open(io.BytesIO(clean)); im.load()
    assert im.size == (4, 4)
    assert b"Midjourney" not in clean
    ok, residual = E.verify_clean(clean)
    assert ok, residual


def test_png_ai_signals_detected():
    s = E.scan_metadata(make_png())
    sigs = " ".join(s["ai_signals"]).lower()
    assert "midjourney" in sigs
    assert "stable diffusion" in sigs
    assert s["png_text"]["Software"] == "Midjourney v6"


# ── WebP surgery ─────────────────────────────────────────────────────────

def test_webp_surgery_drops_exif_xmp_keeps_vp8():
    raw = make_webp()
    clean, meta = E._webp_surgery(raw)
    assert "exif" in meta["removed"] and "xmp" in meta["removed"]
    assert b"VP8 " in clean and b"VP8X" in clean
    assert b"EXIF" not in clean
    (riff_size,) = struct.unpack("<I", clean[4:8])
    assert riff_size == len(clean) - 8
    ok, _ = E.verify_clean(clean)
    assert ok


# ── C2PA / AI signals ────────────────────────────────────────────────────

def test_c2pa_jumb_detected():
    raw = make_jpeg(with_xmp=False) + b"\x00jumb\x00c2pa-manifest"
    s = E.scan_metadata(raw)
    assert any("C2PA" in sig or "JUMBF" in sig for sig in s["ai_signals"])


def test_trained_algorithmic_media_flagged():
    blob = b"\xff\xd8" + b"\x00" * 64 + b"trainedAlgorithmicMedia" + b"\xff\xd9"
    s = E.scan_metadata(blob)
    assert any("trainedAlgorithmicMedia" in sig for sig in s["ai_signals"])


# ── file-level API ───────────────────────────────────────────────────────

def test_strip_exif_file_roundtrip_with_backup(tmp_path):
    from PIL import Image
    p = tmp_path / "photo.jpg"
    img = Image.new("RGB", (16, 16), color="yellow")
    ex = img.getexif(); ex[271] = "LeakyCam"
    img.save(p, exif=ex)
    rep = E.strip_exif(p, backup=True)
    assert rep["had_exif"] is True
    assert rep["backup"] and Path(rep["backup"]).exists()
    assert rep["bytes_after"] < rep["bytes_before"]
    ok, _ = E.verify_clean(p.read_bytes())
    assert ok


def test_strip_exif_backup_collision(tmp_path):
    from PIL import Image
    p = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(p)
    (tmp_path / "a.jpg.bak").write_bytes(b"old")
    rep = E.strip_exif(p, backup=True)
    assert rep["backup"].endswith(".bak.1")


def test_clean_copy_mat2_style(tmp_path):
    from PIL import Image
    p = tmp_path / "shot.png"
    Image.new("RGB", (8, 8)).save(p)
    rep = E.clean_copy(p)
    assert rep["dest"].endswith(".cleaned.png")
    assert p.exists()  # original untouched
    assert rep["verified_clean"] is True


def test_secure_delete_wipes_and_unlinks(tmp_path):
    p = tmp_path / "secret.bak"
    p.write_bytes(b"sensitive-bytes-here")
    assert E.secure_delete(p, passes=2) is True
    assert not p.exists()
    assert E.secure_delete(tmp_path / "missing") is False


def test_exif_summary_backward_compat(tmp_path):
    from PIL import Image
    p = tmp_path / "x.jpg"
    img = Image.new("RGB", (8, 8))
    ex = img.getexif(); ex[271] = "Cam"
    img.save(p, exif=ex)
    s = E.exif_summary(p)
    assert s["has_exif"] is True and s["has_gps"] is False
    assert isinstance(s["tags"], dict)


def test_format_summary_readable():
    s = E.scan_metadata(make_jpeg(gps=(6.5, "N", 3.3, "E")))
    txt = E.format_summary(s)
    assert "JPEG" in txt and "GPS" in txt and "XMP" in txt


def test_strip_exif_to_bytes_alias():
    raw = make_png()
    assert E.strip_exif_to_bytes(raw) == E.strip_metadata_bytes(raw)


# ── STUN codec ───────────────────────────────────────────────────────────

def _synth_stun_response(ip="203.0.113.45", port=54321):
    req, txn = N._build_binding_request()
    xport = port ^ (N._STUN_MAGIC_COOKIE >> 16)
    xaddr = struct.unpack("!I", __import__("socket").inet_aton(ip))[0] \
        ^ N._STUN_MAGIC_COOKIE
    attr = struct.pack("!HH", 0x0020, 8) + b"\x00\x01" \
        + struct.pack("!H", xport) + struct.pack("!I", xaddr)
    resp = struct.pack("!HHI12s", 0x0101, len(attr),
                       N._STUN_MAGIC_COOKIE, txn) + attr
    return resp, txn, req


def test_stun_request_shape():
    req, txn = N._build_binding_request()
    assert len(req) == 20 and len(txn) == 12
    typ, ln, cookie = struct.unpack("!HHI", req[:8])
    assert (typ, ln, cookie) == (0x0001, 0, 0x2112A442)


def test_stun_response_decode():
    resp, txn, _ = _synth_stun_response()
    assert N._parse_binding_response(resp, txn) == "203.0.113.45"


def test_stun_rejects_txn_mismatch():
    resp, _, _ = _synth_stun_response()
    with pytest.raises(ValueError):
        N._parse_binding_response(resp, b"\x00" * 12)


def test_stun_rejects_wrong_type():
    resp, txn, _ = _synth_stun_response()
    bad = struct.pack("!HHI12s", 0x0111, 0, N._STUN_MAGIC_COOKIE, txn)
    with pytest.raises(ValueError):
        N._parse_binding_response(bad, txn)


def test_netleak_verdict_webrtc_leak():
    with mock.patch.object(N, "_http_text", return_value="198.51.100.7"), \
         mock.patch.object(N, "stun_public_ip", return_value="203.0.113.9"), \
         mock.patch.object(N, "local_interface_ips", return_value=[
             {"ip": "192.168.1.5", "class": "rfc1918-private"}]), \
         mock.patch.object(N, "ipv6_egress",
                           return_value={"egress": "", "local": [],
                                         "supported": True}):
        rep = N.check_net_leak()
    assert rep.webrtc_leak is True
    assert rep.stun_ip == "203.0.113.9"
    assert rep.local_exposure is True
    assert rep.ok is False
    assert "🚨" in N.format_report(rep)


def test_netleak_verdict_clean():
    with mock.patch.object(N, "_http_text", return_value="198.51.100.7"), \
         mock.patch.object(N, "stun_public_ip", return_value="198.51.100.7"), \
         mock.patch.object(N, "local_interface_ips", return_value=[]), \
         mock.patch.object(N, "ipv6_egress",
                           return_value={"egress": "", "local": [],
                                         "supported": False}):
        rep = N.check_net_leak()
    assert rep.webrtc_leak is False and rep.ok is True


def test_netleak_never_raises():
    with mock.patch.object(N, "_http_text", side_effect=OSError("down")), \
         mock.patch.object(N, "stun_public_ip", return_value=""), \
         mock.patch.object(N, "local_interface_ips", return_value=[]), \
         mock.patch.object(N, "ipv6_egress", side_effect=RuntimeError("x")):
        rep = N.check_net_leak()
    assert "check failed" in rep.note


# ── dnsleak: real bash.ws protocol ───────────────────────────────────────

def _mock_dns(monkeypatch, items, configured=("9.9.9.9",), public_ip="198.51.100.7",
              asn="9009"):
    probed = []
    monkeypatch.setattr(D, "_bash_ws_id", lambda: "leakid9")
    monkeypatch.setattr(D, "_http_json", lambda url, timeout=15: items)
    monkeypatch.setattr(D.socket, "getaddrinfo",
                        lambda h, p: probed.append(h) or [])
    monkeypatch.setattr(D, "_public_ip", lambda: public_ip)
    monkeypatch.setattr(D, "_public_ipv6", lambda: "")
    monkeypatch.setattr(D, "_ip_info",
                        lambda ip: {"asn": asn, "org": "T", "isp": "T",
                                    "country": "NL"})
    monkeypatch.setattr(D, "configured_resolvers", lambda: list(configured))
    monkeypatch.setattr(D, "detect_smhnr_risk",
                        lambda: {"applicable": False, "risk": False,
                                 "detail": ""})
    return probed


def test_bash_ws_typed_items_parsed(monkeypatch):
    items = [
        {"type": "ip", "ip": "198.51.100.7", "country_name": "NL", "asn": "9009"},
        {"type": "dns", "ip": "9.9.9.9", "country_name": "US", "asn": "19281"},
        {"type": "conclusion", "text": "No leak"},
    ]
    probed = _mock_dns(monkeypatch, items)
    rep = D.check_dns_leak(probes=6)
    assert rep.method == "bash.ws"
    assert rep.resolvers == ["9.9.9.9"]
    assert rep.resolver_info[0]["asn"] == "19281"  # from server, no geo lookup
    assert rep.conclusion == "No leak"
    assert rep.egress_ips == ["198.51.100.7"]
    assert rep.leak is False and rep.ok is True
    # real protocol: N distinct hosts under the server-issued id
    assert len(probed) == 6 and len(set(probed)) == 6
    assert all(h.endswith(".leakid9.bash.ws") for h in probed)


def test_bash_ws_uses_server_id_not_self_nonce(monkeypatch):
    _mock_dns(monkeypatch, [])
    with mock.patch.object(D, "_bash_ws_probe", return_value=None) as probe:
        rep = D.check_dns_leak()
    # empty items → retry path → falls back to local heuristic honestly
    assert rep.method == "local-heuristic"
    assert "bash.ws unreachable" in rep.note


def test_transparent_proxy_detected(monkeypatch):
    items = [
        {"type": "dns", "ip": "203.0.113.99", "country_name": "NG",
         "asn": "29465"},
    ]
    _mock_dns(monkeypatch, items, configured=("1.1.1.1",))
    rep = D.check_dns_leak(probes=2)
    assert rep.hijack is True
    assert "1.1.1.1" in rep.hijack_detail
    assert rep.ok is False


def test_strict_asn_mode(monkeypatch):
    items = [
        {"type": "dns", "ip": "10.8.0.1", "country_name": "NL", "asn": "9009"},
        {"type": "dns", "ip": "203.0.113.99", "country_name": "NG", "asn": "29465"},
    ]
    _mock_dns(monkeypatch, items, configured=())
    rep = D.check_dns_leak(expected_asn="9009", probes=2)
    assert rep.leak is True
    assert "203.0.113.99" in rep.leak_detail


def test_heuristic_isp_leak(monkeypatch):
    items = [
        {"type": "dns", "ip": "203.0.113.99", "country_name": "NG", "asn": "29465"},
    ]
    _mock_dns(monkeypatch, items, configured=(), asn="9009")
    rep = D.check_dns_leak(probes=2)
    assert rep.leak is True  # ASN differs from egress ASN, not public resolver
    assert "203.0.113.99" in rep.leak_detail


def test_public_resolvers_not_flagged(monkeypatch):
    items = [{"type": "dns", "ip": "1.1.1.1", "country_name": "US",
              "asn": "13335"}]
    _mock_dns(monkeypatch, items, configured=(), asn="9009")
    rep = D.check_dns_leak(probes=2)
    assert rep.leak is False and rep.ok is True


def test_dnsleak_never_raises(monkeypatch):
    monkeypatch.setattr(D, "_public_ip", mock.Mock(side_effect=RuntimeError("x")))
    rep = D.check_dns_leak()
    assert "check failed" in rep.note


def test_dnsleak_report_shape_backward_compat():
    r = D.DnsLeakReport()
    d = r.to_dict()
    assert {"leak", "resolvers", "method"} <= set(d)


def test_format_report_marks():
    rep = D.DnsLeakReport(ok=True, method="bash.ws", public_ip="1.2.3.4",
                          public_asn="9009", resolvers=["9.9.9.9"],
                          resolver_info=[{"ip": "9.9.9.9", "asn": "19281",
                                          "country": "US"}],
                          leak_detail="all good")
    txt = D.format_report(rep)
    assert "✅ NO LEAK" in txt and "9.9.9.9" in txt


def test_configured_resolvers_never_raises():
    assert isinstance(D.configured_resolvers(), list)


def test_smhnr_report_shape():
    r = D.detect_smhnr_risk()
    assert set(r) == {"applicable", "risk", "detail"}


# ── package surface ──────────────────────────────────────────────────────

def test_package_exports():
    import nomorals.security as S
    for name in ("check_dns_leak", "check_net_leak", "strip_exif",
                 "strip_exif_to_bytes", "scan_metadata", "clean_copy",
                 "verify_clean", "secure_delete", "stun_public_ip",
                 "local_interface_ips", "DnsLeakReport", "NetLeakReport"):
        assert hasattr(S, name), name
