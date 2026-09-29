"""decoder — the Universal Decoder as a tool.

One tool, three modes:
* ``analyze`` (default) — full forensic report: best decode chain, every
  successful decoder, entropy/charset/magic, hash identification +
  known-secret match, token-pattern detection, cookie parsing, JWT decode.
* ``decode`` — just the winning chain's output (for piping).
* ``decoders`` — list the engine's registered decoders.

Accepts a raw string or a workspace file path (text or binary — gzip,
zlib, images and other binary formats are handled).  Fully offline:
nothing is transmitted, ever.

    decoder(data="cGFzc3dvcmQ=")                    → password
    decoder(path="downloads/blob", mode="analyze")   → forensics report
    decoder(mode="decoders")                         → engine inventory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.decoder import (DECODERS, analyze, decode_any, identify_hash,
                            identify_magic, known_hash_lookup,
                            known_hash_lookup_chained, learn_hash,
                            save_report)
from ..core.errors import ToolError
from ..core.policy import Capability

_MAX_FILE_BYTES = 2_000_000


def _summarize(report: Any) -> str:
    """One-paragraph human summary of a report (for chat replies)."""
    bits: list[str] = []
    if report.best:
        chain = " → ".join(str(c) for c in report.best["chain"])
        out = report.best["output"]
        if isinstance(out, (bytes, bytearray)):
            bits.append(f"decodes as {' → '.join(str(c) for c in report.best['chain'])} "
                        f"(file: {identify_magic(bytes(out)) or 'binary'}, "
                        f"{len(out)} bytes)")
            out = None
        if out is not None and isinstance(out, str):
            shown = out if len(out) <= 120 else out[:120] + " …"
            bits.append(f"decodes as {chain}: {shown!r}")
        elif isinstance(out, dict):
            if "magic" in out:
                bits.append(f"decodes as {chain} (file: {out['magic']}, "
                            f"{out.get('bytes')} bytes)")
            elif "label" in out:
                bits.append(f"decodes as {chain} ({out['label']})")
            elif "columns" in out:
                bits.append(f"table: {out['row_count']} rows × "
                            f"{len(out['columns'])} cols "
                            f"({', '.join(out['columns'][:6])})")
            elif "parts" in out:
                names = [p.get("filename") for p in out["parts"]
                         if p.get("filename")]
                bits.append(f"MIME message with parts: {names or 'body'}")
            elif "header" in out:
                bits.append(f"JWT: alg={out['header'].get('alg')} "
                            f"signed={out['signed']} "
                            f"claims={list(out['payload'])[:6]}")
            else:
                bits.append(f"structured decode via {chain}")
        elif out is not None:
            bits.append(f"decodes as {chain}")
    if report.hash:
        if report.hash.get("known"):
            k = report.hash["known"]
            bits.append(f"hash match: {k['algorithm']} of {k['plaintext']!r}")
        else:
            bits.append("hash: " + "/".join(report.hash["candidates"]))
    for t in report.tokens[:4]:
        bits.append(f"token: {t['label']} ({t['sample']})")
    if report.cookies:
        names = [c["name"] for c in report.cookies if "name" in c]
        flags = [c["flag"] for c in report.cookies if "flag" in c]
        bits.append("cookies: " + ", ".join(names[:6])
                    + (f" [flags: {', '.join(flags[:5])}]" if flags else ""))
    if report.jwt and report.jwt.get("warnings"):
        bits.append("JWT warnings: " + "; ".join(report.jwt["warnings"]))
    f = report.forensics
    bits.append(f"({f['bytes']} bytes, entropy {f['entropy']}, "
                f"charset {f['charset']}"
                + (f", magic {f['magic']}" if f.get("magic") else "") + ")")
    return " ".join(bits) if bits else "nothing decodable found"


#: algorithms the auto-attack will attempt on its own (bounded, quiet)
_AUTO_CRACKABLE = ("md5", "sha1", "sha256", "sha512", "crc32", "ntlm")


#: bounded pass plan for the auto-attack: (label, charset, min_len, max_len,
#: candidates, markov).  Each pass runs hybrid (corpus + rules + the
#: charset's brute force) but charset passes SKIP Markov: a digits pass
#: spending its budget on Markov word-shapes never reaches the PIN space,
#: yet its wordlist phase still catches corpus+rule words (hunter2+1).
#: The final bare hybrid pass owns the full Markov budget.  First hit wins.
_AUTO_PASSES: tuple[tuple[str, str, int, int, int, bool], ...] = (
    ("digits", "0123456789", 3, 8, 120_000, False),
    ("lowercase", "abcdefghijklmnopqrstuvwxyz", 3, 4, 480_000, False),
    ("hybrid", "", 1, 6, 200_000, True),
)


def auto_attack(
    digest: str,
    *,
    algo: str = "",
    charset: str = "",
    min_len: int = 1,
    max_len: int = 6,
    max_candidates: int = 800_000,
    db: Any = None,
) -> dict[str, Any]:
    """Identify → known-secrets → (if unknown) bounded multi-pass crack.

    This is the "auto attack": a digest is never left sitting at
    "candidates: md5/sha1" when a few hundred thousand candidates can settle
    it.  Passes run digits-only, then lowercase, then the hybrid wordlist/
    markov engine — each bounded, first hit wins, whole thing capped at
    ``max_candidates`` total so an interactive call stays fast.  The full
    ``hashcrack`` engine is one flag away for bigger jobs.
    """
    from .hashcrack import crack_hash

    digest = (digest or "").strip()
    candidates = identify_hash(digest)
    out: dict[str, Any] = {"digest": digest, "candidates": candidates}
    if not candidates:
        out["cracked"] = False
        out["note"] = "does not look like a crackable digest"
        return out
    builtin = known_hash_lookup(digest)
    known = builtin if builtin is not None else \
        known_hash_lookup_chained(db, digest)
    if known:
        out["cracked"] = True
        out["plaintext"] = known["plaintext"]
        out["algorithm"] = known["algorithm"]
        out["via"] = "known-secrets"
        out["known_source"] = "built-in" if builtin is not None else "learned"
        return out
    algo = algo or next((c for c in candidates if c in _AUTO_CRACKABLE), "")
    if not algo:
        out["cracked"] = False
        out["note"] = (f"algorithm {candidates[0]} is not auto-crackable "
                       "here — use the hashcrack engine directly")
        return out

    # explicit charset/length → single pass exactly as asked
    if charset:
        passes = (("custom", charset, min_len, max_len,
                   min(int(max_candidates), 2_000_000), False),)
    else:
        passes = _AUTO_PASSES

    total_tested = 0
    total_secs = 0.0
    for label, pcharset, pmin, pmax, pcap, pmarkov in passes:
        if total_tested >= max_candidates:
            break
        room = max_candidates - total_tested
        try:
            result = crack_hash(digest, algo=algo, charset=pcharset,
                                min_len=pmin, max_len=pmax,
                                max_candidates=min(pcap, room),
                                use_markov=pmarkov)
        except Exception as exc:  # noqa: BLE001
            out.setdefault("pass_errors", []).append(f"{label}: {exc}")
            continue
        d = result.as_dict()
        total_tested += d["tested"]
        total_secs += d["elapsed"]
        found = result.found.get(digest)
        out["passes"] = out.get("passes", [])
        out["passes"].append({"pass": label, "tested": d["tested"],
                              "seconds": d["elapsed"], "found": bool(found),
                              "backend": d["backend"]})
        if found:
            out.update({
                "cracked": True,
                "algorithm": algo,
                "plaintext": found,
                "via": f"live-crack:{label}",
                "tested": total_tested,
                "seconds": round(total_secs, 3),
            })
            # chain: persist this solve so the next sighting is instant
            if db is not None:
                out["learned"] = learn_hash(
                    db, digest, found, algorithm=algo,
                    source=f"auto-attack:{label}")
            return out
    out.update({
        "cracked": False,
        "algorithm": algo,
        "plaintext": None,
        "via": "not-found",
        "tested": total_tested,
        "seconds": round(total_secs, 3),
        "note": (f"no match in {total_tested:,} candidates across "
                 f"{len(out.get('passes', []))} passes "
                 f"({round(total_secs, 1)}s) — widen with the hashcrack tool "
                 "(--max-len / --charset / --mode wordlist)"),
    })
    return out


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "decoder",
        description=(
            "Universal decoder: identify + decode almost anything — base64/"
            "hex/base32/base58/binary/rot13/atbash/url/entities/escapes/"
            "morse/leet (incl. nested chains), JSON/YAML/CSV/INI/key-value/"
            "MIME/PEM/JWT/gzip, hash identification with known-secret match, "
            "cookie parsing, API-token detection, entropy + file-magic "
            "forensics. Pass data= (string) or path= (workspace file, text "
            "or binary). mode: analyze (default) | decode | decoders | "
            "crack (digest → identify → known secrets → bounded live "
            "crack). analyze auto-attacks identified unknown hashes "
            "(auto_crack=False to skip)."
        ),
        capability=Capability.FS_READ,
    )
    def decoder(data: str = "", *, path: str = "",
                mode: str = "analyze", max_depth: int = 3,
                auto_crack: bool = True, algo: str = "",
                charset: str = "", min_len: int = 1, max_len: int = 6,
                max_candidates: int = 800_000) -> dict[str, Any]:
        if mode == "decoders":
            return {
                "decoders": [
                    {"name": d.name, "description": d.description}
                    for d in DECODERS
                ],
                "count": len(DECODERS),
            }
        if not data and not path:
            raise ToolError("decoder needs data= (string) or path= (file)")
        max_depth = max(1, min(int(max_depth or 3), 6))

        db = getattr(context, "db", None)
        if mode == "crack":
            if not data:
                raise ToolError("decoder crack needs the digest in data=")
            return auto_attack(data, algo=algo, charset=charset,
                               db=db,
                               min_len=min_len, max_len=max_len,
                               max_candidates=max_candidates)

        if path:
            from .filesystem import safe_path

            target = safe_path(context, path, must_exist=True)
            raw = target.read_bytes()[:_MAX_FILE_BYTES]
            truncated = target.stat().st_size > _MAX_FILE_BYTES
            report = analyze(raw, max_depth=max_depth)
            input_text = raw.decode("utf-8", "ignore")
        else:
            if isinstance(data, str) and len(data) > 2_000_000:
                data = data[:2_000_000]
            report = analyze(data, max_depth=max_depth)
            input_text = data

        # report archive: every decode is queryable later (nm decoder history)
        report_id = save_report(db, report, source=f"decoder:{mode}",
                                kind=mode, input_text=input_text)

        out: dict[str, Any] = {"report": report.to_dict(),
                               "summary": _summarize(report)}
        if report_id:
            out["report_id"] = report_id
        if mode == "decode":
            out["decoded"] = (report.best or {}).get("output")
        elif path:
            out["path"] = str(target)
            out["truncated"] = truncated

        # OSINT feed: successful decodes map cookies/JWTs into the
        # identity graph as person + domain entities (fail-soft —
        # enrichment, never a failure; crack mode never gets here).
        try:
            from ..agents.osint_graph import IdentityGraph

            best_name = getattr(report.best, "name", None) or "analyze"
            osint = IdentityGraph(context).ingest_decoder_findings(
                report, source=f"decoder:{best_name}")
            if osint["persons_found"] or osint["domains_found"]:
                out["osint"] = osint
        except Exception:  # noqa: BLE001
            pass

        # auto-attack: an identified, unknown digest gets a bounded
        # live-crack attempt before the report is returned
        hinfo = report.hash
        if auto_crack and hinfo and not hinfo.get("known"):
            digest = (input_text or "").strip()
            if identify_hash(digest):
                try:
                    out["auto_crack"] = auto_attack(
                        digest, algo=algo, charset=charset, db=db,
                        min_len=min_len, max_len=max_len,
                        max_candidates=max_candidates)
                except Exception as exc:  # noqa: BLE001
                    out["auto_crack"] = {"cracked": False,
                                         "note": f"auto-attack failed: {exc}"}
        return out

    @registry.register(
        "identify_hash",
        description=(
            "Identify a hash/digest by shape (md5/sha1/sha256/sha512/bcrypt/"
            "argon2/NTLM/CRC32/…) and match it against the built-in "
            "common-secrets table."
        ),
        capability=Capability.FS_READ,
    )
    def identify_hash_tool(digest: str) -> dict[str, Any]:
        digest = (digest or "").strip()
        if not digest:
            raise ToolError("identify_hash needs the digest string")
        candidates = identify_hash(digest)
        if not candidates:
            return {"digest": digest, "candidates": [], "known": None,
                    "note": "does not look like a hash digest"}
        return {"digest": digest, "candidates": candidates,
                "known": known_hash_lookup_chained(
                    getattr(context, "db", None), digest)}
