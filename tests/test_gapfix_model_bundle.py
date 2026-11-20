"""Voice-model bundles (.dvmodel) — fakeyou-recast §3B.13.

The format is Devon's own, mined from storytold/fakeyou-recast's
model-serde shape: a plaintext header (fast metadata) + three parts
(unencrypted details, encrypted details, encrypted heavy payload
with acoustic + vocoder bytes). The tests mirror the upstream's own
test shapes (empty file, bad magic, fast read, encrypted metadata,
full read) and add Devon's guarantees: every refusal names the part,
corruption is caught by the per-part hashes and MACs, expiry and
password gates fire, and a bundle registers into the existing
ServiceRegistry as a dvmodel-format artifact.
"""

from __future__ import annotations

import struct

import pytest

from nomorals.voice.model_bundle import (
    MAGIC,
    ModelBundleError,
    VoiceModel,
    _assemble_blob,
    control_model_bundle,
    read_details,
    read_fast,
    read_full,
    read_metadata,
    register_model_bundle,
    unpack_bundle,
    validate_bundle,
    write_bundle,
)

SECRET = "test-bundle-secret"


def _model(**overrides):
    base = dict(
        token="dvm_morgan-1", name="Morgan Freeman", author="Vegito",
        model_version="1.2.0",
        created_at_ms=1666148294000, updated_at_ms=1666148294000,
        expires_at_s=1668737894, invalid_before_s=None,
        bypass_expiry=True, password=None,
        pipeline_version=1,
        acoustic_model_bytes=b"AAAA" + bytes(range(200)),
        vocoder_model_bytes=b"VVVV" + bytes(range(150)))
    base.update(overrides)
    return VoiceModel(**base)


def _packed(tmp_path, **overrides):
    path = tmp_path / "morgan.dvmodel"
    write_bundle(_model(**overrides), SECRET, str(path))
    return str(path)


def _craft(path, model, *, secret=SECRET, salt=b"\x01" * 16):
    """Write bundle bytes for a model WITHOUT validation — the reader
    edge-case path (a bundle another writer produced, or a tampered
    one). Timestamps default to fixed values so the bytes are stable."""
    if model.created_at_ms is None:
        model.created_at_ms = 1666148294000
    if model.updated_at_ms is None:
        model.updated_at_ms = 1666148294000
    blob = _assemble_blob(model, secret, salt)
    with open(path, "wb") as fh:
        fh.write(blob)
    return str(path)


def _flip_byte(path, offset):
    with open(path, "r+b") as fh:
        fh.seek(offset)
        byte = fh.read(1)
        fh.seek(offset)
        fh.write(bytes([byte[0] ^ 0xFF]))


def _heavy_offset(path):
    """Byte offset of the heavy part's envelope inside a packed file."""
    with open(path, "rb") as fh:
        blob = fh.read()
    pos = len(MAGIC) + 2 + 16
    (fast_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4 + fast_len
    for _name in ("details", "encrypted_details"):
        (length,) = struct.unpack_from("<I", blob, pos)
        pos += 4 + length
    (heavy_len,) = struct.unpack_from("<I", blob, pos)
    return pos + 4, heavy_len


def _flip_heavy_byte_and_rehash(path, at=40):
    """Flip a byte inside the heavy envelope AND update the header's
    recorded heavy hash to match — so the tamper reaches the MAC
    check instead of the hash check. (The recorded hash is a fixed
    64-hex string, so the header length is unchanged.)"""
    import hashlib
    offset, length = _heavy_offset(path)
    with open(path, "r+b") as fh:
        blob = bytearray(fh.read())
    blob[offset + at] ^= 0xFF
    envelope = bytes(blob[offset:offset + length])
    new_hash = hashlib.sha256(envelope).hexdigest().encode()
    fast = read_fast(path)
    old_hash = fast.parts["heavy"].encode()
    assert len(old_hash) == len(new_hash) == 64
    idx = bytes(blob).find(old_hash)
    assert idx != -1
    blob[idx:idx + 64] = new_hash
    with open(path, "wb") as fh:
        fh.write(bytes(blob))


# -- round trip ---------------------------------------------------------

class TestRoundTrip:
    def test_pack_read_fast(self, tmp_path):
        path = _packed(tmp_path)
        fast = read_fast(path)
        assert fast.format_version == "1.0.0"
        assert fast.token == "dvm_morgan-1"
        assert fast.name == "Morgan Freeman"
        assert fast.author == "Vegito"
        assert set(fast.parts) == {"details", "encrypted_details", "heavy"}

    def test_pack_read_metadata(self, tmp_path):
        path = _packed(tmp_path)
        meta = read_metadata(path, SECRET)
        assert meta.model_version == "1.2.0"
        assert meta.created_at_ms == 1666148294000
        assert meta.updated_at_ms == 1666148294000
        assert meta.expires_at_s == 1668737894
        assert meta.invalid_before_s is None
        assert meta.bypass_expiry is True
        assert meta.password_set is False
        assert meta.heavy_bytes > 0

    def test_pack_read_full(self, tmp_path):
        path = _packed(tmp_path)
        model = read_full(path, SECRET, now=1667000000)
        assert model.token == "dvm_morgan-1"
        assert model.name == "Morgan Freeman"
        assert model.pipeline_version == 1
        assert model.acoustic_model_bytes.startswith(b"AAAA")
        assert model.vocoder_model_bytes.startswith(b"VVVV")
        assert len(model.acoustic_model_bytes) == 204
        assert len(model.vocoder_model_bytes) == 154

    def test_read_details_needs_no_secret(self, tmp_path):
        path = _packed(tmp_path)
        details = read_details(path)
        assert details["model_version"] == "1.2.0"
        assert details["created_at_ms"] == 1666148294000

    def test_two_packs_differ_by_salt(self, tmp_path):
        _packed(tmp_path)
        second = tmp_path / "second.dvmodel"
        write_bundle(_model(), SECRET, str(second))
        assert (tmp_path / "morgan.dvmodel").read_bytes() != second.read_bytes()
        # ... but both read back identically.
        assert read_full(str(second), SECRET,
                         now=1667000000).acoustic_model_bytes.startswith(b"AAAA")

    def test_unpack_writes_both_model_files(self, tmp_path):
        path = _packed(tmp_path, expires_at_s=None)
        out = tmp_path / "out"
        receipt = unpack_bundle(path, SECRET, str(out))
        assert receipt["token"] == "dvm_morgan-1"
        assert receipt["acoustic_bytes"] == 204
        assert receipt["vocoder_bytes"] == 154
        assert (out / "dvm-morgan-1-acoustic.bin"
                ).read_bytes().startswith(b"AAAA")
        assert (out / "dvm-morgan-1-vocoder.bin"
                ).read_bytes().startswith(b"VVVV")

    def test_validate_bundle_ok(self, tmp_path):
        path = _packed(tmp_path)
        assert validate_bundle(path)["ok"] is True
        assert validate_bundle(path, SECRET)["ok"] is True
        assert validate_bundle(path, SECRET)["pipeline_version"] == 1


# -- refusals name the part ----------------------------------------------

class TestRefusals:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.dvmodel"
        path.write_bytes(b"")
        with pytest.raises(ModelBundleError, match="fast metadata"):
            read_fast(str(path))

    def test_bad_magic(self, tmp_path):
        path = tmp_path / "small.txt"
        path.write_bytes(b"hello world, not a bundle")
        with pytest.raises(ModelBundleError,
                           match="fast metadata: bad magic"):
            read_fast(str(path))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_fast(str(tmp_path / "nope.dvmodel"))

    def test_wrong_secret_names_the_part(self, tmp_path):
        path = _packed(tmp_path)
        with pytest.raises(ModelBundleError,
                           match="encrypted_details: authentication failed"):
            read_metadata(path, "wrong-secret")

    def test_flipped_heavy_byte_fails_mac_naming_heavy(self, tmp_path):
        path = _packed(tmp_path)
        _flip_heavy_byte_and_rehash(path)
        with pytest.raises(ModelBundleError,
                           match="heavy: authentication failed"):
            read_full(path, SECRET, now=1667000000)

    def test_flipped_heavy_byte_fails_hash_first(self, tmp_path):
        # Without re-recording the hash, the header hash check fires
        # first — still naming the heavy part.
        path = _packed(tmp_path)
        offset, _length = _heavy_offset(path)
        _flip_byte(path, offset + 40)
        with pytest.raises(ModelBundleError,
                           match="heavy: sha256 mismatch"):
            read_full(path, SECRET, now=1667000000)

    def test_flipped_details_byte_fails_hash_naming_details(self, tmp_path):
        path = _packed(tmp_path)
        with open(path, "rb") as fh:
            blob = fh.read()
        pos = len(MAGIC) + 2 + 16
        (fast_len,) = struct.unpack_from("<I", blob, pos)
        pos += 4 + fast_len
        (details_len,) = struct.unpack_from("<I", blob, pos)
        _flip_byte(path, pos + 4 + details_len // 2)
        with pytest.raises(ModelBundleError,
                           match="details: sha256 mismatch"):
            read_metadata(path, SECRET)

    def test_trailing_bytes_refused(self, tmp_path):
        path = _packed(tmp_path)
        with open(path, "ab") as fh:
            fh.write(b"junk")
        with pytest.raises(ModelBundleError, match="trailing bytes"):
            read_fast(path)

    def test_missing_acoustic_refused_on_read(self, tmp_path):
        path = _craft(tmp_path / "no-acoustic.dvmodel",
                      _model(acoustic_model_bytes=b""))
        with pytest.raises(ModelBundleError,
                           match="heavy payload: acoustic model missing"):
            read_full(path, SECRET, now=1667000000)

    def test_missing_vocoder_refused_on_read(self, tmp_path):
        path = _craft(tmp_path / "no-vocoder.dvmodel",
                      _model(vocoder_model_bytes=b""))
        with pytest.raises(ModelBundleError,
                           match="heavy payload: vocoder model missing"):
            read_full(path, SECRET, now=1667000000)

    def test_validate_bundle_reports_tamper_without_raising(self, tmp_path):
        path = _packed(tmp_path)
        offset, _length = _heavy_offset(path)
        _flip_byte(path, offset + 40)
        result = validate_bundle(path, SECRET)
        assert result["ok"] is False
        assert "heavy" in result["error"]


# -- expiry + password gates ----------------------------------------------

class TestGates:
    def test_expired_model_refused_naming_expiry(self, tmp_path):
        path = _packed(tmp_path, expires_at_s=1667000000,
                       bypass_expiry=None)
        with pytest.raises(ModelBundleError,
                           match="encrypted details: model expired"):
            read_full(path, SECRET, now=1667000001)

    def test_expired_model_allowed_with_flag(self, tmp_path):
        path = _packed(tmp_path, expires_at_s=1667000000)
        model = read_full(path, SECRET, now=1667000001, allow_expired=True)
        assert model.token == "dvm_morgan-1"

    def test_expired_model_allowed_by_bypass_flag(self, tmp_path):
        path = _packed(tmp_path, expires_at_s=1667000000,
                       bypass_expiry=True)
        model = read_full(path, SECRET, now=1667000001)
        assert model.token == "dvm_morgan-1"

    def test_expired_model_refused_when_bypass_is_false(self, tmp_path):
        path = _packed(tmp_path, expires_at_s=1667000000,
                       bypass_expiry=False)
        with pytest.raises(ModelBundleError,
                           match="encrypted details: model expired"):
            read_full(path, SECRET, now=1667000001)

    def test_not_yet_valid_refused(self, tmp_path):
        path = _packed(tmp_path, invalid_before_s=1700000000,
                       expires_at_s=None)
        with pytest.raises(ModelBundleError,
                           match="encrypted details: model is not valid "
                                 "before"):
            read_full(path, SECRET, now=1667000000)

    def test_password_required(self, tmp_path):
        path = _packed(tmp_path, password="open-sesame")
        with pytest.raises(ModelBundleError,
                           match="encrypted details: this model needs its "
                                 "password"):
            read_full(path, SECRET, now=1667000000)

    def test_password_mismatch(self, tmp_path):
        path = _packed(tmp_path, password="open-sesame")
        with pytest.raises(ModelBundleError,
                           match="encrypted details: password mismatch"):
            read_full(path, SECRET, password="wrong", now=1667000000)

    def test_password_ok(self, tmp_path):
        path = _packed(tmp_path, password="open-sesame")
        model = read_full(path, SECRET, password="open-sesame",
                          now=1667000000)
        assert model.token == "dvm_morgan-1"


# -- writer validation ------------------------------------------------------

class TestWriterValidation:
    def test_token_too_long(self, tmp_path):
        with pytest.raises(ModelBundleError,
                           match="fast metadata: model token too long"):
            write_bundle(_model(token="x" * 33), SECRET,
                         str(tmp_path / "x.dvmodel"))

    def test_token_bad_charset(self, tmp_path):
        with pytest.raises(ModelBundleError,
                           match="fast metadata: bad model token"):
            write_bundle(_model(token="bad token!"), SECRET,
                         str(tmp_path / "x.dvmodel"))

    def test_name_too_long(self, tmp_path):
        with pytest.raises(ModelBundleError,
                           match="fast metadata: model name too long"):
            write_bundle(_model(name="n" * 256), SECRET,
                         str(tmp_path / "x.dvmodel"))

    def test_author_too_long(self, tmp_path):
        with pytest.raises(ModelBundleError,
                           match="fast metadata: author username too long"):
            write_bundle(_model(author="a" * 21), SECRET,
                         str(tmp_path / "x.dvmodel"))

    def test_empty_acoustic_refused_at_pack(self, tmp_path):
        with pytest.raises(ModelBundleError,
                           match="heavy payload: acoustic model missing"):
            write_bundle(_model(acoustic_model_bytes=b""), SECRET,
                         str(tmp_path / "x.dvmodel"))

    def test_empty_vocoder_refused_at_pack(self, tmp_path):
        with pytest.raises(ModelBundleError,
                           match="heavy payload: vocoder model missing"):
            write_bundle(_model(vocoder_model_bytes=b""), SECRET,
                         str(tmp_path / "x.dvmodel"))

    def test_empty_secret_refused(self, tmp_path):
        with pytest.raises(ModelBundleError,
                           match="a bundle needs a secret"):
            write_bundle(_model(), "", str(tmp_path / "x.dvmodel"))

    def test_no_replace_refuses_existing(self, tmp_path):
        path = _packed(tmp_path)
        with pytest.raises(FileExistsError, match="already exists"):
            write_bundle(_model(), SECRET, path)

    def test_replace_overwrites(self, tmp_path):
        path = _packed(tmp_path)
        receipt = write_bundle(_model(name="Morgan Freeman II"), SECRET,
                               path, replace=True)
        assert read_fast(path).name == "Morgan Freeman II"
        assert receipt["path"] == path

    def test_receipt_carries_sha256(self, tmp_path):
        import hashlib
        path = tmp_path / "receipt.dvmodel"
        receipt = write_bundle(_model(), SECRET, str(path))
        assert receipt["sha256"] == hashlib.sha256(
            path.read_bytes()).hexdigest()
        assert receipt["token"] == "dvm_morgan-1"
        assert receipt["parts"]["heavy"] > 0


# -- registry integration ---------------------------------------------------

class TestRegistry:
    def test_register_bundle(self, tmp_path):
        bundle = _packed(tmp_path)
        services_dir = tmp_path / "services"
        decl = register_model_bundle(bundle, services_dir=str(services_dir))
        assert decl.slug == "dvm-morgan-1"
        assert decl.version == "1.2.0"
        assert decl.task == "voice-convert"
        assert decl.artifact.format == "dvmodel"
        assert decl.artifact.path == bundle

    def test_registered_bundle_validates(self, tmp_path):
        bundle = _packed(tmp_path)
        services_dir = tmp_path / "services"
        decl = register_model_bundle(bundle, services_dir=str(services_dir))
        decl.artifact.validate()  # the dvmodel branch: magic + header

    def test_registered_bundle_listed(self, tmp_path):
        bundle = _packed(tmp_path)
        services_dir = tmp_path / "services"
        register_model_bundle(bundle, services_dir=str(services_dir))
        from nomorals.voice.services import ServiceRegistry
        rows = ServiceRegistry(str(services_dir)).list(task="voice-convert")
        assert [d.slug for d in rows] == ["dvm-morgan-1"]

    def test_tampered_bundle_fails_validation(self, tmp_path):
        bundle = _packed(tmp_path)
        services_dir = tmp_path / "services"
        decl = register_model_bundle(bundle, services_dir=str(services_dir))
        offset, _length = _heavy_offset(bundle)
        _flip_byte(bundle, offset + 40)
        with pytest.raises(ValueError, match="sha256 mismatch"):
            decl.artifact.validate()

    def test_register_uses_format_version_when_model_version_is_not_semver(
            self, tmp_path):
        bundle = _packed(tmp_path, model_version="v2-draft")
        services_dir = tmp_path / "services"
        decl = register_model_bundle(bundle, services_dir=str(services_dir))
        assert decl.version == "1.0.0"


# -- CLI + chat ---------------------------------------------------------------

class TestCliAndChat:
    def _args(self, argv):
        from nomorals.cmdline.parser import _parser
        return _parser().parse_args(argv)

    def test_cli_pack_inspect_unpack_register(self, tmp_path, monkeypatch):
        from nomorals.cmdline.commands.voice import _cmd_voice_model
        monkeypatch.setenv("NM_VOICE_BUNDLE_SECRET", SECRET)
        # The CLI register writes to the default service store — point
        # it at a scratch dir so the test never touches the real one.
        monkeypatch.setenv("NOMORALS_VOICE_SERVICES_DIR",
                           str(tmp_path / "services"))
        acoustic = tmp_path / "a.bin"
        vocoder = tmp_path / "v.bin"
        acoustic.write_bytes(b"AAAA-acoustic")
        vocoder.write_bytes(b"VVVV-vocoder")
        out = tmp_path / "m.dvmodel"
        args = self._args(["voice", "model", "pack",
                           "--acoustic", str(acoustic),
                           "--vocoder", str(vocoder),
                           "--name", "Morgan Freeman",
                           "--token", "dvm_morgan-1",
                           "--author", "Vegito",
                           "--out", str(out)])
        assert _cmd_voice_model(args, None) == 0
        args = self._args(["voice", "model", "inspect", str(out)])
        assert _cmd_voice_model(args, None) == 0
        args = self._args(["voice", "model", "inspect", str(out), "--full"])
        assert _cmd_voice_model(args, None) == 0
        args = self._args(["voice", "model", "unpack", str(out),
                           "--out-dir", str(tmp_path / "unpacked")])
        assert _cmd_voice_model(args, None) == 0
        assert (tmp_path / "unpacked" / "dvm-morgan-1-acoustic.bin"
                ).read_bytes() == b"AAAA-acoustic"
        args = self._args(["voice", "model", "register", str(out)])
        assert _cmd_voice_model(args, None) == 0

    def test_cli_pack_without_secret_fails(self, tmp_path, monkeypatch):
        from nomorals.cmdline.commands.voice import _cmd_voice_model
        monkeypatch.delenv("NM_VOICE_BUNDLE_SECRET", raising=False)
        args = self._args(["voice", "model", "pack",
                           "--acoustic", "a", "--vocoder", "v",
                           "--name", "N", "--token", "t",
                           "--author", "A", "--out", str(tmp_path / "o")])
        assert _cmd_voice_model(args, None) == 1

    def test_cli_inspect_missing_file_fails(self, tmp_path):
        from nomorals.cmdline.commands.voice import _cmd_voice_model
        args = self._args(["voice", "model", "inspect",
                           str(tmp_path / "nope.dvmodel")])
        assert _cmd_voice_model(args, None) == 1

    def test_chat_list_inspect_register(self, tmp_path):
        bundle = _packed(tmp_path)
        services_dir = str(tmp_path / "services")
        assert "no voice-model bundles" in control_model_bundle(
            "list", services_dir=services_dir)
        text = control_model_bundle(f"inspect {bundle}",
                                    services_dir=services_dir)
        assert "dvm_morgan-1" in text and "Morgan Freeman" in text
        text = control_model_bundle(f"register {bundle}",
                                    services_dir=services_dir)
        assert "registered dvm-morgan-1@1.2.0" in text
        assert "dvm-morgan-1" in control_model_bundle(
            "list", services_dir=services_dir)

    def test_chat_inspect_missing_file_names_it(self, tmp_path):
        text = control_model_bundle(
            "inspect /nonexistent.dvmodel",
            services_dir=str(tmp_path / "services"))
        assert text.startswith("model inspect failed:")

    def test_chat_unknown_verb_is_usage(self, tmp_path):
        text = control_model_bundle(
            "explode", services_dir=str(tmp_path / "services"))
        assert text.startswith("usage: /voice model")
