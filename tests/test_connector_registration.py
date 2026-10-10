"""Registration + honest-degradation tests for connectors that were wired
into the runtime but missing from the registry.

Regression: ``audd`` and ``duffel`` existed as modules but were never
imported by ``nomorals.connectors.__init__``, so ``create_connector`` /
``get_connector`` raised ``unknown connector`` — the AudD music-recognition
path in the partner runtime (``runtime_media.py``) was dead code in
production. These tests pin the registration.
"""

from __future__ import annotations

import unittest

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.audd import AudDConnector
from nomorals.connectors.duffel import DuffelConnector
from nomorals.connectors.registry import (
    create_connector,
    get_connector,
    list_connectors,
)
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class RegistrationTest(unittest.TestCase):
    def test_audd_registered(self):
        self.assertIs(get_connector("audd"), AudDConnector)
        conn = create_connector("audd", _vault())
        self.assertIsInstance(conn, AudDConnector)

    def test_duffel_registered(self):
        self.assertIs(get_connector("duffel"), DuffelConnector)
        conn = create_connector("duffel", _vault())
        self.assertIsInstance(conn, DuffelConnector)

    def test_both_in_list(self):
        ids = {c["id"] for c in list_connectors()}
        self.assertIn("audd", ids)
        self.assertIn("duffel", ids)

    def test_audd_degrades_honestly_without_key(self):
        conn = create_connector("audd", _vault())
        st = conn.status()
        self.assertFalse(st.connected)
        self.assertIn("audd", st.detail.lower())
        out = conn.recognize("/tmp/does-not-exist.mp3")
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("needs_key"))

    def test_duffel_degrades_honestly_without_key(self):
        conn = create_connector("duffel", _vault())
        st = conn.status()
        self.assertFalse(st.connected)


if __name__ == "__main__":
    unittest.main()
