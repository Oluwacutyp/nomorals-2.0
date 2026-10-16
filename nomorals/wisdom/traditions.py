"""Tradition packs — N-tradition descriptors (texts + timeline +
practice + language + provenance rules), Nigeria first.

Finding 1 fix. The seed corpus measured 47 texts, 0 sub-Saharan
African, 1 Mesoamerican, 45/47 Eurasian; 'Egypt' appeared only as an
origin of Christian/Gnostic lineage, not as African traditions. A
Universal OS with Nigeria/Africa first-class — not an axis — needs
packs that describe traditions *without* pretending to contain them.

Design rules (enforced in code, not in comments):
- A pack is a *descriptor*: texts are listed with provenance +
  restriction metadata. No sacred verses are embedded here.
- Provenance + permission are schema fields, not afterthoughts:
  every pack text carries ``provenance`` (source, license,
  permission_ref) and a ``restriction``.
- Restriction is first-class metadata and the system must STOP:
  :func:`check_ingest` raises :class:`RestrictedCorpusError` for
  ``initiated`` / ``do-not-digitize`` material without a permission
  record — it never scrapes around the restriction.
- Ifa is the canonical case: 256 odu, thousands of verses, initiated
  knowledge. The pack describes the *structure* (256 odu, divination
  system, oral corpus) and points at public scholarship — it does not
  digitize the verses, and :func:`check_ingest` refuses to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "RESTRICTIONS", "Provenance", "PackText", "TraditionPack",
    "RestrictedCorpusError", "nigeria_pack", "all_packs", "pack_by_id",
    "check_ingest", "corpus_coverage",
]

RESTRICTIONS = ("open", "public-scholarship", "community-permission",
                "initiated", "do-not-digitize")
#: Restrictions that STOP ingest without a permission record.
_BLOCKING = ("initiated", "do-not-digitize", "community-permission")


class RestrictedCorpusError(ValueError):
    """Ingest refused: restricted corpus, no permission record.

    This is a STOP, not a detour — the caller must obtain permission
    from the tradition's custodians (or use public scholarship) rather
    than routing around the restriction.
    """


@dataclass
class Provenance:
    """Provenance + permission for one pack text. Schema, not footnote."""

    source: str = ""  # where this descriptor came from
    license: str = ""  # e.g. public-domain, community-permission
    translator: str = ""
    collected_by: str = ""
    permission_ref: str = ""  # pointer to the permission record, if any
    restriction: str = "open"
    access_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PackText:
    """A text *descriptor*: title + kind + restriction + provenance.

    ``summary`` is a scholarly description (what the text *is*), never
    the text's content. Sacred verses are never embedded.
    """

    title: str
    kind: str = "text"  # text | oral-corpus | commentary | translation
    status: str = "descriptor"  # descriptor | public | restricted
    restriction: str = "open"
    provenance: Provenance = field(default_factory=Provenance)
    summary: str = ""
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class TraditionPack:
    """One tradition as a descriptor pack."""

    id: str
    name: str
    region: str  # e.g. "Nigeria / Yorubaland", "Egypt / Nile Valley"
    languages: list[str] = field(default_factory=list)
    description: str = ""
    texts: list[PackText] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    practices: list[dict[str, Any]] = field(default_factory=list)
    provenance_rules: list[str] = field(default_factory=list)
    restriction: str = "open"
    permission_required: bool = False
    notes: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.id:
            problems.append("pack id is required")
        if self.restriction not in RESTRICTIONS:
            problems.append(f"unknown restriction {self.restriction!r}")
        for text in self.texts:
            if text.restriction not in RESTRICTIONS:
                problems.append(
                    f"text {text.title!r}: unknown restriction "
                    f"{text.restriction!r}")
            if text.restriction in _BLOCKING and not text.provenance.permission_ref:
                # descriptor-only is fine; ingest would be blocked.
                # The pack itself is valid — the *ingest* is what stops.
                pass
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "region": self.region,
            "languages": self.languages, "description": self.description,
            "texts": [t.to_dict() for t in self.texts],
            "timeline": self.timeline, "practices": self.practices,
            "provenance_rules": self.provenance_rules,
            "restriction": self.restriction,
            "permission_required": self.permission_required,
            "notes": self.notes,
        }


def nigeria_pack() -> TraditionPack:
    """Nigeria pack — first-class, built first.

    Covers Yoruba (Ifa — restricted, descriptor-only), Igbo (Odinani),
    Hausa, Edo/Benin, Ibibio, and Tiv traditions. No sacred verses are
    included anywhere in this pack; Ifa's 256 odu are described as a
    *structure*, and ingest of odu verses is refused by
    :func:`check_ingest` without a permission record.
    """
    ifa = PackText(
        title="Ifa — 256 Odu (oral corpus)",
        kind="oral-corpus", status="restricted",
        restriction="do-not-digitize",
        provenance=Provenance(
            source="public scholarship on Ifa divination (descriptive)",
            license="not-licensed-for-digitization",
            restriction="do-not-digitize",
            access_note="Initiated knowledge. The 256 odu and their "
                        "verses are transmitted through initiation and "
                        "apprenticeship; they are NOT fully digitizable "
                        "and this pack does not pretend otherwise."),
        summary="The Yoruba divination system: 256 odu (16 major + 240 "
                "minor), each with thousands of associated verses (ese), "
                "proverbs, and sacrifices, interpreted by a babalawo. "
                "Descriptor only — structure, not verses.",
        language="Yoruba")
    return TraditionPack(
        id="nigeria", name="Nigeria — Yoruba, Igbo, Hausa, Edo, Ibibio, Tiv",
        region="Nigeria / West Africa",
        languages=["Yoruba", "Igbo", "Hausa", "Edo (Bini)", "Ibibio",
                   "Tiv", "English", "Nigerian Pidgin"],
        description="Nigeria-first tradition pack: the country's major "
                    "indigenous knowledge systems, described with "
                    "provenance + permission as schema fields. Oral "
                    "traditions are descriptor-first; recordings enter "
                    "only through the oral-ingest pipeline with owner "
                    "verification + permission.",
        texts=[
            ifa,
            PackText(
                title="Ori — Yoruba concept of head / destiny",
                kind="text", status="descriptor",
                restriction="public-scholarship",
                provenance=Provenance(
                    source="public scholarship on Yoruba thought",
                    license="public-scholarship",
                    restriction="public-scholarship",
                    access_note="Public scholarly descriptions only."),
                summary="Ori as inner head, destiny, and personal "
                        "divinity in Yoruba thought; central to "
                        "personhood and moral agency.",
                language="Yoruba"),
            PackText(
                title="Odinani — Igbo traditional religion (descriptor)",
                kind="text", status="descriptor",
                restriction="public-scholarship",
                provenance=Provenance(
                    source="public scholarship on Igbo cosmology",
                    license="public-scholarship",
                    restriction="public-scholarship"),
                summary="Igbo cosmology: Chukwu (the Great Spirit), "
                        "alusi (spirits), ancestors, ofo/ogu (justice), "
                        "and the chi (personal spirit).",
                language="Igbo"),
            PackText(
                title="Hausa — Bori + Islamic scholarship (descriptor)",
                kind="text", status="descriptor",
                restriction="public-scholarship",
                provenance=Provenance(
                    source="public scholarship on Hausa religious life",
                    license="public-scholarship",
                    restriction="public-scholarship"),
                summary="Hausa religious life: Bori spirit possession "
                        "traditions alongside centuries of Islamic "
                        "scholarship in Kano / Sokoto.",
                language="Hausa"),
            PackText(
                title="Edo / Benin — royal + ancestral tradition",
                kind="text", status="descriptor",
                restriction="public-scholarship",
                provenance=Provenance(
                    source="public scholarship on Benin Kingdom",
                    license="public-scholarship",
                    restriction="public-scholarship"),
                summary="Benin Kingdom: Oba kingship, ancestral altars, "
                        "bronze-casting guilds, and oral histories of "
                        "the royal court.",
                language="Edo (Bini)"),
        ],
        timeline=[
            {"start": -1000, "end": 1500,
             "title": "Ifa divination system matures in Yorubaland",
             "summary": "Oral corpus of 256 odu transmitted through "
                        "babalawo apprenticeship.",
             "sources": ["public scholarship on Ifa"]},
            {"start": 1200, "end": 1897,
             "title": "Benin Kingdom (Edo)",
             "summary": "Royal court, bronze guilds, ancestral altars.",
             "sources": ["public scholarship on Benin Kingdom"]},
        ],
        practices=[
            {"name": "Ifa consultation (descriptive)",
             "tradition": "Yoruba / Ifa",
             "summary": "A babalawo casts the opele / ikin to reveal an "
                        "odu and its counsel. Descriptor only — this is "
                        "not a how-to for divination.",
             "source": "public scholarship",
             "restriction": "initiated"},
            {"name": "Morning ori reflection",
             "tradition": "Yoruba",
             "summary": "Quiet reflection on one's head/destiny and the "
                        "day's conduct. A public, non-initiated practice "
                        "suitable for the practice guide.",
             "source": "public scholarship",
             "restriction": "open"},
        ],
        provenance_rules=[
            "Oral material enters only via oral ingest (draft → owner "
            "verifies → ingest) with a permission record.",
            "Restricted material (initiated / do-not-digitize) stops "
            "ingest without permission — never scraped around.",
            "Every passage carries provenance: source, translator, "
            "license, permission_ref, restriction.",
            "Public scholarship is cited as scholarship, never presented "
            "as the tradition speaking for itself.",
        ],
        restriction="community-permission",
        permission_required=True,
        notes="Nigeria first: this pack ships first and sets the schema "
              "every other pack follows.")


def _african_descriptor_pack(pack_id: str, name: str, region: str,
                             languages: list[str], description: str,
                             texts: list[PackText]) -> TraditionPack:
    return TraditionPack(
        id=pack_id, name=name, region=region, languages=languages,
        description=description, texts=texts,
        provenance_rules=[
            "Descriptor-first: structure + public scholarship, never "
            "embedded sacred text.",
            "Restricted material stops without a permission record.",
        ],
        restriction="public-scholarship",
        permission_required=False)


def all_packs() -> list[TraditionPack]:
    """All packs: Nigeria first, then further African + world descriptors."""
    packs = [nigeria_pack()]
    packs.append(_african_descriptor_pack(
        "kemet", "Kemet — ancient Egyptian tradition",
        "Egypt / Nile Valley", ["Egyptian (hieroglyphic)", "Coptic"],
        "Ancient Egyptian religion and wisdom: Ma'at, the Book of the "
        "Dead (Book of Coming Forth by Day), temple wisdom. The seed "
        "corpus's one 'egyptian' text is a start, not coverage.",
        [PackText(title="Book of Coming Forth by Day (descriptor)",
                  kind="text", status="descriptor",
                  restriction="public-scholarship",
                  provenance=Provenance(
                      source="public-domain translations",
                      license="public-domain",
                      restriction="public-scholarship"),
                  summary="Funerary text guiding the deceased through "
                          "the afterlife; public-domain translations "
                          "exist and are ingestible.",
                  language="Egyptian")]))
    packs.append(_african_descriptor_pack(
        "ethiopia", "Ethiopia — Ge'ez Christian + indigenous traditions",
        "Ethiopia / Horn of Africa", ["Ge'ez", "Amharic"],
        "Ethiopian Orthodox canon (including Enoch, Jubilees — texts the "
        "seed corpus already carries via other canons), plus indigenous "
        "traditions.",
        [PackText(title="Book of Enoch — Ethiopian canon (descriptor)",
                  kind="text", status="descriptor",
                  restriction="public-scholarship",
                  provenance=Provenance(
                      source="public-domain translations",
                      license="public-domain",
                      restriction="public-scholarship"),
                  summary="Enoch survives complete in Ge'ez; the seed "
                          "corpus carries it under judaism.",
                  language="Ge'ez")]))
    packs.append(_african_descriptor_pack(
        "akan", "Akan — Ghana / Côte d'Ivoire",
        "Ghana / Côte d'Ivoire", ["Twi", "Fante"],
        "Akan cosmology: Nyame, abosom, ancestors, adinkra symbolism, "
        "proverb-based ethics.",
        [PackText(title="Adinkra + proverb corpus (descriptor)",
                  kind="oral-corpus", status="descriptor",
                  restriction="public-scholarship",
                  provenance=Provenance(
                      source="public scholarship on Akan symbolism",
                      license="public-scholarship",
                      restriction="public-scholarship"),
                  summary="Adinkra symbols and the proverb corpus as "
                          "ethical teaching — descriptor only.",
                  language="Twi")]))
    packs.append(_african_descriptor_pack(
        "swahili-coast", "Swahili Coast — East Africa",
        "Kenya / Tanzania / Zanzibar", ["Swahili", "Arabic"],
        "Swahili-coast Islam, poetry (utendi), and coastal trading-city "
        "knowledge systems.",
        [PackText(title="Utendi poetry (descriptor)",
                  kind="text", status="descriptor",
                  restriction="public-scholarship",
                  provenance=Provenance(
                      source="public scholarship on Swahili poetry",
                      license="public-scholarship",
                      restriction="public-scholarship"),
                  summary="Swahili epic poetry tradition — descriptor only.",
                  language="Swahili")]))
    packs.append(_african_descriptor_pack(
        "zulu", "Zulu / Nguni — Southern Africa",
        "South Africa / Southern Africa", ["isiZulu", "isiXhosa"],
        "Nguni cosmology: uMvelinqangi, amadlozi (ancestors), ubuntu "
        "ethics, praise poetry (izibongo).",
        [PackText(title="Izibongo praise poetry (descriptor)",
                  kind="oral-corpus", status="descriptor",
                  restriction="community-permission",
                  provenance=Provenance(
                      source="public scholarship on izibongo",
                      license="community-permission",
                      restriction="community-permission",
                      access_note="Living oral tradition — community "
                                  "permission required for recordings."),
                  summary="Praise poetry for chiefs, kings, and "
                          "individuals — descriptor only.",
                  language="isiZulu")]))
    return packs


def pack_by_id(pack_id: str) -> TraditionPack:
    for pack in all_packs():
        if pack.id == pack_id:
            return pack
    raise ValueError(f"unknown tradition pack {pack_id!r}; known: "
                     f"{[p.id for p in all_packs()]}")


def check_ingest(pack: TraditionPack, text: PackText, *,
                 permission_ref: str = "") -> Provenance:
    """Gate ingest of one pack text. Returns the provenance to record.

    Raises :class:`RestrictedCorpusError` (STOP) when the text's
    restriction is blocking and no permission record is supplied.
    Open / public-scholarship texts pass with their provenance.
    """
    if text.restriction not in RESTRICTIONS:
        raise ValueError(f"unknown restriction {text.restriction!r}")
    if text.restriction in _BLOCKING and not permission_ref:
        raise RestrictedCorpusError(
            f"STOP — {text.title!r} is restricted "
            f"({text.restriction}); ingest requires a permission record "
            "from the tradition's custodians. The system will not "
            "scrape around this.")
    prov = text.provenance
    return Provenance(
        source=prov.source, license=prov.license,
        translator=prov.translator, collected_by=prov.collected_by,
        permission_ref=permission_ref or prov.permission_ref,
        restriction=text.restriction, access_note=prov.access_note)


def corpus_coverage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Measurable coverage over manifest-style entries.

    ``entries``: dicts with ``tradition`` and ``region`` keys.
    Returns counts by tradition, by region, and the Finding-1 numbers:
    sub-Saharan African texts vs Eurasian vs other — so the corpus can
    be *measured* instead of asserted.
    """
    sub_saharan = {"nigeria", "yoruba", "igbo", "hausa", "edo", "akan",
                   "zulu", "nguni", "swahili", "ethiopia", "kemet",
                   "sub-saharan", "west-africa", "east-africa",
                   "southern-africa"}
    eurasian = {"christianity", "christian-gnostic", "judaism",
                "hermeticism", "taoism", "hinduism", "buddhism",
                "confucianism", "islam", "sufism", "theosophy",
                "stoicism", "interfaith", "european"}
    # NOTE: "egyptian" (North Africa) and "mesoamerican" are NOT Eurasian;
    # they fall into "other", reproducing Finding 1 exactly: 45/47
    # Eurasian, 1 Mesoamerican, 1 Egyptian, 0 sub-Saharan African.
    by_tradition: dict[str, int] = {}
    by_region: dict[str, int] = {}
    counts = {"sub_saharan_african": 0, "eurasian": 0, "other": 0}
    for e in entries:
        trad = str(e.get("tradition", "")).lower()
        region = str(e.get("region", "")).lower()
        by_tradition[trad] = by_tradition.get(trad, 0) + 1
        if region:
            by_region[region] = by_region.get(region, 0) + 1
        if any(k in trad or k in region for k in sub_saharan):
            counts["sub_saharan_african"] += 1
        elif any(k in trad or k in region for k in eurasian):
            counts["eurasian"] += 1
        else:
            counts["other"] += 1
    total = len(entries)
    return {"total": total, "by_tradition": dict(sorted(by_tradition.items())),
            "by_region": dict(sorted(by_region.items())),
            "counts": counts,
            "sub_saharan_share": round(
                counts["sub_saharan_african"] / total, 4) if total else 0.0}
