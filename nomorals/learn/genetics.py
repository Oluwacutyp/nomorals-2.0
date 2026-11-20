"""Genetics learning — virtual fruit-fly crosses, exact, taught.

Pattern-only from echelon/drosophila-lab (AGPL-3.0 — a FlyLab
clone): the DESIGN is taken — build designer parents from a mutant
allele catalogue, cross them, read the offspring — and everything
else is Devon's own work, written from scratch in this module. No
code was ported. The original simulated 1000 random offspring with
an unseeded RNG, ignored linkage entirely, and had no chi-square, no
prediction scoring and no lessons. This organ does the genetics
EXACTLY (rational arithmetic throughout), with real linkage
(Haldane map distances), Drosophila male achiasmy (males do not
recombine — a real fact the original ignored), scores the student's
own predictions against the exact ratios, runs chi-square on counted
broods, and teaches through six guided lessons.

The allele catalogue (trait names, allele codes, chromosomes and map
positions in centimorgans) follows the published FlyLab/Drosophila
mutant set — public genetics facts, not code. Where this module
improves on the original's model it says so in the docstring of the
function that does it.

Named refusals (never a faked ratio): ``unknown_trait``,
``unknown_allele``, ``invalid_genotype``, ``bad_ratio``,
``too_many_loci``, ``no_cross``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "ALLELES",
    "Allele",
    "ChiSquare",
    "CrossResult",
    "GeneticsError",
    "GeneticsSession",
    "Genotype",
    "LESSONS",
    "Phenotype",
    "Trait",
    "TRAITS",
    "alleles_for",
    "chi_square",
    "get_allele",
    "get_session",
    "get_trait",
    "lesson",
    "list_lessons",
    "parse_genotype",
    "parse_ratios",
    "predict",
    "score_prediction",
    "simulate",
]

#: Gamete enumeration cap per parent — realistic crosses use 1-3
#: traits (<= 8 gametes); beyond this the exact table is a wall of
#: noise, so the engine refuses by name instead of stalling.
MAX_GAMETES = 4096


class GeneticsError(Exception):
    """A named refusal from the genetics engine."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ── catalogue (public Drosophila genetics facts) ─────────────────────────
# (code, name, trait abbr, chromosome, map position cM, dominant, lethal)
_ALLELE_ROWS: tuple[tuple[str, str, str, str | int, float, bool, bool], ...] = (
    ("F", "Forked", "BR", "X", 56.3, False, False),
    ("SV", "Shaven", "BR", 4, 3.0, False, False),
    ("SN", "Singed", "BR", "X", 21.0, False, False),
    ("SS", "Spineless", "BR", 3, 58.5, False, False),
    ("SB", "Stubble", "BR", 3, 58.2, True, True),
    ("BL", "Black", "BC", 2, 48.5, False, False),
    ("E", "Ebony", "BC", 2, 70.7, False, False),
    ("S", "Sable", "BC", "X", 43.0, False, False),
    ("T", "Tan", "BC", "X", 27.7, False, False),
    ("Y", "Yellow", "BC", "X", 0.0, False, False),
    ("AR", "Aristapedia", "AN", 3, 47.7, True, False),
    ("BW", "Brown", "EC", 2, 104.5, False, False),
    ("PR", "Purple", "EC", 2, 54.5, False, False),
    ("SE", "Sepia", "EC", 3, 26.0, False, False),
    ("W", "White", "EC", "X", 1.5, False, False),
    ("B", "Bar", "ES", "X", 57.0, True, False),
    ("EY", "Eyeless", "ES", 4, 2.0, False, False),
    ("L", "Lobe", "ES", 2, 72.0, True, False),
    ("ST", "Star", "ES", 2, 1.3, True, True),
    ("AP", "Apterous", "SZ", 2, 55.4, False, False),
    ("M", "Miniature", "SZ", "X", 36.1, False, False),
    ("VG", "Vestigial", "SZ", 2, 67.0, False, False),
    ("CY", "Curly", "SH", 2, 6.1, True, True),
    ("C", "Curved", "SH", 2, 75.5, False, False),
    ("DP", "Dumpy", "SH", 2, 13.0, False, False),
    ("SD", "Scalloped", "SH", "X", 51.5, False, False),
    ("CV", "Crossveinless", "VN", "X", 13.7, False, False),
    ("RI", "Radius Incomplete", "VN", 3, 48.4, False, False),
    ("D", "Dichaete", "AG", 3, 41.0, True, True),
)

_TRAIT_ROWS: tuple[tuple[str, str], ...] = (
    ("BR", "Bristle"),
    ("BC", "Body Color"),
    ("AN", "Antennae"),
    ("EC", "Eye Color"),
    ("ES", "Eye Shape"),
    ("SZ", "Wing Size"),
    ("SH", "Wing Shape"),
    ("VN", "Wing Vein"),
    ("AG", "Wing Angle"),
)


class Trait:
    """One trait (e.g. Body Color) with its mutant alleles."""

    def __init__(self, abbr: str, name: str) -> None:
        self.abbr = abbr
        self.name = name
        self.alleles: dict[str, Allele] = {}

    def __repr__(self) -> str:  # noqa: D105
        return f"Trait({self.abbr}/{self.name})"


class Allele:
    """One mutant allele: code, name, trait, chromosome, map position."""

    def __init__(self, code: str, name: str, trait: Trait,
                 chromo: str | int, position: float,
                 dominant: bool = False, lethal: bool = False) -> None:
        self.code = code
        self.name = name
        self.trait = trait
        self.chromo = "X" if str(chromo).upper() == "X" else int(chromo)
        self.position = position
        self.dominant = dominant
        self.lethal = lethal
        trait.alleles[code.upper()] = self

    def is_autosomal(self) -> bool:
        return self.chromo != "X"

    def __repr__(self) -> str:  # noqa: D105
        return (f"Allele({self.code}/{self.name} @{self.chromo}:"
                f"{self.position})")


def _build_catalogue() -> tuple[dict[str, Trait], dict[str, Allele]]:
    traits = {abbr: Trait(abbr, name) for abbr, name in _TRAIT_ROWS}
    alleles = {}
    for code, name, abbr, chromo, pos, dom, leth in _ALLELE_ROWS:
        alleles[code.upper()] = Allele(code, name, traits[abbr], chromo,
                                       pos, dom, leth)
    return traits, alleles


TRAITS, ALLELES = _build_catalogue()


def get_trait(abbr: str) -> Trait:
    """Look a trait up by abbreviation or name — else unknown_trait."""
    key = (abbr or "").strip().upper()
    if key in TRAITS:
        return TRAITS[key]
    for trait in TRAITS.values():
        if trait.name.upper() == key:
            return trait
    raise GeneticsError("unknown_trait", f"no trait '{abbr}'")


def get_allele(code: str) -> Allele:
    """Look an allele up by code — else unknown_allele."""
    key = (code or "").strip().upper()
    if key in ALLELES:
        return ALLELES[key]
    raise GeneticsError("unknown_allele", f"no allele '{code}'")


def alleles_for(trait: str) -> list[Allele]:
    """Every mutant allele of a trait, in map order."""
    return sorted(get_trait(trait).alleles.values(),
                  key=lambda a: (str(a.chromo), a.position))


# ── genotype / phenotype ─────────────────────────────────────────────────


class Genotype:
    """One fly: sex + per-trait mutant alleles (wild type unrecorded).

    ``genes`` maps trait abbreviation to a tuple of 0-2 alleles. A
    male carries at most one allele per X-linked trait.
    """

    def __init__(self, sex: str = "f",
                 genes: dict[str, tuple[Allele, ...]] | None = None) -> None:
        sex = sex.lower()
        if sex not in ("f", "m"):
            raise GeneticsError("invalid_genotype", f"sex must be f/m, got '{sex}'")
        self.sex = sex
        self.genes: dict[str, tuple[Allele, ...]] = dict(genes or {})

    def valid(self) -> tuple[bool, str]:
        """Lethality + sex rules. Returns (ok, reason)."""
        for abbr, alleles in self.genes.items():
            if len(alleles) > 2:
                return False, f"{abbr}: more than two alleles"
            if len(alleles) == 2 and alleles[0].lethal and alleles[1].lethal:
                return False, f"{abbr}: two lethal alleles"
            for allele in alleles:
                if allele.lethal and not allele.is_autosomal() and self.sex == "m":
                    return False, f"{allele.name}: lethal X-linked male"
            if (len(alleles) == 2 and self.sex == "m"
                    and not alleles[0].is_autosomal()
                    and not alleles[1].is_autosomal()):
                return False, f"{abbr}: male with two X-linked alleles"
        return True, ""

    def check(self) -> None:
        ok, reason = self.valid()
        if not ok:
            raise GeneticsError("invalid_genotype", reason)

    def phenotype(self) -> Phenotype:
        return Phenotype(self)

    def describe(self) -> str:
        """'female, Ebony/Ebony, Brown/wild type'."""
        parts = ["female" if self.sex == "f" else "male"]
        for abbr in sorted(self.genes):
            alleles = self.genes[abbr]
            if len(alleles) == 1:
                parts.append(f"{alleles[0].name}/wild type")
            else:
                parts.append(f"{alleles[0].name}/{alleles[1].name}")
        return ", ".join(parts) if len(parts) > 1 else parts[0] + ", wild type"

    def __repr__(self) -> str:  # noqa: D105
        return f"Genotype({self.describe()})"


class Phenotype:
    """What a genotype LOOKS like: only the alleles that show.

    An allele shows when it is dominant, lethal, homozygous, or —
    for an X-linked allele — when the fly is male. Everything else
    is carried silently. Wild type is the absence of a shown allele.
    """

    def __init__(self, genotype: Genotype) -> None:
        self.sex = genotype.sex
        self.shown: dict[str, Allele] = {}
        for abbr, alleles in genotype.genes.items():
            shown: Allele | None = None
            if len(alleles) == 1:
                allele = alleles[0]
                if (allele.dominant or allele.lethal
                        or (not allele.is_autosomal() and self.sex == "m")):
                    shown = allele
            else:
                a1, a2 = alleles
                if (a1.code == a2.code or a1.dominant or a1.lethal):
                    shown = a1
                elif a2.dominant or a2.lethal:
                    shown = a2
                elif not a1.is_autosomal() and self.sex == "m":
                    shown = a1
                elif not a2.is_autosomal() and self.sex == "m":
                    shown = a2
            if shown is not None:
                self.shown[abbr] = shown

    def label(self) -> str:
        """Code label: 'E+BW', or 'WT' for wild type."""
        if not self.shown:
            return "WT"
        return "+".join(sorted(a.code for a in self.shown.values()))

    def names(self) -> str:
        """Human label: 'Ebony, Brown', or 'wild type'."""
        if not self.shown:
            return "wild type"
        return ", ".join(a.name for _, a in sorted(self.shown.items()))

    def __repr__(self) -> str:  # noqa: D105
        return f"Phenotype({self.label()} {self.sex})"


def parse_genotype(spec: str) -> Genotype:
    """Parse ``f E/E BW/WT`` / ``m W`` / ``female AR``.

    First token is the sex (``f``/``m``/``female``/``male``); each
    following token is one trait as ``CODE`` or ``CODE/CODE`` (``WT``
    = wild type). Allele codes are unique across traits, so a code
    names its trait. Raises GeneticsError by name on bad input.
    """
    tokens = (spec or "").strip().split()
    if not tokens:
        raise GeneticsError("invalid_genotype", "empty genotype spec")
    sex_token = tokens[0].lower()
    if sex_token in ("f", "female"):
        sex = "f"
    elif sex_token in ("m", "male"):
        sex = "m"
    else:
        raise GeneticsError(
            "invalid_genotype",
            f"genotype must start with f/m, got '{tokens[0]}'")
    genes: dict[str, list[Allele]] = {}
    for token in tokens[1:]:
        parts = token.split("/")
        if len(parts) > 2 or not parts[0]:
            raise GeneticsError("invalid_genotype", f"bad allele token '{token}'")
        for code in parts:
            if code.upper() == "WT":
                continue
            allele = get_allele(code)
            genes.setdefault(allele.trait.abbr, []).append(allele)
    genotype = Genotype(sex, {k: tuple(v) for k, v in genes.items()})
    genotype.check()
    return genotype


# ── exact prediction ─────────────────────────────────────────────────────
# Gametes are exact rational distributions, not samples. Per
# chromosome the parent carries two homologues; a gamete takes one
# allele per locus from either homologue. Females recombine between
# adjacent loci with Haldane's map function
# r = 1/2 (1 - e^(-2d/100)); Drosophila males do NOT recombine at
# all (achiasmy), so a male's gametes are his two parental
# haplotypes, 50/50. Phase is coupling by construction — the mutant
# alleles listed for a parent sit on one homologue, wild type on the
# other — because the genotype spec carries no phase information;
# every result names that assumption.


def _haldane(distance_cm: float) -> Fraction:
    """Recombination fraction for a map distance (Haldane, no interference)."""
    if distance_cm <= 0:
        return Fraction(0)
    r = 0.5 * (1.0 - math.exp(-2.0 * distance_cm / 100.0))
    return Fraction(r).limit_denominator(10 ** 9)


def _chromosome_groups(parent: Genotype) -> dict[str | int, list[tuple[str, Allele, Allele | None]]]:
    """Group a parent's loci by chromosome, in map order.

    Returns {chromosome: [(trait_abbr, homologue_A allele, homologue_B
    allele), ...]} where homologue A carries the first-listed allele
    (coupling) and B the second (or wild type). Homozygous loci carry
    the same allele on both homologues.
    """
    by_chromo: dict[str | int, list[tuple[str, Allele, Allele | None]]] = {}
    for abbr, alleles in parent.genes.items():
        allele = alleles[0]
        chromo = allele.chromo
        if len(alleles) == 1:
            pair: tuple[str, Allele, Allele | None] = (abbr, allele, None)
        else:
            pair = (abbr, alleles[0], alleles[1])
        by_chromo.setdefault(chromo, []).append(pair)
    for pairs in by_chromo.values():
        pairs.sort(key=lambda p: p[1].position)
    return by_chromo


def _haplotype_probs(pairs: list[tuple[str, Allele, Allele | None]],
                     recombines: bool) -> dict[tuple[Allele | None, ...], Fraction]:
    """Exact gamete-haplotype distribution for one chromosome.

    Each haplotype is a tuple of one allele per locus (None = wild
    type). Probability comes from the switch pattern: start on either
    homologue (1/2 each), then at each interval either stay (1 - r)
    or switch (r). A non-recombining parent only ever produces the
    two parental haplotypes.
    """
    n = len(pairs)
    if n == 0:
        return {(): Fraction(1)}
    # (homologue A allele, homologue B allele) per locus; None is the
    # wild-type copy of a heterozygous locus. Homozygous loci carry the
    # same allele on both homologues.
    homos = [(a, b) for _, a, b in pairs]
    if not recombines:
        hap_a = tuple(a for a, _ in homos)
        hap_b = tuple(b for _, b in homos)
        if hap_a == hap_b:
            return {hap_a: Fraction(1)}
        return {hap_a: Fraction(1, 2), hap_b: Fraction(1, 2)}
    intervals = [
        _haldane(abs(pairs[i + 1][1].position - pairs[i][1].position))
        for i in range(n - 1)
    ]
    out: dict[tuple[Allele | None, ...], Fraction] = {}
    for switches in range(1 << (n - 1)) if n > 1 else [0]:
        for start in (0, 1):
            prob = Fraction(1, 2)
            side = start
            hap: list[Allele | None] = []
            for i, (a, b) in enumerate(homos):
                if i > 0:
                    switched = bool(switches & (1 << (i - 1)))
                    r = intervals[i - 1]
                    prob *= r if switched else (1 - r)
                    if switched:
                        side = 1 - side
                allele = (a, b)[side]
                hap.append(None if allele is None else allele)
            # Wild-type-only loci on a homologue are stored as the
            # allele itself when homozygous; None marks the WT copy
            # of a heterozygous locus.
            key = tuple(hap)
            out[key] = out.get(key, Fraction(0)) + prob
    # Normalise away zero-probability haplotypes (r == 0 intervals).
    return {k: v for k, v in out.items() if v > 0}


def _gametes(parent: Genotype) -> dict[tuple[tuple[str, Allele | None], ...], Fraction]:
    """Exact gamete distribution for a parent.

    A gamete is a tuple of (trait_abbr, allele-or-None) over every
    locus the parent carries. For an X chromosome in a male the
    gamete is either the whole X haplotype or the Y (no X alleles).
    """
    groups = _chromosome_groups(parent)
    recombines = parent.sex == "f"
    per_chromo: list[dict[tuple[Allele | None, ...], Fraction]] = []
    chromo_order: list[str | int] = []
    for chromo, pairs in groups.items():
        if chromo == "X" and parent.sex == "m":
            hap = tuple(a for _, a, _ in pairs)
            per_chromo.append({hap: Fraction(1, 2), tuple(None for _ in pairs): Fraction(1, 2)})
        else:
            per_chromo.append(_haplotype_probs(pairs, recombines))
        chromo_order.append(chromo)
    if not per_chromo:
        return {(): Fraction(1)}
    combined: dict[tuple[tuple[str, Allele | None], ...], Fraction] = {(): Fraction(1)}
    for chromo, dist in zip(chromo_order, per_chromo, strict=True):
        pairs = groups[chromo]
        new: dict[tuple[tuple[str, Allele | None], ...], Fraction] = {}
        for gamete, prob in combined.items():
            for hap, hprob in dist.items():
                loci = tuple(
                    (pairs[i][0], hap[i]) for i in range(len(pairs)))
                new[gamete + loci] = new.get(gamete + loci, Fraction(0)) + prob * hprob
        combined = new
        if len(combined) > MAX_GAMETES:
            raise GeneticsError(
                "too_many_loci",
                f"more than {MAX_GAMETES} gametes — cross fewer traits at once")
    return combined


@dataclass
class CrossResult:
    """The exact outcome of a cross, as rational proportions."""

    mother: Genotype
    father: Genotype
    combined: dict[str, Fraction] = field(default_factory=dict)
    by_sex: dict[str, dict[str, Fraction]] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    inviable: Fraction = Fraction(0)
    total_gametes: int = 0

    def ratio_lines(self) -> list[str]:
        """'Ebony — 1/4 (25%)' lines, biggest first, ties by label."""
        total = sum(self.combined.values())
        lines = []
        for label, prob in sorted(self.combined.items(),
                                  key=lambda kv: (-kv[1], kv[0])):
            frac = prob / total if total else Fraction(0)
            pct = float(frac * 100)
            lines.append(f"  {self.names.get(label, label)} "
                         f"[{label}] — {_display_fraction(frac)} ({pct:.1f}%)")
        return lines

    def sex_lines(self) -> list[str]:
        """Per-sex ratio lines, only when the sexes differ."""
        if self.by_sex.get("f") == self.by_sex.get("m"):
            return []
        lines = []
        for sex in ("f", "m"):
            dist = self.by_sex.get(sex, {})
            total = sum(dist.values())
            if not total:
                continue
            head = "daughters" if sex == "f" else "sons"
            lines.append(f"  {head}:")
            for label, prob in sorted(dist.items(),
                                      key=lambda kv: (-kv[1], kv[0])):
                frac = prob / total
                pct = float(frac * 100)
                lines.append(f"    {self.names.get(label, label)} "
                             f"[{label}] — {_display_fraction(frac)} ({pct:.1f}%)")
        return lines


def _display_fraction(frac: Fraction) -> str:
    """A readable fraction: exact when small, '~' + near when not.

    Haldane recombination fractions are transcendental, so linked
    ratios carry big denominators; the percentage next to it is the
    number a student reads, and the '~' says the fraction is rounded.
    """
    if frac.denominator <= 64:
        return str(frac)
    return "~" + str(frac.limit_denominator(100))


def predict(mother: Genotype, father: Genotype) -> CrossResult:
    """Exact offspring distribution of a cross (rational arithmetic).

    The mother is the female parent, the father the male — swapped
    automatically when the arguments arrive the other way round.
    Inviable genotypes (lethal combinations) are dropped and their
    share reported as ``inviable``; the surviving ratios are
    renormalised. Phase is coupling (mutants together) — the genotype
    spec carries no phase, and every result says so.
    """
    if mother.sex == "m" and father.sex == "f":
        mother, father = father, mother
    if mother.sex != "f" or father.sex != "m":
        raise GeneticsError(
            "invalid_genotype", "a cross needs one female and one male parent")
    mother.check()
    father.check()
    m_gametes = _gametes(mother)
    f_gametes = _gametes(father)
    result = CrossResult(mother=mother, father=father,
                         total_gametes=len(m_gametes) * len(f_gametes))
    combined: dict[str, Fraction] = {}
    by_sex: dict[str, dict[str, Fraction]] = {"f": {}, "m": {}}
    names: dict[str, str] = {}
    inviable = Fraction(0)
    # Sex comes from the father: an X-bearing paternal gamete makes a
    # daughter, a Y (no X alleles at all) makes a son. A father with
    # no X loci at all has no X/Y split in his gametes, so each of his
    # gametes is split evenly between daughters and sons here.
    father_has_x = any(
        allele is not None and allele.chromo == "X"
        for gamete in f_gametes for _, allele in gamete)
    for m_gam, m_prob in m_gametes.items():
        for f_gam, f_prob in f_gametes.items():
            prob = m_prob * f_prob
            child_genes: dict[str, list[Allele]] = {}
            for abbr, allele in m_gam + f_gam:
                if allele is not None:
                    child_genes.setdefault(abbr, []).append(allele)
            if father_has_x:
                father_gave_x = any(
                    allele is not None and allele.chromo == "X"
                    for _, allele in f_gam)
                sexes: tuple[tuple[str, Fraction], ...] = (
                    ("f" if father_gave_x else "m", prob),)
            else:
                sexes = (("f", prob / 2), ("m", prob / 2))
            for sex, sex_prob in sexes:
                child = Genotype(sex, {k: tuple(v)
                                        for k, v in child_genes.items()})
                ok, _ = child.valid()
                if not ok:
                    inviable += sex_prob
                    continue
                pheno = child.phenotype()
                label = pheno.label()
                names[label] = pheno.names()
                combined[label] = combined.get(label, Fraction(0)) + sex_prob
                by_sex[sex][label] = by_sex[sex].get(label, Fraction(0)) + sex_prob
    # Renormalise the survivors (inviable share excluded).
    total = sum(combined.values())
    if total <= 0:
        raise GeneticsError("invalid_genotype",
                            "this cross has no viable offspring")
    result.combined = {k: v / total for k, v in combined.items()}
    result.by_sex = {
        sex: ({k: v / sum(d.values()) for k, v in d.items()} if d else {})
        for sex, d in by_sex.items()
    }
    result.names = names
    result.inviable = inviable
    return result


def simulate(mother: Genotype, father: Genotype, n: int = 100,
             seed: int = 0) -> dict[str, int]:
    """A counted brood of ``n`` offspring — seeded, so reproducible.

    Draws from the exact distribution instead of re-simulating
    meiosis per fly (same numbers, honest about it): the brood is a
    multinomial sample of the exact ratios, with inviable conceptions
    counted separately under ``INVIABLE``.
    """
    if n <= 0:
        raise GeneticsError("bad_ratio", "brood size must be positive")
    result = predict(mother, father)
    rng = random.Random(seed)
    labels = sorted(result.combined)
    weights = [float(result.combined[label]) for label in labels]
    counts: dict[str, int] = {label: 0 for label in labels}
    counts["INVIABLE"] = 0
    total_weight = sum(weights) + float(result.inviable)
    for _ in range(n):
        roll = rng.random() * total_weight
        acc = 0.0
        picked: str | None = None
        for label, weight in zip(labels, weights, strict=True):
            acc += weight
            if roll < acc:
                picked = label
                break
        if picked is None:
            counts["INVIABLE"] += 1
        else:
            counts[picked] += 1
    return {k: v for k, v in counts.items() if v > 0}


# ── chi-square + prediction scoring ───────────────────────────────────────


def _gammq(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) — Numerical Recipes.

    Pure-Python series / continued-fraction evaluation; no scipy
    anywhere in this repo, so the chi-square p-value is computed
    here, not wished for.
    """
    if x < 0 or a <= 0:
        raise GeneticsError("bad_ratio", "gamma arguments out of range")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        # Series for P(a, x); Q = 1 - P.
        term = 1.0 / a
        total = term
        n = 1
        while True:
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14 or n > 1000:
                break
            n += 1
        p_val = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p_val))
    # Continued fraction for Q(a, x).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q_val = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return max(0.0, min(1.0, q_val))


@dataclass
class ChiSquare:
    """Chi-square goodness of fit: statistic, df, p-value, verdict."""

    statistic: float
    dof: int
    p_value: float
    verdict: str


def chi_square(observed: dict[str, int],
               expected: dict[str, Fraction]) -> ChiSquare:
    """Goodness of fit of counted offspring against exact ratios.

    ``observed`` maps phenotype label to a count; ``expected`` maps
    the same labels to exact proportions. Classes with zero expected
    proportion must have zero observed, or the fit refuses by name.
    """
    total = sum(observed.values())
    if total <= 0:
        raise GeneticsError("bad_ratio", "no offspring counted")
    labels = sorted(set(observed) | set(expected))
    statistic = 0.0
    for label in labels:
        obs = float(observed.get(label, 0))
        exp = float(expected.get(label, Fraction(0))) * total
        if exp <= 0:
            if obs > 0:
                raise GeneticsError(
                    "bad_ratio",
                    f"observed {label} but expected none of them")
            continue
        statistic += (obs - exp) ** 2 / exp
    dof = len([label for label in labels
               if float(expected.get(label, Fraction(0))) > 0]) - 1
    if dof <= 0:
        raise GeneticsError("bad_ratio", "need at least two expected classes")
    p_value = _gammq(dof / 2.0, statistic / 2.0)
    verdict = ("fits the expected ratios — the difference is the kind "
               "chance makes"
               if p_value >= 0.05 else
               "does NOT fit the expected ratios — something beyond "
               "chance is going on; re-check the parents")
    return ChiSquare(statistic=statistic, dof=dof, p_value=p_value,
                     verdict=verdict)


@dataclass
class PredictionScore:
    """How a student's predicted ratios compare to the exact ones."""

    exact: bool
    max_error: float
    mean_error: float
    diffs: dict[str, float]
    verdict: str


def score_prediction(predicted: dict[str, Fraction],
                     expected: dict[str, Fraction]) -> PredictionScore:
    """Score predicted phenotype proportions against the exact ones.

    Proportions need not sum to 1 — they are normalised first. A
    predicted class the cross never produces, or a missing class,
    counts as a full error on that class.
    """
    pred_total = sum(predicted.values())
    exp_total = sum(expected.values())
    if pred_total <= 0 or exp_total <= 0:
        raise GeneticsError("bad_ratio", "empty prediction")
    labels = sorted(set(predicted) | set(expected))
    diffs = {}
    for label in labels:
        p = float(predicted.get(label, Fraction(0))) / float(pred_total)
        e = float(expected.get(label, Fraction(0))) / float(exp_total)
        diffs[label] = p - e
    errors = [abs(v) for v in diffs.values()]
    max_error = max(errors)
    mean_error = sum(errors) / len(errors)
    if max_error == 0:
        verdict = "exact — every ratio right."
    elif max_error <= 0.02:
        verdict = "close — within 2 points on every class."
    elif max_error <= 0.10:
        verdict = "off in places — re-check the classes that moved most."
    else:
        verdict = "off — work the Punnett square again, one trait at a time."
    return PredictionScore(exact=max_error == 0, max_error=max_error,
                           mean_error=mean_error, diffs=diffs,
                           verdict=verdict)


def parse_ratios(text: str) -> dict[str, Fraction]:
    """Parse ``E:1/4 WT:3/4`` / ``E+BW:0.5`` into label -> proportion."""
    out: dict[str, Fraction] = {}
    for token in (text or "").split():
        if ":" not in token:
            raise GeneticsError("bad_ratio", f"expected LABEL:ratio, got '{token}'")
        label, _, value = token.partition(":")
        label = label.strip().upper()
        if not label or not value:
            raise GeneticsError("bad_ratio", f"bad ratio token '{token}'")
        try:
            out[label] = Fraction(value)
        except (ValueError, ZeroDivisionError):
            raise GeneticsError("bad_ratio", f"bad number '{value}'") from None
    if not out:
        raise GeneticsError("bad_ratio", "no ratios given")
    if any(v < 0 for v in out.values()):
        raise GeneticsError("bad_ratio", "ratios cannot be negative")
    return out


# ── lessons ──────────────────────────────────────────────────────────────
# Teaching text is Devon's own. Each lesson carries a worked cross
# (rendered live from the engine, so it can never drift from it) and
# a practice cross the student predicts with `genetics predict`.

LESSONS: tuple[dict[str, Any], ...] = (
    {
        "id": "monohybrid",
        "title": "1 — One trait: dominance is a rule, not a vote",
        "body": (
            "A trait has two copies of its gene, one from each parent. "
            "A dominant allele shows with one copy; a recessive allele "
            "needs two. Ebony body (E) is recessive: E/E flies are ebony, "
            "E/WT flies look wild type but CARRY ebony. Cross two carriers "
            "and the Punnett square gives 1 Ebony : 3 wild type — not "
            "because the flies vote, but because each parent deals each "
            "allele half the time, and only E+E makes an ebony fly."
        ),
        "worked": ("f E/WT", "m E/WT"),
        "practice": ("f E/WT", "m E/E"),
        "hint": "The mother deals E or WT half the time; the father deals E every time.",
    },
    {
        "id": "dihybrid",
        "title": "2 — Two traits: independent assortment",
        "body": (
            "Genes on DIFFERENT chromosomes assort independently — each "
            "pair of alleles is dealt without consulting the other pair. "
            "Lobe eyes (L, chromosome 2, dominant) and Aristapedia "
            "antennae (AR, chromosome 3, dominant) are on different "
            "chromosomes, so a double heterozygote deals four gamete "
            "types equally and the classic 9:3:3:1 falls out: 9 both, 3 "
            "aristapedia only, 3 lobe only, 1 wild type. If the ratio "
            "comes out skewed, the genes are probably NOT independent — "
            "see lesson 4."
        ),
        "worked": ("f AR/WT L/WT", "m AR/WT L/WT"),
        "practice": ("f AR/WT L/WT", "m WT"),
        "hint": "Four gamete types from the mother, one from the father — write all four boxes.",
    },
    {
        "id": "sexlinked",
        "title": "3 — Sex-linked: the X chromosome plays favourites",
        "body": (
            "Males carry ONE X chromosome, so a recessive X allele has "
            "nowhere to hide in a son: White eyes (W, X-linked recessive) "
            "show in every son of a White mother, while daughters get "
            "their father's X and look wild type. This is criss-cross "
            "inheritance — the trait passes from mother to son, father to "
            "daughter — and it is why the sexes of the offspring must be "
            "read separately, never lumped."
        ),
        "worked": ("f W/W", "m WT"),
        "practice": ("f W/WT", "m W"),
        "hint": "Sons get their only X from their mother; daughters get one from each parent.",
    },
    {
        "id": "linkage",
        "title": "4 — Linkage: neighbours travel together",
        "body": (
            "Genes on the SAME chromosome travel together unless a "
            "crossover separates them. Lobe eyes (L, chr 2 at 72.0) and "
            "Ebony body (E, chr 2 at 70.7) sit just 1.3 map units apart, "
            "so barely one gamete in a hundred is recombinant — cross a "
            "double-heterozygous mother to an ebony tester father and "
            "the brood is almost all parental types (Lobe+Ebony and wild "
            "type), with the two recombinant classes (Lobe only, Ebony "
            "only) down near 1% each. And in Drosophila the rule has a "
            "twist: MALES DO NOT RECOMBINE at all, so the same cross "
            "with the sexes swapped gives ZERO recombinants — the "
            "father deals only his two parental haplotypes."
        ),
        "worked": ("f L/WT E/WT", "m E/E"),
        "practice": ("m L/WT E/WT", "f E/E"),
        "hint": "The father deals only his two parental haplotypes — no crossovers in a male fly.",
    },
    {
        "id": "lethal",
        "title": "5 — Lethal alleles: the missing quarter",
        "body": (
            "Some alleles kill when homozygous — Stubble bristles (SB) "
            "is dominant AND lethal. Cross two Stubble carriers and a "
            "quarter of the conceptions never hatch: SB/SB is dead on "
            "arrival. Among the living the ratio is 2 Stubble : 1 wild "
            "type, not 3:1 — the missing quarter is the lesson. A ratio "
            "that looks 'wrong' is often a lethal allele hiding in the "
            "denominator."
        ),
        "worked": ("f SB/WT", "m SB/WT"),
        "practice": ("f SB/WT", "m WT"),
        "hint": "SB/SB never appears — count only the living when you take the ratio.",
    },
    {
        "id": "testcross",
        "title": "6 — Testcross: read the unknown parent",
        "body": (
            "A fly that LOOKS wild type may still carry a recessive "
            "allele. Cross it to a homozygous recessive tester (E/E) and "
            "the offspring confess the parent: if half the brood is "
            "ebony, the parent was a carrier (E/WT); if none is, the "
            "parent was WT/WT. One cross, one question, one clean answer "
            "— this is how every unknown in the lab gets read."
        ),
        "worked": ("f E/WT", "m E/E"),
        "practice": ("f BW/WT", "m BW/BW"),
        "hint": "The tester deals Brown every time — the offspring ratio IS the mother's gamete ratio.",
    },
)


def list_lessons() -> list[dict[str, Any]]:
    """The six lessons, in teaching order."""
    return list(LESSONS)


def lesson(lesson_id: str) -> dict[str, Any]:
    """One lesson by id — else unknown_trait-style refusal by name."""
    key = (lesson_id or "").strip().lower()
    for entry in LESSONS:
        if entry["id"] == key:
            return entry
    known = ", ".join(entry["id"] for entry in LESSONS)
    raise GeneticsError("unknown_trait", f"no lesson '{lesson_id}' — try: {known}")


# ── per-chat session ─────────────────────────────────────────────────────


@dataclass
class GeneticsSession:
    """One chat's genetics bench: cross history + lesson progress."""

    history: list[dict[str, Any]] = field(default_factory=list)
    lessons_done: set[str] = field(default_factory=set)
    last_cross: CrossResult | None = None
    last_practice: tuple[str, str, str] | None = None

    def record_cross(self, mother_spec: str, father_spec: str,
                     result: CrossResult, kind: str = "cross") -> None:
        self.history.append({
            "kind": kind,
            "mother": mother_spec,
            "father": father_spec,
            "combined": {k: str(v) for k, v in result.combined.items()},
            "inviable": str(result.inviable),
        })
        self.last_cross = result


_SESSIONS: dict[str, GeneticsSession] = {}


def get_session(chat_key: str = "") -> GeneticsSession:
    """The bench for one chat (created on first use)."""
    key = chat_key or "default"
    session = _SESSIONS.get(key)
    if session is None:
        session = GeneticsSession()
        _SESSIONS[key] = session
    return session
