"""WAEC/JAMB curriculum alignment + course generation (build-map #48).

The Nigeria wedge: tutoring and materials mapped to the actual WAEC/JAMB
curricula — syllabus codes, topic order, and course generation that never
starts from a bare prompt (clarify -> storyboard -> build).

HONESTY NOTE on codes: WAEC publishes syllabi as topic lists, not with a
single canonical numbering. The ``WAEC <Subject> <block>.<n>`` codes here are
a STABLE INTERNAL scheme (grouped to match the official topic structure) so
the tutor can say "this is WAEC Physics 4.1" and the student can find it.
They are not official WAEC section numbers. Topic CONTENT follows the real
WAEC syllabus structure; anything uncertain is marked.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

__all__ = [
    "SyllabusTopic",
    "SYLLABI",
    "SUBJECTS",
    "find_topic",
    "syllabus_code",
    "subject_topics",
    "curriculum_order",
    "CourseScope",
    "SCOPING_QUESTIONS",
    "Lesson",
    "Course",
    "storyboard",
    "build_course",
    "course_tutor",
    "courses_dir",
]

_log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> set[str]:
    toks = set(_TOKEN.findall((text or "").lower()))
    # Crude de-pluralization so "waves" matches the "wave" keyword.
    for t in list(toks):
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            toks.add(t[:-1])
    return toks


@dataclass(frozen=True)
class SyllabusTopic:
    """One syllabus topic. ``code`` is the internal stable scheme
    (see module docstring) — e.g. ``WAEC Physics 4.1``."""
    code: str
    subject: str
    title: str
    keywords: tuple[str, ...] = ()
    jamb: bool = False  # JAMB specifically examines this (beyond WAEC)
    level: str = "both"  # "waec" | "jamb" | "both"


def _t(code: str, subject: str, title: str, keywords: str,
       *, jamb: bool = False, level: str = "both") -> SyllabusTopic:
    return SyllabusTopic(code=code, subject=subject, title=title,
                         keywords=tuple(keywords.split()),
                         jamb=jamb, level=level)


# ── syllabus data (real WAEC topic structure) ────────────────────────────────

PHYSICS = "Physics"
CHEMISTRY = "Chemistry"
BIOLOGY = "Biology"
MATHEMATICS = "Mathematics"
ENGLISH = "English"
ECONOMICS = "Economics"

SUBJECTS = (PHYSICS, CHEMISTRY, BIOLOGY, MATHEMATICS, ENGLISH, ECONOMICS)

SYLLABI: dict[str, list[SyllabusTopic]] = {
    PHYSICS: [
        _t("WAEC Physics 1.1", PHYSICS, "Measurements and Units",
           "measurement unit SI vernier micrometer screw gauge"),
        _t("WAEC Physics 1.2", PHYSICS, "Motion",
           "motion kinematics velocity acceleration displacement projectile graphs"),
        _t("WAEC Physics 1.3", PHYSICS, "Forces and Equilibrium",
           "force friction moment equilibrium vector resultant"),
        _t("WAEC Physics 1.4", PHYSICS, "Work, Energy and Power",
           "work energy power kinetic potential conservation"),
        _t("WAEC Physics 1.5", PHYSICS, "Simple Harmonic Motion",
           "oscillation pendulum spring SHM period frequency"),
        _t("WAEC Physics 1.6", PHYSICS, "Machines",
           "machine lever pulley inclined plane efficiency mechanical advantage"),
        _t("WAEC Physics 2.1", PHYSICS, "Pressure in Fluids",
           "pressure pascal barometer manometer hydraulic"),
        _t("WAEC Physics 2.2", PHYSICS, "Upthrust and Archimedes' Principle",
           "archimedes upthrust floatation density relative density"),
        _t("WAEC Physics 3.1", PHYSICS, "Temperature and Thermometry",
           "temperature thermometer celsius kelvin absolute zero"),
        _t("WAEC Physics 3.2", PHYSICS, "Thermal Expansion",
           "expansion linear areal cubic bimetallic strip"),
        _t("WAEC Physics 3.3", PHYSICS, "Heat and Calorimetry",
           "heat calorimetry specific heat capacity latent heat"),
        _t("WAEC Physics 3.4", PHYSICS, "Change of State",
           "melting boiling evaporation condensation vapour"),
        _t("WAEC Physics 3.5", PHYSICS, "Gas Laws",
           "boyle charles pressure law ideal gas kinetic theory"),
        _t("WAEC Physics 4.1", PHYSICS, "Wave Motion",
           "wave wavelength frequency amplitude transverse longitudinal"),
        _t("WAEC Physics 4.2", PHYSICS, "Sound Waves",
           "sound echo resonance pitch loudness doppler"),
        _t("WAEC Physics 4.3", PHYSICS, "Light: Reflection and Refraction",
           "light reflection refraction mirror lens snell total internal reflection"),
        _t("WAEC Physics 4.4", PHYSICS, "Optical Instruments",
           "camera microscope telescope projector magnifying glass"),
        _t("WAEC Physics 4.5", PHYSICS, "Dispersion of Light",
           "dispersion prism spectrum rainbow colour"),
        _t("WAEC Physics 5.1", PHYSICS, "Electrostatics",
           "charge coulomb electric field induction electroscope"),
        _t("WAEC Physics 5.2", PHYSICS, "Capacitors",
           "capacitor capacitance dielectric series parallel"),
        _t("WAEC Physics 5.3", PHYSICS, "Current Electricity",
           "current ohm resistance emf potential difference circuit kirchhoff"),
        _t("WAEC Physics 5.4", PHYSICS, "Magnetism",
           "magnet magnetic field compass domain"),
        _t("WAEC Physics 5.5", PHYSICS, "Electromagnetic Induction",
           "faraday lenz induction transformer motor generator"),
        _t("WAEC Physics 5.6", PHYSICS, "Alternating Current",
           "alternating current rms peak frequency"),
        _t("WAEC Physics 5.7", PHYSICS, "Electronics",
           "diode transistor rectifier amplifier logic gate semiconductor"),
        _t("WAEC Physics 6.1", PHYSICS, "Atomic Models",
           "atom rutherford bohr energy levels models"),
        _t("WAEC Physics 6.2", PHYSICS, "Photoelectric Effect",
           "photoelectric photon threshold frequency work function planck"),
        _t("WAEC Physics 6.3", PHYSICS, "Nuclear Physics",
           "radioactivity half-life fission fusion nuclear"),
    ],
    CHEMISTRY: [
        _t("WAEC Chemistry 1.1", CHEMISTRY, "Measurement and Particulate Nature",
           "matter particle diffusion measurement"),
        _t("WAEC Chemistry 2.1", CHEMISTRY, "Atomic Structure and Isotopes",
           "atom proton neutron electron isotope electronic configuration"),
        _t("WAEC Chemistry 2.2", CHEMISTRY, "Periodicity",
           "periodic table groups periods trends electronegativity"),
        _t("WAEC Chemistry 2.3", CHEMISTRY, "Chemical Bonding",
           "bond ionic covalent metallic hydrogen bond dative"),
        _t("WAEC Chemistry 3.1", CHEMISTRY, "The Mole Concept",
           "mole avogadro molar mass"),
        _t("WAEC Chemistry 3.2", CHEMISTRY, "Stoichiometry and Formulae",
           "empirical molecular formula percentage composition stoichiometry"),
        _t("WAEC Chemistry 4.1", CHEMISTRY, "Gas Laws",
           "boyle charles avogadro ideal gas molar volume"),
        _t("WAEC Chemistry 4.2", CHEMISTRY, "Kinetic Theory of Gases",
           "kinetic theory graham diffusion"),
        _t("WAEC Chemistry 5.1", CHEMISTRY, "Acids, Bases and Salts",
           "acid base alkali salt neutralisation amphoteric"),
        _t("WAEC Chemistry 5.2", CHEMISTRY, "pH and Indicators",
           "ph indicator litmus methyl orange phenolphthalein"),
        _t("WAEC Chemistry 5.3", CHEMISTRY, "Volumetric Analysis and Titration",
           "titration titre burette pipette standard solution"),
        _t("WAEC Chemistry 6.1", CHEMISTRY, "Rates of Reaction",
           "rate collision theory activation energy catalyst"),
        _t("WAEC Chemistry 6.2", CHEMISTRY, "Chemical Equilibrium",
           "equilibrium le chatelier reversible"),
        _t("WAEC Chemistry 7.1", CHEMISTRY, "Electrolysis",
           "electrolysis faraday cathode anode electrolyte"),
        _t("WAEC Chemistry 7.2", CHEMISTRY, "Electrochemical Cells",
           "cell emf daniell cell electrode potential"),
        _t("WAEC Chemistry 8.1", CHEMISTRY, "Hydrocarbons",
           "alkane alkene alkyne hydrocarbon cracking homologous series"),
        _t("WAEC Chemistry 8.2", CHEMISTRY, "Functional Groups",
           "alkanol alkanoic acid ester soap functional group"),
        _t("WAEC Chemistry 8.3", CHEMISTRY, "Polymers",
           "polymer polymerisation plastic rubber vulcanisation"),
        _t("WAEC Chemistry 9.1", CHEMISTRY, "Air, Water and the Environment",
           "air water pollution haber contact process extraction"),
    ],
    BIOLOGY: [
        _t("WAEC Biology 1.1", BIOLOGY, "Cells",
           "cell microscope organelle osmosis diffusion plasmolysis"),
        _t("WAEC Biology 1.2", BIOLOGY, "Tissues, Organs and Systems",
           "tissue organ system organisation"),
        _t("WAEC Biology 2.1", BIOLOGY, "Plant Nutrition",
           "photosynthesis chlorophyll mineral nutrition"),
        _t("WAEC Biology 2.2", BIOLOGY, "Animal Nutrition",
           "digestion enzyme teeth food test alimentary canal"),
        _t("WAEC Biology 3.1", BIOLOGY, "Transport in Plants",
           "transpiration xylem phloem translocation"),
        _t("WAEC Biology 3.2", BIOLOGY, "Transport in Mammals",
           "heart blood circulation artery vein capillary"),
        _t("WAEC Biology 4.1", BIOLOGY, "Respiration",
           "respiration aerobic anaerobic breathing gaseous exchange"),
        _t("WAEC Biology 4.2", BIOLOGY, "Excretion",
           "excretion kidney nephron urea skin lungs"),
        _t("WAEC Biology 5.1", BIOLOGY, "Nervous Coordination",
           "neurone reflex arc brain spinal cord"),
        _t("WAEC Biology 5.2", BIOLOGY, "Hormonal Coordination",
           "hormone insulin auxin gibberellin endocrine"),
        _t("WAEC Biology 6.1", BIOLOGY, "Reproduction",
           "reproduction sexual asexual pollination fertilisation"),
        _t("WAEC Biology 6.2", BIOLOGY, "Human Reproduction",
           "menstrual cycle pregnancy placenta birth"),
        _t("WAEC Biology 6.3", BIOLOGY, "Growth and Development",
           "growth metamorphosis growth curve"),
        _t("WAEC Biology 7.1", BIOLOGY, "Heredity",
           "gene chromosome mendel monohybrid dihybrid inheritance"),
        _t("WAEC Biology 7.2", BIOLOGY, "Variation and Selection",
           "variation natural selection adaptation"),
        _t("WAEC Biology 8.1", BIOLOGY, "Evolution",
           "evolution darwin lamarck evidence fossil"),
        _t("WAEC Biology 9.1", BIOLOGY, "Ecosystems",
           "ecosystem food chain food web ecological pyramid"),
        _t("WAEC Biology 9.2", BIOLOGY, "Nutrient Cycles",
           "carbon cycle nitrogen cycle water cycle"),
        _t("WAEC Biology 9.3", BIOLOGY, "Conservation",
           "conservation pollution deforestation erosion"),
    ],
    MATHEMATICS: [
        _t("WAEC Maths 1.1", MATHEMATICS, "Fractions, Decimals and Percentages",
           "fraction decimal percentage ratio proportion"),
        _t("WAEC Maths 1.2", MATHEMATICS, "Indices and Logarithms",
           "indices logarithm antilog laws"),
        _t("WAEC Maths 1.3", MATHEMATICS, "Surds",
           "surd rationalise conjugate"),
        _t("WAEC Maths 1.4", MATHEMATICS, "Standard Form",
           "standard form significant figures approximation"),
        _t("WAEC Maths 2.1", MATHEMATICS, "Linear Equations",
           "linear equation simultaneous substitution elimination"),
        _t("WAEC Maths 2.2", MATHEMATICS, "Quadratic Equations",
           "quadratic factorise formula completing square discriminant"),
        _t("WAEC Maths 2.3", MATHEMATICS, "Inequalities",
           "inequality number line range"),
        _t("WAEC Maths 2.4", MATHEMATICS, "Polynomials",
           "polynomial remainder factor theorem"),
        _t("WAEC Maths 2.5", MATHEMATICS, "Variation",
           "variation direct inverse joint partial"),
        _t("WAEC Maths 3.1", MATHEMATICS, "Areas of Plane Figures",
           "area trapezium circle sector segment"),
        _t("WAEC Maths 3.2", MATHEMATICS, "Volumes and Surface Areas",
           "volume prism cylinder cone sphere frustum"),
        _t("WAEC Maths 4.1", MATHEMATICS, "Angles and Polygons",
           "angle triangle polygon parallel lines"),
        _t("WAEC Maths 4.2", MATHEMATICS, "Circle Theorems",
           "circle chord tangent cyclic quadrilateral theorem"),
        _t("WAEC Maths 4.3", MATHEMATICS, "Constructions and Loci",
           "construction locus bisector perpendicular"),
        _t("WAEC Maths 5.1", MATHEMATICS, "Trigonometric Ratios",
           "sine cosine tangent trigonometry"),
        _t("WAEC Maths 5.2", MATHEMATICS, "Sine and Cosine Rules",
           "sine rule cosine rule triangle"),
        _t("WAEC Maths 5.3", MATHEMATICS, "Bearings",
           "bearing elevation depression"),
        _t("WAEC Maths 6.1", MATHEMATICS, "Data Presentation",
           "histogram pie chart bar chart ogive frequency"),
        _t("WAEC Maths 6.2", MATHEMATICS, "Measures of Central Tendency",
           "mean median mode average"),
        _t("WAEC Maths 6.3", MATHEMATICS, "Probability",
           "probability sample space mutually exclusive independent"),
        _t("WAEC Maths 7.1", MATHEMATICS, "Arithmetic Progression",
           "arithmetic progression AP nth term sum"),
        _t("WAEC Maths 7.2", MATHEMATICS, "Geometric Progression",
           "geometric progression GP nth term sum"),
        _t("WAEC Maths 8.1", MATHEMATICS, "Vectors",
           "vector magnitude direction scalar product", jamb=True),
        _t("WAEC Maths 8.2", MATHEMATICS, "Matrices",
           "matrix determinant inverse simultaneous", jamb=True),
        _t("WAEC Maths 9.1", MATHEMATICS, "Differentiation",
           "differentiation derivative gradient maxima minima rate",
           jamb=True),
        _t("WAEC Maths 9.2", MATHEMATICS, "Integration",
           "integration integral area under curve", jamb=True),
    ],
    ENGLISH: [
        _t("WAEC English 1.1", ENGLISH, "Comprehension",
           "comprehension passage inference question"),
        _t("WAEC English 2.1", ENGLISH, "Summary Writing",
           "summary precis concise sentence"),
        _t("WAEC English 3.1", ENGLISH, "Vocabulary",
           "vocabulary synonym antonym register"),
        _t("WAEC English 3.2", ENGLISH, "Idioms and Phrasal Verbs",
           "idiom phrasal verb collocation expression"),
        _t("WAEC English 4.1", ENGLISH, "Grammar and Concord",
           "grammar concord tense article preposition agreement"),
        _t("WAEC English 4.2", ENGLISH, "Sentence Structure",
           "clause phrase sentence types complex compound"),
        _t("WAEC English 5.1", ENGLISH, "Speech Sounds",
           "vowel consonant diphthong phoneme pronunciation", jamb=True),
        _t("WAEC English 5.2", ENGLISH, "Stress and Intonation",
           "stress intonation rhyme rhythm syllable", jamb=True),
        _t("WAEC English 6.1", ENGLISH, "Essay Writing",
           "essay narrative descriptive argumentative expository"),
        _t("WAEC English 6.2", ENGLISH, "Letters and Reports",
           "formal letter informal letter report article speech"),
    ],
    ECONOMICS: [
        _t("WAEC Economics 1.1", ECONOMICS, "Nature and Scope of Economics",
           "economics scarcity choice opportunity cost scale"),
        _t("WAEC Economics 1.2", ECONOMICS, "Scale of Preference",
           "scale of preference wants needs"),
        _t("WAEC Economics 2.1", ECONOMICS, "Demand",
           "demand demand curve determinants law of demand"),
        _t("WAEC Economics 2.2", ECONOMICS, "Supply",
           "supply supply curve determinants law of supply"),
        _t("WAEC Economics 2.3", ECONOMICS, "Equilibrium",
           "equilibrium price shortage surplus"),
        _t("WAEC Economics 3.1", ECONOMICS, "Elasticity",
           "elasticity price elasticity income cross inelastic"),
        _t("WAEC Economics 4.1", ECONOMICS, "Market Structures",
           "perfect competition monopoly oligopoly market"),
        _t("WAEC Economics 4.2", ECONOMICS, "Price Determination",
           "price mechanism price determination"),
        _t("WAEC Economics 5.1", ECONOMICS, "Factors of Production",
           "land labour capital entrepreneur production"),
        _t("WAEC Economics 5.2", ECONOMICS, "Costs and Revenue",
           "cost fixed variable marginal average revenue"),
        _t("WAEC Economics 6.1", ECONOMICS, "Business Organisations",
           "sole proprietor partnership company cooperative enterprise"),
        _t("WAEC Economics 7.1", ECONOMICS, "National Income",
           "GDP GNP national income per capita"),
        _t("WAEC Economics 8.1", ECONOMICS, "Money",
           "money functions inflation deflation"),
        _t("WAEC Economics 8.2", ECONOMICS, "Banking",
           "central bank commercial bank credit creation"),
        _t("WAEC Economics 9.1", ECONOMICS, "Inflation and Unemployment",
           "inflation unemployment causes control stagflation"),
        _t("WAEC Economics 10.1", ECONOMICS, "International Trade",
           "international trade comparative advantage terms of trade"),
        _t("WAEC Economics 10.2", ECONOMICS, "Balance of Payments",
           "balance of payments exchange rate devaluation"),
        _t("WAEC Economics 11.1", ECONOMICS, "Public Finance",
           "taxation budget fiscal policy public debt"),
        _t("WAEC Economics 12.1", ECONOMICS, "Population",
           "population census malthus demographic"),
        _t("WAEC Economics 12.2", ECONOMICS, "Economic Development",
           "development HDI indicators growth"),
    ],
}


# ── lookup ───────────────────────────────────────────────────────────────────

def subject_topics(subject: str) -> list[SyllabusTopic]:
    """Ordered topics for a subject (case-insensitive name)."""
    for name, topics in SYLLABI.items():
        if name.lower() == (subject or "").strip().lower():
            return list(topics)
    return []


def curriculum_order(subject: str) -> list[SyllabusTopic]:
    """The teaching order — the syllabus order itself."""
    return subject_topics(subject)


def find_topic(query: str, subject: str | None = None) -> SyllabusTopic | None:
    """Fuzzy match a topic: 'waves' -> WAEC Physics 4.1. Token-overlap
    scoring over titles + keywords; never raises."""
    try:
        qtok = _tok(query)
        if not qtok:
            return None
        pool: list[SyllabusTopic] = []
        if subject:
            pool = subject_topics(subject)
        else:
            for topics in SYLLABI.values():
                pool.extend(topics)
        best: SyllabusTopic | None = None
        best_score = 0.0
        for topic in pool:
            tt = _tok(topic.title) | set(topic.keywords)
            overlap = len(qtok & tt)
            if not overlap:
                continue
            score = overlap / max(len(qtok), 1)
            # Title-word hits count double.
            title_hits = len(qtok & _tok(topic.title))
            score += 0.5 * title_hits / max(len(qtok), 1)
            if score > best_score:
                best_score = score
                best = topic
        return best
    except Exception:  # noqa: BLE001
        _log.debug("find_topic failed", exc_info=True)
        return None


def syllabus_code(topic: SyllabusTopic | str) -> str:
    """'WAEC Physics 4.1 — Wave Motion'."""
    if isinstance(topic, str):
        found = find_topic(topic)
        if found is None:
            return topic
        topic = found
    jamb_note = " · JAMB" if topic.jamb else ""
    return f"{topic.code} — {topic.title}{jamb_note}"


# ── course generation: clarify → storyboard → build ──────────────────────────
# Never from a bare prompt. The Mindsmith pattern: 2-3 scoping questions
# first, then a storyboard, then the build.

SCOPING_QUESTIONS: list[dict[str, Any]] = [
    {
        "field": "level",
        "question": "Which exam are you preparing for?",
        "options": ["waec", "jamb", "both"],
        "default": "both",
    },
    {
        "field": "depth",
        "question": "How deep should this course go?",
        "options": ["quick (revision in days)", "full (proper course)"],
        "default": "full",
    },
    {
        "field": "start",
        "question": "Where should we start? (syllabus code like '4.1', a topic name, or 'syllabus order')",
        "options": [],
        "default": "syllabus order",
    },
]


@dataclass
class CourseScope:
    topic: str
    level: str = "both"   # waec | jamb | both
    depth: str = "full"   # quick | full
    start: str = "syllabus order"
    weeks: int = 4
    subject: str = ""

    def __post_init__(self) -> None:
        if self.level not in ("waec", "jamb", "both"):
            raise ValueError(f"level must be waec/jamb/both, got {self.level!r}")
        if self.depth not in ("quick", "full"):
            raise ValueError(f"depth must be quick/full, got {self.depth!r}")


@dataclass
class Lesson:
    n: int
    title: str
    syllabus_code: str
    objectives: list[str] = field(default_factory=list)
    explainer: str = ""
    worked_examples: list[str] = field(default_factory=list)
    quiz: list[dict[str, str]] = field(default_factory=list)  # {q, answer}

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "title": self.title,
                "syllabus_code": self.syllabus_code,
                "objectives": self.objectives, "explainer": self.explainer,
                "worked_examples": self.worked_examples, "quiz": self.quiz}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Lesson":
        return cls(n=int(d.get("n", 0)), title=str(d.get("title", "")),
                   syllabus_code=str(d.get("syllabus_code", "")),
                   objectives=list(d.get("objectives") or []),
                   explainer=str(d.get("explainer", "")),
                   worked_examples=list(d.get("worked_examples") or []),
                   quiz=list(d.get("quiz") or []))


@dataclass
class Course:
    id: str
    title: str
    scope: CourseScope
    lessons: list[Lesson] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title,
                "scope": {"topic": self.scope.topic, "level": self.scope.level,
                          "depth": self.scope.depth, "start": self.scope.start,
                          "weeks": self.scope.weeks,
                          "subject": self.scope.subject},
                "lessons": [ls.to_dict() for ls in self.lessons],
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Course":
        s = d.get("scope") or {}
        scope = CourseScope(topic=str(s.get("topic", "")),
                            level=str(s.get("level", "both")),
                            depth=str(s.get("depth", "full")),
                            start=str(s.get("start", "syllabus order")),
                            weeks=int(s.get("weeks", 4)),
                            subject=str(s.get("subject", "")))
        return cls(id=str(d.get("id", uuid.uuid4().hex[:12])),
                   title=str(d.get("title", "")),
                   scope=scope,
                   lessons=[Lesson.from_dict(x) for x in
                            (d.get("lessons") or [])],
                   created_at=float(d.get("created_at", time.time())))

    def save(self, path: str | Path | None = None) -> str:
        p = Path(path) if path else courses_dir() / f"{self.id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, course_id: str) -> "Course | None":
        p = courses_dir() / f"{course_id}.json"
        if not p.is_file():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            _log.debug("course load failed", exc_info=True)
            return None

    def markdown(self) -> str:
        """The full course as one markdown document (for grounded tutoring)."""
        parts = [f"# {self.title}\n"]
        for ls in self.lessons:
            parts.append(f"## Lesson {ls.n}: {ls.title}\n")
            parts.append(f"*{ls.syllabus_code}*\n")
            if ls.objectives:
                parts.append("**Objectives:** " + "; ".join(ls.objectives) + "\n")
            if ls.explainer:
                parts.append(ls.explainer + "\n")
            for ex in ls.worked_examples:
                parts.append(f"**Worked example:** {ex}\n")
        return "\n".join(parts)


def courses_dir() -> Path:
    return Path.home() / ".nomorals" / "learn" / "courses"


def _scope_topics(scope: CourseScope) -> list[SyllabusTopic]:
    """Topics for the course, in teaching order, honoring the scope.

    When the user named a specific topic (not a whole subject), the course
    centers on it: that topic plus the following syllabus topics (12 for a
    full course, 6 for quick revision). Naming a subject builds the whole
    subject course.
    """
    hit = find_topic(scope.topic)
    if hit is not None and not scope.subject:
        scope.subject = hit.subject
    topics = subject_topics(scope.subject) if scope.subject else []
    if not topics:
        return []
    is_subject_query = (scope.topic.strip().lower()
                        == (scope.subject or "").strip().lower())
    centered = False
    if hit is not None and not is_subject_query and hit in topics:
        idx = topics.index(hit)
        cap = 12 if scope.depth == "full" else 6
        topics = topics[idx:idx + cap]
        centered = True
    elif scope.depth == "quick":
        # Revision mode: cap at 6 lessons.
        topics = topics[:6]
    if scope.level == "jamb" and not centered:
        # JAMB-emphasis topics first, rest follow in syllabus order.
        topics = [t for t in topics if t.jamb] + \
                 [t for t in topics if not t.jamb]
    start = (scope.start or "").strip().lower()
    if start and start != "syllabus order" and topics:
        shit = find_topic(scope.start, subject=scope.subject or None)
        if shit is not None and shit in topics:
            topics = topics[topics.index(shit):]
    return topics


def storyboard(scope: CourseScope,
               llm_fn: Callable[[str, str], str] | None = None
               ) -> list[dict[str, Any]]:
    """Outline lessons + quizzes from the syllabus mapping.

    Never from a bare prompt: the storyboard is ALWAYS grounded in syllabus
    topics. Returns lesson skeletons; :func:`build_course` fills them in.
    """
    topics = _scope_topics(scope)
    if not topics:
        # Unknown topic: honest empty storyboard, not invented lessons.
        return []
    boards: list[dict[str, Any]] = []
    if llm_fn is not None:
        try:
            import json as _json
            listing = "\n".join(
                f"- {t.code}: {t.title}" for t in topics[:20])
            raw = llm_fn(
                "You are a WAEC/JAMB curriculum designer. Reply with ONLY JSON.",
                f"Design a {scope.depth} course on '{scope.topic}' "
                f"({scope.level.upper()}) from these syllabus topics:\n{listing}\n\n"
                'Reply with ONLY a JSON array: [{"title": "<lesson title>", '
                '"code": "<syllabus code>", "objectives": ["<3 objectives>"]}]')
            start, end = raw.find("["), raw.rfind("]")
            items = _json.loads(raw[start:end + 1]) if start != -1 and end > start else []
            for i, item in enumerate(items[:24], 1):
                if not isinstance(item, dict):
                    continue
                boards.append({
                    "n": i, "title": str(item.get("title", "")) or f"Lesson {i}",
                    "syllabus_code": str(item.get("code", "")),
                    "objectives": [str(o) for o in
                                   (item.get("objectives") or [])][:5],
                })
            if boards:
                return boards
        except Exception:  # noqa: BLE001 — fall through to template boards
            _log.debug("LLM storyboard failed, using template", exc_info=True)
    # Template storyboard: one lesson per syllabus topic (grounded, honest).
    for i, topic in enumerate(topics[:24], 1):
        boards.append({
            "n": i, "title": topic.title,
            "syllabus_code": syllabus_code(topic),
            "objectives": [
                f"Define the key terms in {topic.title.lower()}",
                f"Apply {topic.title.lower()} to WAEC-style questions",
                f"Link {topic.title.lower()} to related topics",
            ],
        })
    return boards


def _llm_lesson(board: dict[str, Any],
                llm_fn: Callable[[str, str], str]) -> dict[str, Any]:
    """One model call fills a lesson skeleton. Raises on bad output."""
    import json as _json
    raw = llm_fn(
        "You are a WAEC/JAMB teacher. Reply with ONLY JSON.",
        f"Write lesson {board['n']}: '{board['title']}' "
        f"({board['syllabus_code']}). Objectives: "
        f"{'; '.join(board['objectives'])}\n\n"
        'Reply with ONLY a JSON object: {"explainer": "<~300 words, '
        'clear teaching with Nigerian examples where natural>", '
        '"worked_examples": ["<2 worked WAEC-style examples with '
        'solutions>"], "quiz": [{"q": "<question>", '
        '"answer": "<model answer>"}]} (3 quiz questions)')
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("bad lesson JSON")
    data = _json.loads(raw[start:end + 1])
    return {
        "explainer": str(data.get("explainer", ""))[:4000],
        "worked_examples": [str(x) for x in
                            (data.get("worked_examples") or [])][:4],
        "quiz": [{"q": str(q.get("q", "")), "answer": str(q.get("answer", ""))}
                 for q in (data.get("quiz") or [])[:5]
                 if isinstance(q, dict) and q.get("q")],
    }


def build_course(scope: CourseScope,
                 llm_fn: Callable[[str, str], str] | None = None) -> Course:
    """Storyboard -> full course. Each lesson = explainer + worked examples
    + quiz. Quizzes feed the mistake notebook (#46) when the student gets
    them wrong.

    Without a model this returns an HONEST structured skeleton (objectives +
    quiz prompts + a note that full text needs a model) — never fake prose.
    """
    boards = storyboard(scope, llm_fn=llm_fn)
    if not boards:
        raise ValueError(
            f"no syllabus topics matched '{scope.topic}' — try a subject "
            f"like 'physics' or browse with /course waec <subject>")
    lessons: list[Lesson] = []
    for board in boards:
        if llm_fn is not None:
            try:
                filled = _llm_lesson(board, llm_fn)
            except Exception:  # noqa: BLE001 — one bad lesson, honest stub
                _log.debug("lesson %s failed, stubbing", board["n"],
                           exc_info=True)
                filled = _stub_lesson(board)
        else:
            filled = _stub_lesson(board)
        lessons.append(Lesson(
            n=board["n"], title=board["title"],
            syllabus_code=board["syllabus_code"],
            objectives=board["objectives"],
            explainer=filled["explainer"],
            worked_examples=filled["worked_examples"],
            quiz=filled["quiz"]))
    title_bits = [scope.topic.title()]
    if scope.level != "both":
        title_bits.append(scope.level.upper())
    title_bits.append("Quick Revision" if scope.depth == "quick" else "Course")
    return Course(id=uuid.uuid4().hex[:12], title=" ".join(title_bits),
                  scope=scope, lessons=lessons)


def _stub_lesson(board: dict[str, Any]) -> dict[str, Any]:
    """Honest skeleton when no model is available."""
    title = board["title"]
    return {
        "explainer": (
            f"## {title} ({board['syllabus_code']})\n\n"
            "Objectives:\n" +
            "\n".join(f"- {o}" for o in board["objectives"]) +
            "\n\n[Full lesson text needs a language model — connect one "
            "and rebuild this course for complete explainers.]"),
        "worked_examples": [],
        "quiz": [
            {"q": f"Explain the key ideas in {title.lower()}.",
             "answer": "(model answer needs a language model)"},
            {"q": f"Give one WAEC-style worked example from {title.lower()}.",
             "answer": "(model answer needs a language model)"},
        ],
    }


def course_tutor(course: Course, *, mode: str = "socratic",
                 llm_fn: Callable[[str, str], str] | None = None) -> Any:
    """Embedded tutor that answers from the course content only.

    Grounded like tutor_from_files: the course markdown is indexed and every
    turn carries citations back to the lesson it came from.
    """
    from .tutor import TutorSession, TutorError
    from ..documents.index import DocumentIndex
    from ..documents.model import Document, Section

    if not course.lessons:
        raise TutorError("course is empty")
    text = course.markdown()
    if not text.strip():
        raise TutorError("course is empty")
    idx = DocumentIndex()
    doc = Document(title=course.title, source=f"course:{course.id}",
                   sections=[Section(heading="", text=text)])
    try:
        idx.add(doc)
    except Exception as exc:  # noqa: BLE001
        raise TutorError(f"could not index course: {exc}") from exc
    citations: list[str] = []
    context_bits: list[str] = []
    try:
        hits = idx.search(course.scope.topic or course.title, limit=3)
    except Exception as exc:  # noqa: BLE001
        raise TutorError(f"course search failed: {exc}") from exc
    for hit in hits:
        src = str(hit.get("doc_id", ""))
        snippet = str(hit.get("snippet", ""))[:300]
        citations.append(f"course:{course.id} ({src})")
        if snippet:
            context_bits.append(f"[{src}] {snippet}")
    codes = "; ".join(ls.syllabus_code for ls in course.lessons[:4])
    session = TutorSession(
        topic=f"{course.title} [{codes}]", mode=mode, llm_fn=llm_fn,
        source=f"course:{course.id}", citations=citations)
    if llm_fn is None and context_bits:
        session._current_question = (
            f"From your course '{course.title}': {context_bits[0][:200]}… "
            f"What does this tell you?")
    return session


# ── chat scoping state ───────────────────────────────────────────────────────

_PENDING_SCOPES: dict[str, CourseScope] = {}


def pending_scope(chat_key: str) -> CourseScope | None:
    return _PENDING_SCOPES.get(chat_key)


def set_pending_scope(chat_key: str, scope: CourseScope) -> None:
    _PENDING_SCOPES[chat_key] = scope


def clear_pending_scope(chat_key: str) -> None:
    _PENDING_SCOPES.pop(chat_key, None)


def scoping_prompt(scope: CourseScope) -> str:
    lines = [f"📚 course: **{scope.topic}** — 3 quick questions first:"]
    for i, q in enumerate(SCOPING_QUESTIONS, 1):
        opts = f" ({'/'.join(q['options'])})" if q["options"] else ""
        lines.append(f"{i}. {q['question']}{opts}")
    lines.append("answer with: /course set <field> <value> — "
                 "e.g. /course set level jamb")
    lines.append("then: /course build")
    return "\n".join(lines)
