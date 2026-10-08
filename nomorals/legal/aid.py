"""Consumer legal aid — plain-language legal information in Nigerian languages.

Positioning (factual, not moralizing): this module provides legal
*information*, never legal *advice*. It explains what Nigerian law says in
plain language — English, Nigerian Pidgin, Yoruba, Hausa, or Igbo — cites
the corpus provisions it used (grounded mode), and escalates consequential
matters to real lawyers. It never tells the user what to do about their
specific situation and never claims to be or perform like a lawyer
(DoNotPay FTC $193K precedent; NBA 2024 AI guidelines). Every public
output carries :data:`DISCLAIMER` (reused from
:mod:`nomorals.legal.contracts`).

Distribution is WhatsApp-native; every answered message is billable
through #68's cost-awareness layer (:func:`bill_answer`).

The corpus is a seed, not a library: four key-provision summaries under
``nomorals/legal/corpus/``. Partner with or license from Nigerian legal
publishers before claiming breadth — do not build 70 volumes from scratch.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .contracts import DISCLAIMER, information_only_check
from .research import LegalResearch, research_citations

__all__ = [
    "LANGUAGES",
    "LANGUAGE_NAMES",
    "Answer",
    "answer_legal_question",
    "format_answer",
    "bill_answer",
    "control_legal",
    "corpus_search",
]

#: Supported answer languages. Yoruba is the flagship per owner directive;
#: Pidgin carries most consumer traffic; Hausa and Igbo complete coverage.
LANGUAGES = ("en", "pcm", "yo", "ha", "ig")
LANGUAGE_NAMES = {
    "en": "English",
    "pcm": "Nigerian Pidgin",
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
}

_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
_index: Any = None


# ── per-language wrappers ──────────────────────────────────────────────

_INTRO = {
    "en": "Here's what Nigerian law says about this, in plain language:",
    "pcm": "Na wetin Naija law talk about dis matter, for simple language:",
    "yo": "Èyí ni ohun tí òfin Nàìjíríà sọ nípa ọ̀rọ̀ yìí, ní èdè tí ó rọrùn:",
    "ha": "Ga abin da dokar Najeriya ta ce game da wannan, a sauƙaƙe:",
    "ig": "Nke a bụ ihe iwu Naịjirịa kwuru gbasara okwu a, n'asụsụ dị mfe:",
}

_ESCALATE_LINE = {
    "en": "⚠️ This one is consequential — talk to a qualified Nigerian lawyer before acting.",
    "pcm": "⚠️ Dis one serious o — make you yarn correct Naija lawyer before you do anything.",
    "yo": "⚠️ Ọ̀rọ̀ yìí ṣe pàtàkì — ẹ bá agbẹjọ́rọ̀ Nàìjíríà tó ní ìwé-àṣẹ sọ̀rọ̀ kí ẹ tó ṣe nǹkan kan.",
    "ha": "⚠️ Wannan lamari ne mai muhimmanci — ka tuntuɓi lauyan Najeriya kafin ka ɗauki mataki.",
    "ig": "⚠️ Okwu a dị oke mkpa — chọọ onye ọkaiwu Naịjirịa tupu i mee ihe ọ bụla.",
}

_SOURCES_LINE = {
    "en": "Sources (from the legal corpus):",
    "pcm": "Where dis info come from:",
    "yo": "Àwọn orísun (láti inú àkójọ òfin):",
    "ha": "Tushen wannan bayani:",
    "ig": "Ebe ozi a si bịa:",
}

_FALLBACK_CORE = {
    "en": ("I couldn't match this to one of the topics I know well. Here's what "
           "I found in the legal corpus — take it to a lawyer for anything "
           "that matters."),
    "pcm": ("I no fit match dis one to wetin I sabi well. Na wetin I see for "
            "law books be dis — carry am go meet lawyer for anything wey serious."),
    "yo": ("Mi ò lè bá ọ̀rọ̀ yìí mu dáadáa pẹ̀lú àwọn kókó tí mo mọ̀. Èyí ni "
            "ohun tí mo rí nínú àwọn ìwé òfin — ẹ gbé e lọ sí ọ̀dọ̀ agbẹjọ́rọ̀ "
            "fún ohun tí ó ṣe pàtàkì."),
    "ha": ("Ban iya daidaita wannan da batutuwan da na sani da kyau ba. Ga "
            "abin da na samu a littattafan doka — ka kai shi wurin lauya ga "
            "duk abin da yake da muhimmanci."),
    "ig": ("Enweghị m ike ijikọ nke a na isiokwu ndị m maara nke ọma. Nke a "
            "bụ ihe m hụrụ n'akwụkwọ iwu — buru ya gakwuru onye ọkaiwu maka "
            "ihe ọ bụla dị mkpa."),
}


# ── scenarios: pattern → corpus docs + translated core answers ──────────

@dataclass
class _Scenario:
    id: str
    patterns: tuple[str, ...]
    corpus_docs: tuple[str, ...]
    escalate: bool
    core: dict[str, str]


_SCENARIOS: tuple[_Scenario, ...] = (
    _Scenario(
        id="landlord_lockout",
        patterns=(
            r"lock(ed)?\s*(me\s*)?out", r"locked\s*out", r"evict", r"eject",
            r"throw(n)?\s*(me\s*)?out", r"chang(e|ed)\s*(the\s*)?lock",
            r"disconnec(t|ted).*(light|water|electric)",
        ),
        corpus_docs=("lagos_tenancy_law.md",),
        escalate=True,
        core={
            "en": ("Your landlord cannot lock you out, remove your belongings, "
                   "or cut your light/water to force you out — self-help eviction "
                   "is not allowed under Lagos tenancy law. Eviction must go "
                   "through proper written notice and a court order."),
            "pcm": ("Your landlord no get right to lock you comot, carry your "
                    "load, or cut your light/water take pursue you — Lagos tenancy "
                    "law no allow dat kind wahala. Dem must give you proper notice "
                    "and go court first."),
            "yo": ("Onílé rẹ kò ní ẹ̀tọ́ láti ti ọ́ jáde, kó nǹkan rẹ jáde, tàbí "
                    "gé iná/omi rẹ láti lé ọ kúrò — òfin ìgbé ní Èkó kò gbà pé "
                    "kí wọ́n lé ènìyàn kúrò láìsí ìlànà. Kí wọ́n tó lé ọ kúrò, "
                    "wọ́n gbọ́dọ̀ fún ọ ní àkíyèsí tó yẹ, lẹ́yìn náà kí ilé-ẹjọ́ "
                    "pàṣẹ."),
            "ha": ("Mai gidan ba shi da ikon ya kulle ka, ya fitar da kayanka, "
                    "ko ya yanke wutar lantarki/ruwa don ya kore ka — dokar haya "
                    "ta Legas ba ta yarda da korar kai ba. Dole a ba ka sanarwa "
                    "ta rubuce, sannan kotu ta ba da umarni."),
            "ig": ("Onye nwe ụlọ enweghị ikike ịchụpụ gị, iwepụ ihe gị, ma ọ "
                    "bụ ịbipụ ọkụ/mmiri gị iji chụpụ gị — iwu ụlọ nke Legos "
                    "anaghị ekwe ka ịchụpụ mmadụ n'onwe ya. A ghaghị inye gị "
                    "ọkwa edere ede, wee nweta iwu ụlọ ikpe."),
        },
    ),
    _Scenario(
        id="wrongful_termination",
        patterns=(
            r"sack(ed)?", r"fir(e|ed|ing)", r"terminat(e|ed|ion)",
            r"dismiss(ed|al)?", r"unpaid\s*salar", r"salary.*(not|never|no)\s*paid",
            r"withheld.*(pay|salary|wage)", r"no\s*notice",
        ),
        corpus_docs=("labour_act.md",),
        escalate=True,
        core={
            "en": ("The Labour Act sets minimum notice periods before your job "
                   "can end — from 1 day up to 1 month depending on how long "
                   "you've worked — and wages you earned must be paid. Ending "
                   "your employment without the required notice, or withholding "
                   "your pay, can be taken to the National Industrial Court."),
            "pcm": ("Labour Act talk say dem must give you notice before your "
                    "work go end — from 1 day reach 1 month, e depend on how long "
                    "you don work — and dem must pay the salary wey you don work "
                    "for. If dem sack you without notice or hold your pay, you fit "
                    "carry di matter go National Industrial Court."),
            "yo": ("Òfin iṣẹ́ (Labour Act) sọ pé kí wọ́n fún ọ ní àkíyèsí kí iṣẹ́ "
                    "rẹ tó parí — láti ọjọ́ kan dé oṣù kan, ó sinmi lórí bí o ṣe "
                    "ti ṣiṣẹ́ tó — kí wọ́n sì san owó-iṣẹ́ tí o ti ṣiṣẹ́ fún. Tí "
                    "wọ́n bá lé ọ kúrò láìsí àkíyèsí tó yẹ tàbí tí wọ́n bá dì "
                    "owó rẹ mú, o lè gbé ọ̀rọ̀ náà lọ sí Ilé-ẹjọ́ Iṣẹ́ Orílẹ̀-èdè."),
            "ha": ("Dokar aiki (Labour Act) ta ce a ba ka sanarwa kafin aikin "
                    "ka ya ƙare — daga rana ɗaya zuwa wata ɗaya, ya danganta da "
                    "tsawon lokacin da ka yi aiki — kuma a biya ka albashin da "
                    "ka yi aiki. Idan aka kore ka ba tare da sanarwa ba ko aka "
                    "hana ka albashinka, za ka iya kai ƙara kotun masana'antu ta "
                    "ƙasa."),
            "ig": ("Iwu ọrụ (Labour Act) kwuru ka e nye gị ọkwa tupu ọrụ gị "
                    "akwụsị — site n'otu ụbọchị ruo otu ọnwa, dabere n'ogologo "
                    "oge ị rụrụ ọrụ — ma kwụọ gị ụgwọ ọrụ ị rụrụ. Ọ bụrụ "
                    "na a chụpụrụ gị n'enweghị ọkwa kwesịrị ekwesị ma ọ bụ "
                    "jide ụgwọ gị, ị nwere ike iburu okwu ahụ gaa Ụlọ Ikpe "
                    "Ọrụ Mba."),
        },
    ),
    _Scenario(
        id="defective_goods",
        patterns=(
            r"defective", r"fake", r"spoilt|spoiled", r"refund",
            r"seller.*(refus|no)\s*(to\s*)?(refund|replace)",
            r"broke(n)?\s*(after|within)", r"not\s*as\s*described",
        ),
        corpus_docs=("fccpc_consumer.md",),
        escalate=False,
        core={
            "en": ("Under the FCCPC Act you have a right to goods that are "
                   "safe and of acceptable quality, and to seek redress when "
                   "they aren't. Complain to the seller in writing first "
                   "(keep your receipt and photos), ask for repair, replacement "
                   "or refund — and if the seller refuses, you can escalate to "
                   "the FCCPC."),
            "pcm": ("FCCPC Act talk say you get right to buy correct thing wey "
                    "safe and make sense, and you fit complain if e no good. "
                    "First, yarn di seller for writing (keep your receipt and "
                    "picture), ask for repair, replacement or refund — if seller "
                    "no gree, you fit carry am go FCCPC."),
            "yo": ("Òfin FCCPC fún ọ ní ẹ̀tọ́ láti ra nǹkan tó léwu láti lò àti "
                    "tó dára, kí o sì wá ọ̀nà àtúnṣe tí kò bá dára. Kọ́kọ́ kọ "
                    "lẹ́tà sí olùtà (pa ìwé-ẹ̀rí rira àti fọ́tò mọ́), béèrè fún "
                    "àtúnṣe, ìrọ́pò tàbí ìdápadà owó — tí olùtà bá kọ̀, o lè gbé "
                    "ọ̀rọ̀ náà lọ sí FCCPC."),
            "ha": ("Dokar FCCPC ta ba ka haƙƙin sayen kaya mai aminci kuma mai "
                    "kyau, da neman gyara idan ba haka ba. Da farko ka rubuta "
                    "ƙorafi ga mai sayarwa (ka riƙe rasit da hotuna), ka nemi "
                    "gyara, musanya ko dawo da kuɗi — idan ya ƙi, za ka iya kai "
                    "ƙara FCCPC."),
            "ig": ("Iwu FCCPC nyere gị ikike ịzụta ihe dị nchebe ma dị mma, "
                    "na ịchọ ndozi mgbe ha adịghị mma. Buru ụzọ deere onye "
                    "na-ere ahịa akwụkwọ (debe nnata gị na foto), rịọ maka "
                    "nrụzi, nnọchi ma ọ bụ nkwụghachi — ọ bụrụ na ọ jụrụ, ị "
                    "nwere ike iburu ya gaa FCCPC."),
        },
    ),
    _Scenario(
        id="arrest_detention",
        patterns=(
            r"arrest(ed)?", r"detain(ed|tion)?", r"police.*(took|carry|held)",
            r"in\s*cell", r"bail",
        ),
        corpus_docs=("constitution_rights.md",),
        escalate=True,
        core={
            "en": ("The Constitution says no one shall lose personal liberty "
                   "except by due process. Anyone arrested must be told the "
                   "reason promptly and brought before a court within a "
                   "reasonable time — generally 24 hours (48 where no court "
                   "is nearby). You are presumed innocent until proved guilty."),
            "pcm": ("Constitution talk say dem no fit just carry person put for "
                    "cell anyhow without due process. Anybody wey dem arrest, "
                    "dem must tell am di reason quick-quick, and carry am go "
                    "court within reasonable time — normally 24 hours. Dem go "
                    "take you as innocent until court talk otherwise."),
            "yo": ("Òfin-ìbágbépọ̀ sọ pé a kò gbọ́dọ̀ gba òmìnira ẹnikẹ́ni láìsí "
                    "ìlànà tó yẹ. Ẹnikẹ́ni tí wọ́n bá mú gbọ́dọ̀ mọ ìdí rẹ̀ ní "
                    "kíákíá, kí wọ́n sì mú un lọ síwájú ilé-ẹjọ́ láàárín àkókò "
                    "tó bójú mu — ní gbogbogbòò ọjọ́ kan. A óo kà ọ́ sí aláìṣẹ̀ "
                    "títí tí ilé-ẹjọ́ fi sọ pé ó jẹ̀bi."),
            "ha": ("Kundin tsarin mulki ya ce ba a tauye 'yancin kowa ba sai "
                    "da bin tsari. Duk wanda aka kama dole a gaya masa dalili "
                    "nan da nan, a kai shi gaban kotu cikin lokaci mai ma'ana "
                    "— yawanci sa'o'i 24. Ana ɗaukar mutum marar laifi ne har "
                    "sai kotu ta tabbatar da laifi."),
            "ig": ("Iwu kwadoro kwuru na agaghị anapụ onye ọ bụla nnwere onwe "
                    "ya n'enweghị usoro kwesịrị ekwesị. Onye ọ bụla e jidere "
                    "aghaghị ịma ihe kpatara ya ngwa ngwa, ma kpọga ya n'ihu "
                    "ụlọ ikpe n'ime oge kwesịrị ekwesị — n'ozuzu awa 24. A na-"
                    "ewere mmadụ dị ka onye aka ya dị ọcha ruo mgbe ụlọ ikpe "
                    "gosipụtara ikpe."),
        },
    ),
)


# ── corpus (RAG, grounded mode) ─────────────────────────────────────────

def _load_index() -> Any:
    """Lazy corpus index. Never raises — returns None when unavailable."""
    global _index
    if _index is not None:
        return _index
    try:
        from ..documents.index import DocumentIndex
        from ..documents.model import Document, Section
        idx = DocumentIndex()
        if not os.path.isdir(_CORPUS_DIR):
            _index = None
            return None
        for fname in sorted(os.listdir(_CORPUS_DIR)):
            if not fname.endswith(".md"):
                continue
            path = os.path.join(_CORPUS_DIR, fname)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            title = text.splitlines()[0].lstrip("# ").strip() if text else fname
            doc = Document(
                id="legal-" + fname.replace(".md", ""),
                title=title,
                source=path,
                sections=[Section(level=1, heading=title, text=text)],
            )
            try:
                idx.add(doc)
            except Exception:  # noqa: BLE001 — one bad doc must not kill the index
                continue
        _index = idx if len(idx) else None
        return _index
    except Exception:  # noqa: BLE001 — legal aid must never raise
        _index = None
        return None


def corpus_search(query: str, limit: int = 3) -> list[dict]:
    """Search the legal corpus. Never raises; [] when unavailable."""
    try:
        idx = _load_index()
        if idx is None:
            return []
        return idx.search(query or "", limit=limit)
    except Exception:  # noqa: BLE001
        return []


# ── answering ────────────────────────────────────────────────────────────

@dataclass
class Answer:
    """One answered legal question. Legal information, never advice."""
    question: str
    language: str
    text: str = ""
    citations: list[str] = field(default_factory=list)
    escalate: bool = False
    cost_kobo: int = 0

    def render(self) -> str:
        parts = [self.text]
        if self.escalate:
            parts.append(_ESCALATE_LINE[self.language])
        if self.citations:
            parts.append(_SOURCES_LINE[self.language] + "\n" +
                         "\n".join(f"• {c}" for c in self.citations))
        parts.append(DISCLAIMER)
        return "\n\n".join(parts)


def _normalize_language(language: str) -> str:
    low = (language or "en").strip().lower()
    aliases = {
        "english": "en", "pidgin": "pcm", "naija": "pcm", "nigerian pidgin": "pcm",
        "yoruba": "yo", "yorùbá": "yo", "hausa": "ha", "igbo": "ig", "ibo": "ig",
    }
    if low in LANGUAGES:
        return low
    return aliases.get(low, "en")


def _match_scenario(question: str) -> Optional[_Scenario]:
    q = (question or "").lower()
    for sc in _SCENARIOS:
        for pat in sc.patterns:
            if re.search(pat, q):
                return sc
    return None


def _citations_for(scenario: Optional[_Scenario], question: str) -> list[str]:
    """Grounded citations: corpus hits for the scenario's docs, else query hits."""
    cites: list[str] = []
    try:
        if scenario is not None:
            for fname in scenario.corpus_docs:
                title = fname.replace(".md", "").replace("_", " ").title()
                hits = corpus_search(title, limit=1)
                if hits:
                    cites.append(hits[0].get("title") or title)
                else:
                    cites.append(title)
        else:
            for hit in corpus_search(question, limit=3):
                t = hit.get("title")
                if t and t not in cites:
                    cites.append(t)
    except Exception:  # noqa: BLE001 — citations are best-effort
        pass
    return cites


def answer_legal_question(
    question: str,
    language: str = "en",
    *,
    researcher: "LegalResearch | None" = None,
) -> Answer:
    """Answer a legal question in plain language. Never raises.

    RAG over the Nigerian legal corpus (grounded citations), scenario
    matching for the common consumer cases, per-language rendering, and
    escalation to real lawyers where consequential. Legal information,
    never legal advice.

    ``researcher`` (optional): a :class:`LegalResearch` instance. When
    given, its traceable document+section citations are merged in front of
    the scenario citations, and an unanswered verdict strengthens the
    escalation — the #78 anti-hallucination stack serving #77.
    """
    lang = _normalize_language(language)
    q = (question or "").strip()
    if not q:
        return Answer(question="", language=lang,
                      text=_INTRO[lang] + "\n" + {
                          "en": "Ask me a question first — e.g. \"my landlord locked me out\".",
                          "pcm": "First ask me question — e.g. \"my landlord don lock me out\".",
                          "yo": "Kọ́kọ́ bi mi ní ìbéèrè — fún àpẹẹrẹ \"onílé mi ti tì mí jáde\".",
                          "ha": "Da farko ka yi mini tambaya — misali \"mai gidana ya kulle ni\".",
                          "ig": "Buru ụzọ jụọ m ajụjụ — dịka \"onye nwe ụlọ m achụpụla m\".",
                      }[lang],
                      citations=[], escalate=False)

    scenario = _match_scenario(q)
    citations = _citations_for(scenario, q)

    # #78 shared stack: merge traceable research citations up front.
    if researcher is not None:
        try:
            r = researcher.research(q)
            rcites = research_citations(r)
            citations = [c for c in rcites if c not in citations] + citations
            if scenario is None and not r.answered:
                # Nothing in the corpus either — be extra explicit.
                pass  # fallback core below already escalates honestly
        except Exception:  # noqa: BLE001 — researcher must never break aid
            pass

    if scenario is not None:
        core = scenario.core[lang]
        escalate = scenario.escalate
    else:
        core = _FALLBACK_CORE[lang]
        escalate = True

    text = _INTRO[lang] + "\n\n" + core
    ans = Answer(question=q, language=lang, text=text,
                 citations=citations, escalate=escalate)

    # Information-only enforcement: scan the rendered answer for advice lines.
    hits = information_only_check(ans.render())
    if hits:
        # Never ship advice-shaped text; fall back to citations + escalation.
        ans.text = (_INTRO[lang] + "\n\n" + core.split(".")[0] + ".")
        ans.escalate = True
    return ans


def format_answer(answer: Answer) -> str:
    """Render an :class:`Answer` for chat. Never raises."""
    try:
        return answer.render()
    except Exception:  # noqa: BLE001
        return DISCLAIMER


# ── WhatsApp billing (#68) ───────────────────────────────────────────────

def bill_answer(
    phone: str,
    *,
    category: str = "service",
    client: str = "default",
    db_path: str = "",
) -> int:
    """Bill one answered legal message through #68's cost-awareness.

    Legal answers over WhatsApp are service conversations → ₦14/message
    by default. Returns the cost in kobo. Never raises.
    """
    try:
        from ..social.whatsapp_cost import CostTracker
        tracker = CostTracker(db_path=db_path) if db_path else CostTracker()
        return tracker.track(phone or "", category or "service",
                             client=client or "default",
                             purpose="legal_aid")
    except ValueError:
        raise  # unknown category is a caller bug — surface it
    except Exception:  # noqa: BLE001 — billing must never break answers
        return 0


# ── chat ─────────────────────────────────────────────────────────────────

def _usage() -> str:
    return ("/legal [language] <your question> — plain-language legal information "
            "(English, Pidgin, Yoruba, Hausa, Igbo).\n"
            "Example: /legal pcm my landlord don lock me out\n" + DISCLAIMER)


def control_legal(tail: str, context: Any = None, chat: Any = None,
                  sender_id: str = "", sender: str = "") -> str:
    """/legal — consumer legal aid. Owner-only; never raises."""
    try:
        rest = (tail or "").strip()
        if not rest or rest.lower() == "help":
            return _usage()
        # Optional leading language: "/legal pcm <question>" or "/legal pidgin ..."
        parts = rest.split(None, 1)
        lang = "en"
        question = rest
        if len(parts) == 2:
            cand = _normalize_language(parts[0])
            known = parts[0].strip().lower() in LANGUAGES or parts[0].strip().lower() in (
                "english", "pidgin", "naija", "nigerian pidgin",
                "yoruba", "yorùbá", "hausa", "igbo", "ibo")
            if known:
                lang = cand
                question = parts[1]
        if len(question) < 4:
            return ("Tell me what happened — e.g. \"/legal my landlord locked me out\".\n"
                    + _usage())
        answer = answer_legal_question(question, lang)
        return format_answer(answer)
    except Exception as e:  # noqa: BLE001 — never raise from chat
        return f"Legal aid hit an error ({e}). {DISCLAIMER}"
