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
    "ANSWER_STYLES",
    "Answer",
    "answer_legal_question",
    "format_answer",
    "format_answer_sms",
    "chunk_for_sms",
    "bill_answer",
    "control_legal",
    "corpus_search",
    "triage",
    "refer_lawyer",
    "letter_kinds",
    "render_letter",
]

#: Supported answer languages. Yoruba is the flagship per owner directive;
#: Pidgin carries most consumer traffic; Hausa and Igbo complete coverage.
LANGUAGES = ("en", "pcm", "yo", "ha", "ig")

#: Answer output styles.
ANSWER_STYLES = ("full", "brief")
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
        id="debt_harassment",
        patterns=(
            r"\bdebt\b", r"loan.{0,30}(harass|threat|shame|embarrass)",
            r"arrest.{0,20}debt", r"owing.{0,20}(threat|arrest|police)",
            r"debt.{0,20}collector", r"shame.{0,20}(debt|loan|owe)",
            r"threat.{0,20}(debt|loan|owe|owing)",
        ),
        corpus_docs=("constitution_rights.md",),
        escalate=True,
        core={
            "en": ("Debt is a civil matter — the police cannot arrest you "
                   "for owing money. A lender's remedy is to sue in court, "
                   "not to send the police or shame you publicly. Keep "
                   "records of every payment and every threat; the threats "
                   "and public shaming can themselves be reported."),
            "pcm": ("Debt matter na civil case — police no fit arrest you "
                    "because you dey owe money. Wetin lender fit do na to "
                    "carry you go court, no be to send police or dey shame "
                    "you for public. Keep record of every payment and every "
                    "threat; you fit report di threat dem sef."),
            "yo": ("Ọ̀rọ̀ gbèsè jẹ́ ọ̀rọ̀ ìlú (civil) — ọlọ́pàá kò lè mú ọ "
                   "nítorí pé o jẹ gbèsè. Ohun tí olùfún-gbèsè lè ṣe ni láti "
                   "gbé ọ lọ sí ilé-ẹjọ́, kì í ṣe láti rán ọlọ́pàá sí ọ "
                   "tàbí láti tì ọ lójú ní gbangba. Pa àkọsílẹ̀ gbogbo "
                   "ìsanwó àti gbogbo ìdẹ̀rùbà mọ́; a lè ròyìn àwọn "
                   "ìdẹ̀rùbà náà fúnra wọn."),
            "ha": ("Bashi lamari ne na farar hula — 'yan sanda ba za su iya "
                   "kama ka saboda bashi ba. Hanyar mai ba da bashi ita ce "
                   "ta kai ka kotu, ba ta aiko 'yan sanda ko ta kunyata ka a "
                   "bainar jama'a ba. Ka riƙe bayanan kowane biya da kowace "
                   "barazana; ana iya kai ƙorafi kan barazanar ma."),
            "ig": ("Ụgwọ bụ okwu obodo — ndị uwe ojii enweghị ike ijide gị "
                   "maka na ị ji ụgwọ. Ụzọ onye na-ebinye ego bụ ịkpọga gị "
                   "n'ụlọ ikpe, ọ bụghị iziga ndị uwe ojii ma ọ bụ imenye "
                   "gị ihere n'ihu ọha. Debe ndekọ nke ụgwọ ọ bụla na iyi "
                   "ọ bụla; ị nwere ike ịkọ iyi ndị ahụ n'onwe ha."),
        },
    ),
    _Scenario(
        id="bail_money",
        patterns=(
            r"bail.*(money|pay|fee|₦|naira|cash)", r"pay.*bail",
            r"(money|payment|fee|₦|cash).{0,40}bail",
            r"demand.{0,30}(money|payment).{0,30}bail",
            r"bail.*(demanded|demanding|asked|asking|collect)",
            r"police.*(money|pay|settle).*(releas|bail|free)",
            r"(releas|free).*(money|pay|settle).*police",
        ),
        corpus_docs=("constitution_rights.md",),
        escalate=True,
        core={
            "en": ("Bail is free under Nigerian law — no officer may demand "
                   "money before releasing a person on bail. An officer who "
                   "demands payment for bail is extorting you. Note the "
                   "officer's name, station, and time, and report it to the "
                   "Police Complaint Response Unit."),
            "pcm": ("Bail na free under Naija law — no officer get right to "
                    "demand money before dem release person for bail. Officer "
                    "wey dey demand money for bail na extortion e dey do. "
                    "Write di officer name, station and time, report am to "
                    "Police Complaint Response Unit."),
            "yo": ("Ìdásílẹ̀ lórí ìdáwọ́dú (bail) jẹ́ ọ̀fẹ́ lábẹ́ òfin "
                   "Nàìjíríà — kò sí ọlọ́pàá tó lè béèrè owó kí wọ́n tó dá "
                   "ènìyàn sílẹ̀. Ọlọ́pàá tó bá ń béèrè owó fún bail, "
                   "ìkó-owó nípa ìpayà ni ó ń ṣe. Kọ orúkọ ọlọ́pàá náà, "
                   "tẹ́ṣọ̀n rẹ̀ àti àkókò sílẹ̀, kí o sì ròyìn sí Ẹ̀ka Ìdáhùn "
                   "Ẹ̀dùn Ọlọ́pàá."),
            "ha": ("Beli kyauta ne a ƙarƙashin dokar Najeriya — babu wani ɗan "
                   "sanda da zai iya neman kuɗi kafin ya saki mutum a kan "
                   "beli. Ɗan sandan da ya nemi kuɗi don beli, cin zarafi "
                   "ne yake yi. Ka rubuta sunan ɗan sandan, ofishin da "
                   "lokaci, ka kai ƙorafi ga Sashen Amsa Korafe-korafen "
                   "'Yan Sanda."),
            "ig": ("Bail bụ n'efu n'okpuru iwu Naịjirịa — ọ dịghị onye uwe "
                   "ojii nwere ike ịchọ ego tupu ahapụ mmadụ na bail. Onye "
                   "uwe ojii na-achọ ego maka bail na-emegbu mmadụ. Dee aha "
                   "onye uwe ojii ahụ, ọdụ ha na oge, ma kọọ ya na Ngalaba "
                   "Nzaghachi Mkpesa Ndị Uwe Ojii."),
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
    _Scenario(
        id="inheritance_intestate",
        patterns=(
            r"inherit", r"intestate", r"died without.{0,20}(will|testament)",
            r"\bwill\b.{0,20}(shar|propert|estate)", r"next of kin",
            r"shar.{0,20}(propert|estate|land).{0,20}(famil|brother|sister)",
            r"letters of administration",
        ),
        corpus_docs=(),
        escalate=True,
        core={
            "en": ("When someone dies without a will (intestate), the law "
                   "decides how the estate is shared — not the loudest "
                   "family member. In Lagos and most southern states the "
                   "Administration of Estates Law applies; spouses and "
                   "children have priority claims. Get the death certificate "
                   "and letters of administration before anyone touches the "
                   "property."),
            "pcm": ("If person die without will, na law go decide how dem "
                    "go take share di property — no be di family member wey "
                    "shout pass. For Lagos and most southern states, "
                    "Administration of Estates Law dey work; wife/husband "
                    "and children get first right. Make una get death "
                    "certificate and letters of administration before anybody "
                    "touch di property."),
            "yo": ("Tí ẹnìkan bá kú láìsí ìwé-ìfẹ́-ìní (will), òfin ni yó "
                   "pinnu bí a ṣe níí pín dúkìá — kì í ṣe ẹni tó bá pariwo "
                   "jùlọ nínú ìdílé. Ní Èkó àti ọ̀pọ̀lọpọ̀ àwọn ìpínlẹ̀ gúúsù, "
                   "Òfin Ìṣàkóso Dúkìá ni ó ń ṣiṣẹ́; aya/ọkọ àti ọmọ ní ẹ̀tọ́ "
                   "àkọ́kọ́. Ẹ gba ìwé-ẹ̀rí ikú àti lẹ́tà ìṣàkóso kí ẹnikẹ́ni "
                   "tó fọwọ́ kan dúkìá náà."),
            "ha": ("Idan mutum ya mutu ba tare da wasiyya ba, doka ce za ta "
                   "yanke yadda za a raba dukiya — ba wanda ya fi ƙarfi a "
                   "iyali ba. A Legas da yawancin jihohin kudu, Dokar "
                   "Gudanar da Gado ce ke aiki; mata/miji da 'ya'ya suna da "
                   "fifiko. A samu takardar shaidar mutuwa da wasiƙar "
                   "gudanarwa kafin wani ya taɓa dukiyar."),
            "ig": ("Mgbe mmadụ nwụrụ n'enweghị akwụkwọ nketa, iwu ga-ekpebi "
                   "otu a ga-esi kee ala — ọ bụghị onye kacha mkpu "
                   "n'ezinụlọ. Na Legos na ọtụtụ steeti ndịda, Iwu Nchịkwa "
                   "Ala na-arụ ọrụ; di/nwunye na ụmụaka nwere ikike mbụ. "
                   "Nweta asambodo ọnwụ na akwụkwọ nchịkwa tupu onye ọ bụla "
                   "emetụ ala ahụ aka."),
        },
    ),
    _Scenario(
        id="cac_registration",
        patterns=(
            r"\bcac\b", r"register.{0,20}business", r"business.{0,20}name",
            r"incorporat", r"\bbn\b.{0,10}(number|regist)", r"\brc\b.{0,10}number",
            r"company.{0,20}regist", r"start.{0,20}(business|company)",
        ),
        corpus_docs=(),
        escalate=False,
        core={
            "en": ("You register a business name or company with the "
                   "Corporate Affairs Commission (CAC) — it can be done "
                   "online. A business name is simpler and cheaper; a "
                   "limited company (LTD) separates your personal liability "
                   "from the business. Keep your CAC certificate and tax ID "
                   "(TIN) together — banks ask for both."),
            "pcm": ("Na Corporate Affairs Commission (CAC) dey register "
                    "business name or company — you fit do am online. "
                    "Business name cheap pass and e simple; limited company "
                    "(LTD) separate your personal liability from di "
                    "business. Keep your CAC certificate and tax ID (TIN) "
                    "together — bank go ask for di two."),
            "yo": ("Ilé-iṣẹ́ Ọ̀rọ̀ Ajọṣepọ̀ (CAC) ni ó ń forúkọ sílẹ̀ orúkọ "
                   "iṣẹ́ tàbí ilé-iṣẹ́ — o lè ṣe é lórí ayélujára. Orúkọ iṣẹ́ "
                   "rọrùn ó sì dín owó kù; ilé-iṣẹ́ tó ní ìdáwọ́dú (LTD) yà "
                   "ìdáwọ́dú ara rẹ kúrò lọ́dọ̀ iṣẹ́ náà. Pa ìwé-ẹ̀rí CAC rẹ "
                   "àti nọ́mbà owó-orí (TIN) mọ́ papọ̀ — ilé-ìfowópamọ́ máa ń "
                   "béèrè fún méjèèjì."),
            "ha": ("Hukumar Harkokin Kamfanoni (CAC) ce ke rajistar sunan "
                   "kasuwanci ko kamfani — ana iya yi a yanar gizo. Sunan "
                   "kasuwanci ya fi sauƙi kuma ya fi arha; kamfani mai iyaka "
                   "(LTD) yana raba alhakinka na kanka da kasuwancin. Ka "
                   "riƙe takardar shaidar CAC da lambar haraji (TIN) tare — "
                   "banki zai nemi duka biyun."),
            "ig": ("Corporate Affairs Commission (CAC) bụ ebe a na-edebanye "
                   "aha azụmahịa ma ọ bụ ụlọ ọrụ — ị nwere ike ime ya "
                   "n'ịntanetị. Aha azụmahịa dị mfe ma dị ọnụ ala; ụlọ "
                   "ọrụ nwere oke (LTD) na-ekewa ụgwọ nke onwe gị na "
                   "azụmahịa ahụ. Debe asambodo CAC gị na nọmba ụtụ isi "
                   "(TIN) ọnụ — ụlọ akụ ga-ajụ maka ha abụọ."),
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
        if scenario is not None and scenario.corpus_docs:
            for fname in scenario.corpus_docs:
                title = fname.replace(".md", "").replace("_", " ").title()
                hits = corpus_search(title, limit=1)
                if hits:
                    cites.append(hits[0].get("title") or title)
                else:
                    cites.append(title)
        if not cites:
            # Scenario has no corpus docs (or the index is unavailable):
            # fall back to a query search so citations stay grounded.
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


def format_answer(answer: Answer, style: str = "full") -> str:
    """Render an :class:`Answer` for chat. Never raises.

    ``style`` is "full" (default) or "brief" (core points only).
    """
    try:
        if (style or "full").lower() == "brief":
            short = answer.text
            # keep intro + the first sentence of the core
            parts = short.split("\n\n", 1)
            if len(parts) == 2:
                sents = re.split(r"(?<=[.!?])\s+", parts[1])
                short = parts[0] + "\n\n" + (sents[0] if sents else parts[1])
            brief = Answer(question=answer.question, language=answer.language,
                           text=short[:500], citations=answer.citations[:2],
                           escalate=answer.escalate)
            return brief.render()
        return answer.render()
    except Exception:  # noqa: BLE001
        return DISCLAIMER


def format_answer_sms(answer: Answer) -> list[str]:
    """Render an :class:`Answer` as numbered SMS-sized chunks (low-bandwidth
    fallback — the Ask-Attorney pattern). Never raises."""
    try:
        return chunk_for_sms(answer.render())
    except Exception:  # noqa: BLE001
        return [DISCLAIMER]


# ── triage: clarifying questions (guided pathway, DoNotPay pattern) ───

#: When no scenario matches outright, these questions steer the user to
#: the right one — a guided digital pathway instead of a dead end.
_TRIAGE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "landlord_lockout": (
        "Did your landlord lock you out, remove your things, or cut your light/water?",
        "Did you receive a written notice before this happened?",
    ),
    "wrongful_termination": (
        "Were you sacked or did your appointment end — and was any notice given?",
        "Is any salary or benefit still unpaid?",
    ),
    "defective_goods": (
        "What did you buy, and what is wrong with it?",
        "Have you complained to the seller in writing (with receipt/photos)?",
    ),
    "arrest_detention": (
        "Were you or someone you know arrested — and were you told the reason?",
        "How long has the person been held without seeing a court?",
    ),
    "bail_money": (
        "Is an officer demanding money before releasing someone on bail?",
        "Do you have the officer's name and station?",
    ),
    "inheritance_intestate": (
        "Did the person leave a will?",
        "Who is trying to share or take the property?",
    ),
    "cac_registration": (
        "Do you want a business name or a limited company?",
        "What line of business is it for?",
    ),
    "debt_harassment": (
        "Who is threatening you over the debt — a lender, an agent, or the police?",
        "Do you have records of payments and the threats?",
    ),
}


def triage(question: str) -> dict:
    """Guided triage: match a question to scenarios, with clarifying
    questions when the match is unclear. Never raises.

    Returns {"match": scenario_id|None, "candidates": [...],
    "questions": [...]}.
    """
    try:
        q = (question or "").lower()
        direct = _match_scenario(q)
        if direct is not None:
            return {"match": direct.id, "candidates": [direct.id],
                    "questions": list(_TRIAGE_QUESTIONS.get(direct.id, ()))}
        candidates: list[str] = []
        for sc in _SCENARIOS:
            score = sum(1 for pat in sc.patterns
                        if re.search(pat, q))
            # partial credit: any single keyword hit
            if score == 0:
                words = {w for w in re.findall(r"[a-z]{4,}", q)}
                pats_words = {w for pat in sc.patterns
                              for w in re.findall(r"[a-z]{4,}", pat)}
                if words & pats_words:
                    score = 0.5
            if score > 0:
                candidates.append(sc.id)
        questions: list[str] = []
        for cid in candidates[:3]:
            questions.extend(_TRIAGE_QUESTIONS.get(cid, ()))
        return {"match": None, "candidates": candidates[:3],
                "questions": questions[:4]}
    except Exception:  # noqa: BLE001
        return {"match": None, "candidates": [], "questions": []}


# ── lawyer referrals (LawPadi/MyJustice pattern) ───────────────────────

#: Real institutions only — names + official sites, no invented contacts.
_LEGAL_AID_DIRECTORY: tuple[dict, ...] = (
    {
        "name": "Legal Aid Council of Nigeria (LACON)",
        "site": "https://legalaidcouncil.gov.ng",
        "blurb": ("Federal body mandated to provide free legal aid to "
                  "indigent Nigerians — criminal defence, civil claims, "
                  "and legal advice through state offices."),
    },
    {
        "name": "Nigerian Bar Association (NBA)",
        "site": "https://www.nigerianbar.org.ng",
        "blurb": ("Umbrella body of Nigerian lawyers; branch offices in "
                  "every state run pro-bono and lawyer-referral schemes."),
    },
    {
        "name": "National Human Rights Commission",
        "site": "https://www.nigeria.nhri.org",
        "blurb": ("Takes complaints on rights violations — police abuse, "
                  "unlawful detention, discrimination."),
    },
    {
        "name": "FCCPC (consumer complaints)",
        "site": "https://fccpc.gov.ng",
        "blurb": ("Federal Competition and Consumer Protection Commission "
                  "— escalate unresolved consumer complaints here."),
    },
)


def refer_lawyer(topic: str = "") -> str:
    """Referral list: real Nigerian legal-aid institutions. Information
    only — Devon does not recommend a specific lawyer. Never raises."""
    try:
        lines = ["🏛️ Where to get a real lawyer (free or affordable):"]
        for entry in _LEGAL_AID_DIRECTORY:
            lines.append(f"\n• *{entry['name']}*\n  {entry['site']}\n  {entry['blurb']}")
        if topic:
            lines.append(f"\nTell them your matter concerns: {topic[:80]}")
        lines.append("\n" + DISCLAIMER)
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return DISCLAIMER


# ── letter templates (DoNotPay guided-pathway pattern) ────────────────

#: Fill-in-the-blank document templates. Information templates, not filed
#: documents — the user completes and sends them.
_LETTERS: dict[str, dict] = {
    "fccpc": {
        "title": "Complaint letter to the FCCPC (defective goods / services)",
        "fields": ("name", "address", "seller", "purchase_date",
                   "item", "problem", "relief"),
        "template": (
            "From: {name}\n{address}\n\nDate: {today}\n\n"
            "To: The Executive Vice Chairman\nFederal Competition and "
            "Consumer Protection Commission (FCCPC)\n\n"
            "Dear Sir/Madam,\n\nCOMPLAINT AGAINST {seller}\n\n"
            "On {purchase_date}, I purchased {item} from {seller}. "
            "The problem is as follows: {problem}\n\n"
            "I have complained to the seller without resolution. "
            "I therefore seek the following relief: {relief}\n\n"
            "Attached are my receipt and photographs of the item.\n\n"
            "Yours faithfully,\n{name}\n\n"
            "— Template for information only; confirm details with a lawyer "
            "before sending."
        ),
    },
    "landlord": {
        "title": "Formal notice to a landlord (unlawful eviction / lockout)",
        "fields": ("name", "address", "landlord", "issue", "demand"),
        "template": (
            "From: {name}\n{address}\n\nDate: {today}\n\n"
            "To: {landlord}\n\nDear Sir/Madam,\n\n"
            "RE: UNLAWFUL INTERFERENCE WITH MY TENANCY\n\n"
            "I write regarding the following: {issue}\n\n"
            "Self-help eviction — lockout, removal of belongings, or "
            "disconnection of utilities — is not permitted under the Lagos "
            "Tenancy Law 2011; eviction requires proper notice and a court "
            "order. I therefore demand the following: {demand}\n\n"
            "Take notice that I reserve all my rights in this matter.\n\n"
            "Yours faithfully,\n{name}\n\n"
            "— Template for information only; confirm details with a lawyer "
            "before sending."
        ),
    },
    "salary": {
        "title": "Demand letter for unpaid salary / entitlements",
        "fields": ("name", "address", "employer", "amount", "period"),
        "template": (
            "From: {name}\n{address}\n\nDate: {today}\n\n"
            "To: {employer}\n\nDear Sir/Madam,\n\n"
            "RE: DEMAND FOR UNPAID SALARY/ENTITLEMENTS\n\n"
            "I write to demand payment of {amount}, being my unpaid "
            "salary/entitlements for {period}.\n\n"
            "Under the Labour Act, wages earned must be paid, and disputes "
            "may be taken to the National Industrial Court. Kindly remit "
            "payment within 14 days of this letter.\n\n"
            "Yours faithfully,\n{name}\n\n"
            "— Template for information only; confirm details with a lawyer "
            "before sending."
        ),
    },
}


def letter_kinds() -> list[str]:
    """Available letter template kinds. Never raises."""
    return sorted(_LETTERS)


def render_letter(kind: str, **fields: str) -> str:
    """Render a fill-in-the-blank letter template. Missing fields are
    left as [FIELD] placeholders. Never raises."""
    try:
        import datetime
        spec = _LETTERS.get((kind or "").lower())
        if spec is None:
            return ("Unknown letter kind. Available: "
                    + ", ".join(letter_kinds()) + "\n\n" + DISCLAIMER)
        data = {f: str(fields.get(f, "")).strip() or f"[{f.upper()}]"
                for f in spec["fields"]}
        data["today"] = datetime.date.today().isoformat()
        return (f"📝 {spec['title']}\n\n"
                + spec["template"].format(**data)
                + "\n\n" + DISCLAIMER)
    except Exception:  # noqa: BLE001
        return DISCLAIMER


# ── SMS chunking (Uganda Ask-Attorney pattern: low-bandwidth fallback) ─

def chunk_for_sms(text: str, limit: int = 155) -> list[str]:
    """Split a long answer into numbered SMS-sized chunks.

    For low-bandwidth / SMS fallback delivery. Never raises.
    """
    try:
        words = (text or "").split()
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for w in words:
            add = len(w) + (1 if cur else 0)
            if cur and cur_len + add > limit:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
                add = len(w)
            cur.append(w)
            cur_len += add
        if cur:
            chunks.append(" ".join(cur))
        n = len(chunks)
        if n <= 1:
            return chunks
        return [f"({i + 1}/{n}) {c}" for i, c in enumerate(chunks)]
    except Exception:  # noqa: BLE001
        return [text or ""]


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
            "Example: /legal pcm my landlord don lock me out\n"
            "/legal triage <your question> — guided clarifying questions\n"
            "/legal refer [topic] — real Nigerian legal-aid institutions\n"
            "/legal letters — fill-in-the-blank letter templates (FCCPC complaint, landlord notice, salary demand)\n"
            "/legal brief [language] <question> — short version of the answer\n"
            + DISCLAIMER)


def control_legal(tail: str, context: Any = None, chat: Any = None,
                  sender_id: str = "", sender: str = "") -> str:
    """/legal — consumer legal aid. Owner-only; never raises."""
    try:
        rest = (tail or "").strip()
        if not rest or rest.lower() == "help":
            return _usage()
        low = rest.lower()
        if low == "letters":
            kinds = ", ".join(letter_kinds())
            return (f"📝 Letter templates (fill in the blanks, then have a "
                    f"lawyer confirm before sending): {kinds}.\n"
                    f"Usage: /legal letter <kind> — e.g. /legal letter fccpc\n"
                    + DISCLAIMER)
        if low.startswith("letter "):
            kind = rest[len("letter "):].strip().split()[0]
            return render_letter(kind)
        if low.startswith("refer"):
            topic = rest[len("refer"):].strip()
            return refer_lawyer(topic)
        if low.startswith("triage"):
            q = rest[len("triage"):].strip()
            if len(q) < 4:
                return "Tell me what happened first — /legal triage <your question>."
            t = triage(q)
            if t["match"]:
                answer = answer_legal_question(q)
                return format_answer(answer)
            lines = ["🔍 I need a bit more detail to point you right:"]
            lines += [f"  • {qq}" for qq in t["questions"]]
            if t["candidates"]:
                lines.append("\nPossible topics: " + ", ".join(t["candidates"]))
            return "\n".join(lines) + "\n\n" + DISCLAIMER
        # Optional leading style: "/legal brief pcm <question>"
        style = "full"
        if low.startswith("brief ") or low.startswith("brief\n"):
            style = "brief"
            rest = rest[5:].strip()
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
        return format_answer(answer, style=style)
    except Exception as e:  # noqa: BLE001 — never raise from chat
        return f"Legal aid hit an error ({e}). {DISCLAIMER}"
