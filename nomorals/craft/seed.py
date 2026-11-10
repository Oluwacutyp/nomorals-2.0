"""Taxonomy seed data for the CraftGraph (L5) — docs 22+23, P1.

The graph's occupational skeleton starts from data, not from a
hand-maintained structure: the branches below are doc 22 §1's own
example branches, shaped the way the public taxonomies (O*NET,
ESCO/ISCO) shape theirs — domain → occupation → role — and recorded
here as the seed those nodes resolve to.  This file *is* the record
for seed nodes: a seed node cites ``craft/seed`` the way a contract
node cites the delegation store.

The seed is a floor, never the ceiling (the mined-systems
doctrine): it is US/EU-taxonomy-shaped, and Devon's real territory
— Nigerian, chat-native, informal-sector work — is expected to
extend it.  Extension is a data edit here (or a pack declaring its
domain), never a code change in the projection.  Nothing in this
file claims a capability: a seeded occupation is a place in the
ontology, not a promise that a cell can do the work — capability
claims read packs and records, not the seed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SEED_DOMAINS", "seed_domain", "seed_occupation"]

#: Each domain: ``id``, ``title`` and its occupations; each
#: occupation: ``id``, ``title`` and the role names it names.
SEED_DOMAINS: tuple[dict[str, Any], ...] = (
    {
        "id": "hospitality",
        "title": "Hospitality & cafes",
        "occupations": (
            {"id": "menu-engineering", "title": "Menu engineering",
             "roles": ("menu engineer",)},
            {"id": "kitchen-ops", "title": "Kitchen operations",
             "roles": ("kitchen lead", "prep cook")},
            {"id": "supplier-purchasing", "title": "Supplier purchasing",
             "roles": ("purchasing officer",)},
            {"id": "staffing", "title": "Staffing & rota",
             "roles": ("rota planner",)},
            {"id": "hygiene-compliance", "title": "Hygiene compliance",
             "roles": ("hygiene officer",)},
            {"id": "pos-operations", "title": "POS operations",
             "roles": ("pos operator",)},
            {"id": "service-design", "title": "Service design",
             "roles": ("service designer",)},
            {"id": "reviews-marketing", "title": "Reviews & marketing",
             "roles": ("listings steward",)},
        ),
    },
    {
        "id": "mental-health-support",
        "title": "Mental-health support",
        "occupations": (
            {"id": "psychoeducation", "title": "Psychoeducation",
             "roles": ("psychoeducation guide",)},
            {"id": "cbt-skills-practice", "title": "CBT skills practice",
             "roles": ("skills practice coach",)},
            {"id": "mood-activity-tracking",
             "title": "Mood & activity tracking",
             "roles": ("tracking steward",)},
            {"id": "relapse-prevention-planning",
             "title": "Relapse-prevention planning",
             "roles": ("prevention planner",)},
            {"id": "crisis-escalation", "title": "Crisis escalation",
             "roles": ("escalation steward",)},
        ),
    },
    {
        "id": "software",
        "title": "Software",
        "occupations": (
            {"id": "product", "title": "Product & requirements",
             "roles": ("product analyst",)},
            {"id": "architecture", "title": "Architecture",
             "roles": ("systems architect",)},
            {"id": "backend", "title": "Backend engineering",
             "roles": ("backend engineer",)},
            {"id": "frontend", "title": "Frontend engineering",
             "roles": ("frontend engineer",)},
            {"id": "database", "title": "Data & storage",
             "roles": ("database engineer",)},
            {"id": "security", "title": "Security & privacy",
             "roles": ("security engineer",)},
            {"id": "qa", "title": "Quality assurance",
             "roles": ("test engineer",)},
            {"id": "sre", "title": "Site reliability",
             "roles": ("sre",)},
            {"id": "release-support", "title": "Release & support",
             "roles": ("release captain", "support diagnostician")},
        ),
    },
    {
        "id": "visual",
        "title": "Visual & design",
        "occupations": (
            {"id": "photo-retouch", "title": "Photo retouch",
             "roles": ("retoucher",)},
            {"id": "color", "title": "Color",
             "roles": ("colorist",)},
            {"id": "composition", "title": "Composition & layout",
             "roles": ("layout designer",)},
            {"id": "packaging", "title": "Packaging",
             "roles": ("packaging designer",)},
            {"id": "export", "title": "Export & delivery",
             "roles": ("export specialist",)},
        ),
    },
    {
        "id": "video",
        "title": "Video",
        "occupations": (
            {"id": "script", "title": "Script",
             "roles": ("scriptwriter",)},
            {"id": "storyboard", "title": "Storyboard",
             "roles": ("storyboard artist",)},
            {"id": "capture", "title": "Capture",
             "roles": ("camera operator",)},
            {"id": "edit", "title": "Edit",
             "roles": ("editor",)},
            {"id": "motion-graphics", "title": "Motion graphics",
             "roles": ("motion designer",)},
            {"id": "sound", "title": "Sound",
             "roles": ("sound designer",)},
            {"id": "color-grading", "title": "Color grading",
             "roles": ("colorist",)},
            {"id": "captions", "title": "Captions",
             "roles": ("captioner",)},
            {"id": "delivery", "title": "Delivery",
             "roles": ("delivery steward",)},
        ),
    },
    {
        "id": "commerce-finance",
        "title": "Commerce & finance",
        "occupations": (
            {"id": "sourcing", "title": "Sourcing",
             "roles": ("sourcing agent",)},
            {"id": "negotiation", "title": "Negotiation",
             "roles": ("negotiator",)},
            {"id": "catalog", "title": "Catalog",
             "roles": ("catalog steward",)},
            {"id": "transaction", "title": "Transactions",
             "roles": ("payments operator",)},
            {"id": "compliance", "title": "Compliance",
             "roles": ("compliance checker",)},
            {"id": "fulfillment", "title": "Fulfillment",
             "roles": ("fulfillment coordinator",)},
        ),
    },
)


def seed_domain(domain_id: str) -> dict[str, Any] | None:
    """One seed domain by id, or ``None``."""
    for domain in SEED_DOMAINS:
        if domain["id"] == domain_id:
            return domain
    return None


def seed_occupation(domain_id: str,
                    occupation_id: str) -> dict[str, Any] | None:
    """One seed occupation inside a seed domain, or ``None``."""
    domain = seed_domain(domain_id)
    if domain is None:
        return None
    for occupation in domain["occupations"]:
        if occupation["id"] == occupation_id:
            return occupation
    return None
