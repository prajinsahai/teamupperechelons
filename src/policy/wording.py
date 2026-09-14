"""Policy wording extraction (Policy Analyst). Regex + keyword lookup over PDF text — no embeddings.

The LLM never reads the whole PDF blind; it calls these tools and gets the clauses
with their surrounding text, then reasons over them.
"""

from __future__ import annotations

import re
from typing import Any

MONEY = r"(?:USD|US\$|\$|INR|₹|EUR|€|GBP|£)\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|mn|m|k|thousand|bn|billion))?"

CLAUSE_PATTERNS: dict[str, list[str]] = {
    "exclusions": [r"\bexclu(?:ded|sion|sions)\b", r"\bdoes not cover\b", r"\bnot covered\b", r"\bshall not (?:apply|be liable)\b", r"\bwar\b", r"\bterrorism\b", r"\bnuclear\b", r"\bcyber\b.*\bexclu"],
    "deductibles": [r"\bdeductible", r"\bexcess\b", r"\bretention\b", r"\bself[- ]insured\b"],
    "limits": [r"\blimit of (?:liability|indemnity)\b", r"\baggregate limit\b", r"\bper occurrence\b", r"\beach and every\b", r"\bsub-?limit"],
    "waiting_periods": [r"\bwaiting period\b", r"\b\d+\s?(?:hours|hour|days|day)\b.*\b(?:waiting|before)", r"\btime deductible\b"],
    "conditions": [r"\bcondition precedent\b", r"\bnotif(?:y|ication) within\b", r"\bwarrant(?:y|ies)\b", r"\breasonable precautions\b"],
    "endorsements": [r"\bendorsement\b", r"\brider\b", r"\bextension\b"],
    "territory_period": [r"\bterritor(?:y|ial)\b", r"\bpolicy period\b", r"\bperiod of insurance\b", r"\bjurisdiction\b"],
}


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z0-9(])", text) if len(s.strip()) > 20]


def extract_clauses(documents: dict[str, str], categories: list[str] | None = None, max_per_category: int = 8) -> dict[str, Any]:
    """Find sentences matching each clause category, with any money amounts they mention."""
    cats = categories or list(CLAUSE_PATTERNS)
    out: dict[str, Any] = {"documents": list(documents), "clauses": {}}
    for cat in cats:
        pats = [re.compile(p, re.I) for p in CLAUSE_PATTERNS.get(cat, [cat])]
        hits = []
        for doc, text in documents.items():
            for s in _sentences(text):
                if any(p.search(s) for p in pats):
                    hits.append({"document": doc, "text": s[:400], "amounts": re.findall(MONEY, s, re.I)[:4]})
                    if len(hits) >= max_per_category:
                        break
            if len(hits) >= max_per_category:
                break
        out["clauses"][cat] = hits
    out["counts"] = {c: len(v) for c, v in out["clauses"].items()}
    out["method"] = "regex clause categories over sentence-split PDF text"
    return out


def search_wording(documents: dict[str, str], query: str, max_hits: int = 10) -> dict[str, Any]:
    """Keyword search: every sentence containing all query words (case-insensitive)."""
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
    hits = []
    for doc, text in documents.items():
        for s in _sentences(text):
            if all(w in s.lower() for w in words):
                hits.append({"document": doc, "text": s[:400]})
                if len(hits) >= max_hits:
                    break
    return {"query": query, "hits": hits, "count": len(hits)}


def _to_number(amount: str) -> float | None:
    m = re.search(r"([\d,]+(?:\.\d+)?)\s?(million|mn|m|k|thousand|bn|billion)?", amount, re.I)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    mult = {"million": 1e6, "mn": 1e6, "m": 1e6, "k": 1e3, "thousand": 1e3, "bn": 1e9, "billion": 1e9}.get((m.group(2) or "").lower(), 1)
    return v * mult


def coverage_gaps(documents: dict[str, str], modeled: dict[str, float]) -> dict[str, Any]:
    """Compare stated limits/deductibles/waiting periods with modeled exposure figures.

    modeled: {"severity_p99": ..., "largest_claim": ..., "expected_annual_loss": ..., "var_99_5": ...}
    """
    clauses = extract_clauses(documents, ["limits", "deductibles", "waiting_periods", "exclusions"])
    limit_amounts = [n for h in clauses["clauses"]["limits"] for a in h["amounts"] if (n := _to_number(a))]
    ded_amounts = [n for h in clauses["clauses"]["deductibles"] for a in h["amounts"] if (n := _to_number(a))]
    gaps = []
    stated_limit = max(limit_amounts) if limit_amounts else None
    if stated_limit is not None:
        for key in ("largest_claim", "severity_p99", "var_99_5"):
            if key in modeled and modeled[key] > stated_limit:
                gaps.append({"type": "limit_below_exposure", "detail": f"stated limit {stated_limit:,.0f} < modeled {key} {modeled[key]:,.0f}",
                             "uncovered": round(modeled[key] - stated_limit, 2)})
    stated_ded = min(ded_amounts) if ded_amounts else None
    if stated_ded is not None and "expected_annual_loss" in modeled and stated_ded > modeled["expected_annual_loss"]:
        gaps.append({"type": "deductible_above_expected_loss", "detail": f"deductible {stated_ded:,.0f} exceeds expected annual loss {modeled['expected_annual_loss']:,.0f}: cover rarely responds"})
    if clauses["counts"]["waiting_periods"]:
        gaps.append({"type": "waiting_period_present", "detail": clauses["clauses"]["waiting_periods"][0]["text"][:200]})
    excl_words = ["cyber", "war", "terrorism", "pandemic", "contingent", "supplier", "nuclear", "pollution"]
    excl_text = " ".join(h["text"].lower() for h in clauses["clauses"]["exclusions"])
    hit_excl = [w for w in excl_words if w in excl_text]
    if hit_excl:
        gaps.append({"type": "notable_exclusions", "detail": f"exclusion wording mentions: {', '.join(hit_excl)}"})
    return {
        "stated_limit": stated_limit,
        "stated_deductible": stated_ded,
        "modeled": modeled,
        "gaps": gaps,
        "gap_count": len(gaps),
        "method": "amounts parsed from limit/deductible clauses vs modeled severity and VaR; keyword exclusions",
    }
