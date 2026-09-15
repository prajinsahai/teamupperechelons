"""Mechanical hallucination check: every number in a narrative must exist in the tool results.

Unit-aware: `$43.4M` in the text is grounded by `43414189.96` in a tool result (the tool value
rounded to the narrative's precision matches), `12.3%` by `12.3`, `$556K` by `556234.0`.
This is the one implementation used by the Core, the manual tracks, the CLI runners and the
replay cohort — so before/after comparisons measure the model, not checker drift.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

# A trailing "." is allowed when it ends a sentence ("$41,000,000.0." must not backtrack to "$41,000").
NUMBER_RE = re.compile(r"(?<![\w.])\$?\s?(\d[\d,]*(?:\.\d+)?)\s?(M|MN|K|B|BN|%|million|thousand|billion)?(?!\d)(?!\.\d)(?![A-Za-z])", re.I)
SUFFIX = {"K": 1e3, "THOUSAND": 1e3, "M": 1e6, "MN": 1e6, "MILLION": 1e6, "B": 1e9, "BN": 1e9, "BILLION": 1e9}
# Structural numbers that carry no claim: bullets, thresholds, percentiles, years.
IGNORE = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 95.0, 99.0, 99.5, 100.0, 120.0, 200.0, 12.0, 24.0, 36.0, 48.0, 60.0} | {float(y) for y in range(2010, 2031)}


def numbers_in(text: str) -> list[dict[str, Any]]:
    """[{raw, value, digits}] for every number-looking token in the text."""
    out = []
    for m in NUMBER_RE.finditer(text or ""):
        raw, suf = m.group(1), (m.group(2) or "").upper()
        digits = raw.replace(",", "")
        try:
            val = float(digits)
        except ValueError:
            continue
        if suf in SUFFIX:
            val *= SUFFIX[suf]
        sig = len(digits.replace(".", "").lstrip("0")) or 1
        out.append({"raw": m.group(0).strip(), "value": val, "sig": sig, "suffix": suf})
    return out


def _leaf_numbers(obj: Any, acc: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)) and not (isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj))):
        acc.add(abs(float(obj))); return  # sign lives outside the "$" in prose ("-$10.0M" vs -10039407.84)
    if isinstance(obj, str):
        for n in numbers_in(obj):
            acc.add(abs(n["value"]))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _leaf_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _leaf_numbers(v, acc)


def numbers_in_tools(tool_records: Iterable[dict[str, Any]], extra_texts: Iterable[str] | None = None) -> set[float]:
    """All numeric leaves in tool results (JSON parsed when possible) plus numbers in extra texts."""
    acc: set[float] = set()
    for rec in tool_records:
        res = rec.get("result", "") if isinstance(rec, dict) else str(rec)
        try:
            _leaf_numbers(json.loads(res), acc)
        except (json.JSONDecodeError, TypeError):
            _leaf_numbers(str(res), acc)
    for t in extra_texts or []:
        _leaf_numbers(t, acc)
    return acc


def _matches(n: dict[str, Any], pool: set[float]) -> bool:
    v = n["value"]
    if v in pool:
        return True
    # rounded forms: the narrative says $43.4M, the tool said 43414189.96
    sig = max(n["sig"], 1)
    for p in pool:
        if p == 0:
            continue
        if abs(p - v) <= 0.5 * 10 ** (math.floor(math.log10(abs(v))) - sig + 1) if v else p == v:
            return True
        # percent given as fraction (0.123 vs 12.3%)
        if n["suffix"] == "%" and abs(p * 100 - v) < 0.05:
            return True
    return False


def grounding(text: str, tool_records: Iterable[dict[str, Any]], extra_texts: Iterable[str] | None = None) -> dict[str, Any]:
    """{numbers, checked, ungrounded, rate, grounded}. `extra_texts` = sub-agent reports (already grounded upstream)."""
    pool = numbers_in_tools(tool_records, extra_texts)
    found = numbers_in(text)
    checked = [n for n in found if n["value"] not in IGNORE]
    missing = [n["raw"] for n in checked if not _matches(n, pool)]
    return {
        "numbers": len(found),
        "checked": len(checked),
        "ungrounded": missing,
        "rate": round((1 - len(missing) / len(checked)) * 100, 1) if checked else 100.0,
        "grounded": not missing,
    }
