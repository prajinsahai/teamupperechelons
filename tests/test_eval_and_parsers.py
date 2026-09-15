"""Grounding checker, synthesis parser, router fallback, ingest normalisation. No LLM."""

import json

import pandas as pd

from src.core.router import keyword_route, _parse_route
from src.core.synthesizer import parse_synthesis, strip_scratchpad
from src.data.ingest import build_bundle
from src.eval.grounding import grounding, numbers_in


# ---------------------------------------------------------------- grounding
TOOLS = [{"name": "t", "args": {}, "result": json.dumps({"ibnr": 43414189.96, "rate_pct": 12.3, "leak": 556234.0, "n": 60, "method": "x"})}]


def test_numbers_in_parses_units():
    got = {n["raw"]: n["value"] for n in numbers_in("IBNR is $43.4M, leakage $556K, rate 12.3%, 60 claims, 43,414,189.96")}
    assert got["$43.4M"] == 43_400_000.0
    assert got["$556K"] == 556_000.0
    assert got["12.3%"] == 12.3
    assert got["43,414,189.96"] == 43414189.96


def test_rounded_money_is_grounded_by_exact_tool_value():
    g = grounding("IBNR stands at **$43.4M** with leakage **$556K** and a 12.3% rate across 60 claims.", TOOLS)
    assert g["ungrounded"] == [], g
    assert g["rate"] == 100.0


def test_invented_number_is_flagged():
    g = grounding("IBNR is $43.4M but we also lose $9.9M to fraud.", TOOLS)
    assert g["ungrounded"] == ["$9.9M"]
    assert g["rate"] == 50.0


def test_sentence_final_period_and_negative_sign():
    tools = [{"name": "t", "args": {}, "result": json.dumps({"max": 41000000.0, "deficit": -10039407.84, "exposure": 730965000.0})}]
    g = grounding("Max claim $41,000,000.0. Exposure $730,965,000.0, deficit **-$10,039,407.84**.", tools)
    assert g["ungrounded"] == [], g
    raws = [n["raw"] for n in numbers_in("max $41,000,000.0. next")]
    assert raws == ["$41,000,000.0"]


def test_structural_numbers_are_ignored():
    g = grounding("At the 99.5% level, 1-in-200, over 2018-2024, 3 actions.", TOOLS)
    assert g["checked"] == 0 and g["grounded"]


# ---------------------------------------------------------------- synthesizer parser
GOOD = '{"executive_summary": "ok **$1**", "active_agents_cited": ["Actuary"], "strategic_risk_friction": "None identified", "key_figures": {"x": "1"}}'


def test_parse_clean_fenced_and_scratchpad():
    assert parse_synthesis(GOOD)["_parse"] == "json"
    assert parse_synthesis("```json\n" + GOOD + "\n```")["_parse"] == "json"
    out = parse_synthesis("<scratchpad>\nthinking...\n</scratchpad>\n" + GOOD)
    assert out["_parse"] == "json" and out["executive_summary"] == "ok **$1**"


def test_parse_truncated_recovers_summary():
    trunc = '{"executive_summary": "The answer is **$5M**.\\n\\n**Actions**\\n- do x", "active_agents_cited": ["Actuary", "Risk_Manager"], "strategic_risk_friction": "None identified", "key_fig'
    out = parse_synthesis(trunc)
    assert out["_parse"] == "recovered"
    assert out["executive_summary"].startswith("The answer is **$5M**.")
    assert out["active_agents_cited"] == ["Actuary", "Risk_Manager"]


def test_parse_no_json_falls_back_to_text():
    out = parse_synthesis("Just prose, no braces.")
    assert out["_parse"] == "no-json" and out["executive_summary"] == "Just prose, no braces."


def test_strip_scratchpad():
    assert strip_scratchpad("<scratchpad>a</scratchpad>  body") == "body"


# ---------------------------------------------------------------- router
def test_keyword_router_locks_actuary_and_ranks():
    r = keyword_route("Is our reinsurance retention right and is capital solvent?", has_documents=False)
    assert r.active[0] == "actuary"
    assert {"reinsurance", "capital"} <= set(r.active)
    assert "policy" not in r.active  # no documents loaded


def test_keyword_router_generic_question_defaults_to_risk():
    assert keyword_route("How are we doing?", has_documents=True).active == ["actuary", "risk"]


def test_parse_route_extracts_json_from_prose():
    d = _parse_route('Sure. {"active_nodes": ["Actuary", "Claims_Analyst"], "synthesized_objective": "x"} thanks')
    assert d["active_nodes"] == ["Actuary", "Claims_Analyst"]


# ---------------------------------------------------------------- ingest
def test_ingest_maps_foreign_schema_and_sanitises():
    csv = ("Claim No,Loss Date,LOB,Gross Incurred,Paid,Sum Insured,Risk Grade\n"
           "C1,2023-02-01,Property,50000,30000,3000000,high\n"
           "C2,2024-03-01,Liability,80000,20000,3000000,2\n"
           "C3,2024-04-01,Liability,0,0,3000000,A\n").encode()
    b = build_bundle([("export.csv", csv)])
    df = b.claims
    assert list(df["accident_year"]) == [2023, 2024]  # zero-amount row dropped
    assert sorted(df["risk_class"]) == ["B", "C"]  # high -> C, 2 -> B
    assert df["claim_amount"].tolist() == [50000.0, 80000.0]
    assert df.attrs["content_hash"] and len(df.attrs["content_hash"]) == 16
    assert any("dropped" in n for n in b.notes)
