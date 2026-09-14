"""Turn whatever the user uploads (CSV / XLSX / PDF, several at once) into a DataBundle.

- Tabular files are classified (claims / policies / treaties / exposure / other) by
  their columns and normalised to the engine's canonical column names.
- PDFs become plain text documents for the Policy Analyst's keyword tools.
No embeddings, no vector store — a table, a text, and a profile.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# canonical claims column -> accepted synonyms (lower-case, punctuation stripped)
CLAIMS_SYNONYMS: dict[str, list[str]] = {
    "claim_id": ["claim_id", "claimid", "claim_no", "claim_number", "claimnumber", "id", "ref"],
    "accident_year": ["accident_year", "acc_year", "loss_year", "year_of_loss", "ay", "occurrence_year", "policy_year", "year"],
    "line_of_business": ["line_of_business", "lob", "line", "class", "coverage", "coverage_type", "product", "peril"],
    "risk_class": ["risk_class", "risk_grade", "rating_class", "risk_band", "grade"],
    "exposure": ["exposure", "sum_insured", "tiv", "insured_value", "exposure_value", "revenue"],
    "previous_claims": ["previous_claims", "prior_claims", "claims_history", "num_prior_claims"],
    "deductible": ["deductible", "excess", "retention_per_claim"],
    "coverage_limit": ["coverage_limit", "limit", "policy_limit", "limit_of_liability", "sum_insured_limit"],
    "development_month": ["development_month", "dev_month", "development", "months_developed", "age_months", "dev"],
    "claim_amount": ["claim_amount", "loss_amount", "gross_loss", "incurred_ultimate", "ultimate", "total_loss", "loss", "amount", "gross_incurred"],
    "reported_amount": ["reported_amount", "incurred", "reported", "incurred_amount", "total_incurred", "case_incurred"],
    "paid_amount": ["paid_amount", "paid", "paid_loss", "paid_to_date", "total_paid"],
    "reserve": ["reserve", "case_reserve", "outstanding", "os_reserve", "reserves", "outstanding_reserve"],
}

REQUIRED_FOR_ENGINE = ["accident_year", "claim_amount"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalise_claims(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Rename columns to canonical names; derive what can be derived. Returns (df, mapping, notes)."""
    cols = {_norm(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    for canon, syns in CLAIMS_SYNONYMS.items():
        if canon in mapping.values():
            continue
        for s in syns:
            if s in cols and cols[s] not in mapping:
                mapping[cols[s]] = canon
                break
    out = df.rename(columns=mapping).copy()
    notes: list[str] = []

    # derive the amounts the engine needs from whatever is present
    if "claim_amount" not in out and "reported_amount" in out:
        out["claim_amount"] = out["reported_amount"]; notes.append("claim_amount := reported_amount")
    if "claim_amount" not in out and {"paid_amount", "reserve"} <= set(out):
        out["claim_amount"] = out["paid_amount"] + out["reserve"]; notes.append("claim_amount := paid + reserve")
    if "reported_amount" not in out and "claim_amount" in out:
        out["reported_amount"] = out["claim_amount"]; notes.append("reported_amount := claim_amount")
    if "paid_amount" not in out and "reported_amount" in out:
        out["paid_amount"] = out["reported_amount"] * 0.6; notes.append("paid_amount := 60% of reported (assumed)")
    if "reserve" not in out and {"reported_amount", "paid_amount"} <= set(out):
        out["reserve"] = (out["reported_amount"] - out["paid_amount"]).clip(lower=0); notes.append("reserve := reported - paid")
    if "accident_year" not in out:
        for c in out.columns:
            if "date" in _norm(c):
                out["accident_year"] = pd.to_datetime(out[c], errors="coerce").dt.year; notes.append(f"accident_year := year({c})"); break
    if "development_month" not in out and "accident_year" in out:
        latest = int(pd.to_numeric(out["accident_year"], errors="coerce").max())
        out["development_month"] = (latest - pd.to_numeric(out["accident_year"], errors="coerce") + 1) * 12
        notes.append("development_month := months since accident year (assumed)")
    if "claim_id" not in out:
        out["claim_id"] = [f"CLM-{i:05d}" for i in range(1, len(out) + 1)]
    if "line_of_business" not in out:
        out["line_of_business"] = "unspecified"
    for c in ("exposure", "deductible", "coverage_limit", "previous_claims"):
        if c not in out:
            default = {"exposure": 1_000_000, "deductible": 0, "coverage_limit": out["claim_amount"].max() if "claim_amount" in out else 1e7, "previous_claims": 0}[c]
            out[c] = default; notes.append(f"{c} := {default} (assumed, column missing)")
    if "risk_class" not in out:
        out["risk_class"] = "B"; notes.append("risk_class := B (assumed)")

    for c in ("accident_year", "claim_amount", "reported_amount", "paid_amount", "reserve", "development_month", "exposure", "deductible", "coverage_limit", "previous_claims"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=[c for c in REQUIRED_FOR_ENGINE if c in out])
    out["accident_year"] = out["accident_year"].astype(int)
    return out, mapping, notes


def classify_table(df: pd.DataFrame) -> str:
    cols = {_norm(c) for c in df.columns}
    score = {
        "claims": len(cols & {"claim_amount", "loss_amount", "paid", "paid_amount", "reserve", "incurred", "claim_id", "claim_no"}),
        "treaties": len(cols & {"retention", "attachment", "attachment_point", "limit", "layer", "reinsurer", "treaty", "premium_rate", "ceded"}),
        "policies": len(cols & {"policy_id", "policy_number", "premium", "sum_insured", "inception", "expiry", "insured"}),
        "exposure": len(cols & {"exposure", "tiv", "location", "region", "site", "revenue", "headcount"}),
    }
    best = max(score, key=score.get)
    return best if score[best] > 0 else "other"


@dataclass
class DataBundle:
    claims: pd.DataFrame | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # name -> df (policies, treaties, ...)
    documents: dict[str, str] = field(default_factory=dict)  # filename -> text
    notes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def profile(self) -> dict[str, Any]:
        """Small, LLM-readable description of what was loaded (no row data)."""
        p: dict[str, Any] = {"sources": self.sources, "normalisation_notes": self.notes}
        if self.claims is not None:
            c = self.claims
            p["claims"] = {
                "rows": int(len(c)),
                "accident_years": [int(c["accident_year"].min()), int(c["accident_year"].max())],
                "lines_of_business": sorted(map(str, c["line_of_business"].unique()))[:12],
                "total_claim_amount": float(c["claim_amount"].sum()),
                "columns": list(map(str, c.columns)),
            }
        for name, t in self.tables.items():
            p[name] = {"rows": int(len(t)), "columns": list(map(str, t.columns))[:20]}
        if self.documents:
            p["documents"] = {k: f"{len(v):,} chars" for k, v in self.documents.items()}
        return p


def read_tabular(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def read_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def build_bundle(files: list[tuple[str, bytes]]) -> DataBundle:
    """files: [(filename, bytes), ...] from st.file_uploader or disk."""
    b = DataBundle()
    claim_frames: list[pd.DataFrame] = []
    for name, data in files:
        lower = name.lower()
        try:
            if lower.endswith(".pdf"):
                b.documents[name] = read_pdf_text(data)
                b.sources.append(f"{name} (pdf)")
            elif lower.endswith((".csv", ".xlsx", ".xls", ".txt")):
                if lower.endswith(".txt"):
                    b.documents[name] = data.decode("utf-8", errors="ignore"); b.sources.append(f"{name} (text)"); continue
                df = read_tabular(name, data)
                kind = classify_table(df)
                if kind == "claims":
                    norm, mapping, notes = normalise_claims(df)
                    claim_frames.append(norm)
                    b.notes += [f"{name}: {n}" for n in notes]
                    b.sources.append(f"{name} (claims, {len(norm):,} rows, mapped {len(mapping)} cols)")
                else:
                    key = kind if kind not in b.tables else f"{kind}_{len(b.tables)}"
                    b.tables[key] = df
                    b.sources.append(f"{name} ({kind}, {len(df):,} rows)")
            else:
                b.notes.append(f"{name}: unsupported type, skipped")
        except Exception as e:  # keep going; report per file
            b.notes.append(f"{name}: failed to read ({e})")
    if claim_frames:
        b.claims = pd.concat(claim_frames, ignore_index=True)
        if len(claim_frames) > 1:
            b.notes.append(f"{len(claim_frames)} claims files concatenated")
    return b


def bundle_from_paths(paths: list[str]) -> DataBundle:
    from pathlib import Path

    return build_bundle([(Path(p).name, Path(p).read_bytes()) for p in paths])
