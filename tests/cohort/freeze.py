"""Golden snapshot of the deterministic engine for the four sample businesses.

    .venv/Scripts/python -m tests.cohort.freeze           # write tests/cohort/golden/<slug>.json
    .venv/Scripts/python -m tests.cohort.freeze --check   # diff against the goldens, exit 1 on change

Every refactor must leave these byte-identical: the math is the product.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from src.actuarial import capital as cap
from src.actuarial import development as dev
from src.actuarial import exposure as expo
from src.actuarial import tcor as tc
from src.actuarial.frequency_severity import fit_frequency, fit_severity, pure_premium, stress_test
from src.actuarial.ibnr import calculate_ibnr
from src.actuarial.reinsurance import calculate_reinsurance_recovery
from src.data.ingest import DataBundle, bundle_from_paths
from src.data.samples import list_samples, sample_files
from src.ml.predict import get_claims_intelligence
from src.policy import wording

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def engine_outputs(bundle: DataBundle, programme: dict[str, float]) -> dict[str, Any]:
    """The same 19 calls app.py's engine() makes, plus the parameterless track tools."""
    df = bundle.claims
    r, l, c = programme["retention"], programme["limit"], programme["capital_held"]
    out = {
        "ibnr": calculate_ibnr(df), "rec": calculate_reinsurance_recovery(df, r, l), "intel": get_claims_intelligence(df),
        "freq": fit_frequency(df), "sev": fit_severity(df), "tri": dev.loss_triangle(df), "stress": stress_test(df),
        "sim": cap.public(cap.simulate_aggregate(df, r, l)), "solv": cap.solvency_position(df, c, r, l),
        "adq": cap.capital_adequacy(df, c, r, l), "expo": expo.exposure_movement(df),
        "conc": expo.concentration(df), "emerg": expo.emerging_signals(df), "sweep": tc.retention_sweep(df, l),
        "cmp": tc.compare_structures(df, r, l), "qs": tc.quota_share_vs_xol(df, r, l),
        "devp": dev.development_pattern(df), "adv": dev.adverse_development(df), "leak": dev.claims_leakage(df),
        # parameterless tools the tracks also expose
        "pricing": pure_premium(df), "large_loss": dev.large_loss_indicators(df), "shift": dev.claim_pattern_shift(df),
        "adq_gross": cap.capital_adequacy(df, c), "rrc": cap.retained_risk_cost(df, r, l),
        "sim_gross": cap.public(cap.simulate_aggregate(df)),
        "clauses": wording.extract_clauses(bundle.documents) if bundle.documents else None,
    }
    out["intel"] = {k: v for k, v in out["intel"].items() if k != "anomalous_claim_ids"} | {"anomalous_top5": out["intel"]["anomalous_claim_ids"][:5]}
    return out


def _clean(x: Any) -> Any:
    """JSON-safe, stable: no private keys, NaN -> None, numpy -> python."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def dump(obj: Any) -> str:
    return json.dumps(_clean(obj), sort_keys=True, indent=1, default=str)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    changed = []
    for slug, prof in list_samples().items():
        bundle = bundle_from_paths(sample_files(slug))
        text = dump(engine_outputs(bundle, prof["programme"]))
        path = GOLDEN_DIR / f"{slug}.json"
        if args.check:
            if not path.exists():
                changed.append(f"{slug}: no golden"); continue
            old = path.read_text(encoding="utf-8")
            if old != text:
                # first differing top-level key, for a useful message
                a, b = json.loads(old), json.loads(text)
                diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
                changed.append(f"{slug}: differs in {diff}")
            else:
                print(f"{slug}: OK")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"{slug}: wrote {path.name} ({len(text):,} chars)")
    if changed:
        print("\n".join(changed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
