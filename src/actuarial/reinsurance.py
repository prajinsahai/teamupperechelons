"""Reinsurance recovery under a per-occurrence excess-of-loss (XoL) treaty.

Pure Python / pandas. Deterministic. No ML, no LLM.

Per claim:  recovery = min( max(loss - retention, 0), limit )
The reinsurer pays the slice of each loss above the retention, up to the layer limit.
"""

from typing import TypedDict

import pandas as pd

# Illustrative treaty: $4.5M xs $500K per occurrence.
DEFAULT_RETENTION = 500_000.0
DEFAULT_LIMIT = 4_500_000.0


class RecoveryResult(TypedDict):
    retention: float
    limit: float
    gross_loss: float
    expected_recovery: float
    net_loss: float  # gross - recovery
    claims_in_layer: int  # claims exceeding retention
    claims_exhausting_layer: int  # claims exceeding retention + limit
    largest_gross_claim: float
    uncovered_above_layer: float  # loss above retention + limit, back on the insured
    method: str


def calculate_reinsurance_recovery(
    df: pd.DataFrame,
    retention: float = DEFAULT_RETENTION,
    limit: float = DEFAULT_LIMIT,
    loss_column: str = "claim_amount",
) -> RecoveryResult:
    """Apply a per-occurrence XoL layer to every claim and total the recoveries."""
    if loss_column not in df.columns:
        raise ValueError(f"claims data missing column: {loss_column}")
    if retention < 0 or limit <= 0:
        raise ValueError("retention must be >= 0 and limit > 0")

    loss = df[loss_column].astype(float)
    excess = (loss - retention).clip(lower=0.0)
    recovery = excess.clip(upper=limit)
    above_layer = (excess - limit).clip(lower=0.0)

    gross = float(loss.sum())
    rec = float(recovery.sum())
    return RecoveryResult(
        retention=float(retention),
        limit=float(limit),
        gross_loss=round(gross, 2),
        expected_recovery=round(rec, 2),
        net_loss=round(gross - rec, 2),
        claims_in_layer=int((loss > retention).sum()),
        claims_exhausting_layer=int((loss > retention + limit).sum()),
        largest_gross_claim=round(float(loss.max()), 2),
        uncovered_above_layer=round(float(above_layer.sum()), 2),
        method=f"per-occurrence excess of loss, {limit:,.0f} xs {retention:,.0f}",
    )
