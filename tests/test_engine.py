"""Hand-checked values for the deterministic engine. No LLM, no network."""

import numpy as np
import pandas as pd
import pytest

from src.actuarial import capital as cap
from src.actuarial.frequency_severity import fit_frequency, fit_severity
from src.actuarial.ibnr import calculate_ibnr, pct_reported
from src.actuarial.reinsurance import calculate_reinsurance_recovery


def _claims(rows):
    cols = ["claim_id", "accident_year", "reported_amount", "paid_amount", "reserve", "claim_amount"]
    return pd.DataFrame(rows, columns=cols)


def test_pct_reported_step_function():
    assert pct_reported(12) == 0.40
    assert pct_reported(13) == 0.70  # anything above 12 months uses the next step
    assert pct_reported(60) == 1.00
    assert pct_reported(120) == 1.00


def test_ibnr_ldf_by_hand():
    # Valuation 2024. AY2024 is 12 months old -> 40% reported -> LDF 2.5 -> IBNR = 1.5 x reported.
    # AY2020 is 60 months old -> fully reported -> IBNR 0.
    df = _claims([
        ["a", 2024, 1000.0, 400.0, 600.0, 1000.0],
        ["b", 2024, 3000.0, 1000.0, 2000.0, 3000.0],
        ["c", 2020, 5000.0, 5000.0, 0.0, 5000.0],
    ])
    r = calculate_ibnr(df)
    by = {row["accident_year"]: row for row in r["by_accident_year"]}
    assert by[2024]["ldf"] == 2.5
    assert by[2024]["ibnr"] == pytest.approx(1.5 * 4000.0)
    assert by[2020]["ibnr"] == 0.0
    assert r["portfolio_loss"] == 9000.0
    assert r["ibnr"] == pytest.approx(6000.0)
    assert r["ultimate_loss"] == pytest.approx(15000.0)


def test_xol_recovery_straddling_the_layer():
    # retention 100, limit 500: 50 -> 0; 300 -> 200; 1000 -> 500 (exhausts), 400 uncovered above.
    df = pd.DataFrame({"claim_amount": [50.0, 300.0, 1000.0]})
    r = calculate_reinsurance_recovery(df, retention=100, limit=500)
    assert r["expected_recovery"] == 700.0
    assert r["claims_in_layer"] == 2
    assert r["claims_exhausting_layer"] == 1
    assert r["uncovered_above_layer"] == 400.0
    assert r["net_loss"] == 1350.0 - 700.0


def test_frequency_fit_on_known_counts():
    rows = [["x", y, 1.0, 1.0, 0.0, 1.0] for y, n in [(2021, 2), (2022, 4), (2023, 6)] for _ in range(n)]
    f = fit_frequency(_claims(rows))
    assert f["poisson_lambda"] == 4.0
    assert f["variance"] == 4.0  # sample variance of 2,4,6
    assert f["dispersion_ratio"] == 1.0
    assert f["distribution"] == "poisson"


def test_severity_lognormal_fit():
    amounts = [np.e ** 1, np.e ** 2, np.e ** 3]  # log = 1, 2, 3 -> mu 2, sigma 1
    s = fit_severity(pd.DataFrame({"claim_amount": amounts}))
    assert s["lognormal_mu"] == pytest.approx(2.0, abs=1e-4)
    assert s["lognormal_sigma"] == pytest.approx(1.0, abs=1e-4)
    assert s["lognormal_mean"] == pytest.approx(np.exp(2.5), abs=0.01)  # the function rounds to 2 dp
    assert s["tail"] == "moderate"


@pytest.fixture
def small_book():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "claim_id": [f"c{i}" for i in range(n)],
        "accident_year": rng.integers(2019, 2025, n),
        "claim_amount": rng.lognormal(10, 1, n).round(0),
    })


def test_simulation_is_deterministic_and_cached_across_copies(small_book):
    cap._CACHE.clear()
    a = cap.simulate_aggregate(small_book, 50_000, 500_000)
    b = cap.simulate_aggregate(small_book.copy(), 50_000, 500_000)  # a fresh object, same content
    assert a is b, "content-keyed cache should return the same object for a copy"
    cap._CACHE.clear()
    c = cap.simulate_aggregate(small_book, 50_000, 500_000)
    assert c["gross"] == a["gross"] and c["net_of_reinsurance"] == a["net_of_reinsurance"]


def test_public_strips_private_arrays(small_book):
    sim = cap.simulate_aggregate(small_book)
    pub = cap.public(sim)
    assert "_gross_paths" in sim and "_gross_paths" not in pub
    assert pub["gross"]["var_99_5"] >= pub["gross"]["var_99"] >= pub["gross"]["var_95"]
