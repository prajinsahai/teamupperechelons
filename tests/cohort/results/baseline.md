# cohort `baseline`

| metric | value |
|---|---|
| Runs (cases x repeats) | 16 |
| Grounding rate (mean) | 98.3% |
| Runs fully grounded | 69% |
| Ungrounded figures (total) | 12 |
| Routing: expected agents present | 100% |
| Routing: mean Jaccard | 0.59 |
| Routing via LLM | 100% |
| Synthesis parsed as JSON | 100% |
| Agent runs OK | 96% of 48 |
| Runs with a timeout | 12% |
| Determinism (computed_* == golden) | 100% |
| Wall time p50 / p90 | 110s / 343s |

| case | routed | grounding | ungrounded | parse | wall |
|---|---|---|---|---|---|
| vel_full | actuary, capital, risk | 100.0% | — | json | 72.2s |
| vel_reserves_capital | actuary, capital, claims | 100.0% | — | json | 110.1s |
| vel_reinsurance | actuary, capital, reinsurance | 92.5% | $67,530, $93,026, $8,791 | json | 80.8s |
| vel_leakage | actuary, claims | 100.0% | — | json | 74.2s |
| nex_full | actuary, capital, risk | 93.0% | $41,000, $730,965, $41,000 | json | 78.5s |
| nex_reserves_capital | actuary, capital, claims | 100.0% | — | json | 128.7s |
| nex_reinsurance | actuary, reinsurance, risk | 100.0% | — | json | 82.8s |
| nex_cat_policy | actuary, capital, policy, risk | 100.0% | — | json | 220.3s |
| aeg_full | actuary, capital, risk | 100.0% | — | json | 65.7s |
| aeg_reserves_capital | actuary, capital, claims | 100.0% | — | json | 136.0s |
| aeg_reinsurance | actuary, claims, reinsurance | 96.9% | $48,018 | json | 85.6s |
| aeg_deteriorating | actuary, claims, risk | 100.0% | — | json | 137.3s |
| ter_full | actuary, capital, risk | 100.0% | — | json | 85.4s |
| ter_reserves_capital | actuary, capital, risk | 93.8% | $10,039,407.84, $10,039,407.84 | json | 267.0s |
| ter_reinsurance | actuary, reinsurance, risk | 96.2% | $1,997, $122,176 | json | 342.9s |
| ter_deteriorating | actuary, claims, risk | 100.0% | — | json | 416.9s |
