# bizmax — AI Actuary

**AI-native insurance optimization for corporate risk.** ForgeAI Hackathon · Team Upper Echelons
(Prajin Sahai S — lead, Jyotiraditya Barik, Archit Anand).

Companies spend millions on insurance without knowing whether they are protected. bizmax turns a
company's own claims and exposure data into a continuously updated answer: what the portfolio is
worth, what reserves it needs, how much reinsurance is actually recovering, and what to change.

> The math runs deterministically in code. The AI reasons and explains — it never invents a figure.

## What it does

A one-page Streamlit dashboard, the "AI Actuary":

| Section | What you see | Where it comes from |
|---|---|---|
| **Portfolio overview** | Portfolio loss · IBNR · Expected reinsurance recovery · Net exposure | Pure-Python actuarial engine (`src/actuarial/`) |
| **Claims intelligence** | ⚠️ Severity trend vs prior years · ⚠️ Anomalous claims detected · RF-predicted vs historical severity, chart, anomaly table | Two scikit-learn models (`src/ml/`) |
| **AI Actuary recommendation** | A concise 3-sentence recommendation on reserve adequacy and reinsurance protection | NVIDIA NIM LLM via a tool call that reads the engine + model outputs |

The LLM is given exactly one tool, `get_actuarial_summary`, and is instructed to quote its figures
verbatim. Every number on screen is computed in Python; the model explains, it does not calculate.
An expander on the page shows the exact JSON the model saw.

## Architecture

```
data/claims.csv ──► Actuarial engine (pure Python)  ──► IBNR, reinsurance recovery, net exposure
                ──► ML models (scikit-learn, joblib) ──► severity prediction, anomaly flags
                                       │
                                       ▼
                   NVIDIA NIM LLM via langchain-openai (one tool, manual loop)
                                       │
                                       ▼
                              Streamlit dashboard (app.py)
```

- **Frontend:** Streamlit, single `app.py`
- **Actuarial math:** pure Python / pandas — loss-development-factor IBNR, per-occurrence excess-of-loss recovery
- **ML (exactly two models):**
  - `RandomForestRegressor` — claim severity from `exposure, risk_class, previous_claims, deductible, coverage_limit`
  - `IsolationForest` — flags the ~5% most anomalous claims from `claim_amount, reported_amount, paid_amount, reserve, development_month`
- **LLM:** one NVIDIA NIM model (`meta/muse-glimmer-30b`, OpenAI-compatible endpoint) through `langchain-openai`, tool calling only
- **No heavy infra:** no LangGraph, no databases, no vector stores, no RAG. Local CSVs + local `.joblib` files.
- **Evaluation:** instrumented for PRISM (Block Convey) — computed figures are attached as trace metadata so an evaluator can mechanically check the narrative against the engine.

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.data.make_sample_claims   # writes data/claims.csv (illustrative, seeded)
python -m src.ml.train_models           # trains both models -> models/*.joblib

copy .env.example .env                   # then fill in NVIDIA_API_KEY and PRISMTRACE_*
streamlit run app.py
```

`train_models.py` prints the hold-out RMSE for the severity model and the number of anomalies
flagged. The dashboard works without an API key; only the recommendation section needs it.

### PRISM tracing

Every recommendation run is traced to PRISM (Block Convey) as one session: the tool call that
reads the engine, the LLM call(s), and the `computed_*` figures as span metadata. Set
`PRISMTRACE_API_KEY`, `PRISMTRACE_PROJECT_ID`, `PRISMTRACE_HOST` in `.env`. To emit a trace
without the UI and check the pipeline:

```bash
python -m src.prism.smoke        # one live session from the real app path
python -m prismtrace.verify      # credential handshake + live-trace doctor
```

## Repository layout

```
app.py                       Streamlit dashboard (entry point)
src/
  actuarial/
    ibnr.py                  calculate_ibnr — LDF method, per accident year
    reinsurance.py           calculate_reinsurance_recovery — XoL layer
  ml/
    features.py              shared feature definitions / encodings
    train_models.py          trains the two models
    predict.py               loads models; get_claims_intelligence, anomaly_table
  llm/
    config.py                model id + prompts (single source of truth)
    agent.py                 make_llm() + one tool + manual tool loop
  data/
    make_sample_claims.py    seeded synthetic claims generator
data/claims.csv              sample portfolio (hypothetical mid-cap manufacturer)
models/                      severity_rf.joblib, anomaly_iforest.joblib
CLAUDE.md                    project rules for AI-assisted development
```

## Disclaimer

All data and figures are **illustrative, modeled estimates** on a synthetic portfolio. Nothing here
is actuarial advice, and the agent never binds cover — a human approval step sits before any decision.
