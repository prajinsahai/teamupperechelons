# bizmax — AI Actuary

**AI-native insurance optimization for corporate risk.** ForgeAI Hackathon · Team Upper Echelons
(Prajin Sahai S — lead, Jyotiraditya Barik, Archit Anand).

Companies spend millions on insurance without knowing whether they are protected. bizmax turns a
company's own claims and exposure data into a continuously updated answer: what the portfolio is
worth, what reserves it needs, how much reinsurance is actually recovering, and what to change.

> The math runs deterministically in code. The AI reasons and explains — it never invents a figure.

## What it does

Two modes. **Auto — bizmax Core**: ask one question; a router picks which specialist agents
are needed (the Actuary always runs), they run **in parallel** with their own deterministic
tools, and a synthesizer merges the reports into one executive summary with evidence charts.
The MAGI-style panel shows the six agents lighting up as they work. **Manual**: run one track
at a time with its own question box, plus a chart-heavy Overview with downloadable data sheets.

Four illustrative businesses ship with the app, each with a distinct risk profile:

| Business | Profile | Built-in story |
|---|---|---|
| **Velocity Fleet & Freight** | ~870 claims/yr, $8K median | 4% claims leakage, 680 overpaid claims, late reporting |
| **Nexus Cloud Sec** | ~20 claims/yr, $41M cyber cat | Layer breached by the top claim; SCR ratio 118% (warning) |
| **Aegis Healthcare** | ~150 claims/yr, 96-month tail | $125M chain-ladder IBNR, 33% litigation, +4%/yr social inflation |
| **TerraFirma Civil** | ~75 claims/yr, dispersion 20 | 2022/2024 weather spikes, SCR ratio 110% (warning) |


Upload the company's data — claims CSV/XLSX, policy and treaty tables, policy wordings as PDF —
or import tables straight from Supabase, then ask any of six specialist tracks. Each track is a
role prompt plus its **own deterministic tools** over the loaded data; the model decides which
tools to call, reads their JSON, and writes the advice. It never computes a figure.

| Track | What it answers | Tools (pure Python) |
|---|---|---|
| **AI Actuary** | Loss forecasting, frequency/severity, reserving, IBNR, pricing, capital, stress tests | Poisson/NB frequency, lognormal severity, LDF IBNR, pure premium, stress scenarios, 1-in-200 capital test |
| **AI Claims Analyst** | Changing claim patterns, adverse development, drivers, large-loss early indicators | Severity trend + Isolation-Forest anomalies, development pattern, adverse-development z-scores, early large claims, per-line shift |
| **AI Reinsurance Manager** | Retentions, attachment points, limits, transfer cost, alternative structures | XoL recovery, Monte Carlo gross vs net, TCoR retention sweep, XoL vs aggregate stop-loss, limit sensitivity |
| **AI Capital Manager** | Capital sufficiency, shortfall probability, cost of retained risk | 99.5% VaR capital test (gross and net), retained-risk economics, loss distribution, stress |
| **AI Risk Manager** | Exposure movement, concentration, emerging threat signals | Exposure by year/line, HHI concentration, red/amber/green emerging signals, anomalies |
| **AI Policy Analyst** | Exclusions, deductibles, limits, waiting periods, coverage gaps | Regex clause extraction, keyword search, gap check vs modeled severity/VaR |

The **Overview** tab is the deterministic dashboard (portfolio loss, IBNR, recovery, net
exposure, severity trend, anomalies). Every track run is one PRISM session carrying the engine's
`computed_*` figures as span metadata, so an evaluator can check the narrative against the math.

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

copy .env.example .env                   # fill in NVIDIA_API_KEY, PRISMTRACE_*, optional SUPABASE_*
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
python -m src.core.run nexus "Is our reinsurance right for a cyber cat?"   # bizmax Core from the CLI
python -m src.tracks.run all             # every track from the CLI, traced, with a grounding check
python -m src.data.supabase_import claims policies   # pull tables into data/imported/
python -m prismtrace.verify              # credential handshake + live-trace doctor
```

### Tests and the replay cohort

```bash
python -m pytest                                   # 48 tests, no API key: engine hand-checks, tool JSON contract, parsers, grounding
python -m tests.cohort.freeze --check              # the deterministic engine must match the golden snapshots byte for byte
python -m tests.cohort.replay --tag baseline --repeat 2   # 16 fixed questions x 4 businesses through the Core, each a PRISM session
python -m tests.cohort.compare baseline fix1       # before/after table: grounding, routing, parse, timeouts, determinism, latency
```

Every sub-agent's spans carry its own `computed_*` truth values, the synthesizer span carries the
union, and the executive summary is checked mechanically against every tool result (unit-aware:
`$43.4M` is grounded by `43414189.96`). Sub-agents run under a 180 s budget; a hung model call
yields a timeout report and synthesis proceeds on what arrived.

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
