# bizmax — AI Actuary (ForgeAI Hackathon, 4-hour build)

An AI agent that keeps a company's insurance programme matched to its measured risk.
Input: the company's own claims + exposure data (CSV). Output: an auditable recommendation
(retention, coverage to cut/add, confidence interval, assumptions, Total Cost of Risk).
Every run is traced and evaluated with PRISM (Block Convey).

Team: Prajin Sahai S (lead), Jyotiraditya Barik, Archit Anand.

**This is a 4-hour hackathon.** Ship the demo path. Anything not on it is out of scope.

---

## 1. Architecture — hybrid data + AI

| Layer | Technology | Rule |
|---|---|---|
| Frontend | **Streamlit** | Single `app.py`. Upload CSV → run → show recommendation + charts. |
| ML | **Scikit-learn** | Exactly two models (see §4). Saved/loaded as `.joblib`. |
| Actuarial math | **Pure Python** (`numpy`/`pandas` allowed) | Deterministic, seeded, unit-tested functions. |
| Reasoning | **One LLM — NVIDIA NIM** (OpenAI-compatible endpoint via `langchain-openai` `ChatOpenAI`) | Tool calling only. Explains and reasons; never computes. |
| Charts | **matplotlib** | Rendered into Streamlit with `st.pyplot`. |

Model: `meta/muse-glimmer-30b` (override with `NVIDIA_MODEL`). Provider + model live only in
`src/llm/config.py` (`make_llm()`); nothing else imports a chat-model class.

Tool-calling loop is a plain manual loop: `llm.bind_tools(tools)` → invoke → if the
response has `tool_calls`, run the matching Python function, append a `ToolMessage`,
repeat until no tool calls. ~20 lines. No agent framework.

## 1a. bizmax Core — auto mode (`src/core/`)

`router.py` (LLM JSON routing per the blueprint, Actuary locked in, keyword fallback) →
`orchestrator.py` (selected tracks run **in parallel** threads; contextvars copied so the PRISM
session and computed_* metadata follow) → `synthesizer.py` (executive summary as JSON after a
`<scratchpad>`; scratchpad stripped; truncated JSON recovered). Charts are NEVER produced by the
synthesizer — `src/charts/figures.py` renders from engine output only. The MAGI-style panel is
`src/ui/magi.py` (nodes strobe cyan↔green while running, lock green when done).
Manual mode = one track at a time (the tabs). CLI: `python -m src.core.run <business> "<question>"`.

Sample businesses (`src/data/samples.py` → `data/samples/<slug>/`): velocity (high freq/low sev,
fraud), nexus (cyber cat, low freq/high sev), aegis (long-tail healthcare), terrafirma (volatile
projects). Each has claims.csv + policy.txt + profile.json with programme defaults.

## 1b. The six tracks (`src/tracks/registry.py`)

AI Actuary · AI Claims Analyst · AI Reinsurance Manager · AI Capital Manager · AI Risk Manager ·
AI Policy Analyst. A track = a role system prompt + its own deterministic tools bound to the
loaded `DataBundle` + a `metadata()` of `computed_*` truth values sent to PRISM. Adding a
capability means adding a **tool** (pure Python in `src/actuarial/` or `src/policy/`) to a
track — never a new prompt trick. All tools return JSON dicts with a `method` key.

Data enters only through `src/data/ingest.py` (multi-file CSV/XLSX/PDF -> `DataBundle` with
column normalisation) or `src/data/supabase_import.py` (tables -> `data/imported/*.csv`).

## 2. NO heavy infrastructure

Do **not** use, install, or suggest:
- LangGraph (or any agent-orchestration framework)
- PostgreSQL, SQLite-as-a-service, or any database server
- Vector databases (Chroma, FAISS, Pinecone, pgvector, …). Supabase is an **import source only**
  (REST -> CSV); the app never queries it at analysis time.
- Complex RAG pipelines, embeddings, chunking, rerankers
- Docker, Redis, Celery, FastAPI backends, auth

Instead:
- **Data = local CSVs** in `data/`. Read with `pandas`. That's the whole data layer.
- **Models = local `joblib` files** in `models/`. Train once with `train.py`, load at app start.
- **Policy wording** = PDF text extracted with `pypdf`, queried through regex/keyword tools in
  `src/policy/wording.py`. No embeddings.
- **State** = Streamlit `st.session_state`. Nothing persists between sessions.

## 3. Core Rule 1 — the LLM NEVER does actuarial math

The LLM must never add, multiply, average, fit, simulate, or estimate a number itself.
Every figure it reports must come from a Python tool call and be quoted verbatim.

- All calculations live in `src/actuarial/` and `src/ml/` as Python functions.
- Each is exposed to the LLM as a tool (`@tool` from `langchain_core.tools`) that returns
  structured JSON: `{"value": ..., "unit": ..., "inputs": {...}, "method": "..."}`.
- The system prompt states this rule explicitly and instructs the LLM to call a tool for
  any quantity, and to say "I need to compute that" rather than guess.
- The explanation prompt receives the engine output as structured data and is told to
  quote figures exactly as given.
- **Check:** if a number appears in the narrative that isn't in a tool result, that's a
  bug. This is exactly the failure PRISM catches (`computed_expected_loss` vs narrative).

Why: deterministic math is what lets PRISM verify the AI against a right answer instead
of a judgement — and it's our answer to "isn't this just a GPT wrapper".

## 4. Core Rule 2 — exactly TWO machine learning models

| # | Model | Class | Purpose | File |
|---|---|---|---|---|
| 1 | Claim severity | `sklearn.ensemble.RandomForestRegressor` | Predict claim cost from claim/exposure features | `models/severity_rf.joblib` |
| 2 | Anomaly detection | `sklearn.ensemble.IsolationForest` | Flag unusual claims / outliers in the book | `models/anomaly_iforest.joblib` |

- Trained by `src/ml/train.py` from `data/claims.csv`; `random_state=42` on both.
- Loaded once via `joblib.load` in `src/ml/predict.py` and cached with `@st.cache_resource`.
- **No third model.** No gradient boosting, no neural nets, no clustering, no "quick
  logistic regression". If it feels like it needs another model, it doesn't — use a
  pure-Python actuarial function instead.
- Frequency (Poisson), development (chain-ladder), Monte Carlo aggregation, and TCoR are
  **not** ML — they're deterministic functions in `src/actuarial/`.

## 5. Actuarial engine (`src/actuarial/`) — pure Python

- `frequency_severity.py` — Poisson/NB frequency, lognormal severity, pure premium, stress tests.
- `ibnr.py` — loss-development-factor IBNR per accident year.
- `development.py` — development pattern, adverse development, large-loss indicators, pattern shift.
- `capital.py` — Monte Carlo aggregate (seeded, cached), VaR/TVaR, capital adequacy, retained-risk cost.
- `tcor.py` — TCoR per layer, retention sweep, XoL vs aggregate stop-loss.
- `exposure.py` — exposure movement, concentration (HHI), emerging signals.
- `reinsurance.py` — per-occurrence XoL recovery on actual claims.
- Every function: type hints, docstring, one pytest with a hand-checked expected value.
- Same CSV in → same answer out. Non-determinism is a monitored failure.

## 6. PRISM integration (`src/prism/`)

- Instrument via `prismtrace.PRISMtraceCallbackHandler` (`pip install "prismtrace-sdk>=0.4.3"`),
  built once per process in `src/prism/tracing.py` and passed as
  `config={"callbacks": callbacks()}` on every chat-model / tool invoke.
- `agent_name = "ai-actuary"`. One `prismtrace.session(...)` per analysis run via
  `analysis_run(session_id, metadata)`; it flushes on exit. Never build a handler per request.
- Run the whole agent loop as ONE `@chain` (see `run_recommendation`) and pass callbacks only
  on that root invoke — child model/tool calls inherit them. One root = one trace with nested
  spans and one flush. Passing callbacks again inside the chain double-counts spans.
- Flushes are non-blocking (thread) so a slow PRISM ingest never stalls the UI. Scripts that
  exit right away must use `analysis_run(..., blocking=True)`.
- Staging live-trace path: `python -m src.prism.smoke`. Verify: `python -m prismtrace.verify`.
- Attach the engine's computed figures as trace metadata so evaluators can compare
  narrative vs truth:
  `computed_expected_loss`, `computed_optimal_retention`, `claim_count`, `data_years`,
  `confidence_tier`.
- Failure modes we monitor: hallucinated figure · explanation–math contradiction ·
  ungrounded coverage claim · overconfidence on thin data · non-determinism.
- Improvement loop: baseline → root cause → one fix (prompt or tool) → re-run the same
  fixed cohort in `tests/cohort/` → record the delta.
- The agent **never binds cover**; a human approval step sits before the CFO.

## 7. Environment & commands

- Python 3.12, venv at `.venv/` (Windows: `.\.venv\Scripts\Activate.ps1`).
- Always run through `.venv/Scripts/python`.
- `.env` (never commit): `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, `PRISMTRACE_API_KEY`, `PRISMTRACE_PROJECT_ID`,
  `PRISMTRACE_HOST`. Loaded with `python-dotenv`. Key names only in `.env.example`.

```bash
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m src.ml.train            # trains the two models → models/*.joblib
.venv/Scripts/python -m pytest                  # actuarial + tool tests
.venv/Scripts/python -m streamlit run app.py    # the demo
.venv/Scripts/python -m src.tracks.run all      # every track from the CLI, traced, with grounding check
.venv/Scripts/python -m src.data.supabase_import claims policies   # pull tables to data/imported/
```

## 8. Layout

```
app.py                Streamlit UI — the only entry point
src/
  actuarial/          pure-Python math (see §5)
  ml/
    train.py          trains RF severity + IsolationForest → models/
    predict.py        loads .joblib, exposes predict_severity(), flag_anomalies()
  llm/
    config.py         make_llm() (NVIDIA NIM), model name, system prompt
    tools.py          @tool wrappers around actuarial/ and ml/ functions
    agent.py          make_llm() + bind_tools + manual tool loop
  prism/
    tracing.py        handler setup, session_id, metadata builder, flush
data/
  claims.csv          sample claims (illustrative mid-cap manufacturer)
  exposure.csv        sample exposure / premium schedule
models/               *.joblib (gitignored)
tests/
  test_actuarial.py
  cohort/             fixed replay cases: input CSV + expected computed_* values
```

## 9. Conventions

- Type hints. `snake_case` functions/vars, `PascalCase` classes. Small modules.
- Tools return JSON-serializable dicts; never raw numpy types (cast with `float()`/`int()`).
- All demo numbers are labelled **illustrative / modeled estimate**. Never present sample
  output as real client data.
- Don't refactor working code in the last hour. Commit small and often.

## 10. Demo path (build in this order)

1. `data/claims.csv` + `train.py` → two `.joblib` files exist.
2. `src/actuarial/` functions + tests pass.
3. `tools.py` + `agent.py` → the LLM answers "what's our expected annual loss?" via a tool call.
4. `app.py` → upload CSV, click Run, see recommendation table + matplotlib chart + anomalies.
5. PRISM handler wired, metadata attached, one trace visible in the PRISM dashboard.
6. One detected failure → prompt fix → re-run cohort → before/after delta on a slide.

## PRISM tracing (do not remove)

This project sends traces to PRISM. Env vars: `PRISMTRACE_API_KEY`,
`PRISMTRACE_PROJECT_ID`, `PRISMTRACE_HOST`.

Tracing is currently wired at: `src/prism/tracing.py` (handler, session, computed_* metadata), `src/llm/agent.py` (`run_agent` — one chain root, callbacks on it), `src/core/orchestrator.py` (one session per Core question; router, each parallel sub-agent and the synthesizer are traced runs inside it), `src/core/router.py` and `src/core/synthesizer.py` (plain invokes — inherit the ambient session), `app.py` (`analysis_run` around every Core run and every manual track run), `src/core/run.py` and `src/tracks/run.py` (CLI runners), `src/prism/smoke.py` (staging live-trace path)

**Standing rule.** Whenever you add or change an agent, chain, graph, tool,
retriever, or any entry point that calls a model, wire it to PRISM before you
finish. Unwired code is invisible in the dashboard. If you are unsure whether
something is covered, assume it is not and wire it.
