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
| Reasoning | **One LLM — Claude via Anthropic API** (`langchain-anthropic` `ChatAnthropic`) | Tool calling only. Explains and reasons; never computes. |
| Charts | **matplotlib** | Rendered into Streamlit with `st.pyplot`. |

Model: `claude-opus-5`. One constant in `src/llm/config.py` — nowhere else.

Tool-calling loop is a plain manual loop: `llm.bind_tools(tools)` → invoke → if the
response has `tool_calls`, run the matching Python function, append a `ToolMessage`,
repeat until no tool calls. ~20 lines. No agent framework.

## 2. NO heavy infrastructure

Do **not** use, install, or suggest:
- LangGraph (or any agent-orchestration framework)
- PostgreSQL, SQLite-as-a-service, or any database server
- Vector databases (Chroma, FAISS, Pinecone, pgvector, …)
- Complex RAG pipelines, embeddings, chunking, rerankers
- Docker, Redis, Celery, FastAPI backends, auth

Instead:
- **Data = local CSVs** in `data/`. Read with `pandas`. That's the whole data layer.
- **Models = local `joblib` files** in `models/`. Train once with `train.py`, load at app start.
- **Policy wording** (if needed at all) = a small CSV/text file the LLM queries through a
  simple keyword-lookup tool. No embeddings.
- **State** = Streamlit `st.session_state`. Nothing persists between sessions.

## 3. Core Rule 1 — the LLM NEVER does actuarial math

The LLM must never add, multiply, average, fit, simulate, or estimate a number itself.
Every figure it reports must come from a Python tool call and be quoted verbatim.

- All calculations live in `src/actuarial/` and `src/ml/` as Python functions.
- Each is exposed to Claude as a tool (`@tool` from `langchain_core.tools`) that returns
  structured JSON: `{"value": ..., "unit": ..., "inputs": {...}, "method": "..."}`.
- The system prompt states this rule explicitly and instructs Claude to call a tool for
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

- `frequency.py` — Poisson fit (mean claims/year), trend adjustment.
- `severity.py` — lognormal fit on historical severities (RF model supplies per-claim
  predictions; lognormal supplies the distribution for simulation).
- `development.py` — chain-ladder factors, simple IBNR.
- `monte_carlo.py` — aggregate annual loss, `numpy.random.default_rng(42)`, N paths.
- `tcor.py` — `TCoR = premium + retained_losses + deductibles + admin`; retention sweep
  to find the minimum-TCoR retention.
- Every function: type hints, docstring, one pytest with a hand-checked expected value.
- Same CSV in → same answer out. Non-determinism is a monitored failure.

## 6. PRISM integration (`src/prism/`)

- Instrument via the PRISM **Python SDK callback handler** passed to `ChatAnthropic`
  invocations (`config={"callbacks": [handler]}`).
- `agent_id = "ai-actuary"`. One stable `session_id` per analysis run, set at run start
  (cannot be backfilled). Call `handler.flush()` before the run ends.
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
- `.env` (never commit): `ANTHROPIC_API_KEY`, `PRISM_API_KEY`. Loaded with `python-dotenv`.

```bash
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m src.ml.train            # trains the two models → models/*.joblib
.venv/Scripts/python -m pytest                  # actuarial + tool tests
.venv/Scripts/python -m streamlit run app.py    # the demo
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
    config.py         MODEL = "claude-opus-5", system prompt
    tools.py          @tool wrappers around actuarial/ and ml/ functions
    agent.py          ChatAnthropic + bind_tools + manual tool loop
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
3. `tools.py` + `agent.py` → Claude answers "what's our expected annual loss?" via a tool call.
4. `app.py` → upload CSV, click Run, see recommendation table + matplotlib chart + anomalies.
5. PRISM handler wired, metadata attached, one trace visible in the PRISM dashboard.
6. One detected failure → prompt fix → re-run cohort → before/after delta on a slide.
