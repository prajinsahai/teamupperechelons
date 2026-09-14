import os, json, time, random
from dotenv import load_dotenv
import prismtrace
from prismtrace import PRISMtraceCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

handler = PRISMtraceCallbackHandler(
    api_key=os.environ["PRISMTRACE_API_KEY"],
    project_id=os.environ["PRISMTRACE_PROJECT_ID"],
    host=os.environ["PRISMTRACE_HOST"],
)

llm = ChatOpenAI(
    model="meta/muse-glimmer-30b",
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url=os.environ["NVIDIA_BASE_URL"],
    max_tokens=800,
    temperature=0.3,
    top_p=0.95,
    timeout=90,
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an actuarial analyst. Explain the computed figures to a CFO in plain "
     "English. Use ONLY the figures provided. Never invent or estimate numbers."),
    ("human",
     "Computed results:\n{figures}\n\nExplain what this means for our retention level."),
])

chain = prompt | llm


def compute(claims, years=5):
    n = len(claims)
    freq = n / years
    sev = sum(claims) / n
    expected_loss = freq * sev
    return {
        "computed_frequency": round(freq, 2),
        "computed_severity": round(sev),
        "computed_expected_loss": round(expected_loss),
        "computed_optimal_retention": round(expected_loss * 1.5),
        "claim_count": n,
        "data_years": years,
        "confidence_tier": "full_monte_carlo" if n >= 100 else "benchmark_thin_data",
    }


def synthetic_claims():
    n = random.randint(6, 14)
    return [random.randint(40000, 950000) for _ in range(n)]


def run_once(session_id, claims):
    math_out = compute(claims)
    with prismtrace.session(session_id):
        result = chain.invoke(
            {"figures": json.dumps(math_out, indent=2)},
            config={"callbacks": [handler]},
        )
    return math_out, result.content


if __name__ == "__main__":
    RUNS = 1

    for i in range(1, RUNS + 1):
        sid = "actuary-demo-%03d" % i
        claims = synthetic_claims()
        t0 = time.time()
        try:
            math_out, text = run_once(sid, claims)
            print("\n===== %s  (%.1fs) =====" % (sid, time.time() - t0))
            print("COMPUTED:", json.dumps(math_out))
            print("NARRATIVE:", text[:400])
        except Exception as e:
            print("[%s] FAILED after %.1fs: %s" % (sid, time.time() - t0, e))
        if i < RUNS:
            time.sleep(3)

    handler.close()
    print("\nDone. %d run(s) sent to PRISM." % RUNS)