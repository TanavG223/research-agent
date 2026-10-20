# Grounded Research Agent

A bounded Python research agent that can search the web, run calculations, retrieve local documents,
and return only per-claim answers that survive citation verification. It includes a FastAPI service,
a dependency-free frontend, a fixed 50-question benchmark, controlled step-budget and planning
experiments, and append-only step logs.

## What is implemented

- `web_search(query)` uses Tavily and returns URL, title, and snippet records.
- `run_python(code)` runs calculation-oriented Python in a restricted, resource-limited subprocess
  with a hard timeout and captured output.
- `search_docs(query)` retrieves top ChromaDB chunks with their document source.
- The model emits exactly one tool call or one final structured response per loop iteration.
- The loop stops after `MAX_STEPS` (default 8) and returns `insufficient information` at the limit.
- Tool failures become history entries so the model can choose a different approach.
- Final claims cite one `source_id` each and pass a fail-closed LLM entailment check.
- Unknown, failed, unsupported, or unjudgeable claims are deleted and logged.
- If every claim is deleted, the outward answer is exactly `I could not verify an answer.`

The live adapter uses the OpenAI Responses API with SDK-parsed Pydantic structured outputs. The local
benchmark adapter uses Ollama's native chat endpoint with JSON-schema constrained responses and
`think: false`, following Ollama's [structured-output](https://docs.ollama.com/capabilities/structured-outputs)
and [chat API](https://docs.ollama.com/api/chat) documentation.

## Architecture

```text
question
  -> bounded ResearchAgent loop
       -> OpenAI or Ollama structured decision
       -> Tavily/recorded web evidence | local Python | ChromaDB
       -> structured ToolResult + unique source_id
       -> append-only JSONL history
  -> generated claims
  -> CitationVerifier
       -> source_id membership check
       -> source success check
       -> OpenAI entailment judge
  -> verified claims only | fixed safe fallback
```

The same agent and verifier are used by the CLI, FastAPI endpoint, and benchmark runner. Provider
adapters are separated from orchestration so offline fixtures can test the complete control flow
without pretending to be live model evaluation.

## Setup

Python 3.11–3.13 is supported. Python 3.11 is recommended for the ChromaDB dependency set.

```bash
git clone https://github.com/TanavG223/research-agent.git
cd research-agent
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
```

For the exact Python 3.11/macOS arm64 environment used by the shipped local-model run, install
`requirements.lock` first and then install this project with `pip install -e . --no-deps`. Live run
manifests record resolved package versions and the lockfile hash.

For a live OpenAI + Tavily run, fill in at least:

```dotenv
OPENAI_API_KEY=...
TAVILY_API_KEY=...
OPENAI_MODEL=gpt-5-mini
OPENAI_JUDGE_MODEL=gpt-5-mini
ALLOWED_HOSTS=127.0.0.1,localhost
```

`.env` is loaded without overriding variables already exported by the shell. To report non-zero cost,
set the three current rate fields in `.env`; provider prices are intentionally not hardcoded because
they change over time.

For a local-model benchmark, install and start Ollama, pull `qwen3:4b`, and keep the default local
settings from `.env.example`. No model API key is required. Local benchmark mode uses real Ollama
inference and the fixed recorded web corpus, so it does not require or silently emulate a web-search
API key. `local-live` mode combines that same local model with real Tavily search and therefore
requires only `TAVILY_API_KEY`.

## Run the app

```bash
.venv/bin/research-agent serve
```

Open [http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/). The health badge reports “Setup needed”
until live credentials are configured. The benchmark comparison remains viewable without credentials.
Interactive API docs are disabled by default; set `ENABLE_API_DOCS=true` for local development.

Other commands:

```bash
# One live, citation-verified question
.venv/bin/research-agent ask "What is the latest stable Python release?"

# Add local Markdown/text files to ChromaDB
.venv/bin/research-agent ingest docs/one.md docs/two.txt --chunk-size 1200

# Reproduce the credential-free orchestration experiment
.venv/bin/research-agent experiment --mode offline

# Run both 50-question suites with real local Ollama inference
.venv/bin/research-agent experiment --mode local

# Run one complete local-Ollama + live-Tavily baseline
.venv/bin/research-agent benchmark --mode local-live --max-steps 8 \
  --output results_baseline_live.json

# Run the planning-subgoal experiment; this mode enforces MAX_STEPS=8
.venv/bin/research-agent benchmark --mode local-live --max-steps 8 \
  --planning-subgoals --output results_experiment_v2.json

# Run the real OpenAI + Tavily benchmark
.venv/bin/research-agent experiment --mode live
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Readiness, credential presence, and active step limit |
| `POST /api/ask` | Run one live question from `{"question": "..."}` |
| `GET /api/results` | Return baseline and experiment result documents |

`POST /api/ask` returns the safe answer text, verified claims, rejected-claim counts and reasons,
step trace, raw source results, token/cost usage, wall latency, run ID, and log path. Rejected claim
text is retained only in the append-only run log, never in the outward answer payload. The endpoint
returns 503 without an OpenAI key rather than silently switching to a mock model.

## Benchmark

[`benchmark.json`](benchmark.json) contains exactly 50 fixed records:

| Category | Count | Contract |
|---|---:|---|
| `factual_lookup` | 15 | One lookup |
| `multi_step` | 15 | Two or more chained lookups |
| `computational` | 10 | Python execution |
| `unanswerable` | 10 | No publicly verifiable answer |

Every record has `id`, `question`, `known_correct_answer`, `category`, and `answerable`. Loading fails
if the count, category mix, IDs, answerability mapping, or reference answers are invalid.

The runner computes:

- answerable accuracy using the judge model against `known_correct_answer`;
- unanswerable hallucination rate from confident, verified answers;
- verifier catch rate as deleted generated claims divided by all generated claims;
- average loop steps, estimated cost, and end-to-end latency per question.

Accuracy also enforces the category's tool contract: factual items require a successful web search,
multi-step items explicitly request and require at least two distinct successful web searches, and
computational items require a successful Python run. A memorized or direct answer cannot satisfy
those benchmark categories. Distinct-call compliance is recorded per item; it is a process check,
not proof that a live model reasoned causally across the two results.

The recorded-evidence experiment changes exactly one field: baseline `MAX_STEPS=8` versus
experiment `MAX_STEPS=3`. The later live experiment keeps `MAX_STEPS=8` and changes exactly one
intentional behavior: planning sub-goals off versus on. Enabled mode requires a one-sentence
sub-goal for every tool call, stores it in the trace, and carries it into the next model turn.

## Shipped results and their boundary

- [`results_baseline.json`](results_baseline.json): `MAX_STEPS=8`, 65.0% accuracy (26/40), 0.0%
  hallucination, 30.16% verifier catch rate, 2.42 average steps, and 9.39 s average latency.
- [`results_experiment.json`](results_experiment.json): `MAX_STEPS=3`, 65.0% accuracy (26/40), 0.0%
  hallucination, 27.12% verifier catch rate, 2.36 average steps, and 8.75 s average latency.
- [`results_baseline_live.json`](results_baseline_live.json): local qwen3:4b plus live Tavily,
  `MAX_STEPS=8`, 47.5% accuracy (19/40), and 10.0% hallucination (1/10).
- [`results_experiment_v2.json`](results_experiment_v2.json): the same live stack with planning
  sub-goals, 55.0% accuracy (22/40), and 0.0% hallucination (0/10).
- [`verifier_audit.md`](verifier_audit.md): all 19 claims stripped from the live baseline, each marked
  Y/N: the verifier was right on 13 and wrongly removed 6 supported claims.
- [`RESULTS.md`](RESULTS.md): three-run comparison, category breakdown, cost, and interpretation.
- [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md): live/experiment failure buckets, every experiment miss,
  and direct JSONL links.

The first two result files are marked `mode: local_fixture_tools`: agent and judge calls are genuine
qwen3:4b inference, but web evidence is recorded. The live files are marked
`mode: local_live_tavily` and are valid end-to-end search runs. Both live artifacts completed all 50
questions; their root JSON matches the corresponding run summary.

Local inference has no provider-billed token cost; electricity and hardware depreciation are
excluded. Live-run dollar values are Tavily list-price-equivalent estimates based on the configured
per-call assumption. The planning change improved live accuracy by 7.5 percentage points but raised
average latency from 15.88 to 29.38 seconds and estimated search cost from $0.00800 to $0.01392 per
question. See `RESULTS.md` before drawing a deployment conclusion.

## Reproducibility

Each benchmark creates an immutable, collision-checked `runs/<run_id>/` with:

- one `<question_id>.jsonl` log per benchmark item;
- model decisions, complete tool results/errors, verification events, and final evaluation;
- `summary.json`, a copy of the aggregate result.

Each result embeds a non-secret run manifest with model/judge IDs, timeout and pricing assumptions,
resolved core package versions, platform/Python versions, and SHA-256 hashes for the benchmark,
agent source tree, tool prompt, fixture set when applicable, and dependency lockfile.

Every shipped complete benchmark contains 50 JSONL logs. `results_*.json` also embeds each
question's step trace and raw sources, but no rejected claim text. Rejected text remains in the
append-only audit logs. Do not put secrets or sensitive documents in questions: logs are
intentionally detailed.

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests scripts
node --check src/research_agent/static/app.js
```

For the browser smoke test, install a Playwright browser or use Google Chrome on macOS, start the API,
then run `scripts/ui_smoke.py`. It verifies desktop/mobile rendering, benchmark metrics, a mocked
verified answer, source badges, the deletion audit, and absence of browser console errors.

## Security boundary

`run_python` first rejects non-calculation syntax and imports, including filesystem, network,
subprocess, dynamic-code, private-attribute, class, and function-definition escape paths. It then
uses an isolated interpreter, sanitized environment, trusted in-child resource bootstrap, temporary
working directory, wall timeout, output caps, process-group termination, an independent parent-side
RSS watchdog, and a macOS Seatbelt profile when available. The only importable modules are `math`,
`statistics`, `decimal`, and `fractions`. The RSS watchdog is a denial-of-service guard and can briefly
overshoot its configured bound between samples. This is defense in depth, not a hardened multi-tenant
boundary; run hostile code inside a dedicated container/VM with no secrets and no network.

The server binds to loopback by default, validates Host headers, disables cross-origin access, and
sets CSP, clickjacking, MIME-sniffing, referrer, and permissions headers. It has no user authentication
or rate limiter because it is a local research harness. Do not expose it publicly without adding both,
isolating code execution, setting request limits, and reviewing what raw sources/logs may disclose.

Citation verification checks whether a cited result supports a claim; it does not prove that the
source itself is correct. Search snippets can be incomplete, retrieved documents can be malicious,
and an LLM judge can err. The verifier therefore fails closed on judge errors, but consequential use
still requires source inspection and domain review.
