# Live-search baseline

## Technical summary

The complete `MAX_STEPS=8` live-search run scored **47.5%** on answerable questions and
**10.0%** hallucination on unanswerable questions. Relative to the recorded-evidence baseline,
accuracy fell 17.5 percentage points and one previously declined unanswerable item produced a
verified confident answer. All 50 Tavily calls succeeded and returned at least one result, so this
movement is model/retrieval/evaluation behavior rather than a search outage.

The run used real `qwen3:4b` inference for the agent, citation verifier, and correctness judge; real
Tavily basic search; and the real local Python sandbox. Its immutable run ID is
`local-live-steps8-20260819T025824193833Z`.

## Metric comparison

| Metric | Recorded baseline | Live-search baseline | Change |
|---|---:|---:|---:|
| Accuracy (answerable) | 65.0% (26/40) | 47.5% (19/40) | -17.5 pp |
| Hallucination rate (unanswerable) | 0.0% (0/10) | 10.0% (1/10) | +10.0 pp |
| Verifier catch rate | 30.16% (19/63) | 33.33% (19/57) | +3.17 pp |
| Average steps / question | 2.42 | 2.24 | -0.18 |
| Average list-price-equivalent cost / question | $0.000 | $0.008 | +$0.008 |
| Average latency / question | 9.39 s | 15.88 s | +6.49 s |

The live run made 50 basic Tavily calls, equivalent to 50 API credits or $0.40 at the configured
$0.008 pay-as-you-go list-price assumption. Because the supplied key is on the free tier, this is a
comparison estimate rather than a claim that $0.40 was actually billed. Local Ollama token cost is
$0; electricity and hardware depreciation are excluded.

## Category outcomes

| Category | Recorded baseline | Live-search baseline | Difference |
|---|---:|---:|---:|
| Factual lookup | 15/15 | 12/15 | -3 |
| Multi-step | 7/15 | 3/15 | -4 |
| Computational | 4/10 | 4/10 | 0 |
| Unanswerable handled correctly | 10/10 | 9/10 | -1 |

## Questions whose scored outcome changed

These are the IDs whose benchmark correctness classification changed. Answer wording alone is not
treated as an accuracy change.

| Question ID | Recorded outcome | Live outcome |
|---|---|---|
| `factual_003` | correct | incorrect |
| `factual_008` | correct | incorrect |
| `factual_014` | correct | incorrect |
| `multi_step_002` | correct | incorrect |
| `multi_step_007` | incorrect | correct |
| `multi_step_008` | correct | incorrect |
| `multi_step_009` | correct | incorrect |
| `multi_step_014` | correct | incorrect |
| `multi_step_015` | correct | incorrect |
| `unanswerable_008` | correct decline | confident verified answer |

Five agent statuses also changed: `factual_003`, `factual_008`, `factual_014`,
`computational_005`, and `unanswerable_008`. Outward answer text changed on 29/50 questions because
live snippets and generated wording differ; the canonical before/after text is preserved in
`results_baseline.json` and `results_baseline_live.json`.

## Completeness and reproducibility

- 50/50 benchmark IDs are present.
- The run directory contains exactly 50 JSONL traces and every trace ends with
  `benchmark_evaluation`.
- The root result is byte-equivalent as parsed JSON to the run's `summary.json`.
- Manifest mode is `local_live_tavily`; backend is `ProductionToolExecutor`; web mode/provider are
  `live_web` / `tavily`; no fixture hash or recorded source ID is present.
- The result and run logs contain no `tvly-` credential pattern.
