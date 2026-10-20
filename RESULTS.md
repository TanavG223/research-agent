# Research-agent benchmark results

## Executive summary

The complete live planning experiment improved the official pipeline score from **47.5% to 55.0%**
(+7.5 percentage points) and reduced the measured unanswerable hallucination rate from **10.0% to
0.0%** (-10.0 points). The selected change was option **(a)**: require one nonblank,
single-sentence sub-goal before every tool call and carry it into the next model turn. It was selected
because broken multi-step chains were the largest live-baseline failure bucket (10), ahead of the
judge path (6) and code/evidence failures (5).

The change helped, but at a meaningful price. Broken/repeated chains fell from 10 to 6 and
multi-step accuracy rose from 3/15 to 6/15. Average steps rose 29.5%, the Tavily cost estimate rose
74.0%, and latency rose 85.0%. The experiment therefore improved grounding behavior without making
the pipeline either cheap or consistently accurate.

## Three-run comparison

| Metric | Baseline (recorded) | Baseline (live) | Experiment (live + planning) |
|---|---:|---:|---:|
| `MAX_STEPS` | 8 | 8 | 8 |
| Planning sub-goals | Off | Off | **On** |
| Accuracy on answerable items | 65.0% (26/40) | 47.5% (19/40) | **55.0% (22/40)** |
| Hallucination rate on unanswerable items | 0.0% (0/10) | 10.0% (1/10) | **0.0% (0/10)** |
| Verifier catch rate | 30.2% (19/63) | 33.3% (19/57) | 34.5% (19/55) |
| Average steps / question | 2.42 | 2.24 | 2.90 |
| Average estimated cost / question | $0.00000 | $0.00800 | $0.01392 |
| Average latency / question | 9.39 s | 15.88 s | 29.38 s |
| Live Tavily calls | 0 | 50 | 87 |
| Total estimated run cost | $0.000 | $0.400 | $0.696 |

| Category | Baseline (recorded) | Baseline (live) | Experiment (live + planning) |
|---|---:|---:|---:|
| Factual lookup | 15/15 (100.0%) | 12/15 (80.0%) | 11/15 (73.3%) |
| Multi-step | 7/15 (46.7%) | 3/15 (20.0%) | **6/15 (40.0%)** |
| Computational | 4/10 (40.0%) | 4/10 (40.0%) | 5/10 (50.0%) |
| Unanswerable handled correctly | 10/10 (100.0%) | 9/10 (90.0%) | 10/10 (100.0%) |

The recorded column used real local qwen3:4b inference with fixed recorded web evidence. The two
live columns used the same local qwen3:4b agent, citation judge, and correctness judge with real
Tavily searches. Consequently, only the two live columns form the experiment comparison; the
recorded run is a useful deterministic reference, not a live-search control.

## What changed

The only intentional experimental behavior change was `planning_subgoals: false -> true`. In the
experiment, every one of the 99 tool decisions contained a nonempty immediate sub-goal, that value
was stored in the step trace, and the next qwen3:4b turn received it with the tool result. Terminal
punctuation was optional, but clear multi-sentence values were rejected. Final decisions could not
carry a sub-goal.

All other controlled settings stayed fixed: `MAX_STEPS=8`, qwen3:4b for agent and both judges,
Ollama 0.21.1, temperature 0, seed 42, fail-closed citation verification, verbose tool descriptions,
the benchmark hash, Tavily basic search, pricing assumption, timeouts, dependencies, and tool
implementations. The source-tree hash necessarily changed because code was added to implement the
flag. The live baseline predates that flag and therefore represents disabled planning by omission;
the experiment records it explicitly as enabled.

Live Tavily evidence is time-varying, so this is a controlled configuration comparison rather than
a bit-for-bit paired replay. The full source results and decisions are preserved in the run logs.

## Effect on the targeted failure

| Targeted outcome | Baseline (recorded) | Baseline (live) | Experiment (live + planning) |
|---|---:|---:|---:|
| Multi-step tool contract completed | 11/15 | 5/15 | **9/15** |
| Broken/repeated multi-step chains | 4 | 10 | **6** |
| Multi-step questions correct | 7/15 | 3/15 | **6/15** |

Planning repaired the chain mechanics for `multi_step_002`, `multi_step_003`, `multi_step_005`, and
`multi_step_014`. Only `multi_step_003` and `multi_step_014` received credit: the correctness judge
rejected `multi_step_002` for including accurate country context with the requested capital, and its
written reason called `multi_step_005` correct while the boolean verdict was false.
`multi_step_015` also became correct, but its chain was already complete in the live baseline; that
flip is judge/run variance rather than evidence that planning repaired its chain.

Six chains still failed: `multi_step_001`, `multi_step_006`, `multi_step_008`, `multi_step_009`,
`multi_step_012`, and `multi_step_013`. Four repeated the same first-hop query until all eight turns
were exhausted. A stated plan therefore improved progression but did not reliably make the 4B model
act on its own plan.

## Scored changes by question

Live baseline to experiment improvements:

- `multi_step_003`, `multi_step_014`, and `multi_step_015` became correct.
- `computational_005` became correct.
- `unanswerable_008` changed from a confident off-target answer to a verifier-blocked answer,
  removing the only measured hallucination.

Regression:

- `factual_012` changed from correct to verifier-blocked.

Recorded baseline to live baseline changes are documented in [LIVE_BASELINE.md](LIVE_BASELINE.md).
The sole improvement was `multi_step_007`; regressions were `factual_003`, `factual_008`,
`factual_014`, `multi_step_002`, `multi_step_008`, `multi_step_009`, `multi_step_014`,
`multi_step_015`, and `unanswerable_008`.

## Interpretation boundaries

- The 55.0% figure is the official fail-closed **pipeline** score. Judge/schema errors count as
  incorrect, so it is not a clean measure of the agent model alone.
- The experiment's 0/10 hallucination result came from nine verifier-blocked answers and one
  fail-closed model/schema error, not ten clean explicit declines. It is safer than the live
  baseline but not strong evidence of calibrated abstention.
- The verifier removed 19 claims in both live runs. Its catch rate rose only because generated
  claims fell from 57 to 55.
- Local Ollama inference has no provider-billed token charge. Dollar figures are Tavily
  list-price-equivalent estimates: basic search costs one credit, configured here at the current
  $0.008 pay-as-you-go rate. The supplied free-tier key may have incurred no actual bill. See the
  official [Tavily credit and pricing documentation](https://docs.tavily.com/documentation/api-credits).
- Electricity and hardware depreciation are excluded.

## Reproducibility

- Recorded baseline: `local-steps8-20260819T021841552079Z`.
- Live baseline: `local-live-steps8-20260819T025824193833Z`.
- Planning experiment: `experiment-v2-planning-live-20260819T040000Z`.
- Each complete run has 50 LF-delimited JSONL traces ending in `benchmark_evaluation` and a
  `summary.json` matching its root result file.
- The live experiment recorded 87/87 successful Tavily calls, 429 returned URLs across 159 hosts,
  and 99/99 unique tool-result source IDs. One successful search legitimately returned zero hits.
- No Tavily credential pattern appears in the result or run logs.

See [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) for every miss and [verifier_audit.md](verifier_audit.md)
for the review of all 19 claims stripped from the live baseline: 13 correct removals and 6
wrong removals of supported claims.
