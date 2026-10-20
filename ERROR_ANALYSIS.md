# Error analysis

## Summary

The live baseline had **22 incorrect outcomes**. Its largest primary bucket was broken or repeated
multi-step lookup chains (10), so the experiment changed exactly one behavior: every tool call had
to state a one-sentence sub-goal that was carried into the next model turn. The experiment reduced
the targeted bucket to 6 and finished with **18 incorrect outcomes**.

| Mutually exclusive primary bucket | Baseline (live) | Experiment | Change |
|---|---:|---:|---:|
| Broken/repeated multi-step chain | 10 | 6 | **-4** |
| Judge/verifier/schema path | 6 | 8 | +2 |
| Python/code/evidence failure | 5 | 4 | -1 |
| Confident answer on an unanswerable item | 1 | 0 | **-1** |
| **Total incorrect outcomes** | **22** | **18** | **-4** |

These buckets assign one primary cause to every incorrect outcome so the totals are additive. The
judge/verifier bucket includes explicit structured-output failures, boolean verdicts that contradict
their written reasons, and support decisions that removed otherwise endpoint-bearing answers. The
official score is not manually corrected: fail-closed pipeline outcomes remain incorrect.

The separate [verifier_audit.md](verifier_audit.md) reviews the 19 claims stripped from the
**live baseline**: 13 were correct removals and 6 removed claims the cited source supported.

## Live-baseline buckets

### Broken or repeated chain — 10

`multi_step_001`, `multi_step_002`, `multi_step_003`, `multi_step_005`, `multi_step_006`,
`multi_step_008`, `multi_step_009`, `multi_step_012`, `multi_step_013`, and `multi_step_014`.

All ten failed the requirement for two distinct successful searches. Seven stopped after an
intermediate fact; three repeated the same first-hop query instead of advancing.

### Judge/verifier/schema path — 6

`factual_003`, `factual_008`, `factual_014`, `multi_step_010`, `multi_step_015`, and
`computational_007`.

Three direct factual answers were removed by the support verifier. Two complete multi-step answers
failed in correctness evaluation, and one successful Python retry was followed by an invalid
citation-judge response.

### Python/code/evidence — 5

`computational_003`, `computational_005`, `computational_006`, `computational_009`, and
`computational_010`.

The programs hard-coded a wrong value, failed validation or an import, printed insufficiently
self-contained evidence, switched to web search instead of retrying Python, or printed an incorrect
conversion result.

### Confident unanswerable answer — 1

`unanswerable_008` collided with an unrelated game entity and returned an adjacent Helios Academy
fact instead of declining.

## What the planning change fixed

Multi-step contract completion rose from **5/15 to 9/15**. The four repaired chains were
`multi_step_002`, `multi_step_003`, `multi_step_005`, and `multi_step_014`. Only
`multi_step_003` and `multi_step_014` became correct; `multi_step_002` and `multi_step_005` completed
their chains but were rejected by the correctness judge.

Six chains remained broken. `multi_step_001` repeated its first search twice and finalized the
intermediate result. `multi_step_008` repeated its first search three times and then produced a
verifier-blocked answer. `multi_step_006`, `multi_step_009`, `multi_step_012`, and
`multi_step_013` repeated one query until `MAX_STEPS=8`.

The planning field improved chain progression but also encouraged more tool use: Tavily calls rose
from 50 to 87 and four questions exhausted the eight-step budget, compared with none in the live
baseline.

## Every experiment miss

| ID | Primary bucket | What happened |
|---|---|---|
| [`factual_003`](runs/experiment-v2-planning-live-20260819T040000Z/factual_003.jsonl) | Judge/verifier | The support judge treated Tokyo's legal-designation nuance as insufficient and removed the answer. |
| [`factual_008`](runs/experiment-v2-planning-live-20260819T040000Z/factual_008.jsonl) | Judge/verifier | The support judge rejected the short endpoint answer as too vague, despite the retrieved atomic-number context. |
| [`factual_012`](runs/experiment-v2-planning-live-20260819T040000Z/factual_012.jsonl) | Judge/verifier | The written support reason says the source directly supports the answer, but the boolean verdict removed it. This was the sole scored regression. |
| [`factual_014`](runs/experiment-v2-planning-live-20260819T040000Z/factual_014.jsonl) | Judge/verifier | The written reason explains that the above-sea-level evidence supports the answer, but the boolean verdict removed it. |
| [`multi_step_001`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_001.jsonl) | Broken chain | Repeated the first lookup twice and finalized the intermediate number instead of finding the element. |
| [`multi_step_002`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_002.jsonl) | Judge/verifier | Completed both lookups; the correctness judge rejected the requested capital because accurate country context was also present. |
| [`multi_step_005`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_005.jsonl) | Judge/verifier | Completed both lookups; the written reason calls the language answer correct while the boolean verdict is false. |
| [`multi_step_006`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_006.jsonl) | Broken chain | Repeated the Hamlet-author lookup eight times and reached the hard limit. |
| [`multi_step_008`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_008.jsonl) | Broken chain | Repeated the Sofia-country lookup three times, never performed the sea lookup, and lost its final claim in verification. |
| [`multi_step_009`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_009.jsonl) | Broken chain | Repeated the Wellington lookup eight times and reached the hard limit. |
| [`multi_step_010`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_010.jsonl) | Judge/verifier | Completed both lookups, but the correctness judge returned invalid structured output. |
| [`multi_step_012`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_012.jsonl) | Broken chain | Repeated the Iceland-capital lookup eight times and reached the hard limit. |
| [`multi_step_013`](runs/experiment-v2-planning-live-20260819T040000Z/multi_step_013.jsonl) | Broken chain | Repeated the Marie Curie birthplace lookup eight times and reached the hard limit. |
| [`computational_003`](runs/experiment-v2-planning-live-20260819T040000Z/computational_003.jsonl) | Code/evidence | Python ran, but the generated calculation hard-coded the wrong Fibonacci value. |
| [`computational_006`](runs/experiment-v2-planning-live-20260819T040000Z/computational_006.jsonl) | Code/evidence | Python produced the correct number without the principal, rate, and duration needed to support the full claim. |
| [`computational_007`](runs/experiment-v2-planning-live-20260819T040000Z/computational_007.jsonl) | Judge/verifier | The first Python call failed, the retry succeeded, and then the citation judge returned invalid structured output, so verification failed closed. |
| [`computational_009`](runs/experiment-v2-planning-live-20260819T040000Z/computational_009.jsonl) | Code/evidence | Used web search instead of the required Python tool and failed the computational tool contract. |
| [`computational_010`](runs/experiment-v2-planning-live-20260819T040000Z/computational_010.jsonl) | Code/evidence | An operator-precedence error produced 80.82 rather than the correct Celsius conversion. |

## Scored outcome changes

Live baseline to experiment improvements: `multi_step_003`, `multi_step_014`, `multi_step_015`,
`computational_005`, and `unanswerable_008`. Regression: `factual_012`.

The answerable changes reconcile to +3 correct (four gains, one loss), taking accuracy from 19/40
to 22/40. `unanswerable_008` accounts for the hallucination movement from 1/10 to 0/10.

The unanswerable result needs a conservative reading: nine experiment answers were blocked by the
verifier and one ended in a fail-closed model/schema error. None was a clean explicit decline.

## Overlapping operational signals

These counts help diagnose the pipeline but overlap and must not be added together.

| Signal | Baseline (recorded) | Baseline (live) | Experiment |
|---|---:|---:|---:|
| Reached `MAX_STEPS` | 0 | 0 | 4 |
| Model-decision/schema error items | 0 | 0 | 1 |
| Correctness-judge schema error items | 1 | 2 | 1 |
| Citation-judge schema error items | 2 | 1 | 1 |
| Incorrect outcomes ending unverified | 6 | 8 | 10 |
| Failed tool calls | 7 | 4 | 3 |
| Failed Tavily calls | 2 | 0 | 0 |
| Failed Python calls | 5 | 4 | 3 |
| Tool-contract failure items | 6 | 12 | 7 |

The experiment's three failed Python calls occurred in `computational_004`, `computational_005`, and
`computational_007`; the first two recovered and were scored correct. All 87 Tavily calls succeeded.

## Measurement limits

- The classifications identify the primary pipeline failure visible in logs; they do not manually
  override official correctness or verifier booleans.
- Search results are live and time-varying. The configuration was controlled, but evidence is not a
  deterministic replay.
- A fixed seed and temperature zero do not guarantee bit-for-bit deterministic Ollama output.
- The multi-step contract checks two distinct successful queries, not whether the second query was
  causally derived from the first. The item-level trace review above supplies that missing context.
