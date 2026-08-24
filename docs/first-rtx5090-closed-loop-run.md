# First RTX 5090 scale and closed-loop run (2026-08-24)

This report records the first completed 10,000-step, GPT-2-small-class run and
the first fixed-budget P0-P2 comparison on its checkpoint.  It separates
teacher-forced prediction, internally certified search, and independent
Metamath verification.

## Artifacts

The remote experiment root is
`/root/autodl-tmp/peano-run/scale-5090-v1`.  Small reports and the important
checkpoints were also copied to the Git-ignored local directory
`outputs/remote-scale-5090-v1/`.

- supervised checkpoint: `model-104m/final.pt`;
- candidate-head checkpoint: `model-104m/rl-1-local.pt`;
- training metrics: `model-104m/metrics.json`;
- held-out test: `evaluation-test-5k.json`;
- corpus manifest and audit: `corpus-1.1m/manifest.json` and
  `corpus-audit-10k.json`;
- public benchmark: `benchmark.json`;
- SFT and RL closed-loop reports: `closed-loop-sft-local.json` and
  `closed-loop-rl-local.json`;
- per-attempt journals: the corresponding `*.progress.jsonl` files;
- exported successful certificates: `certificates/`.

The supervised checkpoint is 1,255,789,187 bytes.  Its local and remote
SHA-256 values both equal
`02017461dbd759f5165ef437217867ea9041cd37b1f6496b177c8f912e56cf10`.
The local RL checkpoint SHA-256 is
`319cbfb16e2de70cdd2de39adbd3893c49ebe09e896c23e492d7981bbae684b6`.

## Corpus and benchmark

The scale corpus contains 1,100,000 action records: 1,080,000 train, 10,000
validation and 10,000 test.  The compressed shards occupy 658,152,070 bytes
(about 628 MiB), cover proof depths 1 through 13 and use contexts up to 2,304
tokens.  A deterministic audit replayed 10,000/10,000 sampled actions through
the typed kernel, with zero invalid actions and zero duplicate IDs in the
sample.  These are action records augmented from 1,004 verified base proof
templates, not 1.1 million mathematically distinct theorems.

The public benchmark contains 51 cases: 12 easy, 17 medium, 18 hard and four
frontier.  Forty are in the research score group and 37 of those pass the
training-leakage eligibility filter.  Reference rules are absent from the
public file and retained only in a private sidecar.

## Model and supervised training

The model has 105,312,496 parameters: width 768, 12 heads, six encoder layers,
six decoder layers, feed-forward width 3,072 and maximum state/action context
2,304.  The parameter count is slightly above the earlier 104M model because
this version includes the finite candidate-index head.

Training used an RTX 5090, BF16, gradient checkpointing, micro-batch 8, 10,000
optimizer steps, a 200-step warmup and cosine decay from `1e-4` to `1e-5`.
It consumed 80,000 examples and 73,038,051 target tokens in 2,875.6 seconds.
Mean throughput was 27.82 examples/s and 25,400 target tokens/s; the full
2,304-token context was exercised.

On 5,000 held-out test examples:

| Metric | Result |
| --- | ---: |
| Policy loss | 0.21236 |
| Perplexity | 1.23660 |
| Token accuracy | 92.256% |
| Exact reference-action accuracy | 0.000% |
| Evaluation throughput | 119.13 examples/s |

The high token accuracy is therefore only a local imitation diagnostic.  It
does not imply that the model can spell an entire valid action or close a
multi-step proof by itself.

## Fixed-budget closed-loop evaluation

Both checkpoints used the same nine leakage-eligible research targets: three
each from easy, medium and hard.  Each target used seed 17, 40 MCTS simulations,
maximum depth 16 and branching 16.  The four policies were uniform, heuristic,
neural and hybrid.  This gives 36 attempts per checkpoint.  Attempts here are
policy-target pairs, not 36 independent theorems.

| Policy | SFT certified | RL certified | SFT seconds | RL seconds | Time speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 0/9 | 0/9 | 77.02 | 79.20 | 0.97x |
| Heuristic | 4/9 | 4/9 | 340.02 | 348.09 | 0.98x |
| Neural | 0/9 | 0/9 | 147.38 | 63.62 | 2.32x |
| Hybrid | 3/9 | 3/9 | 646.51 | 558.76 | 1.16x |
| All policies | 7/36 | 7/36 | 1,210.93 | 1,049.67 | 1.15x |

Difficulty results were unchanged by the RL update.  Heuristic solved 3/3
easy, 0/3 medium and 1/3 hard.  Hybrid solved 3/3 easy and 0/3 in both medium
and hard.  Uniform and neural solved 0/3 in every bucket.

The SFT checkpoint used `autoregressive_bootstrap` candidate scoring.  The RL
checkpoint used `candidate_index`.  Neural scoring became 2.32 times faster,
but the small candidate-head update did not improve solve rate.  Hybrid search
remained dominated by symbolic enumeration and lookahead, so its end-to-end
speedup was smaller.

## P2 replay and external verification

Replay collection used 12 training-only synthetic targets under the same
40-simulation, depth-16, branching-16 search limits.  Ten targets produced
internally verified proofs and the buffer contained 50 states.  Budget-exhausted
hard searches remained censored instead of becoming false negative rewards.
Three candidate-head epochs reduced policy loss from 1.0577 to 1.0391 and value
loss from 0.08563 to 0.000048.

All seven successful attempts in the final closed-loop run were replayed at
their exact policy, seed and search configuration.  Their exported certificates
passed both the project verifier and the official C Metamath executable.  The
executable was version `0.199.pre` (29-Jan-2022), with SHA-256
`f22429bc4cd4e371621dbdbe02d89e4d3e89c15cbd39393072c64636204722aa`.
External verification was fail-closed while the local WSL transport was being
configured; only runs that explicitly printed that all proofs were verified
were counted as passed.

## Conclusions and limits

The first scale run establishes that the million-record pipeline, 105M model,
long-context training, structured candidate head, MCTS replay and dual
certificate verification all work end to end.  The learned candidate head also
removes a material autoregressive inference bottleneck.

It does **not** establish that the learned policy improves proof ability.  On
this sample, neural search solved no goals and the RL update changed solve rate
by zero.  P1 and P2 research acceptance therefore remain unmet.  This run also
uses only nine of 37 eligible research cases and one seed; its Wilson intervals
are wide and it is not the planned three-seed full benchmark.

The next experiment should collect substantially more successful replay,
separate replay-generation and evaluation theorem families, batch candidate
scoring, and then run all 37 eligible cases with seeds 17/19/23.  P3 continuous
latent reasoning should remain deferred until that controlled experiment shows
a solve-rate improvement over the heuristic and SFT baselines.
