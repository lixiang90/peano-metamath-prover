# First large-scale depth run (2026-08-11)

This report records the first million-example, GPT-2-small-class training run.
It deliberately separates verified symbolic depth, teacher-forced policy
accuracy, and complete-action success.

Historical report, retained during the 2026-09-04 documentation review. Its test
counts and next-step recommendations describe that run, not the current branch.
For later results, see the [RTX 5090 report](first-rtx5090-closed-loop-run.md),
[PA+ local GPU validation](pa-plus-neural-training.md), and
[current roadmap](progress-and-roadmap.md).

## Artifacts

- Database: `formal/peano-number-theory.mm`
- Base proof pool: `outputs/scale-gpt2-depth-v1/base`
- Million-example corpus: `outputs/scale-gpt2-depth-v1/corpus-1m`
- Corpus audit: `outputs/scale-gpt2-depth-v1/corpus-1m-audit-10k.json`
- 1,000-step baseline: `outputs/scale-gpt2-depth-v1/model-104m/final.pt`
- 5,000-step resumable run: `outputs/scale-gpt2-depth-v1/model-104m-10k/latest.pt`
- Natural held-out evaluation: `outputs/scale-gpt2-depth-v1/evaluation-test-5k-step5000.json`
- Depth stress pool: `outputs/scale-gpt2-depth-v1/depth-stress-32`
- Depth stress evaluation corpus: `outputs/scale-gpt2-depth-v1/depth-stress-scale-30k`
- Depth stress evaluation: `outputs/scale-gpt2-depth-v1/evaluation-depth-stress-test-5k-step5000.json`

The `outputs/` directory is intentionally ignored by Git. Preserve or publish
large artifacts separately when reproducing this run.

## Corpus

The scale corpus contains 1,000,000 replayable action examples:

| Split | Examples | Maximum proof depth |
| --- | ---: | ---: |
| Train | 990,000 | 16 |
| Validation | 5,000 | 14 |
| Test | 5,000 | 14 |

The training split has 19,576 depth-16 examples. Sampling is balanced by proof
depth instead of reproducing the much more shallow-skewed base-template
frequency. A deterministic audit sampled 10,000 examples and replayed all
10,000 successfully in the symbolic environment; it found zero invalid
actions and zero duplicate IDs.

The corpus occupies 564,443,599 bytes on disk. It is an action corpus generated
from verified proof templates, not one million mathematically distinct theorem
statements.

## Model and training

The model has 104,130,543 parameters:

- model width 768;
- 12 attention heads;
- 6 encoder and 6 decoder layers;
- feed-forward width 3,072;
- maximum state/action context 2,304 tokens.

The first baseline ran for 1,000 optimizer steps with micro-batch 1 and gradient
accumulation 4. It exercised the full 2,304-token context and completed without
CUDA OOM.

The longer run uses a 10,000-step learning-rate schedule and was paused exactly
at step 5,000 so it can be resumed without changing its trajectory. It used
micro-batch 8, consumed 40,000 examples and 30,992,201 target tokens, and took
3,907 seconds on an RTX 3090. Mean throughput including checkpoint overhead was
10.24 examples/s. The final observed training loss was 0.320. The checkpoint is
about 1.25 GB and includes the optimizer, scheduler, data cursor, RNG states,
metrics history, and exact-resume metadata.

## Held-out results

Both evaluations below use the same 5,000-example natural test split. The two
runs differ in batch size and learning-rate schedule as well as training amount,
so the comparison is an engineering baseline rather than a controlled ablation.

| Run | Policy perplexity | Token accuracy | Exact action accuracy |
| --- | ---: | ---: | ---: |
| 1,000-step baseline | 3.617 | 46.62% | 0.00% |
| 5,000-step long run | 1.386 | 88.31% | 0.00% |

At step 5,000, natural-test token accuracy ranges from 77.81% to 95.66% across
proof-depth buckets 1 through 14. It does not collapse monotonically with depth,
although sparse deep buckets have high uncertainty.

## Depth stress test

A separate symbolic stress run allowed depth 32, enlarged structural limits,
used six seeds and made 50,000 composition attempts per seed. Per-seed maximum
proof depths were 14, 15, 16, 13, 15, and 19. This establishes that the previous
depth-16 result was partly a configured cap, not a hard generator limit.

The deepest proof that fit reliably into the 2,304-token scale representation
was depth 17 in the stress test split. A 2,000-example audit replayed 2,000/2,000
actions successfully. On all 5,000 stress-test examples, the step-5,000 model
obtained 84.78% token accuracy. The deeper buckets were:

| Proof depth | Examples | Token accuracy |
| ---: | ---: | ---: |
| 14 | 55 | 80.37% |
| 15 | 46 | 80.69% |
| 16 | 22 | 80.03% |
| 17 | 38 | 76.50% |

This is evidence of gradual local-policy degradation rather than an abrupt
depth failure. It is not evidence that the neural prover can autonomously solve
depth-17 goals: exact full-action accuracy remained zero.

Exact reference-action accuracy is only an imitation diagnostic. A valid proof
may choose a different rule sequence, while exact reproduction may merely
memorize a template. This run did not yet perform the definitive closed-loop
evaluation in which the reference proof is hidden and success requires a newly
searched, independently verified certificate.

## Conclusions and next bottlenecks

The symbolic generator is verified beyond depth 16 and reached depth 19 under
the stress configuration. The neural policy learns useful token-level structure
across depth 1-17, but complete action generation is still the blocking metric.
The current 2,304-token serialization also drops some of the deepest generated
proof states.

The next run should prioritize:

1. length-bucketed or token-budgeted batches to reduce padding and improve GPU
   throughput;
2. structured action decoding or constrained symbolic candidate scoring, rather
   than unconstrained whole-action text generation;
3. a depth-stratified search benchmark reporting solved goals and externally
   verified certificates;
4. longer-context or more compact proof-state encoding for depth 18+;
5. resuming the existing exact checkpoint to step 10,000 only after the search
   metric is in place, so additional compute answers a stronger question than
   teacher-forced token accuracy alone.

All 41 project tests and `compileall` passed after the run. During evaluation, a
JSON-export bug caused by checkpoint RNG tensors was found and fixed; a regression
test now ensures evaluation reports contain a compact JSON-safe checkpoint
summary.

The closed-loop evaluation, reinforcement-learning and latent-reasoning plan is
documented in [progress-and-roadmap.md](progress-and-roadmap.md).
