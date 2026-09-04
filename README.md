[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-lightgrey)](README-zh.md)

# Peano Metamath Prover

A strictly typed theorem generator and neural-symbolic prover for `peano.mm`.
The project is currently in beta: neural networks and search algorithms may
only propose candidate actions. A successful proof must compile into a
certificate that the project verifier can replay. Important results should
also be cross-checked with an external Metamath implementation.

## The `htps` branch: hypergraph search and extended PA+

This section was checked on 2026-09-04 against `htps` commit `eefcd09`.
`main` retains the base neural-symbolic prover, MCTS, intermediate lemmas, and
the continuous latent reasoning prototype. Subsequent HTPS and generated PA+
development lives on the
[`htps` branch](https://github.com/lixiang90/peano-metamath-prover/tree/htps)
of this same repository. These additions have not been merged into `main`;
they are not a separate external dependency and do not require a neighboring
copy of the project.

### What it adds to `main`

- **HTPS search and training:** `src/htps_prover/` provides AND/OR hypergraph
  search with shared goal nodes, PUCT, soft critic backups, and guarded
  hyperedges that require an intermediate lemma to be proved before use.
  Forward proof DAGs are split into training and evaluation sets by recursive
  proof skeleton. Latent supervised training and a synchronous search/replay
  loop are also available.
- **Extended PA+ formal library:** `formal/peano-pa-plus.mm` and an editable
  JSON definition catalog add 68 conservative high-level relations and 35
  non-logical `statement` goals, covering number theory, finite recurrences,
  integers/rationals, and elementary analysis certificates. Goal formulas are
  neither axioms nor proved theorems.
- **Theorem generation:** 136 replayable unfolding/folding bridges are
  mechanically derived from definitions, with bounded natural-number
  instances, structural guidance from 35 goals, and quotas on single-definition
  wrappers. Goals are sampling hints only, never proof premises.
- **Base neural pipeline integration:** PA+ context, goal-label hints, typed
  binding slots, an in-batch candidate contrastive loss, and weighted sampling
  by generation type. Inference can use definition bridges and bounded terms;
  certificates inline the bridges back into source-library rules. Existing
  checkpoints support append-only vocabulary expansion, but new weights still
  need training.

### What has been validated

On 2026-08-31, a local RTX 3060 Laptop GPU trained a 64-dimensional model with
two encoder and two decoder layers for three epochs on 616 PA+ examples.
Validation loss decreased from 5.9908 to 3.8880. Hybrid search closed one
definition-bridge instance in a single step; its exported 33-label certificate
passed both the project kernel and official Metamath verification. All 81
regression tests on the branch passed as of this review.

These are engineering checks, not a 35-goal proving experiment or
million-example PA+ or latent/HTPS training. The small model still ranked the
correct bridge 14th out of 17 candidates, so improved neural proving ability
has not been established. The RTX 5090 and 1.1M-example results under
"Current progress" below come from the earlier number-theory/Scale experiment;
they are not performance evidence for extended PA+ or HTPS.

Current limitations also matter: HTPS evaluation automatically runs only the
project verifier; mandatory external verification is not yet wired in. PA+
generated bridges are not yet supported by the general corpus-audit and Scale
template-parsing paths, and catalog/bounded-instance options are not yet
available in `peano-htps generate`. A formal four-policy PA+ comparison must
first give every policy the same definition bridges, bounded terms, and
candidate-enumeration settings, so differences in search environments are not
mistaken for model improvements.

### Usage and documentation

Save or commit local changes before switching branches from the repository
root. A separate Python virtual environment is recommended:

```bash
git fetch origin
git switch htps
python -m pip install -e ".[neural,dev]"
peano-htps --help
python -m neural_prover build-corpus --help
```

These additional commands and PA+ options require the `htps` branch.
Checkpoints, training data, and `outputs/` are not distributed through Git.
Existing models must remain paired with their original tokenizers and use the
explicit upgrades described in the branch documentation; token IDs cannot be
mixed arbitrarily. The links below point across branches to avoid referencing
files that do not exist on `main`:

- [HTPS branch README and quick start](https://github.com/lixiang90/peano-metamath-prover/blob/htps/README.md)
- [HTPS data, training, search, and limitations](https://github.com/lixiang90/peano-metamath-prover/blob/htps/docs/htps-design.md)
- [PA+ conservative definitions and expressive scope](https://github.com/lixiang90/peano-metamath-prover/blob/htps/docs/pa-plus-definitions.md)
- [PA+ random theorem generation](https://github.com/lixiang90/peano-metamath-prover/blob/htps/docs/pa-plus-random-generation.md)
- [PA+ neural training, local validation, and integration gaps](https://github.com/lixiang90/peano-metamath-prover/blob/htps/docs/pa-plus-neural-training.md)
- [Branch progress and research roadmap](https://github.com/lixiang90/peano-metamath-prover/blob/htps/docs/progress-and-roadmap.md)

## Features

- A strict separation between syntax rules and logical assertions:
  - `term`, `wff`, `BINOP`, and similar rules only construct and type-check ASTs.
  - Only assertions whose conclusions start with `|-` may participate in proof
    composition.
- A uniform `Γ ⊢ C` representation supporting partial premise discharge and
  open proof states.
- First-order unification, occurs checks, exact type checking, and Metamath
  `$d` constraints.
- Vacuous-quantifier filtering, premise subsumption, duplicate detection, and
  quality scoring based on mathematical structure.
- An encoder-decoder Transformer policy/value model using atomic formal tokens.
- A structured policy head that scores a finite set of kernel-valid candidates.
- Verifiable intermediate-lemma cut actions: prove a lemma first, then use it
  to reach the final goal.
- Optional implicit continuous-vector reasoning and a learned halting head,
  with compatibility for existing model files and checkpoints.
- Hybrid search combining traditional action enumeration, auxiliary-lemma
  construction, and PUCT/MCTS.
- AlphaZero-style MCTS replay; exhausted search budgets are treated as censored
  outcomes rather than fabricated negative rewards.
- Resumable million-example generation in gzip shards, plus approximately 100M
  parameter training with 2,304-token contexts.
- Uncompressed Metamath proof export, project-kernel replay, and external
  Metamath cross-verification.

## Repository layout

```text
formal/                  peano.mm and conservative number-theory definitions
src/metamath_generator/  Parsing, unification, composition, quality control, export
src/neural_prover/       Transformer, MCTS, Scale data and training
tests/                   Kernel, number-theory, and neural-symbolic regression tests
docs/                    Architecture, number-theory definitions, Scale experiments
```

Training data, model checkpoints, and experiment outputs are not committed to
Git. They are written to `outputs/` by default.

## Current progress

As of 2026-08-24, the project had completed a 1.1M-example dataset, 10,000
training steps for a GPT-2 Small-class model, and its first fixed-budget
closed-loop evaluation:

- Generated 1,100,000 replayable action examples, split into
  1,080,000/10,000/10,000 training/validation/test examples.
- A random audit replayed 10,000/10,000 sampled actions successfully in the
  project symbolic environment.
- Trained a 105,312,496-parameter model on an RTX 5090 for 10,000 optimizer
  steps, consuming 80,000 examples and approximately 73.04 million target tokens.
- On 5,000 held-out examples, teacher-forced token accuracy was 92.26%, while
  exact reference-action accuracy remained zero.
- On nine fixed research goals with equal search budgets, the
  uniform/heuristic/neural/hybrid policies certified 0/9, 4/9, 0/9, and 3/9
  goals respectively. All seven successful certificates passed official
  Metamath verification.
- Updating the candidate head from 50 verifiable replay records did not
  improve solve rate, but reduced pure-neural search time from 147.4 to 63.6
  seconds, a 2.32× speedup.

These results demonstrate an operational engineering loop spanning
million-example data, a 100M-class model, long contexts, candidate scoring,
MCTS replay, and dual verification. They **do not establish that the learned
policy improves proving ability**. Pure-neural search solved no goals in this
small sample, so the P1/P2 solve-rate acceptance criteria remain unmet. The
next step is to expand replay and complete the formal evaluation on 37
leakage-eligible goals with three seeds.

## Installation

For the symbolic generator only:

```bash
python -m pip install -e .
```

For Transformer training and GPU inference:

```bash
python -m pip install -e ".[neural]"
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick start

Generate quality-filtered theorems, inference rules, and open proof states:

```bash
python -m metamath_generator formal/peano.mm \
  --mode random --steps 5000 --seed 7 \
  --output-dir outputs/generated
```

Outputs include:

- `closed_theorems.jsonl`
- `inference_rules.jsonl`
- `proof_states.jsonl`
- `quality_summary.json`

Build a strictly typed neural-policy corpus:

```bash
python -m neural_prover build-corpus \
  formal/peano-number-theory.mm outputs/corpus \
  --seeds 7,11,19,23 --steps-per-seed 5000 \
  --max-state-tokens 384 --max-action-tokens 384

python -m neural_prover audit-corpus \
  formal/peano-number-theory.mm outputs/corpus \
  --output outputs/corpus/audit.json
```

Build a deduplicated million-example sharded corpus:

```bash
python -m neural_prover build-scale-corpus \
  formal/peano-number-theory.mm outputs/corpus outputs/scale-1m \
  --train-examples 990000 --validation-examples 5000 \
  --test-examples 5000 --shard-size 20000 \
  --max-state-tokens 2304 --max-action-tokens 2304 --workers 8

python -m neural_prover audit-scale-corpus \
  formal/peano-number-theory.mm outputs/scale-1m \
  --sample-size 1000 --output outputs/scale-1m/audit.json
```

Train an approximately 104M-parameter model with 2,304-token contexts:

```bash
python -m neural_prover train-scale \
  outputs/scale-1m outputs/model-104m \
  --max-steps 1000 --micro-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --initial-context-tokens 512 --context-warmup-steps 50 \
  --checkpoint-every 250 --device cuda
```

Further reading:

- [System architecture](docs/architecture.md)
- [Conservative number-theory definitions](docs/number-theory.md)
- [First million-example training and depth experiment](docs/first-large-scale-run.md)
- [First RTX 5090 scale training and closed-loop experiment](docs/first-rtx5090-closed-loop-run.md)
- [Intermediate lemmas and continuous latent reasoning](docs/lemma-and-latent-reasoning.md)
- [Current progress and research roadmap](docs/progress-and-roadmap.md)
- [Historical Scale baseline](docs/scale-baseline.md)

## Next stage: closed-loop proving

The engineering paths for P0 trustworthy evaluation, P1 finite-candidate
policy scoring, and P2 AlphaZero replay are implemented. Formal research
acceptance still requires three-seed experiments on a fixed benchmark.
When building a public benchmark, also provide the training corpus; reference
rules are written only to a private sidecar:

```bash
python -m neural_prover build-benchmark \
  formal/peano-number-theory.mm outputs/benchmark-v2.json \
  --training-corpus outputs/corpus \
  --reference-output outputs/benchmark-reference.private.json

python -m neural_prover evaluate-mcts \
  outputs/model/final.pt outputs/corpus outputs/benchmark-v2.json \
  formal/peano-number-theory.mm outputs/baselines.json \
  --policies uniform,heuristic,neural,hybrid --seeds 17,19,23 \
  --external-verifier /path/to/metamath \
  --require-external-verification
```

The `METAMATH_EXECUTABLE` environment variable can also specify the external
verifier. When external verification is required, a missing executable,
timeout, or ambiguous output prevents the result from being counted as
certified.

Upgrade an existing checkpoint to support intermediate lemmas and continuous
latent reasoning without overwriting the original files:

```bash
python -m neural_prover init-latent \
  outputs/model/final.pt outputs/corpus/tokenizer.json \
  outputs/model-latent/initial.pt outputs/model-latent/tokenizer.json

python -m neural_prover prove-decomposed \
  outputs/model-latent/initial.pt outputs/model-latent/tokenizer.json \
  outputs/benchmark-v2.json CASE_ID formal/peano-number-theory.mm \
  --simulations 100 --max-depth 20 --branching 24
```

The upgrade preserves all existing token IDs and their associated weights,
appending only lemma-action tokens and new modules. New weights still need
training on certified replay. This engineering prototype alone is not
evidence of a solve-rate improvement. See
[Intermediate lemmas and continuous latent reasoning](docs/lemma-and-latent-reasoning.md)
for the design and acceptance criteria.

The intended end-to-end proving workflow is:

```text
Goal and open premises
    ↓
Symbolic kernel enumerates legal candidate actions
    ↓
Neural policy ranks candidates, optionally after bounded latent reasoning
    ↓
Beam/MCTS exploration, execution, and backtracking
    ↓
All open premises discharged
    ↓
Compile the proof DAG and Metamath certificate
    ↓
Cross-check with the project verifier and an external Metamath implementation
```

Reinforcement learning will reward verified proofs, not reproduction of
training actions. Continuous latent reasoning is used only for internal
planning before an action. Every step that changes the formal state must
remain a discrete, recorded, verifiable Metamath action. See the
[current progress and research roadmap](docs/progress-and-roadmap.md)
for detailed stages and acceptance criteria.

## Scope of the number-theory language

`formal/peano-number-theory.mm` adds explicit conservative definitions for
primes, powers, finite sequences, rational inequalities, and finite
certificates for logarithms/Li. Fermat's Last Theorem, Goldbach's conjecture,
the Prime Number Theorem, and the von Koch formulation equivalent to the
Riemann Hypothesis are named using the `statement` type. They are neither
axioms nor proved theorems in this library. Open conjectures are never used
as successful training labels.

## Trust boundary

Neural scores are not proofs. A result is accepted only if:

1. Every action passes type and `$d` constraint checks.
2. The proof DAG compiles into a Metamath proof.
3. The project verifier can replay that proof against the original formal
   library, respecting declaration order and active scope.

The project verifier currently supports only uncompressed proofs and is not
a replacement for a mature, extensively audited general-purpose Metamath
verifier. The CLI can invoke `metamath-exe` for independent verification.
When publishing or citing important conclusions, use
`--require-external-verification` to make it a mandatory certification gate.

## Project history

Early experiments encountered mixed syntax/theorem handling, excessive
vacuous-quantifier and open-premise variants, cross-type substitutions, and
approximately 1.21% exact duplicates in a Scale corpus. Those implementations
and experimental artifacts are not included in this release directory. The
current version retains the formal pipeline with strict typing fixes,
quality filtering, and global deduplication.

## License

This project is released under [GPL-3.0](LICENSE). `formal/peano.mm` retains
Robert Solovay's original GPL copyright notice; see its file header and
[NOTICE](NOTICE).
