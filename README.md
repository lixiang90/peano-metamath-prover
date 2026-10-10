[![English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-lightgrey)](README-zh.md)

# Peano Metamath Prover

A research toolkit for synthetic theorem generation and neural-symbolic proving over
Peano arithmetic and conservative PA+ definitions. Training data comes from constructed
formal proofs, not natural-language reasoning traces. Neural models propose or rank
actions; the symbolic kernel checks them and replays the final Metamath certificate.

PA+, MCTS, HTPS, intermediate lemmas, and continuous latent reasoning are maintained
together on `main`. No algorithm-specific branch or separate HTPS installation is needed.
This is a Beta research system: the pipelines work, but improved theorem-solving ability
from learning has not yet been established.

The isolated [v2](v2/README.md) adds RoPE decoder-only models, a causal lemma
encoder, a verified theorem library, explicit scratchpad agents, standard next-token
pretraining, and RLVR. It keeps the same PA+ kernel and does not migrate v1 checkpoints.

Data generation now defaults to random proof graphs shared by v1, HTPS, Scale, and
v2: compose source or derived rules through automatic unification, with a 10%
auxiliary instantiation probability. Scale generates fresh graphs instead of
expanding a fixed template bank. See the [algorithm and commands](docs/random-proof-graphs.md).
An optional [generic Metamath token sampler](docs/generic-metamath-generation.md)
is available with `--mode generic`; the default algorithm and training pipelines remain unchanged.

## 1. What the system contains

| Layer | Implementation | Role |
| --- | --- | --- |
| Shared formal system | `formal/`, `metamath_generator` | PA / number theory / PA+; parsing, unification, types, `$d`, proof construction |
| Synthetic data | `metamath_generator`, `htps_prover.forward` | Quality-filtered theorems, bounded instances, definition bridges, proof DAG datasets |
| Models and training | `neural_prover`, `htps_prover.training` | Symbolic tokens, policy/value heads, optional lemma actions and continuous latent computation |
| Search backends | `neural_prover`, `htps_prover.hypergraph` | Best-first / PUCT-MCTS over multi-goal states, or HTPS over shared AND/OR goal nodes |
| Certification | Shared environment and certificate verifier | Replay against the source library; optional mandatory external Metamath verification |
| V2 proof agent | `v2/pa_prover_v2` | Causal LM with in-context lemma vectors; NTP/SFT/RLVR and verified scratchpad actions |

MCTS and HTPS share mathematical semantics, but are not just different numeric parameters:
their search structures and replay targets differ. Existing CLI and dataset formats remain
separate selectable workflows. PA+ catalog-driven generation is wired into `build-corpus`;
it is also wired into HTPS `generate`, with policy/lemma action audits and training.
The Scale generator also replays PA+ templates with certified bridges. See the
[workflow guide](docs/training-and-evaluation.md) for the integration matrix and commands.

## 2. Formal scope and trust

The libraries form a hierarchy:

- `formal/peano.mm`: the base Peano arithmetic library.
- `formal/peano-number-theory.mm`: conservative number-theory definitions and named targets.
- `formal/peano-pa-plus.mm`: 68 additional high-level definitions and 35 closed target
  formulas, generated from `formal/pa-plus-definitions.json`.

PA+ covers arithmetic relations, finite recurrences, integers/rationals, and finite
certificates for elementary analysis. It is not a general real-analysis library.
`statement` names a formula without asserting it: the 35 targets are neither extra axioms
nor a list of solved theorems. Target-guided sampling does not prove its targets.

Definition compilation checks dependencies, free variables, and freshness constraints.
The 136 fold/unfold bridges are proof-backed search macros, inlined into source-library
rules in the final certificate. Proposed lemmas must be proved before use; latent vectors
never create facts. See [PA+ definitions](docs/pa-plus-definitions.md) and
[architecture](docs/architecture.md).

The project verifier currently supports uncompressed proofs. Important results should
also pass an external Metamath implementation. A neural score, a solved search node, or
high token accuracy is not a proof.

## 3. Install and run

Use Python 3.10+ from the repository root. Symbolic generation needs no PyTorch:

```bash
python -m pip install -e .
python -m metamath_generator formal/peano.mm \
  --mode random --steps 100 --seed 7 --output-dir outputs/smoke
```

For neural training and the complete Python test suite:

```bash
python -m pip install -e ".[neural,dev]"
python -m pytest
```

If PyTorch is already installed, no reinstall is needed for source-tree testing:

```powershell
$env:PYTHONPATH='src;v2'
python -m pytest
```

Bash examples use `\` for line continuation; in PowerShell use a backtick or a single line.
External-verifier integration tests require a separate Metamath executable; a skipped test
does not establish external certification. These tests do not start GPU training.
Outputs and checkpoints are ignored by Git.

Try shared PA+ generation with bounded natural numbers and target guidance:

```bash
python -m metamath_generator formal/peano-pa-plus.mm \
  --mode random --steps 1000 --seed 7 \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 --max-definition-only-search-per-predicate 1 \
  --output-dir outputs/pa-plus-guided
```

Outputs include `closed_theorems.jsonl`, `inference_rules.jsonl`, `proof_states.jsonl`, and
quality reports. Continue with [training and evaluation](docs/training-and-evaluation.md)
for PA+ corpus audit/training, MCTS baselines, or HTPS supervised/closed-loop experiments.

## 4. Evaluation and current evidence

The primary metric is certified solve rate under a stated compute budget, not action-token
accuracy. Keep reference proofs out of search inputs and check training overlap. Compare
uniform, heuristic, neural, and hybrid policies with the same goals, permitted rules,
seeds, and budgets; report certificate status and environment fingerprints. Equal
MCTS/HTPS simulation counts are not equal compute budgets.

`neural_prover evaluate` / `evaluate-mcts` and HTPS `evaluate` / `collect` / `closed-loop`
support `--external-verifier /path/to/metamath --require-external-verification`.
When required, missing, failed, timed-out, or ambiguous external verification does not
count as certified. Keep `internal_only` and `internal+external` results separate.
Generation guidance, model target hints, and inference ordering have separate ablations;
see [certification and fair evaluation](docs/pa-plus-certification.md).

Evidence as of 2026-09-04:

- Certification: a regenerated 616-example PA+ corpus passed full action replay;
  three definition-bridge certificates and an HTPS sanity target passed external replay.
  Old PA+ bridge actions used unstable variable slots and must be regenerated with
  `sorted-v1`; historical checkpoints are preserved, not silently repaired.
- Historical Scale/MCTS run: 1.1M generated records, a 105M-parameter model, and 10,000
  optimizer steps; only 80,000 records were consumed. On nine fixed goals,
  uniform/heuristic/neural/hybrid solved 0/9, 4/9, 0/9, and 3/9; all seven successful
  attempts passed external verification. This was not a PA+ or HTPS training run.
- PA+ GPU and latent/HTPS work remain engineering validation. There is no demonstrated
  solve-rate improvement from learning, no 35-target success result, and no completed
  large-scale PA+/HTPS experiment.

PA+ → HTPS data/training and base-checkpoint fine-tuning are now connected. Next: build
held-out target families and run multi-seed controlled evaluations before scaling up. Evidence and
acceptance criteria are in the [progress and roadmap](docs/progress-and-roadmap.md).

## 5. Repository and documentation

```text
formal/                  Shared PA, number theory, PA+ library and definition catalog
src/metamath_generator/  Formal kernel, conservative definitions, synthetic generation
src/neural_prover/       Models, MCTS, Scale, shared environments and certification
src/htps_prover/         Forward DAG data, HTPS, latent training and synchronous replay
tests/                   Shared-kernel and both-backend regression tests
tools/                   Verification utilities
docs/                    Guides, designs, audits, and historical experiment reports
outputs/                 Local data/checkpoints/reports (not tracked)
```

The [documentation index](docs/README.md) is organized by task rather than branch.
Most detailed design documents are currently in Chinese.

- Run experiments: [training and evaluation](docs/training-and-evaluation.md).
- Understand the implementation: [architecture](docs/architecture.md),
  [HTPS](docs/htps-design.md), [lemma and latent reasoning](docs/lemma-and-latent-reasoning.md).
- Work on PA+: [definitions](docs/pa-plus-definitions.md),
  [generation](docs/pa-plus-random-generation.md), [neural pipeline](docs/pa-plus-neural-training.md).
- Review results and plans: [roadmap](docs/progress-and-roadmap.md),
  [formal-system evolution](docs/formal-system-evolution.md).

The historical `htps` branch is retained, but new development belongs on `main` or short-lived
feature branches. Old commands remain available; see the workflow guide for migrating an
installation named `peano-metamath-prover-htps`. Contribution rules are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## 6. License

[GPL-3.0](LICENSE). `formal/peano.mm` retains Robert Solovay's original copyright notice;
see [NOTICE](NOTICE).
