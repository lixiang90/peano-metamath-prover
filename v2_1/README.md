# PA+ v2.1: certified classical ATP and frozen-model orchestration

This independent package leaves all v1 (`src/neural_prover`) and v2
(`v2/pa_prover_v2`) implementations unchanged. It contains **no training command**.

The initial backend is Vampire. Supported formulas contain natural-number terms
`0,S,+,*`, equality/order atoms, conjunction, implication, universal quantifiers,
and a small explicit whitelist of quantifier-free PA+ definitions. Existentials,
formula/operator metavariables, arbitrary Skolemization and general first-order
proof import are **not** implemented. Unsupported input is reported explicitly.

Vampire receives actual first-order formulas, with free target parameters rigid
and source arithmetic axioms universally closed. It is an untrusted search
backend. We preserve its TPTP proof and used input names, then independently
reconstruct the original goal from PA axioms. This is **bounded reconstruction,
not a general TSTP step-by-step importer**: some legitimate Vampire proofs cannot
yet be reconstructed. Every helper is exported as a proved `$p`, including the
Hilbert-logic plumbing, quantifier instantiation and induction. The official C
Metamath verifier checks the entire generated DAG; a prover claim alone never
counts as success. No new mathematical axioms or human theorem library is loaded.

Four arms share an end-to-end wall-clock budget:

* `atp`: Vampire, followed by reconstruction only if Vampire claims a proof.
* `induction`: a short Vampire attempt, then conventional single-variable PA
  induction with the original predicate; no learned strengthening or lemmas.
* `model`: the existing frozen v1 checkpoint, bounded legal-candidate best-first
  search; this is not a retrained high-level policy or a reproduction of old MCTS scores.
* `hybrid`: short ATP and reconstruction first; the same frozen model handles
  remaining unsolved work. High-level learned induction/lemma synthesis remains
  future work and is not claimed by this initial integration.

Timing includes translation, search, reconstruction, export and official
verification. Source/model loading is excluded consistently. A process watchdog
stops excessive overruns; no solution completed after the budget counts. Outcomes
include unsupported, reconstruction gaps, timeout, verifier failure and unknown,
none of which asserts mathematical unprovability.

The self-generated diagnostic benchmark combines random arithmetic terms and
certified rewrites, conditional equality/Horn tasks, and conjectures obtained by
enumerating small terms and grouping their numerical evaluations. Testing only
proposes conjectures: all admitted references have an official certificate.
Reference proofs are stored separately and never passed to evaluation workers.
Generated tasks and ATP inputs do not consult the named human target statements.
The October 10 frozen checkpoint is a historical comparator: its old training
records contain named target hints. Inference disables these hints, but cannot
erase their influence from existing weights. Do not describe that checkpoint
or its hybrid arm as an independently established strict no-human-target-prior run.
If training corpora are supplied, exact and alpha-normalized conclusion overlaps
are excluded before admission. This does not exclude all common proof skeletons.
Selection favors the currently reconstructable fragment; results cannot be
generalized to arbitrary PA+, number theory, or unrestricted ATP performance.

Install the package with `pip install -e .`, or set `PYTHONPATH=src;v2;v2_1`
on Windows. Obtain Vampire from its official release and provide paths to both
executables. Do not install theorem databases with it.
On Windows, the official Vampire 5.0.0 archive supplies `cygwin1.dll` but also
requires `cygiconv-2.dll` and `cygintl-8.dll`. Keep the Cygwin runtime DLLs beside
the isolated executable; do not replace Windows system DLLs.

```text
python -m pa_prover_v21 build-benchmark formal/peano-pa-plus.mm outputs/v21-data \
  --metamath outputs/tools/metamath/metamath.exe --count 36 --seed 21012 \
  --training-corpus outputs/pa-plus-graph-rtx3090-20261010/corpus

python -m pa_prover_v21 evaluate formal/peano-pa-plus.mm outputs/v21-data outputs/v21-eval \
  --vampire outputs/tools/vampire-5.0.0/vampire.exe \
  --metamath outputs/tools/metamath/metamath.exe --seconds 10 \
  --checkpoint outputs/pa-plus-graph-rtx3090-20261010/model/final.pt \
  --tokenizer outputs/pa-plus-graph-rtx3090-20261010/corpus/tokenizer.json
```

Use fresh output directories. `cases.jsonl`, `manifest.json`, `results.jsonl`
and `report.json` preserve the frozen goals, source/tool/checkpoint hashes,
per-case outcomes, Wilson intervals and paired exact McNemar comparisons.

See the [October 12 implementation and evaluation report](../docs/v21-atp-evaluation.md)
for the frozen benchmark, results and limitations.

References: [Vampire](https://github.com/vprover/vampire),
[induction in saturation](https://arxiv.org/abs/2402.18954).
