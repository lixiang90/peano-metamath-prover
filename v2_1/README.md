# PA+ v2.1: certified ATP and high-level neural proof strategies

This independent package leaves all v1 (`src/neural_prover`) and v2
(`v2/pa_prover_v2`) implementations unchanged. The historical frozen-model
comparison is retained. New commands implement a separate macro policy,
self-generated curriculum, NTP/SFT, expert iteration and RLVR. No command starts
training implicitly; training and expert iteration require explicit invocation.

## Optional structured feedback and scratchpads (B/C)

`--reasoning-mode baseline` keeps the original A policy and search as the default.
`feedback` (B) supplies role-delimited, continuous ENGINE/MODEL transcripts with
real execution feedback. `scratchpad` (C) also offers one-macro TRY trials,
COMMIT/BACKTRACK and bounded attempt memory. A rollback restores the parent scope;
attempt records never become facts. Trial closure defaults to 0.25 seconds,
at most 4 trials and 24 interaction commands per trajectory, all charged to the
same total proof budget. Source axioms and mathematical action proposals are unchanged.

For B/C, NTP trains on whole successful interaction streams. SFT includes a
MODEL-only trajectory loss, multi-positive candidate loss, and value examples.
RLVR inherits the corpus mode and accounts for every sampled control command.
Official root verification remains the reward gate. Oversized complete histories
are reported rather than truncated. The implementation still scores bounded
commands; arbitrary formula generation and recursive latent reasoning are deferred.

Generation, proof and expert-iteration commands accept `--reasoning-mode`.
Training reads the mode from the corpus manifest. Evaluation accepts suffixed
arms such as `heuristic-a,heuristic-b,heuristic-c` and `learned-a,learned-b,learned-c`,
with `--checkpoint`, `--checkpoint-feedback`, `--checkpoint-scratchpad` for separate
weights. All arms use the same frozen cases/library and end-to-end wall budget.
See [the protocol, audit rules and commands](../docs/v21-symbolic-scratchpad.md).

## High-level strategy workflow

Classical proof-producing closure alternates with a bounded high-level search.
The available actions are `INDUCT` (optionally generalizing object parameters),
`HAVE`, `STRENGTHEN`, `USE` and `UNFOLD`. States contain **all** remaining goals,
their local assumptions, types, distinct-variable constraints and remaining
budget. HAVE continuations stay blocked until the lemma is proved. A strengthened
statement creates two obligations: prove it, and prove that it implies the old
goal. No proposal becomes an axiom. Each branch owns its fresh-variable context.

The initial policy is a fresh RoPE/RMSNorm/SwiGLU autoregressive Transformer with
a budget-conditioned value head and fixed UTF-8 byte vocabulary. Candidate
continuations share a prompt KV cache. The decoder assigns sequence likelihoods
to a **bounded typed action grammar**; a small uniform exploration mixture is
included in the actual categorical policy. The current implementation does not
generate arbitrary unbounded auxiliary formulas. Lemmas are retrieved as text
from a frozen, proof-carrying symbolic library. A bidirectional lemma encoder and
Loop Transformer are deferred until this baseline is measured.

Data generation combines existing random proof graphs and small arithmetic
conjectures grouped by finite evaluations. Numeric agreement proposes a problem;
only Metamath verification proves it. Expression-size strata retain bootstrap
problems. Cases are split before teacher searches, grouping normalized statements,
conclusions, formula skeletons and graph ancestry. Unsolved proposals remain in
the evaluation population. This guards the recorded families, not every possible
semantic equivalence or common proof pattern.

The teacher records complete macro trajectories. Verified alternatives at the
same state form multi-positive targets. Equal-budget forced-action trials record
the marginal gain over classical closure, including the cost of proving a new
lemma. Randomized rollouts separately estimate initial-state completion frequency;
verifier failures and context overflow are censored. A timeout is zero **only for
the event “this solver completes within this budget”**, never a proof of falsity.
The first round estimates the random policy; model-driven rounds recalibrate on
the frozen learned policy. This value head is a heuristic, not a proof oracle.

NTP uses ordinary next-token cross entropy over complete synthetic sequences.
SFT combines multi-positive categorical likelihood, action-token likelihood and
separately sampled value examples. Sampling weights balance observed difficulty,
operator structure and premise count. RLVR samples one trajectory per rollout,
uses official complete-proof rewards and stores the exact action distribution.
Its clipped policy objective and reference KL use that same candidate set and
temperature. Best-first teacher traces cannot be used as on-policy PPO samples.
Zero-advantage groups produce no policy update. Context overflow is reported;
unsolved goals are never silently truncated.

All commands below use fresh output directories. They are examples, **not** a
recommendation to start a full training run without first auditing a pilot corpus.
PowerShell accepts each example as one line.

```text
python -m pa_prover_v21 generate-strategy formal/peano-pa-plus.mm outputs/v21-strategy --metamath outputs/tools/metamath/metamath.exe --count 24 --seconds 10 --trials 4 --counterfactuals 4
python -m pa_prover_v21 audit-strategy formal/peano-pa-plus.mm outputs/v21-strategy --metamath outputs/tools/metamath/metamath.exe
python -m pa_prover_v21 train-policy formal/peano-pa-plus.mm outputs/v21-strategy outputs/v21-ntp --stage ntp --steps 100 --device cuda
python -m pa_prover_v21 train-policy formal/peano-pa-plus.mm outputs/v21-strategy outputs/v21-sft --stage sft --steps 100 --device cuda --checkpoint outputs/v21-ntp/policy.pt
python -m pa_prover_v21 rlvr formal/peano-pa-plus.mm outputs/v21-strategy outputs/v21-rlvr --metamath outputs/tools/metamath/metamath.exe --checkpoint outputs/v21-sft/policy.pt --groups 10 --group-size 4 --seconds 10 --device cuda
python -m pa_prover_v21 evaluate-strategy formal/peano-pa-plus.mm outputs/v21-strategy outputs/v21-comparison --metamath outputs/tools/metamath/metamath.exe --vampire outputs/tools/vampire-5.0.0/vampire.exe --checkpoint outputs/v21-sft/policy.pt --methods atp,induction,heuristic,random,learned --split test --seconds 10 --device cuda
python -m pa_prover_v21 expert-iterate formal/peano-pa-plus.mm outputs/v21-round2 --metamath outputs/tools/metamath/metamath.exe --checkpoint outputs/v21-sft/policy.pt --count 24 --steps 100 --seconds 10 --device cuda
python -m pa_prover_v21 promote-library formal/peano-pa-plus.mm outputs/v21-strategy outputs/v21-library --metamath outputs/tools/metamath/metamath.exe
```

`generate-strategy --checkpoint` uses both heuristic and frozen learned teachers.
`expert-iterate` performs one generation/audit/SFT/library-promotion round. Pass
`--library` explicitly to the next generation round to accumulate prior proved
training lemmas. Promotion never reads validation/test proofs. Old split
assignments persist in checkpoint provenance; conflicting ancestry or training
overlap is rejected. Evaluation freezes the library and checks checkpoint overlap.
Tiny grouped corpora may have empty validation/test splits: enlarge the proposal
pool, rather than moving training examples into the test set.

`prove-strategy SOURCE CASE.json OUTPUT --metamath PATH` accepts one serialized
case in the `cases.jsonl` schema, with optional `--policy learned --checkpoint`.
`evaluate-strategy` compares classical-only closure, conventional induction, and
heuristic/random/learned macro policies under the same wall budget. The latter
three share the action generator, limits and library. Verification and export
count toward the budget; workers have an external watchdog. The legacy `evaluate`
command retains the old frozen-v1 comparison with its original meaning.
Classical-only and conventional-induction arms also retrieve and replay the same
certified library, so the learned arm has no exclusive historical knowledge.

Checkpoints include optimizer state for inspection, but `--checkpoint` currently
warm-starts weights with a fresh optimizer; exact optimizer-resume is not claimed.
Artifacts (`live.json`, traces, certificates, checksums and reports) stay in
ignored `outputs/`. No generated theorem, model weight or executable is committed.

## Restricted ATP bridge and historical benchmark

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

The historical `evaluate` command has four arms sharing an end-to-end budget:

* `atp`: Vampire, followed by reconstruction only if Vampire claims a proof.
* `induction`: a short Vampire attempt, then conventional single-variable PA
  induction with the original predicate; no learned strengthening or lemmas.
* `model`: the existing frozen v1 checkpoint, bounded legal-candidate best-first
  search; this is not a retrained high-level policy or a reproduction of old MCTS scores.
* `hybrid`: short ATP and reconstruction first; the same frozen model handles
  remaining unsolved work. The separate macro workflow above is the new
  high-level policy; this historical arm still uses the unchanged frozen adapter.

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
