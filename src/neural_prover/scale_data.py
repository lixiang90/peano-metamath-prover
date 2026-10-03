from __future__ import annotations

import gzip
import hashlib
import json
import math
import multiprocessing
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from metamath_generator.model import (
    Hypothesis,
    Node,
    Theorem,
    normalized_pair,
)
from metamath_generator.parser import parse
from metamath_generator.unification import substitute_simultaneous

from .data import ProverExample, load_examples
from .environment import (
    BackwardEnvironment,
    ProofState,
    Tactic,
    parse_tactic_tokens,
)
from .tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class ScaleCorpusConfig:
    train_examples: int = 990_000
    validation_examples: int = 5_000
    test_examples: int = 5_000
    shard_size: int = 20_000
    max_state_tokens: int = 2304
    max_action_tokens: int = 2304
    seed: int = 20260729
    gzip_level: int = 1
    workers: int = 8
    kernel_validation_interval: int = 1000
    max_new_records: int = 0
    depth_balanced: bool = True


@dataclass(frozen=True, slots=True)
class _Template:
    example: ProverExample
    theorem: Theorem
    tactic: Tactic


@dataclass(frozen=True, slots=True)
class _TemplatePool:
    all_templates: tuple[_Template, ...]
    by_depth: dict[int, tuple[_Template, ...]]
    depths: tuple[int, ...]


def _term(rng: random.Random, budget: int) -> Node:
    if budget <= 1:
        return Node("0")
    if budget == 2 or rng.random() < 0.38:
        return Node("S", (_term(rng, budget - 1),))
    left = max(1, (budget - 1) // 2)
    right = max(1, budget - 1 - left)
    return Node(
        rng.choice(("+", "*")),
        (_term(rng, left), _term(rng, right)),
    )


def _wff(rng: random.Random, budget: int) -> Node:
    if budget <= 5:
        return Node(
            rng.choice(("=", "<")),
            (_term(rng, 1), _term(rng, max(1, budget - 2))),
        )
    choice = rng.random()
    if choice < 0.22:
        return Node("not", (_wff(rng, budget - 1),))
    if choice < 0.72:
        left = max(3, (budget - 1) // 2)
        right = max(3, budget - 1 - left)
        return Node(
            rng.choice(("implies", "iff", "and", "or")),
            (_wff(rng, left), _wff(rng, right)),
        )
    left = max(1, (budget - 1) // 2)
    right = max(1, budget - 1 - left)
    return Node(
        rng.choice(("=", "<")),
        (_term(rng, left), _term(rng, right)),
    )


def _budget(rng: random.Random) -> int:
    value = rng.random()
    if value < 0.72:
        return rng.randint(1, 24)
    if value < 0.93:
        return rng.randint(25, 128)
    if value < 0.992:
        return rng.randint(129, 512)
    return rng.randint(513, 900)


def _instantiation(
    theorem: Theorem,
    rng: random.Random,
) -> dict[str, Node]:
    result: dict[str, Node] = {}
    for variable, typecode in theorem.variable_types.items():
        budget = _budget(rng)
        if typecode == "term":
            result[variable] = _term(rng, budget)
        elif typecode == "wff":
            result[variable] = _wff(rng, budget)
        elif typecode == "BINOP":
            result[variable] = Node(rng.choice(("+", "*")))
        elif typecode == "BINPRED":
            result[variable] = Node(rng.choice(("=", "<")))
        elif typecode == "LOGBINOP":
            result[variable] = Node(
                rng.choice(("implies", "iff", "and", "or"))
            )
        elif typecode == "QUANT":
            result[variable] = Node(rng.choice(("forall", "exists")))
        # Metamath var substitutions must remain object variables.  Keeping
        # them schematic is both sound and sufficient for term/wff diversity.
    return result


def _variables_in(
    expression: Node,
    variable_types: dict[str, str],
) -> set[str]:
    return {
        node.op
        for node in expression.walk()
        if node.op in variable_types
    }


def _instantiate_theorem(
    theorem: Theorem,
    substitution: dict[str, Node],
    name: str,
) -> Theorem:
    hypotheses = [
        Hypothesis(
            hypothesis.label,
            substitute_simultaneous(hypothesis.expr, substitution),
        )
        for hypothesis in theorem.hypotheses
    ]
    conclusion = substitute_simultaneous(
        theorem.conclusion, substitution
    )
    remaining_types = {
        variable: typecode
        for variable, typecode in theorem.variable_types.items()
        if variable not in substitution
    }
    used = {
        node.op
        for expression in [
            *(hypothesis.expr for hypothesis in hypotheses),
            conclusion,
        ]
        for node in expression.walk()
        if node.op in remaining_types
    }
    remaining_types = {
        variable: remaining_types[variable] for variable in used
    }
    constraints: set[tuple[str, str]] = set()
    for left, right in theorem.d_constraints:
        left_value = substitute_simultaneous(Node(left), substitution)
        right_value = substitute_simultaneous(Node(right), substitution)
        left_variables = _variables_in(left_value, remaining_types)
        right_variables = _variables_in(right_value, remaining_types)
        if left_variables & right_variables:
            raise ValueError("instantiation collapses a $d constraint")
        for a in left_variables:
            for b in right_variables:
                constraints.add(normalized_pair(a, b))
    return Theorem(
        name,
        hypotheses,
        conclusion,
        d_constraints=constraints,
        variable_types=remaining_types,
        kind="scale_augmented",
    )


def _instantiate_tactic(
    tactic: Tactic,
    substitution: dict[str, Node],
) -> Tactic:
    return Tactic.create(
        tactic.rule,
        {
            variable: substitute_simultaneous(value, substitution)
            for variable, value in tactic.substitution
        },
    )


def _action_tokens(
    tactic: Tactic,
    theorem: Theorem,
    tokenizer: MetamathTokenizer,
    environment: BackwardEnvironment,
) -> list[str]:
    if tactic.rule == "<ASSUMPTION>":
        return [
            "<BOS>",
            "<ACTION>",
            "<ASSUMPTION>",
            "<END_ACTION>",
            "<EOS>",
        ]
    assertion = environment.assertions[tactic.rule]
    canonical = tokenizer.canonical_variables(theorem)
    return tokenizer.assertion_tactic_tokens(
        assertion, tactic.substitution_dict(), canonical,
    )


def _templates(
    split_path: Path,
    tokenizer: MetamathTokenizer,
    database,
    environment: BackwardEnvironment,
) -> _TemplatePool:
    result: list[_Template] = []
    for example in load_examples(split_path):
        theorem = tokenizer.theorem_from_state_tokens(
            example.state_tokens,
            database,
            name=f"scale_base_{example.example_id}",
        )
        tactic = parse_tactic_tokens(
            example.action_tokens,
            theorem,
            tokenizer,
            database,
            environment=environment,
        )
        environment.apply(ProofState.from_theorem(theorem), tactic)
        if any(
            typecode in {
                "term", "wff", "BINOP", "BINPRED", "LOGBINOP", "QUANT"
            }
            for typecode in theorem.variable_types.values()
        ):
            result.append(_Template(example, theorem, tactic))
    if not result:
        raise ValueError(f"no augmentable templates in {split_path}")
    by_depth: dict[int, list[_Template]] = {}
    for template in result:
        by_depth.setdefault(
            template.example.proof_depth, []
        ).append(template)
    return _TemplatePool(
        tuple(result),
        {
            depth: tuple(templates)
            for depth, templates in by_depth.items()
        },
        tuple(sorted(by_depth)),
    )


def _seed_for(
    seed: int,
    split: str,
    index: int,
    attempt: int,
) -> int:
    digest = hashlib.sha256(
        f"{seed}:{split}:{index}:{attempt}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _length_bucket(length: int) -> str:
    for bound in (64, 128, 256, 512, 1024, 2048, 2304):
        if length <= bound:
            return f"<= {bound}"
    return "> 2304"


def _one_record(
    split: str,
    index: int,
    templates: _TemplatePool,
    tokenizer: MetamathTokenizer,
    environment: BackwardEnvironment,
    config: ScaleCorpusConfig,
) -> dict:
    for attempt in range(128):
        rng = random.Random(
            _seed_for(config.seed, split, index, attempt)
        )
        if config.depth_balanced:
            depth = templates.depths[rng.randrange(len(templates.depths))]
            depth_templates = templates.by_depth[depth]
            template = depth_templates[rng.randrange(len(depth_templates))]
        else:
            template = templates.all_templates[
                rng.randrange(len(templates.all_templates))
            ]
        substitution = _instantiation(template.theorem, rng)
        theorem = _instantiate_theorem(
            template.theorem,
            substitution,
            f"scale_{split}_{index}",
        )
        tactic = _instantiate_tactic(template.tactic, substitution)
        try:
            if (
                config.kernel_validation_interval > 0
                and index % config.kernel_validation_interval == 0
            ):
                environment.apply(
                    ProofState.from_theorem(theorem), tactic
                )
            state_ids = tokenizer.encode(
                tokenizer.state_tokens(theorem)
            )
            action_ids = tokenizer.encode(
                _action_tokens(
                    tactic, theorem, tokenizer, environment
                )
            )
        except (KeyError, ValueError):
            continue
        if (
            len(state_ids) > config.max_state_tokens
            or len(action_ids) > config.max_action_tokens
        ):
            continue
        payload = (
            bytes(str(state_ids), "ascii")
            + b"|"
            + bytes(str(action_ids), "ascii")
        )
        return {
            "id": hashlib.sha256(payload).hexdigest()[:20],
            "state": state_ids,
            "action": action_ids,
            "value": template.example.value_target,
            "difficulty": template.example.difficulty,
            "base": template.example.example_id,
            "proof_depth": template.example.proof_depth,
            "generation_kind": template.example.generation_kind,
            "definition_support": list(template.example.definition_support),
        }
    raise RuntimeError(
        f"could not generate a valid {split} record at index {index}"
    )


_WORKER_CONTEXT: dict = {}


def _initialize_worker(
    database_path: str,
    base_corpus: str,
    config_payload: dict,
) -> None:
    database = parse(database_path)
    base = Path(base_corpus)
    tokenizer = MetamathTokenizer.load(base / "tokenizer.json")
    environment = BackwardEnvironment(database)
    environment.configure_from_tokenizer(tokenizer)
    _WORKER_CONTEXT.update({
        "tokenizer": tokenizer,
        "environment": environment,
        "config": ScaleCorpusConfig(**config_payload),
        "templates": {
            split: _templates(
                base / f"{split}.jsonl",
                tokenizer,
                database,
                environment,
            )
            for split in ("train", "validation", "test")
        },
    })


def _worker_record(item: tuple[str, int]) -> dict:
    split, index = item
    return _one_record(
        split,
        index,
        _WORKER_CONTEXT["templates"][split],
        _WORKER_CONTEXT["tokenizer"],
        _WORKER_CONTEXT["environment"],
        _WORKER_CONTEXT["config"],
    )


def _write_manifest(output: Path, manifest: dict) -> None:
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_scale_corpus(
    database_path: str | Path,
    base_corpus_directory: str | Path,
    output_directory: str | Path,
    config: ScaleCorpusConfig | None = None,
) -> dict:
    """Build or resume a compact million-record, sharded corpus."""

    cfg = config or ScaleCorpusConfig()
    if cfg.shard_size <= 0:
        raise ValueError("shard_size must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    targets = {
        "train": cfg.train_examples,
        "validation": cfg.validation_examples,
        "test": cfg.test_examples,
    }
    stable_config = {
        key: value
        for key, value in asdict(cfg).items()
        if key not in {"max_new_records", "workers"}
    }
    if manifest_path.exists():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest["configuration"] != stable_config:
            raise ValueError(
                "existing scale corpus configuration does not match"
            )
        if manifest.get("complete"):
            return manifest
    else:
        manifest = {
            "format": "peano-scale-corpus-v3",
            "database": str(Path(database_path).resolve()),
            "base_corpus": str(
                Path(base_corpus_directory).resolve()
            ),
            "configuration": stable_config,
            "targets": targets,
            "counts": {split: 0 for split in targets},
            "next_candidate_index": {
                split: 0 for split in targets
            },
            "shards": {split: [] for split in targets},
            "length_histogram": {},
            "proof_depth_histogram": {
                split: {} for split in targets
            },
            "complete": False,
            "leakage_control": (
                "each split is augmented only from templates already "
                "assigned to that conclusion-family split"
            ),
        }
        _write_manifest(output, manifest)
    manifest.setdefault(
        "next_candidate_index",
        {
            split: int(manifest["counts"][split])
            for split in targets
        },
    )
    manifest.setdefault(
        "proof_depth_histogram",
        {split: {} for split in targets},
    )

    base = Path(base_corpus_directory)
    tokenizer = MetamathTokenizer.load(base / "tokenizer.json")
    tokenizer.save(output / "tokenizer.json")
    seen_ids: set[str] = set()
    for split in targets:
        for shard in manifest["shards"][split]:
            with gzip.open(
                output / shard["path"],
                "rt",
                encoding="utf-8",
            ) as stream:
                for line in stream:
                    if line.strip():
                        seen_ids.add(str(json.loads(line)["id"]))
    pool = None
    if cfg.workers > 1:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            processes=cfg.workers,
            initializer=_initialize_worker,
            initargs=(
                str(Path(database_path).resolve()),
                str(base.resolve()),
                asdict(cfg),
            ),
        )
        split_templates = None
        environment = None
    else:
        database = parse(database_path)
        environment = BackwardEnvironment(database)
        environment.configure_from_tokenizer(tokenizer)
        split_templates = {
            split: _templates(
                base / f"{split}.jsonl",
                tokenizer,
                database,
                environment,
            )
            for split in targets
        }
    remaining_this_call = (
        cfg.max_new_records
        if cfg.max_new_records > 0
        else sum(
            targets[split] - manifest["counts"][split]
            for split in targets
        )
    )
    try:
        for split, target in targets.items():
            directory = output / split
            directory.mkdir(exist_ok=True)
            while (
                manifest["counts"][split] < target
                and remaining_this_call > 0
            ):
                start = manifest["counts"][split]
                count = min(
                    cfg.shard_size,
                    target - start,
                    remaining_this_call,
                )
                part = len(manifest["shards"][split])
                path = directory / f"part-{part:05d}.jsonl.gz"
                histogram: dict[str, int] = {}
                depth_histogram: dict[str, int] = {}
                accepted = 0
                with gzip.open(
                    path,
                    "wt",
                    encoding="utf-8",
                    compresslevel=cfg.gzip_level,
                ) as stream:
                    while accepted < count:
                        remaining = count - accepted
                        candidate_count = max(
                            128,
                            int(math.ceil(remaining * 1.03)),
                        )
                        candidate_start = int(
                            manifest["next_candidate_index"][split]
                        )
                        candidate_stop = (
                            candidate_start + candidate_count
                        )
                        manifest["next_candidate_index"][split] = (
                            candidate_stop
                        )
                        indices = (
                            (split, index)
                            for index in range(
                                candidate_start,
                                candidate_stop,
                            )
                        )
                        if pool is not None:
                            records = pool.imap(
                                _worker_record,
                                indices,
                                chunksize=16,
                            )
                        else:
                            records = (
                                _one_record(
                                    item[0],
                                    item[1],
                                    split_templates[item[0]],  # type: ignore[index]
                                    tokenizer,
                                    environment,  # type: ignore[arg-type]
                                    cfg,
                                )
                                for item in indices
                            )
                        for record in records:
                            record_id = str(record["id"])
                            if record_id in seen_ids:
                                continue
                            seen_ids.add(record_id)
                            length = max(
                                len(record["state"]),
                                len(record["action"]),
                            )
                            bucket = _length_bucket(length)
                            histogram[bucket] = (
                                histogram.get(bucket, 0) + 1
                            )
                            proof_depth = str(
                                int(record.get("proof_depth", 0))
                            )
                            depth_histogram[proof_depth] = (
                                depth_histogram.get(proof_depth, 0) + 1
                            )
                            stream.write(
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            accepted += 1
                            if accepted == count:
                                break
                manifest["counts"][split] += count
                manifest["shards"][split].append({
                    "path": str(path.relative_to(output)),
                    "records": count,
                    "bytes": path.stat().st_size,
                    "proof_depth_histogram": dict(sorted(
                        depth_histogram.items(),
                        key=lambda item: int(item[0]),
                    )),
                })
                for bucket, value in histogram.items():
                    manifest["length_histogram"][bucket] = (
                        manifest["length_histogram"].get(bucket, 0)
                        + value
                    )
                split_depths = manifest["proof_depth_histogram"][split]
                for depth, value in depth_histogram.items():
                    split_depths[depth] = split_depths.get(depth, 0) + value
                remaining_this_call -= count
                _write_manifest(output, manifest)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    manifest["total"] = sum(manifest["counts"].values())
    manifest["complete"] = all(
        manifest["counts"][split] == target
        for split, target in targets.items()
    )
    manifest["maximum_proof_depth"] = max(
        (
            int(depth)
            for histogram in manifest["proof_depth_histogram"].values()
            for depth, count in histogram.items()
            if count > 0
        ),
        default=0,
    )
    _write_manifest(output, manifest)
    return manifest


def iter_scale_records(
    corpus_directory: str | Path,
    split: str,
):
    corpus = Path(corpus_directory)
    manifest = json.loads(
        (corpus / "manifest.json").read_text(encoding="utf-8")
    )
    for shard in manifest["shards"][split]:
        with gzip.open(
            corpus / shard["path"],
            "rt",
            encoding="utf-8",
        ) as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
