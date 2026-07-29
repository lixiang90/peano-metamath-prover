from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from metamath_generator.parser import parse

from .certificate import compile_certificate, verify_certificate
from .data import load_examples
from .environment import BackwardEnvironment, ProofState, Tactic
from .hybrid import HybridActionGenerator
from .mcts import MCTSExperience, MCTSResult
from .mcts import MCTSConfig, ProofMCTS
from .model import ProofTransformer
from .search import HybridPolicy, TransformerPolicy
from .tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class ReplayExample:
    state_ids: tuple[int, ...]
    action_ids: tuple[tuple[int, ...], ...]
    policy_target: tuple[float, ...]
    value_target: float

    def to_record(self) -> dict:
        return {
            "state_ids": list(self.state_ids),
            "action_ids": [list(ids) for ids in self.action_ids],
            "policy_target": list(self.policy_target),
            "value_target": self.value_target,
        }

    @classmethod
    def from_record(cls, record: dict) -> "ReplayExample":
        return cls(
            tuple(record["state_ids"]),
            tuple(tuple(ids) for ids in record["action_ids"]),
            tuple(record["policy_target"]),
            float(record["value_target"]),
        )


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.examples: list[ReplayExample] = []

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _tactic_tokens(
        tactic: Tactic,
        state: ProofState,
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
        theorem = state.as_theorem()
        canonical = tokenizer.canonical_variables(theorem)
        assertion = environment.assertions[tactic.rule]
        order = [
            floating.expr.args[0].op
            for floating in assertion.floating
        ] or list(assertion.variable_types)
        return tokenizer.tactic_tokens(
            tactic.rule,
            tactic.substitution_dict(),
            canonical,
            variable_order=order,
        )

    def add_mcts(
        self,
        result: MCTSResult,
        tokenizer: MetamathTokenizer,
        environment: BackwardEnvironment,
        *,
        max_state_tokens: int | None = None,
        max_action_tokens: int | None = None,
    ) -> int:
        added = 0
        for experience in result.experiences:
            theorem = experience.state.as_theorem()
            state_ids = tuple(tokenizer.encode(
                tokenizer.state_tokens(theorem)
            ))
            if (
                max_state_tokens is not None
                and len(state_ids) > max_state_tokens
            ):
                continue
            action_ids: list[tuple[int, ...]] = []
            probabilities: list[float] = []
            for tactic, probability in zip(
                experience.tactics,
                experience.visit_probabilities,
            ):
                try:
                    tokens = self._tactic_tokens(
                        tactic,
                        experience.state,
                        tokenizer,
                        environment,
                    )
                    ids = tuple(tokenizer.encode(tokens))
                except (KeyError, ValueError):
                    continue
                if (
                    max_action_tokens is not None
                    and len(ids) > max_action_tokens
                ):
                    continue
                action_ids.append(ids)
                probabilities.append(probability)
            total = sum(probabilities)
            if not action_ids or total <= 0:
                continue
            replay = ReplayExample(
                state_ids,
                tuple(action_ids),
                tuple(value / total for value in probabilities),
                experience.value_target,
            )
            self.examples.append(replay)
            added += 1
        if len(self.examples) > self.capacity:
            del self.examples[:-self.capacity]
        return added

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            for example in self.examples:
                stream.write(
                    json.dumps(example.to_record()) + "\n"
                )

    @classmethod
    def load(
        cls,
        path: str | Path,
        capacity: int = 50_000,
    ) -> "ReplayBuffer":
        buffer = cls(capacity)
        buffer.examples = [
            ReplayExample.from_record(json.loads(line))
            for line in Path(path).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ][-capacity:]
        return buffer


@dataclass(frozen=True, slots=True)
class ReinforcementConfig:
    epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    value_loss_weight: float = 1.0
    gradient_clip: float = 1.0
    gradient_accumulation: int = 8
    seed: int = 7
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class ReplayCollectionConfig:
    examples: int = 32
    simulations: int = 100
    max_depth: int = 16
    branching: int = 32
    capacity: int = 50_000
    seed: int = 7
    device: str = "auto"


def _curriculum_sample(candidates, count: int, seed: int):
    """Choose a reproducible easy-heavy mixture for early self-play."""

    randomizer = random.Random(seed)
    buckets = {
        difficulty: [
            example for example in candidates
            if example.difficulty == difficulty
        ]
        for difficulty in ("easy", "medium", "hard")
    }
    for examples in buckets.values():
        randomizer.shuffle(examples)
        examples.sort(key=lambda item: (
            item.proof_depth,
            len(item.state_tokens),
        ))
    schedule = ("easy", "easy", "medium", "easy", "hard")
    selected = []
    cursors = {key: 0 for key in buckets}
    while len(selected) < count:
        progressed = False
        for difficulty in schedule:
            cursor = cursors[difficulty]
            if cursor >= len(buckets[difficulty]):
                continue
            selected.append(buckets[difficulty][cursor])
            cursors[difficulty] += 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def collect_replay_from_corpus(
    checkpoint: str | Path,
    corpus_directory: str | Path,
    database_path: str | Path,
    output_path: str | Path,
    config: ReplayCollectionConfig | None = None,
) -> dict:
    """Run MCTS only on training states and save verified RL experience."""

    cfg = config or ReplayCollectionConfig()
    random.seed(cfg.seed)
    device = _device(cfg.device)
    database = parse(database_path)
    corpus = Path(corpus_directory)
    tokenizer = MetamathTokenizer.load(corpus / "tokenizer.json")
    candidates = [
        example for example in load_examples(corpus / "train.jsonl")
        if example.origin == "synthetic"
    ]
    candidates = _curriculum_sample(
        candidates, cfg.examples, cfg.seed
    )
    model, _ = ProofTransformer.load_checkpoint(
        checkpoint,
        map_location=device,
    )
    environment = BackwardEnvironment(database)
    neural = TransformerPolicy(
        model,
        tokenizer,
        environment,
        device,
    )
    policy = HybridPolicy(environment, neural)
    prover = ProofMCTS(
        environment,
        HybridActionGenerator(environment),
        policy,
        MCTSConfig(
            simulations=cfg.simulations,
            max_depth=cfg.max_depth,
            branching=cfg.branching,
            seed=cfg.seed,
        ),
    )
    replay = ReplayBuffer(cfg.capacity)
    attempted = 0
    solved = 0
    rejected_certificates = 0
    added = 0
    by_difficulty: dict[str, dict[str, int]] = {}
    episodes: list[dict] = []
    for example in candidates:
        theorem = tokenizer.theorem_from_state_tokens(
            example.state_tokens,
            database,
            name=f"rl_{example.example_id}",
        )
        result = prover.prove(ProofState.from_theorem(theorem))
        attempted += 1
        certified = False
        certificate_error: str | None = None
        if result.search.solved:
            try:
                certificate = compile_certificate(
                    theorem,
                    result.search,
                    database,
                    name=f"rl_cert_{example.example_id}",
                )
                verify_certificate(certificate, database)
                certified = True
                solved += 1
            except Exception as exc:
                # An unverified success must never become a positive reward.
                rejected_certificates += 1
                certificate_error = str(exc)
        if certified or not result.search.solved:
            added += replay.add_mcts(
                result,
                tokenizer,
                environment,
                max_state_tokens=model.config.max_state_tokens,
                max_action_tokens=model.config.max_action_tokens,
            )
        bucket = by_difficulty.setdefault(
            example.difficulty,
            {"attempted": 0, "certified": 0},
        )
        bucket["attempted"] += 1
        bucket["certified"] += int(certified)
        episodes.append({
            "example_id": example.example_id,
            "difficulty": example.difficulty,
            "proof_depth": example.proof_depth,
            "source_rule": example.rule,
            "certified": certified,
            "certificate_error": certificate_error,
            "simulations": result.search.simulations,
            "nodes": result.search.nodes,
            "experiences": len(result.experiences),
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    replay.save(output)
    summary = {
        "format": "peano-mcts-replay-v1",
        "configuration": asdict(cfg),
        "device": str(device),
        "data_policy": (
            "only synthetic examples from train.jsonl; validation, test, "
            "benchmark, and famous frontier statements are excluded"
        ),
        "attempted": attempted,
        "certified": solved,
        "rejected_certificates": rejected_certificates,
        "replay_examples": len(replay),
        "added": added,
        "by_difficulty": by_difficulty,
        "episodes": episodes,
        "replay": str(output.resolve()),
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def reinforce_model(
    checkpoint: str | Path,
    replay: ReplayBuffer,
    output_checkpoint: str | Path,
    config: ReinforcementConfig | None = None,
) -> dict:
    """Train policy on MCTS visits and value on verified search outcomes."""

    cfg = config or ReinforcementConfig()
    if not replay.examples:
        raise ValueError("replay buffer is empty")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _device(cfg.device)
    model, payload = ProofTransformer.load_checkpoint(
        checkpoint,
        map_location=device,
    )
    model.to(device)
    compatible: list[ReplayExample] = []
    skipped = 0
    for example in replay.examples:
        if len(example.state_ids) > model.config.max_state_tokens:
            skipped += 1
            continue
        pairs = [
            (ids, probability)
            for ids, probability in zip(
                example.action_ids, example.policy_target
            )
            if len(ids) <= model.config.max_action_tokens
        ]
        total = sum(probability for _, probability in pairs)
        if not pairs or total <= 0:
            skipped += 1
            continue
        compatible.append(ReplayExample(
            example.state_ids,
            tuple(ids for ids, _ in pairs),
            tuple(probability / total for _, probability in pairs),
            example.value_target,
        ))
    if not compatible:
        raise ValueError("replay has no examples compatible with model limits")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        examples = list(compatible)
        random.shuffle(examples)
        optimizer.zero_grad(set_to_none=True)
        policy_total = 0.0
        value_total = 0.0
        for index, example in enumerate(examples, 1):
            state = torch.tensor(
                [example.state_ids],
                dtype=torch.long,
                device=device,
            )
            memory, memory_padding, value = model.encode(state)
            if example.value_target > 0:
                action_log_probabilities: list[torch.Tensor] = []
                for ids in example.action_ids:
                    action = torch.tensor(
                        [ids],
                        dtype=torch.long,
                        device=device,
                    )
                    logits = model.decode(
                        action[:, :-1],
                        memory,
                        memory_padding,
                    )
                    target = action[:, 1:]
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs.gather(
                        -1, target.unsqueeze(-1)
                    ).squeeze(-1)
                    action_log_probabilities.append(selected.mean())
                action_scores = torch.stack(action_log_probabilities)
                target_policy = torch.tensor(
                    example.policy_target,
                    dtype=torch.float32,
                    device=device,
                )
                policy_loss = -(target_policy * action_scores).sum()
            else:
                # A failed single-player search has a useful value target but
                # no trustworthy policy target: its visit distribution merely
                # describes where an unsuccessful search spent time.
                policy_loss = value.sum() * 0.0
            value_target = torch.tensor(
                [example.value_target],
                dtype=torch.float32,
                device=device,
            )
            value_loss = F.mse_loss(value, value_target)
            loss = (
                policy_loss + cfg.value_loss_weight * value_loss
            ) / cfg.gradient_accumulation
            loss.backward()
            policy_total += float(policy_loss.item())
            value_total += float(value_loss.item())
            if (
                index % cfg.gradient_accumulation == 0
                or index == len(examples)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.gradient_clip,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        history.append({
            "epoch": epoch,
            "policy_loss": policy_total / len(examples),
            "value_loss": value_total / len(examples),
        })
    model.save_checkpoint(
        output_checkpoint,
        optimizer_state=optimizer.state_dict(),
        metadata={
            **payload.get("metadata", {}),
            "reinforcement": {
                "configuration": asdict(cfg),
                "replay_examples": len(compatible),
                "skipped_incompatible": skipped,
                "history": history,
            },
        },
    )
    return {
        "device": str(device),
        "replay_examples": len(compatible),
        "skipped_incompatible": skipped,
        "history": history,
        "checkpoint": str(Path(output_checkpoint).resolve()),
    }
