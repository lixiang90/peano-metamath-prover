"""Reproducible generate / pretrain / SFT / RLVR / evaluate commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _emit(value, destination=None):
    text = json.dumps(value, indent=2, ensure_ascii=False)
    if destination:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(text, encoding="utf-8")
    print(text)


def _load_model(path, kernel, device):
    from .model import load_checkpoint
    model, metadata = load_checkpoint(path, device=device)
    if metadata.get("theory_sha256") != kernel.theory_fingerprint:
        raise ValueError("checkpoint has no matching PA+ theory fingerprint")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="random PA+ proofs, certified Agent episodes and a train-only library")
    gen.add_argument("database")
    gen.add_argument("output")
    gen.add_argument("--seeds", default="7,11,19")
    gen.add_argument("--steps-per-seed", type=int, default=200)
    gen.add_argument("--max-examples", type=int, default=120)
    gen.add_argument("--library-items", type=int, default=32)
    gen.add_argument("--library-bytes", type=int, default=2_000_000)
    gen.add_argument("--axiom-instances-per-seed", type=int, default=8)
    for mode in ("pretrain", "sft"):
        train = sub.add_parser(mode)
        train.add_argument("database")
        train.add_argument("corpus")
        train.add_argument("output")
        train.add_argument("--checkpoint")
        train.add_argument("--steps", type=int, default=100)
        train.add_argument("--batch-size", type=int, default=2)
        train.add_argument("--learning-rate", type=float, default=3e-4)
        train.add_argument("--d-model", type=int, default=128)
        train.add_argument("--layers", type=int, default=4)
        train.add_argument("--heads", type=int, default=4)
        train.add_argument("--lemma-layers", type=int, default=2)
        train.add_argument("--lemma-slots", type=int, default=2)
        train.add_argument("--context", type=int, default=4096)
        train.add_argument("--device", default="cpu")
        train.add_argument("--seed", type=int, default=7)
        train.add_argument("--threads", type=int, default=1)
    rl = sub.add_parser("rlvr")
    ev = sub.add_parser("evaluate")
    for cmd in (rl, ev):
        cmd.add_argument("database")
        cmd.add_argument("corpus")
        cmd.add_argument("checkpoint")
        cmd.add_argument("output")
        cmd.add_argument("--library", help="defaults to corpus/library.json")
        cmd.add_argument("--max-steps", type=int, default=24)
        cmd.add_argument("--max-candidates", type=int, default=16)
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--seed", type=int, default=7)
        cmd.add_argument("--threads", type=int, default=1)
        cmd.add_argument("--require-external", action="store_true")
    rl.add_argument("--groups", type=int, default=8)
    rl.add_argument("--group-size", type=int, default=4)
    rl.add_argument("--learning-rate", type=float, default=1e-5)
    rl.add_argument("--update-epochs", type=int, default=1)
    ev.add_argument("--split", choices=("validation", "test"), default="test")
    ev.add_argument("--limit", type=int, default=10)
    ev.add_argument("--samples", type=int, default=1)
    ev.add_argument("--neural-retrieval", action="store_true")
    audit = sub.add_parser("audit", help="replay all teacher episodes; optionally cross-check official Metamath")
    audit.add_argument("database")
    audit.add_argument("corpus")
    audit.add_argument("--require-external", action="store_true")
    audit.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "generate":
        from .data import DataConfig, generate_corpus
        _emit(generate_corpus(args.database, args.output, DataConfig(
            seeds=tuple(int(s) for s in args.seeds.split(",")), steps_per_seed=args.steps_per_seed,
            max_examples=args.max_examples, library_items=args.library_items, library_bytes=args.library_bytes,
            axiom_instances_per_seed=args.axiom_instances_per_seed)))
        return
    from .kernel import ProofKernel, theorem_from_data
    from .library import TheoremLibrary
    from .data import load_episodes
    kernel = ProofKernel(args.database)
    library = TheoremLibrary.load(getattr(args, "library", None) or Path(args.corpus) / "library.json", kernel)
    if args.command == "audit":
        from .agent import AgentSession, AgentAction, AgentBudget
        counts, external = {}, 0
        groups, statements, skeletons = {}, {}, {}
        for split in ("train", "validation", "test"):
            episodes = load_episodes(args.corpus, split, kernel=kernel)
            counts[split] = len(episodes)
            for episode in episodes:
                for table, key in ((groups, "split_group"), (statements, "statement_sha256"), (skeletons, "skeleton_sha256")):
                    digest = episode[key]
                    if digest in table and table[digest] != split:
                        raise ValueError("cross-split leakage")
                    table[digest] = split
                cert = theorem_from_data(episode["certificate"])
                actions = [AgentAction.from_dict(a) for a in episode["actions"]]
                session = AgentSession(kernel, cert, library, AgentBudget(max_steps=max(128, len(actions) + 8), required_external=args.require_external))
                for action in actions:
                    if session.status != "running":
                        raise ValueError("teacher actions continue after terminal episode")
                    session.step(action)
                if session.status != "success":
                    raise ValueError("uncertified corpus episode")
                external += int(session.certification["external_verified"])
        _emit({"counts": counts, "independently_replayed": sum(counts.values()), "external_verified": external,
               "theory_sha256": kernel.theory_fingerprint, "library_count": len(library)}, args.output)
        return
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.command in {"pretrain", "sft"}:
        from .model import LemmaDecoderLM, ModelConfig
        from .training import TrainConfig, build_training_samples, train_language_model
        model = _load_model(args.checkpoint, kernel, args.device) if args.checkpoint else LemmaDecoderLM(ModelConfig(
            d_model=args.d_model, n_heads=args.heads, n_layers=args.layers, lemma_layers=args.lemma_layers,
            lemma_slots=args.lemma_slots, max_seq_len=args.context))
        episodes = load_episodes(args.corpus, kernel=kernel)
        mode = "ntp" if args.command == "pretrain" else "sft"
        samples, stats = build_training_samples(episodes, kernel, library, model.config, mode)
        summary = train_language_model(model, samples, args.output, TrainConfig(
            steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=args.seed, device=args.device), mode=mode,
            metadata={"theory_sha256": kernel.theory_fingerprint, "data_stats": stats})
        library.select()
        library.save(Path(args.output) / "library.json")
        _emit(summary)
        return
    from .inference import PolicyConfig, rollout, rollout_record
    model = _load_model(args.checkpoint, kernel, args.device)
    policy = PolicyConfig(max_candidates=args.max_candidates)
    if args.command == "rlvr":
        from .rlvr import RLVRConfig, train_rlvr
        _emit(train_rlvr(model, kernel, library, load_episodes(args.corpus, kernel=kernel), args.output,
                         RLVRConfig(groups=args.groups, group_size=args.group_size, update_epochs=args.update_epochs,
                                    learning_rate=args.learning_rate, seed=args.seed, max_steps=args.max_steps,
                                    require_external=args.require_external), policy))
        return
    from .agent import AgentBudget
    from .memory import FrozenLemmaRetriever
    from neural_prover.certificate import export_certificate
    if args.limit <= 0 or args.samples <= 0:
        raise ValueError("evaluation limit and samples must be positive")
    if args.neural_retrieval:
        library = FrozenLemmaRetriever(library, model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, solved = [], 0
    episodes = load_episodes(args.corpus, args.split, kernel=kernel)[:args.limit]
    if not episodes:
        raise ValueError("requested evaluation split is empty")
    for index, episode in enumerate(episodes):
        target = theorem_from_data(episode["certificate"])
        target.proof = None
        successes = 0
        for sample in range(args.samples):
            result = rollout(model, kernel, target, library, budget=AgentBudget(max_steps=args.max_steps, required_external=args.require_external),
                             policy=policy, seed=args.seed + index * args.samples + sample)
            records.append({"target_id": episode["id"], "sample": sample, **rollout_record(result)})
            if result.reward:
                successes += 1
                export_certificate(result.session.certificate, output / f"{episode['id']}-{sample}.mm", ambient_database=kernel.database)
        solved += int(successes > 0)
    (output / "rollouts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    _emit({"split": args.split, "targets": len(episodes), "solved": solved, "pass_at_samples": solved / len(episodes),
           "samples_per_target": args.samples, "max_steps": args.max_steps, "policy": vars(policy),
           "neural_retrieval": args.neural_retrieval, "require_external": args.require_external,
           "theory_sha256": kernel.theory_fingerprint}, output / "summary.json")
