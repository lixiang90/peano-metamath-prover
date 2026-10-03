"""A frozen retrieval snapshot; live memory injection still uses current weights."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import torch

from .tokenizer import ByteTokenizer
from .model import ARCHITECTURE, CausalDecoder, LemmaDecoderLM
from .library import LibraryBudgetError, LibraryLeakageError


class _LemmaEncoderSnapshot(torch.nn.Module):
    """Only the lemma modules, allocated on CPU before copying live weights."""

    encode_lemmas = LemmaDecoderLM.encode_lemmas

    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.lemma_embedding = torch.nn.Embedding(self.config.vocab_size, self.config.d_model,
                                                   padding_idx=ByteTokenizer.pad_id, device="cpu")
        with torch.device("cpu"):
            self.lemma_decoder = CausalDecoder(self.config, self.config.lemma_layers)
        self.summary_slots = torch.nn.Parameter(torch.empty(
            self.config.lemma_slots, self.config.d_model, device="cpu",
        ))
        source = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                  if name.startswith("lemma_") or name == "summary_slots"}
        self.load_state_dict(source, strict=True)
        self.requires_grad_(False)
        self.eval()


class FrozenLemmaRetriever:
    """Freeze the index encoder explicitly, avoiding stale-vector comparisons.

    Refresh explicitly between training rounds. Only lemma modules are copied
    to CPU, avoiding a transient second main model on the training GPU. The
    view never recursively encodes dependencies or changes library truth.
    """

    def __init__(self, library, model):
        self.library = library
        self.refresh(model)

    def refresh(self, model=None):
        """Reindex library changes, optionally taking a new trained-weight snapshot.

        Without ``model``, the frozen encoder is unchanged. Passing the live
        model after training replaces the snapshot and its fingerprint. Query
        and indexed vectors always use the same exact snapshot.
        """
        if model is not None:
            # Constructing CPU modules consumes RNG; retrieval must not perturb
            # the caller's training/dropout RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.encoder = _LemmaEncoderSnapshot(model)
            identity = {"architecture": ARCHITECTURE, "config": asdict(model.config),
                        "tokenizer": ByteTokenizer().fingerprint()}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
            for name, tensor in self.encoder.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
            self.encoder_fingerprint = digest.hexdigest()
        return self.library.build_index(self.encode, self.encoder_fingerprint)

    def encode(self, statements):
        tokenizer = ByteTokenizer()
        self.encoder.eval()
        with torch.no_grad():
            # Bounded batches; pooling summary slots only serves retrieval.
            batches = []
            for start in range(0, len(statements), 16):
                ids = [tokenizer.encode(text, bos=True, eos=True) for text in statements[start:start + 16]]
                batches.append(self.encoder.encode_lemmas(ids).mean(dim=1))
            return torch.cat(batches) if batches else torch.empty((0, self.encoder.config.d_model))

    def search(self, query, limit=8, **kwargs):
        return self.library.search(query, limit, encoder=self.encode,
                                   encoder_fingerprint=self.encoder_fingerprint, **kwargs)

    def __getattr__(self, name):
        return getattr(self.library, name)

    def __len__(self):
        return len(self.library)


def incorporate_verified_rollouts(library, rollouts, *, step):
    """Online train-only accumulation after rollout groups, never during evaluation."""
    # Local import avoids a module-initialization cycle; reuse the exact reward
    # boundary rather than maintaining a weaker second definition of success.
    from .rlvr import verified_reward

    uses = {}
    skipped_budget, skipped_visibility = 0, 0
    before = set(library.record_ids)
    for result in rollouts:
        session = result.session
        for event in session.trace if session is not None else ():
            action = event["action"]
            if isinstance(action, dict) and action.get("op") == "USE" and event["observation"]["status"] in {"running", "success"}:
                key = action["args"].get("lemma_id")
                if key in library.record_ids:
                    uses[key] = uses.get(key, 0) + 1
        if verified_reward(result, library.kernel):
            try:
                library.add(session.certificate, provenance={"split": "train", "source": "rlvr_verified"},
                            importance=1.0, reuse_count=1, step=step)
            except LibraryBudgetError:
                skipped_budget += 1
            except LibraryLeakageError:
                skipped_visibility += 1
    records = {r.id: r for r in library.records}
    library.update_scores({key: {"reuse_count": records[key].reuse_count + count}
                           for key, count in uses.items() if key in records})
    return {"added": len(set(library.record_ids) - before), "library_count": len(library.record_ids),
            "bytes_used": library.bytes_used, "skipped_budget": skipped_budget,
            "skipped_visibility": skipped_visibility}
