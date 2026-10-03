"""A bounded, train-only library of independently replayed proof certificates.

Retrieval indexes encode complete statements in one batch.  They never
recursively embed proof dependencies, and require an explicit encoder
fingerprint plus the exact indexed library version on every neural query.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping

from metamath_generator.model import Node, Theorem
from .kernel import (
    ProofKernel, canonical_json, content_hash, statement_fingerprint,
    statement_text, theorem_from_data, theorem_to_data,
)


class LibraryError(ValueError):
    pass


class LibraryBudgetError(LibraryError):
    pass


class LibraryLeakageError(LibraryError):
    pass


class StaleIndexError(LibraryError):
    pass


@dataclass(frozen=True)
class LibraryRecord:
    id: str
    statement_fingerprint: str
    content_hash: str
    certificate_json: str
    provenance_json: str
    importance: float = 1.0
    reuse_count: int = 0
    step: int = 0

    @property
    def theorem(self) -> Theorem:
        # Callers cannot mutate an already verified library record in place.
        return theorem_from_data(json.loads(self.certificate_json))

    @property
    def provenance(self) -> dict:
        return json.loads(self.provenance_json)

    @property
    def statement(self) -> str:
        return statement_text(self.theorem)

    @property
    def proof_length(self) -> int:
        return len(self.theorem.proof.source_labels)

    @property
    def proof_length_saving(self) -> int:
        theorem = self.theorem
        return max(0, self.proof_length - len(theorem.hypotheses) - len(theorem.floating) - 1)

    def to_data(self) -> dict:
        data = {
            "id": self.id, "statement_fingerprint": self.statement_fingerprint,
            "content_hash": self.content_hash, "certificate": json.loads(self.certificate_json),
            "provenance": self.provenance, "importance": self.importance,
            "reuse_count": self.reuse_count, "step": self.step,
        }
        data["record_hash"] = content_hash(data)
        return data


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9-]*|[=<>+*|/-]+", text))


def _vectors(values, expected_rows: int) -> list[list[float]]:
    # A CPU torch encoder is optional; lexical retrieval imports no torch.
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    rows = [[float(value) for value in row] for row in values]
    if len(rows) != expected_rows:
        raise LibraryError("encoder returned the wrong number of vectors")
    width = len(rows[0]) if rows else 0
    if rows and (not width or any(len(row) != width for row in rows)):
        raise LibraryError("encoder vectors have inconsistent dimensions")
    normalized = []
    for row in rows:
        if any(not math.isfinite(value) for value in row):
            raise LibraryError("encoder returned non-finite values")
        norm = math.sqrt(sum(value * value for value in row))
        normalized.append([value / norm if norm else 0.0 for value in row])
    return normalized


class TheoremLibrary:
    FORMAT = "pa-prover-v2-library-v2"

    def __init__(
        self, kernel: ProofKernel | str | Path, *, max_items: int = 128,
        max_bytes: int = 8_000_000, exclude_fingerprints: Iterable[str] = (),
        max_step: int | None = None,
    ):
        self.kernel = kernel if isinstance(kernel, ProofKernel) else ProofKernel(kernel)
        if max_items <= 0 or max_bytes <= 0:
            raise LibraryBudgetError("library budgets must be positive")
        self.max_items, self.max_bytes = int(max_items), int(max_bytes)
        self.exclude_fingerprints = frozenset(exclude_fingerprints)
        self.max_step = max_step
        self._records: dict[str, LibraryRecord] = {}
        self._index: dict | None = None
        self._index_stale = False
        if self.bytes_used > self.max_bytes:
            raise LibraryBudgetError("byte budget cannot hold library metadata")

    def __len__(self) -> int:
        return len(self._records)

    @property
    def theory_fingerprint(self) -> str:
        return self.kernel.theory_fingerprint

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    @property
    def records(self) -> tuple[LibraryRecord, ...]:
        return tuple(self._records.values())

    @property
    def statements(self) -> tuple[Theorem, ...]:
        return tuple(record.theorem for record in self._records.values())

    @property
    def version(self) -> str:
        return content_hash(sorted(self._records))

    @property
    def indexed_version(self) -> str | None:
        return self._index.get("library_version") if self._index else None

    def _payload(self, records=None, index=...):
        records = self._records if records is None else records
        return {
            "format": self.FORMAT, "theory_fingerprint": self.theory_fingerprint,
            "max_items": self.max_items, "max_bytes": self.max_bytes,
            "exclude_fingerprints": sorted(self.exclude_fingerprints), "max_step": self.max_step,
            "records": [record.to_data() for record in records.values()],
            "index": self._index if index is ... else index,
        }

    @property
    def bytes_used(self) -> int:
        return len(canonical_json(self._payload()).encode("utf-8"))

    def _invalidate_index(self):
        if self._index is not None:
            self._index_stale = True
        self._index = None

    def get(self, record_id: str) -> Theorem:
        try:
            return self._records[record_id].theorem
        except KeyError as exc:
            raise LibraryError(f"unknown library record {record_id}") from exc

    def _check_provenance(self, provenance, fingerprint, step):
        provenance = {"split": provenance} if isinstance(provenance, str) else dict(provenance)
        if provenance.get("split") != "train":
            raise LibraryLeakageError("only train-provenance certificates may enter the library")
        if fingerprint in self.exclude_fingerprints:
            raise LibraryLeakageError("certificate is excluded as a held-out/target statement")
        if step < 0 or (self.max_step is not None and step > self.max_step):
            raise LibraryLeakageError("certificate comes from an unavailable future step")
        return provenance

    def add(self, theorem: Theorem, *, provenance="train", importance=1.0, reuse_count=0, step=0) -> str:
        self.kernel.verify(theorem)
        fingerprint = statement_fingerprint(theorem)
        provenance = self._check_provenance(provenance, fingerprint, int(step))
        if not math.isfinite(float(importance)) or importance < 0 or reuse_count < 0:
            raise LibraryError("importance and reuse scores must be finite and nonnegative")
        certificate = theorem_to_data(theorem)
        digest = content_hash(certificate)
        record = LibraryRecord(
            digest, fingerprint, digest, canonical_json(certificate), canonical_json(provenance),
            float(importance), int(reuse_count), int(step),
        )
        if len(canonical_json(self._payload({digest: record}, index=None)).encode("utf-8")) > self.max_bytes:
            raise LibraryBudgetError("one certificate exceeds the complete library byte budget")
        old = self._records.get(digest)
        if old is not None:
            # Re-observing a proof cannot rewrite its original visibility time.
            record = replace(record, step=min(old.step, record.step), reuse_count=max(old.reuse_count, record.reuse_count))
        prior_version = self.version
        self._records[digest] = record
        self.select()
        if self.version != prior_version:
            self._invalidate_index()
        return digest

    @staticmethod
    def _utility(record):
        return record.importance + math.log1p(record.reuse_count) + 0.1 * math.log1p(record.proof_length_saving)

    def select(self) -> tuple[str, ...]:
        """Greedy value/diversity selection under count AND serialized-byte caps."""
        before = self.version
        candidates = list(self._records.values())
        selected: dict[str, LibraryRecord] = {}
        selected_terms: list[set[str]] = []
        signatures = {record.id: _terms(record.statement) for record in candidates}
        while candidates and len(selected) < self.max_items:
            def score(record):
                terms = signatures[record.id]
                overlap = max((len(terms & other) / max(1, len(terms | other)) for other in selected_terms), default=0.0)
                return self._utility(record) + 0.25 * (1.0 - overlap), record.id
            record = max(candidates, key=score)
            candidates.remove(record)
            trial = {**selected, record.id: record}
            if len(canonical_json(self._payload(trial, index=None)).encode("utf-8")) > self.max_bytes:
                continue
            selected = trial
            selected_terms.append(signatures[record.id])
        self._records = selected
        if self.version != before:
            self._invalidate_index()
        # A score update may change encoded size even with the same identities.
        if self.bytes_used > self.max_bytes:
            self._invalidate_index()
        return self.record_ids

    def update_scores(self, scores: Mapping[str, Mapping | float]) -> tuple[str, ...]:
        for record_id, values in scores.items():
            if record_id not in self._records:
                raise LibraryError(f"unknown library record {record_id}")
            values = {"importance": values} if isinstance(values, (float, int)) else dict(values)
            if set(values) - {"importance", "reuse_count"}:
                raise LibraryError("only importance and reuse_count are mutable scores")
            record = self._records[record_id]
            importance = float(values.get("importance", record.importance))
            reuse = int(values.get("reuse_count", record.reuse_count))
            if not math.isfinite(importance) or importance < 0 or reuse < 0:
                raise LibraryError("scores must be finite and nonnegative")
            self._records[record_id] = replace(record, importance=importance, reuse_count=reuse)
        return self.select()

    def build_index(self, encoder: Callable, encoder_fingerprint: str) -> dict:
        if not encoder_fingerprint:
            raise StaleIndexError("an explicit encoder fingerprint is required")
        ids = list(self.record_ids)
        # Intentionally one flat batch: statement encoding cannot recurse
        # through mutually referring or self-referential proof metadata.
        vectors = _vectors(encoder([self._records[key].statement for key in ids]), len(ids)) if ids else []
        index = {"encoder_fingerprint": str(encoder_fingerprint), "library_version": self.version,
                 "record_ids": ids, "vectors": vectors}
        if len(canonical_json(self._payload(index=index)).encode("utf-8")) > self.max_bytes:
            raise LibraryBudgetError("embedding index would exceed the library byte budget")
        self._index = index
        self._index_stale = False
        return {key: value for key, value in index.items() if key != "vectors"}

    def search(
        self, query: str | Node | Theorem, limit: int = 8, *,
        exclude_fingerprints: Iterable[str] = (), max_step: int | None = None,
        encoder: Callable | None = None, encoder_fingerprint: str | None = None,
    ) -> list[LibraryRecord]:
        if limit <= 0:
            return []
        text = statement_text(query) if isinstance(query, Theorem) else query.to_prefix() if isinstance(query, Node) else str(query)
        forbidden = self.exclude_fingerprints | frozenset(exclude_fingerprints)
        visible_step = self.max_step if max_step is None else max_step if self.max_step is None else min(self.max_step, max_step)
        candidates = [record for record in self._records.values()
                      if record.statement_fingerprint not in forbidden
                      and (visible_step is None or record.step <= visible_step)]
        if encoder is None:
            if encoder_fingerprint is not None:
                raise StaleIndexError("neural retrieval needs the matching encoder, not only its fingerprint")
            terms = _terms(text)
            scores = {record.id: len(terms & _terms(record.statement)) / max(1, len(terms | _terms(record.statement))) for record in candidates}
        else:
            index = self._index
            if index is None or not encoder_fingerprint or index["encoder_fingerprint"] != encoder_fingerprint or index["library_version"] != self.version:
                raise StaleIndexError("missing or stale retrieval index; rebuild with the current encoder and library")
            vector = _vectors(encoder([text]), 1)[0]
            if index["vectors"] and len(vector) != len(index["vectors"][0]):
                raise StaleIndexError("query encoder dimension differs from indexed vectors")
            scores = {key: sum(a * b for a, b in zip(vector, row)) for key, row in zip(index["record_ids"], index["vectors"])}
        return sorted(candidates, key=lambda record: (scores.get(record.id, -math.inf), self._utility(record), record.id), reverse=True)[:limit]

    def save(self, destination: str | Path) -> None:
        payload = canonical_json(self._payload())
        if len(self._records) > self.max_items or len(payload.encode("utf-8")) > self.max_bytes:
            raise LibraryBudgetError("refusing to persist an over-budget library")
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(
        cls, source: str | Path, kernel: ProofKernel | str | Path, *,
        max_items: int | None = None, max_bytes: int | None = None,
        exclude_fingerprints: Iterable[str] = (), max_step: int | None = None,
    ) -> "TheoremLibrary":
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        if data.get("format") != cls.FORMAT:
            raise LibraryError("unsupported theorem library format")
        stored_step = data.get("max_step")
        visible_step = stored_step if max_step is None else max_step if stored_step is None else min(stored_step, max_step)
        library = cls(kernel, max_items=data["max_items"] if max_items is None else max_items,
                      max_bytes=data["max_bytes"] if max_bytes is None else max_bytes,
                      exclude_fingerprints=set(data.get("exclude_fingerprints", ())) | set(exclude_fingerprints),
                      max_step=visible_step)
        if data["theory_fingerprint"] != library.theory_fingerprint:
            raise LibraryError("library was certified under a different theory")
        for raw in data["records"]:
            record = dict(raw)
            claimed = record.pop("record_hash", None)
            if claimed != content_hash(record):
                raise LibraryError("library record metadata hash mismatch")
            cert = record["certificate"]
            if content_hash(cert) != record["content_hash"] or record["id"] != record["content_hash"]:
                raise LibraryError("certificate content hash mismatch")
            theorem = theorem_from_data(cert)
            library.kernel.verify(theorem)
            fingerprint = statement_fingerprint(theorem)
            if fingerprint != record["statement_fingerprint"]:
                raise LibraryError("statement fingerprint mismatch")
            provenance = library._check_provenance(record["provenance"], fingerprint, int(record["step"]))
            importance, reuse = float(record["importance"]), int(record["reuse_count"])
            if not math.isfinite(importance) or importance < 0 or reuse < 0 or record["id"] in library._records:
                raise LibraryError("invalid or duplicate library record")
            library._records[record["id"]] = LibraryRecord(
                record["id"], fingerprint, record["content_hash"], canonical_json(cert), canonical_json(provenance), importance, reuse, int(record["step"]),
            )
        index = data.get("index")
        if index is not None:
            if index.get("library_version") != library.version or set(index.get("record_ids", [])) != set(library.record_ids) or len(index["record_ids"]) != len(library):
                raise StaleIndexError("persisted index does not match the certified library")
            if not index.get("encoder_fingerprint"):
                raise StaleIndexError("persisted index has no encoder identity")
            _vectors(index["vectors"], len(library))
            library._index = index
        if len(library) > library.max_items or library.bytes_used > library.max_bytes:
            raise LibraryBudgetError("persisted library exceeds the requested hard budget")
        return library
